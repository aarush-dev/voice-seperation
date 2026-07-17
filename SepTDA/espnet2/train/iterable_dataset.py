"""Iterable dataset module.

:class:`IterableESPnetDataset` is a ``torch.utils.data.IterableDataset``
counterpart of :class:`espnet2.train.dataset.ESPnetDataset`: instead of
random-accessing each loader by key, it streams the configured scp-style
files line by line in lockstep (matching by utterance id), which avoids
loading a full index into memory and works with data too large to look up
by key efficiently. Loader types that can't be streamed (see
``DATA_TYPES`` here vs. ``espnet2.train.dataset.DATA_TYPES``) fall back to
an internal :class:`ESPnetDataset` keyed random-access lookup per utterance.
"""
import copy
from io import StringIO
from pathlib import Path
from typing import Callable, Collection, Dict, Iterator, List, Tuple, Union

import kaldiio
import numpy as np
import soundfile
import torch
from torch.utils.data.dataset import IterableDataset
from typeguard import check_argument_types

from espnet2.train.dataset import ESPnetDataset
from espnet2.fileio.msvs_json import read_msvs_json_from_yaml


def load_kaldi(input):
    """Load one Kaldi ark/matrix entry, returning a bare ndarray (rate stripped)."""
    retval = kaldiio.load_mat(input)
    if isinstance(retval, tuple):
        assert len(retval) == 2, len(retval)
        if isinstance(retval[0], int) and isinstance(retval[1], np.ndarray):
            # sound scp case
            rate, array = retval
        elif isinstance(retval[1], int) and isinstance(retval[0], np.ndarray):
            # Extended ark format case
            array, rate = retval
        else:
            raise RuntimeError(f"Unexpected type: {type(retval[0])}, {type(retval[1])}")

        # Multichannel wave fie
        # array: (NSample, Channel) or (Nsample)

    else:
        # Normal ark case
        assert isinstance(retval, np.ndarray), type(retval)
        array = retval
    return array


DATA_TYPES = {
    "sound": lambda x: soundfile.read(x)[0],
    "kaldi_ark": load_kaldi,
    "npy": np.load,
    "text_int": lambda x: np.loadtxt(
        StringIO(x), ndmin=1, dtype=np.long, delimiter=" "
    ),
    "csv_int": lambda x: np.loadtxt(StringIO(x), ndmin=1, dtype=np.long, delimiter=","),
    "text_float": lambda x: np.loadtxt(
        StringIO(x), ndmin=1, dtype=np.float32, delimiter=" "
    ),
    "csv_float": lambda x: np.loadtxt(
        StringIO(x), ndmin=1, dtype=np.float32, delimiter=","
    ),
    "text": lambda x: x,
}


class IterableESPnetDataset(IterableDataset):
    """Streaming Pytorch Dataset class for ESPNet.

    Each entry in ``path_name_type_list`` is ``(path, data_name, loader_type)``.
    Streamable ``loader_type``\\ s (see module-level ``DATA_TYPES``) are read
    line-by-line as this dataset is iterated; other types fall back to an
    internal :class:`espnet2.train.dataset.ESPnetDataset` looked up by key.
    With multiple DataLoader workers, utterance ids are sharded
    round-robin by worker id.

    Examples:
        >>> dataset = IterableESPnetDataset([('wav.scp', 'input', 'sound'),
        ...                                  ('token_int', 'output', 'text_int')],
        ...                                )
        >>> for uid, data in dataset:
        ...     data
        {'input': per_utt_array, 'output': per_utt_array}
    """

    def __init__(
        self,
        path_name_type_list: Collection[Tuple[str, str, str]],
        preprocess: Callable[
            [str, Dict[str, np.ndarray]], Dict[str, np.ndarray]
        ] = None,
        float_dtype: str = "float32",
        int_dtype: str = "long",
        key_file: str = None,
    ):
        assert check_argument_types()
        if len(path_name_type_list) == 0:
            raise ValueError(
                '1 or more elements are required for "path_name_type_list"'
            )

        path_name_type_list = copy.deepcopy(path_name_type_list)
        self.preprocess = preprocess

        self.float_dtype = float_dtype
        self.int_dtype = int_dtype
        self.key_file = key_file

        self.debug_info = {}
        non_iterable_list = []
        self.path_name_type_list = []

        for path, name, _type in path_name_type_list:
            if name in self.debug_info:
                raise RuntimeError(f'"{name}" is duplicated for data-key')
            self.debug_info[name] = path, _type
            if _type not in DATA_TYPES:
                non_iterable_list.append((path, name, _type))
            else:
                self.path_name_type_list.append((path, name, _type))

        if len(non_iterable_list) != 0:
            # Some types doesn't support iterable mode
            self.non_iterable_dataset = ESPnetDataset(
                path_name_type_list=non_iterable_list,
                preprocess=preprocess,
                float_dtype=float_dtype,
                int_dtype=int_dtype,
            )
        else:
            self.non_iterable_dataset = None

        if Path(Path(path_name_type_list[0][0]).parent, "utt2category").exists():
            self.apply_utt2category = True
        else:
            self.apply_utt2category = False

    def has_name(self, name) -> bool:
        """Whether this dataset has a data field called ``name``."""
        return name in self.debug_info

    def names(self) -> Tuple[str, ...]:
        """Return every data field name this dataset provides."""
        return tuple(self.debug_info)

    def __repr__(self):
        _mes = self.__class__.__name__
        _mes += "("
        for name, (path, _type) in self.debug_info.items():
            _mes += f'\n  {name}: {{"path": "{path}", "type": "{_type}"}}'
        _mes += f"\n  preprocess: {self.preprocess})"
        return _mes

    def _build_uid_iterator(self) -> Iterator[str]:
        """Return the ordered utterance-id iterator driving this dataset's iteration.

        Prefers an explicit ``key_file``; otherwise uses the first
        streamable file's keys; otherwise falls back to the non-iterable
        dataset's own key order.
        """
        if self.key_file is not None:
            if self.key_file.endswith("yaml"):
                return read_msvs_json_from_yaml(self.key_file)
            return (
                line.rstrip().split(maxsplit=1)[0]
                for line in open(self.key_file, encoding="utf-8")
            )
        elif len(self.path_name_type_list) != 0:
            return (
                line.rstrip().split(maxsplit=1)[0]
                for line in open(self.path_name_type_list[0][0], encoding="utf-8")
            )
        else:
            return iter(self.non_iterable_dataset)

    @staticmethod
    def _read_matching_line_values(
        files: List, uid: str, linenum_box: List[int]
    ) -> List[str]:
        """Advance ``files`` in lockstep until each yields a line keyed ``uid``.

        Lines belonging to uids not requested (e.g. owned by another
        DataLoader worker) are silently skipped over. ``linenum_box[0]`` is
        a shared line counter, used only for error messages.
        """
        while True:
            keys = []
            values = []
            for f in files:
                linenum_box[0] += 1
                try:
                    line = next(f)
                except StopIteration:
                    raise RuntimeError(f"{uid} is not found in the files")
                sps = line.rstrip().split(maxsplit=1)
                if len(sps) != 2:
                    raise RuntimeError(
                        f"This line doesn't include a space:"
                        f" {f}:L{linenum_box[0]}: {line})"
                    )
                key, value = sps
                keys.append(key)
                values.append(value)

            for k_idx, k in enumerate(keys):
                if k != keys[0]:
                    raise RuntimeError(
                        f"Keys are mismatched. Text files (idx={k_idx}) is "
                        f"not sorted or not having same keys at L{linenum_box[0]}"
                    )

            # If the key is matched, break the loop
            if len(keys) == 0 or keys[0] == uid:
                return values

    def _load_streamed_fields(self, values: List[str]) -> Dict[str, np.ndarray]:
        """Parse each streamed file's raw line-value via its ``DATA_TYPES`` loader."""
        data = {}
        for value, (path, name, _type) in zip(values, self.path_name_type_list):
            func = DATA_TYPES[_type]
            data[name] = func(value)
        return data

    def _cast_to_desired_dtype(
        self, data: Dict[str, np.ndarray]
    ) -> Dict[str, np.ndarray]:
        """Cast every (post-preprocessing) field to ``float_dtype``/``int_dtype``."""
        for name in data:
            value = data[name]
            if not isinstance(value, np.ndarray):
                raise RuntimeError(
                    f"All values must be converted to np.ndarray object "
                    f'by preprocessing, but "{name}" is still {type(value)}.'
                )

            # Cast to desired type
            if value.dtype.kind == "f":
                value = value.astype(self.float_dtype)
            elif value.dtype.kind == "i":
                value = value.astype(self.int_dtype)
            else:
                raise NotImplementedError(f"Not supported dtype: {value.dtype}")
            data[name] = value
        return data

    def __iter__(self) -> Iterator[Tuple[Union[str, int], Dict[str, np.ndarray]]]:
        """Stream ``(uid, data)`` pairs, sharding uids across DataLoader workers.

        Reads each configured scp-style file line-by-line in lockstep
        (matching by key), merges in any non-iterable-loader fields,
        applies the optional preprocessor, and casts to the configured
        dtypes.
        """
        uid_iter = self._build_uid_iterator()

        files = [open(lis[0], encoding="utf-8") for lis in self.path_name_type_list]

        worker_info = torch.utils.data.get_worker_info()

        linenum_box = [0]
        count = 0
        for count, uid in enumerate(uid_iter, 1):
            # If num_workers>=1, split keys
            if worker_info is not None:
                if (count - 1) % worker_info.num_workers != worker_info.id:
                    continue

            # 1. Read a line from each file
            values = self._read_matching_line_values(files, uid, linenum_box)

            # 2. Load the entry from each line and create a dict
            # 2.a. Load data streamingly
            data = self._load_streamed_fields(values)
            if self.non_iterable_dataset is not None:
                # 2.b. Load data from non-iterable dataset
                _, from_non_iterable = self.non_iterable_dataset[uid]
                data.update(from_non_iterable)

            # 3. [Option] Apply preprocessing
            #   e.g. espnet2.train.preprocessor:CommonPreprocessor
            if self.preprocess is not None:
                data = self.preprocess(uid, data)

            # 4. Force data-precision
            data = self._cast_to_desired_dtype(data)

            yield uid, data

        if count == 0:
            raise RuntimeError("No iteration")
