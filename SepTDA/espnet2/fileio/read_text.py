"""Readers for Kaldi-style key-value text files (scp/text/label files).

Most of these read a file with one entry per line, formatted as::

    <key><delimiter><value...>

where ``<delimiter>`` is either a single tab or run(s) of whitespace
depending on the reader, and ``<value...>`` is either a single free-form
string, or itself split into columns/numbers.
"""

import collections.abc
import logging
from mmap import mmap
from pathlib import Path
from random import randint
from typing import Dict, List, Optional, Set, Tuple, Union

from typeguard import typechecked


def _read_two_columns(
    path: Union[Path, str],
    keys_to_load: Optional[Set[Union[str, int]]],
    split_delimiter: Optional[str],
) -> Dict[str, str]:
    """Shared implementation for the tab- and whitespace-delimited 2-column readers.

    Each line is split into at most 2 fields using ``split_delimiter``
    (``None`` means "any run of whitespace", matching ``str.split()``).
    """
    if keys_to_load is not None:
        logging.info(
            f"keys_to_load is not None, only loading {len(keys_to_load)} keys "
            f"from {path}"
        )

    data: Dict[str, str] = {}
    with Path(path).open("r", encoding="utf-8") as f:
        for linenum, line in enumerate(f, 1):
            fields = line.rstrip().split(split_delimiter, maxsplit=1)
            if len(fields) == 1:
                key, value = fields[0], ""
            else:
                key, value = fields

            if keys_to_load is not None and key not in keys_to_load:
                continue

            if key in data:
                raise RuntimeError(f"{key} is duplicated ({path}:{linenum})")
            data[key] = value
    return data


def _read_multi_columns(
    path: Union[Path, str],
    return_unsplit: bool,
    split_delimiter: Optional[str],
) -> Tuple[Dict[str, List[str]], Optional[Dict[str, str]]]:
    """Shared implementation for the tab- and whitespace-delimited multi-column readers.

    The first field (up to ``split_delimiter``) is the key; the remainder of
    the line is whitespace-split into a list of column values.
    """
    data: Dict[str, List[str]] = {}
    unsplit_data: Optional[Dict[str, str]] = {} if return_unsplit else None

    with Path(path).open("r", encoding="utf-8") as f:
        for linenum, line in enumerate(f, 1):
            fields = line.rstrip().split(split_delimiter, maxsplit=1)
            if len(fields) == 1:
                key, value = fields[0], ""
            else:
                key, value = fields

            if key in data:
                raise RuntimeError(f"{key} is duplicated ({path}:{linenum})")

            data[key] = value.split() if value != "" else [""]
            if return_unsplit:
                unsplit_data[key] = value

    return data, unsplit_data


@typechecked
def read_2columns_text_improved(
    path: Union[Path, str],
    keys_to_load: Optional[Set[Union[str, int]]] = None,
) -> Dict[str, str]:
    """Read a tab-delimited text file having 2 columns as dict object.

    Only load the keys in keys_to_load if it is not None.

    Examples:
        wav.scp:
            key1	/some/path/a.wav
            key2	/some/path/b.wav

        >>> read_2columns_text_improved('wav.scp')
        {'key1': '/some/path/a.wav', 'key2': '/some/path/b.wav'}

    """
    return _read_two_columns(path, keys_to_load, split_delimiter="\t")


@typechecked
def read_multi_columns_text_improved(
    path: Union[Path, str], return_unsplit: bool = False
) -> Tuple[Dict[str, List[str]], Optional[Dict[str, str]]]:
    """Read a tab-delimited text file having 2 or more columns as dict object.

    Examples:
        wav.scp:
            key1	/some/path/a1.wav /some/path/a2.wav
            key2	/some/path/b1.wav /some/path/b2.wav  /some/path/b3.wav
            key3	/some/path/c1.wav
            ...

        >>> read_multi_columns_text_improved('wav.scp')
        {'key1': ['/some/path/a1.wav', '/some/path/a2.wav'],
         'key2': ['/some/path/b1.wav', '/some/path/b2.wav', '/some/path/b3.wav'],
         'key3': ['/some/path/c1.wav']}

    """
    return _read_multi_columns(path, return_unsplit, split_delimiter="\t")


@typechecked
def read_2columns_text(
    path: Union[Path, str],
    keys_to_load: Optional[Set[Union[str, int]]] = None,
) -> Dict[str, str]:
    """Read a text file having 2 columns as dict object.

    Only load the keys in keys_to_load if it is not None.

    Examples:
        wav.scp:
            key1 /some/path/a.wav
            key2 /some/path/b.wav

        >>> read_2columns_text('wav.scp')
        {'key1': '/some/path/a.wav', 'key2': '/some/path/b.wav'}

    """
    return _read_two_columns(path, keys_to_load, split_delimiter=None)


@typechecked
def read_multi_columns_text(
    path: Union[Path, str], return_unsplit: bool = False
) -> Tuple[Dict[str, List[str]], Optional[Dict[str, str]]]:
    """Read a text file having 2 or more columns as dict object.

    Examples:
        wav.scp:
            key1 /some/path/a1.wav /some/path/a2.wav
            key2 /some/path/b1.wav /some/path/b2.wav  /some/path/b3.wav
            key3 /some/path/c1.wav
            ...

        >>> read_multi_columns_text('wav.scp')
        {'key1': ['/some/path/a1.wav', '/some/path/a2.wav'],
         'key2': ['/some/path/b1.wav', '/some/path/b2.wav', '/some/path/b3.wav'],
         'key3': ['/some/path/c1.wav']}

    """
    return _read_multi_columns(path, return_unsplit, split_delimiter=None)


@typechecked
def load_num_sequence_text(
    path: Union[Path, str], loader_type: str = "csv_int"
) -> Dict[str, List[Union[float, int]]]:
    """Read a text file indicating sequences of number

    Examples:
        key1 1 2 3
        key2 34 5 6

        >>> d = load_num_sequence_text('text')
        >>> np.testing.assert_array_equal(d["key1"], np.array([1, 2, 3]))
    """
    if loader_type == "text_int":
        delimiter = " "
        dtype = int
    elif loader_type == "text_float":
        delimiter = " "
        dtype = float
    elif loader_type == "csv_int":
        delimiter = ","
        dtype = int
    elif loader_type == "csv_float":
        delimiter = ","
        dtype = float
    else:
        raise ValueError(f"Not supported loader_type={loader_type}")

    # path looks like:
    #   utta 1,0
    #   uttb 3,4,5
    # -> return {'utta': np.ndarray([1, 0]),
    #            'uttb': np.ndarray([3, 4, 5])}
    key2value = read_2columns_text(path)

    # Using for-loop instead of dict-comprehension for debuggability
    retval: Dict[str, List[Union[float, int]]] = {}
    for key, value in key2value.items():
        try:
            retval[key] = [dtype(i) for i in value.split(delimiter)]
        except TypeError:
            logging.error(
                f'Error happened with path="{path}", id="{key}", value="{value}"'
            )
            raise
    return retval


@typechecked
def read_label(
    path: Union[Path, str]
) -> Dict[str, List[List[Union[str, float, int]]]]:
    """Read a text file indicating sequences of number

    Examples:
        key1 start_time_1 end_time_1 phone_1 start_time_2 end_time_2 phone_2 ....\n
        key2 start_time_1 end_time_1 phone_1 \n

        >>> d = load_num_sequence_text('label')
        >>> np.testing.assert_array_equal(d["key1"], [0.1, 0.2, "啊"]))
    """
    label_file = open(path, "r", encoding="utf-8")

    retval: Dict[str, List[List[Union[str, float, int]]]] = {}
    for label_line in label_file.readlines():
        fields = label_line.strip().split()
        key = fields[0]
        phn_info = fields[1:]
        segments = []
        for i in range(len(phn_info) // 3):
            segments.append(
                [phn_info[i * 3], phn_info[i * 3 + 1], phn_info[i * 3 + 2]]
            )
        retval[key] = segments
    return retval


class RandomTextReader(collections.abc.Mapping):
    """Reader class for random access to text.

    Simple text reader for non-pair text data (for unsupervised ASR)
        Instead of loading the whole text into memory (often large for UASR),
        the reader consumes text which stores in byte-offset of each text file
        and randomly selected unpaired text from it for training using mmap.

    Examples:
        text
            text1line
            text2line
            text3line
        scp
            11
            00000000000000000010
            00000000110000000020
            00000000210000000030
        scp explanation
            (number of digits per int value)
            (text start at bytes 0 and end at bytes 10 (including "\n"))
            (text start at bytes 11 and end at bytes 20 (including "\n"))
            (text start at bytes 21 and end at bytes 30 (including "\n"))
    """

    @typechecked
    def __init__(
        self,
        text_and_scp: str,
    ):
        super().__init__()

        text_path, scp_path = text_and_scp.split("-")

        text_file = Path(text_path).open("r+b")
        scp_file = Path(scp_path).open("r+b")

        self.text_mm = mmap(text_file.fileno(), 0)
        self.scp_mm = mmap(scp_file.fileno(), 0)

        max_num_digits_line = self.scp_mm.readline()
        max_num_digits = int(max_num_digits_line)
        assert max_num_digits > 0

        self.first_line_offset = len(max_num_digits_line)
        self.max_num_digits = max_num_digits
        self.stride = 2 * max_num_digits + 1

        num_text_bytes = len(self.scp_mm) - len(max_num_digits_line)
        assert num_text_bytes % self.stride == 0
        num_lines = num_text_bytes // self.stride
        self.num_lines = num_lines

    def __getitem__(self, key) -> str:
        # choose random line from scp
        # the first line defines the max number of digits
        random_line_number = randint(0, self.num_lines - 1)

        # get the number of bytes of corresponding line in text
        scp_start_bytes = self.first_line_offset
        scp_start_bytes += random_line_number * self.stride
        scp_end_bytes = scp_start_bytes + self.stride - 1

        text_start_bytes = int(
            self.scp_mm[scp_start_bytes : scp_start_bytes + self.max_num_digits]
        )
        text_end_bytes = int(
            self.scp_mm[scp_start_bytes + self.max_num_digits : scp_end_bytes]
        )

        # retrieve text line
        text = self.text_mm[text_start_bytes:text_end_bytes].decode("utf-8")
        return text

    def __contains__(self, item):
        return True

    def __len__(self):
        return self.num_lines

    def __iter__(self):
        return None

    def keys(self):
        return None
