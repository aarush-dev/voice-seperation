"""Reader/writer for ``wav.scp``-style scp files pointing at audio files.

``wav.scp`` is a Kaldi-style scp file mapping an utterance id to one (or,
with ``multi_columns=True``, several space-separated) audio file path(s)::

    key1 /some/path/a.wav
    key2 /some/path/b.wav
    key3 /some/path/c.wav

With ``multi_columns=True``::

    key1 /some/path/a.wav /some/path/a2.wav
    key2 /some/path/b.wav /some/path/b2.wav

Each key's files are read with ``soundfile`` and concatenated along the
channel axis. The "Improved" reader/writer variants use a tab as the
column delimiter instead of whitespace (see :mod:`espnet2.fileio.read_text`).
"""

import collections.abc
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import soundfile
from typeguard import typechecked

from espnet2.fileio.read_text import (
    read_2columns_text,
    read_2columns_text_improved,
    read_multi_columns_text,
    read_multi_columns_text_improved,
)


def soundfile_read(
    wavs: Union[str, List[str]],
    dtype=None,
    always_2d: bool = False,
    concat_axis: int = 1,
    start: int = 0,
    end: int = None,
    return_subtype: bool = False,
) -> Tuple[np.array, int]:
    """Read one or more audio files with ``soundfile`` and concatenate them.

    Args:
        wavs: A single audio file path, or a list of paths to concatenate
            along ``concat_axis``. All files must share the same sample rate
            and matching size along the non-concatenated axis.
        dtype: Target array dtype. ``"float16"`` is emulated by reading as
            ``float32`` and casting down, since soundfile has no native
            float16 support.
        always_2d: Forwarded to ``soundfile.SoundFile.read``.
        concat_axis: Axis along which multiple files are concatenated
            (default 1, i.e. the channel axis for ``(Time, Channel)`` data).
        start: First frame to read (applied to every file).
        end: One-past-the-last frame to read, or ``None`` to read to the end.
        return_subtype: If True, also return each file's soundfile subtype.

    Returns:
        ``(array, samplerate)``, or ``(array, samplerate, subtypes)`` if
        ``return_subtype`` is True.
    """
    if isinstance(wavs, str):
        wavs = [wavs]

    arrays = []
    subtypes = []
    prev_rate = None
    prev_wav = None
    for wav_path in wavs:
        with soundfile.SoundFile(wav_path) as sound_file:
            sound_file.seek(start)
            if end is not None:
                num_frames = end - start
            else:
                num_frames = -1
            if dtype == "float16":
                array = sound_file.read(
                    num_frames,
                    dtype="float32",
                    always_2d=always_2d,
                ).astype(dtype)
            else:
                array = sound_file.read(num_frames, dtype=dtype, always_2d=always_2d)
            rate = sound_file.samplerate
            subtype = sound_file.subtype
            subtypes.append(subtype)

        if len(wavs) > 1 and array.ndim == 1 and concat_axis == 1:
            # array: (Time, Channel)
            array = array[:, None]

        if prev_wav is not None:
            if prev_rate != rate:
                raise RuntimeError(
                    f"'{prev_wav}' and '{wav_path}' have mismatched sampling rate: "
                    f"{prev_rate} != {rate}"
                )

            dim1 = arrays[0].shape[1 - concat_axis]
            dim2 = array.shape[1 - concat_axis]
            if dim1 != dim2:
                raise RuntimeError(
                    "Shapes must match with "
                    f"{1 - concat_axis} axis, but gut {dim1} and {dim2}"
                )

        prev_rate = rate
        prev_wav = wav_path
        arrays.append(array)

    if len(arrays) == 1:
        array = arrays[0]
    else:
        array = np.concatenate(arrays, axis=concat_axis)

    if return_subtype:
        return array, rate, subtypes
    else:
        return array, rate


def _normalize_rate_and_signal(
    value: Union[Tuple[int, np.ndarray], Tuple[np.ndarray, int]]
) -> Tuple[int, np.ndarray]:
    """Validate and unpack a ``writer[key] = value`` assignment.

    ``value`` must be a 2-tuple containing exactly one ``int`` (the sample
    rate) and one ``np.ndarray`` (the signal), in either order. The signal
    is returned as a 2D ``(Time, Channel)`` array.
    """
    value = list(value)
    if len(value) != 2:
        raise ValueError(f"Expecting 2 elements, but got {len(value)}")
    if isinstance(value[0], int) and isinstance(value[1], np.ndarray):
        rate, signal = value
    elif isinstance(value[1], int) and isinstance(value[0], np.ndarray):
        signal, rate = value
    else:
        raise TypeError("value shoulbe be a tuple of int and numpy.ndarray")

    if signal.ndim not in (1, 2):
        raise RuntimeError(f"Input signal must be 1 or 2 dimension: {signal.ndim}")
    if signal.ndim == 1:
        signal = signal[:, None]
    return rate, signal


class SoundScpReader(collections.abc.Mapping):
    """Reader class for 'wav.scp'.

    Examples:
        wav.scp is a text file that looks like the following:

        key1 /some/path/a.wav
        key2 /some/path/b.wav
        key3 /some/path/c.wav
        key4 /some/path/d.wav
        ...

        >>> reader = SoundScpReader('wav.scp')
        >>> rate, array = reader['key1']

        If multi_columns=True is given and
        multiple files are given in one line
        with space delimiter, and  the output array are concatenated
        along channel direction

        key1 /some/path/a.wav /some/path/a2.wav
        key2 /some/path/b.wav /some/path/b2.wav
        ...

        >>> reader = SoundScpReader('wav.scp', multi_columns=True)
        >>> rate, array = reader['key1']

        In the above case, a.wav and a2.wav are concatenated.

        Note that even if multi_columns=True is given,
        SoundScpReader still supports a normal wav.scp,
        i.e., a wav file is given per line,
        but this option is disable by default
        because dict[str, list[str]] object is needed to be kept,
        but it increases the required amount of memory.
    """

    @typechecked
    def __init__(
        self,
        fname,
        dtype=None,
        always_2d: bool = False,
        multi_columns: bool = False,
        concat_axis=1,
    ):
        self.fname = fname
        self.dtype = dtype
        self.always_2d = always_2d

        if multi_columns:
            self.data, _ = read_multi_columns_text(fname)
        else:
            self.data = read_2columns_text(fname)
        self.multi_columns = multi_columns
        self.concat_axis = concat_axis

    def __getitem__(self, key) -> Tuple[int, np.ndarray]:
        wavs = self.data[key]

        array, rate = soundfile_read(
            wavs,
            dtype=self.dtype,
            always_2d=self.always_2d,
            concat_axis=self.concat_axis,
        )
        # Returned as scipy.io.wavread's order
        return rate, array

    def get_path(self, key):
        return self.data[key]

    def __contains__(self, item):
        return item

    def __len__(self):
        return len(self.data)

    def __iter__(self):
        return iter(self.data)

    def keys(self):
        return self.data.keys()


class SoundScpWriter:
    """Writer class for 'wav.scp'

    Args:
        outdir:
        scpfile:
        format: The output audio format
        multi_columns: Save multi channel data
            as multiple monaural audio files
        output_name_format: The naming formam of generated audio files
        output_name_format_multi_columns: The naming formam of generated audio files
            when multi_columns is given
        dtype:
        subtype:

    Examples:
        >>> writer = SoundScpWriter('./data/', './data/wav.scp')
        >>> writer['aa'] = 16000, numpy_array
        >>> writer['bb'] = 16000, numpy_array

        aa ./data/aa.wav
        bb ./data/bb.wav

        >>> writer = SoundScpWriter(
            './data/', './data/feat.scp', multi_columns=True,
        )
        >>> numpy_array.shape
        (100, 2)
        >>> writer['aa'] = 16000, numpy_array

        aa ./data/aa-CH0.wav ./data/aa-CH1.wav

    """

    @typechecked
    def __init__(
        self,
        outdir: Union[Path, str],
        scpfile: Union[Path, str],
        format="wav",
        multi_columns: bool = False,
        output_name_format: str = "{key}.{audio_format}",
        output_name_format_multi_columns: str = "{key}-CH{channel}.{audio_format}",
        subtype: Optional[str] = None,
    ):
        self.dir = Path(outdir)
        self.dir.mkdir(parents=True, exist_ok=True)
        scpfile = Path(scpfile)
        scpfile.parent.mkdir(parents=True, exist_ok=True)
        self.fscp = scpfile.open("w", encoding="utf-8")
        self.format = format
        self.subtype = subtype
        self.output_name_format = output_name_format
        self.multi_columns = multi_columns
        self.output_name_format_multi_columns = output_name_format_multi_columns

        self.data: Dict[str, Union[str, List[str]]] = {}

    def __setitem__(
        self, key: str, value: Union[Tuple[int, np.ndarray], Tuple[np.ndarray, int]]
    ):
        rate, signal = _normalize_rate_and_signal(value)

        if signal.shape[1] > 1 and self.multi_columns:
            wav_paths = []
            for channel in range(signal.shape[1]):
                wav_path = self.dir / self.output_name_format_multi_columns.format(
                    key=key, audio_format=self.format, channel=channel
                )
                wav_path.parent.mkdir(parents=True, exist_ok=True)
                wav_path = str(wav_path)
                soundfile.write(
                    wav_path, signal[:, channel], rate, subtype=self.subtype
                )
                wav_paths.append(wav_path)

            self.fscp.write(f"{key} {' '.join(wav_paths)}\n")

            # Store the file path
            self.data[key] = wav_paths
        else:
            wav_path = self.dir / self.output_name_format.format(
                key=key, audio_format=self.format
            )
            wav_path.parent.mkdir(parents=True, exist_ok=True)
            wav_path = str(wav_path)
            soundfile.write(wav_path, signal, rate, subtype=self.subtype)
            self.fscp.write(f"{key} {wav_path}\n")

            # Store the file path
            self.data[key] = wav_path

    def get_path(self, key):
        return self.data[key]

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def close(self):
        self.fscp.close()


class SoundScpReaderImproved(collections.abc.Mapping):
    """Reader class for 'wav.scp'.

    Examples:
        wav.scp is a text file that looks like the following:

        key1 /some/path/a.wav
        key2 /some/path/b.wav
        key3 /some/path/c.wav
        key4 /some/path/d.wav
        ...

        >>> reader = SoundScpReader('wav.scp')
        >>> rate, array = reader['key1']

        If multi_columns=True is given and
        multiple files are given in one line
        with space delimiter, and  the output array are concatenated
        along channel direction

        key1 /some/path/a.wav /some/path/a2.wav
        key2 /some/path/b.wav /some/path/b2.wav
        ...

        >>> reader = SoundScpReader('wav.scp', multi_columns=True)
        >>> rate, array = reader['key1']

        In the above case, a.wav and a2.wav are concatenated.

        Note that even if multi_columns=True is given,
        SoundScpReader still supports a normal wav.scp,
        i.e., a wav file is given per line,
        but this option is disable by default
        because dict[str, list[str]] object is needed to be kept,
        but it increases the required amount of memory.
    """

    @typechecked
    def __init__(
        self,
        fname,
        dtype=None,
        always_2d: bool = False,
        multi_columns: bool = False,
        concat_axis=1,
    ):
        self.fname = fname
        self.dtype = dtype
        self.always_2d = always_2d

        if multi_columns:
            self.data, _ = read_multi_columns_text_improved(fname)
        else:
            self.data = read_2columns_text_improved(fname)
        self.multi_columns = multi_columns
        self.concat_axis = concat_axis

    def __getitem__(self, key) -> Tuple[int, np.ndarray]:
        wavs = self.data[key]

        array, rate = soundfile_read(
            wavs,
            dtype=self.dtype,
            always_2d=self.always_2d,
            concat_axis=self.concat_axis,
        )
        # Returned as scipy.io.wavread's order
        return rate, array

    def get_path(self, key):
        return self.data[key]

    def __contains__(self, item):
        return item

    def __len__(self):
        return len(self.data)

    def __iter__(self):
        return iter(self.data)

    def keys(self):
        return self.data.keys()


class SoundScpWriterImproved:
    """Writer class for 'wav.scp'

    Args:
        outdir:
        scpfile:
        format: The output audio format
        multi_columns: Save multi channel data
            as multiple monaural audio files
        output_name_format: The naming formam of generated audio files
        output_name_format_multi_columns: The naming formam of generated audio files
            when multi_columns is given
        dtype:
        subtype:

    Examples:
        >>> writer = SoundScpWriter('./data/', './data/wav.scp')
        >>> writer['aa'] = 16000, numpy_array
        >>> writer['bb'] = 16000, numpy_array

        aa ./data/aa.wav
        bb ./data/bb.wav

        >>> writer = SoundScpWriter(
            './data/', './data/feat.scp', multi_columns=True,
        )
        >>> numpy_array.shape
        (100, 2)
        >>> writer['aa'] = 16000, numpy_array

        aa ./data/aa-CH0.wav ./data/aa-CH1.wav

    """

    @typechecked
    def __init__(
        self,
        outdir: Union[Path, str],
        scpfile: Union[Path, str],
        format="wav",
        multi_columns: bool = False,
        output_name_format: str = "{key}.{audio_format}",
        output_name_format_multi_columns: str = "{key}-CH{channel}.{audio_format}",
        subtype: Optional[str] = None,
    ):
        self.dir = Path(outdir)
        self.dir.mkdir(parents=True, exist_ok=True)
        scpfile = Path(scpfile)
        scpfile.parent.mkdir(parents=True, exist_ok=True)
        self.fscp = scpfile.open("w", encoding="utf-8")
        self.format = format
        self.subtype = subtype
        self.output_name_format = output_name_format
        self.multi_columns = multi_columns
        self.output_name_format_multi_columns = output_name_format_multi_columns

        self.data: Dict[str, Union[str, List[str]]] = {}

    def __setitem__(
        self, key: str, value: Union[Tuple[int, np.ndarray], Tuple[np.ndarray, int]]
    ):
        rate, signal = _normalize_rate_and_signal(value)

        if signal.shape[1] > 1 and self.multi_columns:
            wav_paths = []
            for channel in range(signal.shape[1]):
                wav_path = self.dir / self.output_name_format_multi_columns.format(
                    key=key, audio_format=self.format, channel=channel
                )
                wav_path.parent.mkdir(parents=True, exist_ok=True)
                wav_path = str(wav_path)
                if self.format == "npy":
                    np.save(wav_path, signal[:, channel])
                else:
                    soundfile.write(
                        wav_path, signal[:, channel], rate, subtype=self.subtype
                    )
                wav_paths.append(wav_path)

            self.fscp.write(f"{key}" + "\t" + "\t".join(wav_paths) + "\n")

            # Store the file path
            self.data[key] = wav_paths
        else:
            wav_path = self.dir / self.output_name_format.format(
                key=key, audio_format=self.format
            )
            wav_path.parent.mkdir(parents=True, exist_ok=True)
            wav_path = str(wav_path)
            if self.format == "npy":
                np.save(wav_path, signal)
            else:
                soundfile.write(wav_path, signal, rate, subtype=self.subtype)
            self.fscp.write(f"{key}\t{wav_path}\n")

            # Store the file path
            self.data[key] = wav_path

    def get_path(self, key):
        return self.data[key]

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def close(self):
        self.fscp.close()
