"""Reader/writer for ``vad.scp``: per-utterance voice-activity time spans.

Unlike ``segments`` (which slices a whole recording session into
utterances), ``vad.scp`` is utterance-scoped: each entry lists the
active-speech time spans (in seconds, ``start:end`` pairs, space
separated) within a single utterance. This is mainly used to guide
silence trimming for UASR::

    key1 0:1.2000
    key2 3.0000:4.5000 7.0000:9.0000
"""

import collections.abc
from pathlib import Path
from typing import List, Tuple, Union

import numpy as np
from typeguard import check_argument_types

from espnet2.fileio.read_text import read_2columns_text


class VADScpReader(collections.abc.Mapping):
    """Reader class for 'vad.scp'.

    Different from `segments`, the `vad.scp` would focus on utterance-level,
    while the `segments` are expected to focus on a whole session. The major
    usage in ESPnet is to guide the silence trim for UASR.

    Examples:
        key1 0:1.2000
        key2 3.0000:4.5000 7.0000:9:0000
        ...

        >>> reader = VADScpReader('wav.scp')
        >>> array = reader['key1']

    """

    def __init__(
        self,
        fname,
        dtype=np.float32,
    ):
        assert check_argument_types()
        self.fname = fname
        self.dtype = dtype
        self.data = read_2columns_text(fname)

    def __getitem__(self, key: str) -> List[Tuple[float, float]]:
        """Parse the ``start:end`` spans for ``key`` into a list of tuples."""
        spans = self.data[key]
        spans = spans.split(" ")
        vad_info = []
        for span in spans:
            start, end = span.split(":")
            vad_info.append((float(start), float(end)))
        return vad_info

    def __contains__(self, item):
        return item

    def __len__(self):
        return len(self.data)

    def __iter__(self):
        return iter(self.data)

    def keys(self):
        return self.data.keys()


class VADScpWriter:
    """Writer class for 'vad.scp'

    Examples:
        key1 0:1.2000
        key2 3.0000:4.5000 7.0000:9:0000
        ...

        >>> writer = VADScpWriter('./data/vad.scp')
        >>> writer['aa'] = list of tuples
        >>> writer['bb'] = list of tuples

    """

    def __init__(
        self,
        scpfile: Union[Path, str],
        dtype=None,
    ):
        assert check_argument_types()
        scpfile = Path(scpfile)
        scpfile.parent.mkdir(parents=True, exist_ok=True)
        self.fscp = scpfile.open("w", encoding="utf-8")
        self.dtype = dtype

        self.data = {}

    def __setitem__(self, key: str, value: List[Tuple[float, float]]):
        """Write the ``start:end`` spans for ``key`` as one space-separated line."""
        assert (
            key not in self.data.keys()
        ), "found duplicate key (key: {}) in your vad values".format(key)
        assert isinstance(value, List), type(value)

        span_strs = []
        for span in value:
            assert (
                len(span) == 2
            ), "each vad tuple should contains exact the start time and end time"
            span_strs.append("{.4f}:{}".format(span[0], span[1]))
        output_str = " ".join(span_strs)

        self.fscp.write(f"{key} {output_str}\n")

        # Store the file path
        self.data[key] = str(output_str)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def close(self):
        self.fscp.close()
