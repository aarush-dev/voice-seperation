"""Reader/writer for scp files that point at individual ``.npy`` arrays.

A "npy.scp" file is a Kaldi-style 2-column text file mapping a key to the
path of a numpy array saved on disk, one entry per line::

    key1 /some/path/a.npy
    key2 /some/path/b.npy
    key3 /some/path/c.npy

``NpyScpWriter`` saves each assigned array as ``{outdir}/{key}.npy`` and
appends the corresponding line to the scp file; ``NpyScpReader`` loads the
array back with ``np.load`` on lookup.
"""

import collections.abc
from pathlib import Path
from typing import Dict, Union

import numpy as np
from typeguard import check_argument_types

from espnet2.fileio.read_text import read_2columns_text, read_2columns_text_improved


class NpyScpWriter:
    """Writer class for a scp file of numpy file.

    Examples:
        key1 /some/path/a.npy
        key2 /some/path/b.npy
        key3 /some/path/c.npy
        key4 /some/path/d.npy
        ...

        >>> writer = NpyScpWriter('./data/', './data/feat.scp')
        >>> writer['aa'] = numpy_array
        >>> writer['bb'] = numpy_array

    """

    def __init__(
        self,
        outdir: Union[Path, str],
        scpfile: Union[Path, str],
        read_text_improved: bool = False,
    ):
        assert check_argument_types()
        self.dir = Path(outdir)
        self.dir.mkdir(parents=True, exist_ok=True)
        scpfile = Path(scpfile)
        scpfile.parent.mkdir(parents=True, exist_ok=True)
        self.fscp = scpfile.open("w", encoding="utf-8")

        self.data: Dict[str, str] = {}

    def get_path(self, key: str) -> str:
        """Return the on-disk ``.npy`` path previously written for ``key``."""
        return self.data[key]

    def __setitem__(self, key: str, value: np.ndarray):
        assert isinstance(value, np.ndarray), type(value)
        npy_path = self.dir / f"{key}.npy"
        npy_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(str(npy_path), value)
        self.fscp.write(f"{key} {npy_path}\n")

        # Store the file path
        self.data[key] = str(npy_path)

    def __enter__(self) -> "NpyScpWriter":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def close(self):
        self.fscp.close()


class NpyScpReader(collections.abc.Mapping):
    """Reader class for a scp file of numpy file.

    Examples:
        key1 /some/path/a.npy
        key2 /some/path/b.npy
        key3 /some/path/c.npy
        key4 /some/path/d.npy
        ...

        >>> reader = NpyScpReader('npy.scp')
        >>> array = reader['key1']

    """

    def __init__(self, fname: Union[Path, str], read_text_improved: bool = False):
        assert check_argument_types()
        self.fname = Path(fname)
        if read_text_improved:
            self.data = read_2columns_text_improved(fname)
        else:
            self.data = read_2columns_text(fname)

    def get_path(self, key: str) -> str:
        """Return the on-disk ``.npy`` path recorded for ``key``."""
        return self.data[key]

    def __getitem__(self, key: str) -> np.ndarray:
        npy_path = self.data[key]
        return np.load(npy_path)

    def __contains__(self, item):
        return item

    def __len__(self):
        return len(self.data)

    def __iter__(self):
        return iter(self.data)

    def keys(self):
        return self.data.keys()
