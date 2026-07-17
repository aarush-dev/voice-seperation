"""Writer for Kaldi-style data directories made of many key-value text files.

A Kaldi-style data directory is a folder containing several small text files
(e.g. ``wav.scp``, ``utt2spk``, ``text``), each mapping an utterance/speaker
id to some value, one entry per line::

    wav.scp:
        uttidA some/where/a.wav
        uttidB some/where/b.wav
    utt2spk:
        uttidA spk1
        uttidB spk2

``DatadirWriter`` builds directory trees of such files lazily: indexing the
writer with a filename creates (or reuses) a nested writer for that file,
and assigning to a key writes one line to it.
"""

import warnings
from pathlib import Path
from typing import Dict, Optional, Union

from typeguard import check_argument_types, check_return_type


class DatadirWriter:
    """Writer class to create a kaldi-like data directory.

    Indexing the writer with a filename returns a child ``DatadirWriter``
    scoped to that file (created under ``self.path`` on first access).
    Assigning a value to a string key on a "leaf" writer appends a line of
    the form ``"{key}{combine}{value}\\n"`` to the underlying file.

    Examples:
        >>> with DatadirWriter("output") as writer:
        ...     # output/sub.txt is created here
        ...     subwriter = writer["sub.txt"]
        ...     # Write "uttidA some/where/a.wav"
        ...     subwriter["uttidA"] = "some/where/a.wav"
        ...     subwriter["uttidB"] = "some/where/b.wav"

    """

    def __init__(self, p: Union[Path, str], combine: str = " "):
        assert check_argument_types()
        self.path = Path(p)
        self.children: Dict[str, "DatadirWriter"] = {}
        self.fd = None
        self.has_children = False
        self.keys = set()
        self.combine = combine

    def __enter__(self) -> "DatadirWriter":
        return self

    def __getitem__(self, key: str) -> "DatadirWriter":
        """Get (or lazily create) the child writer for the file named ``key``."""
        assert check_argument_types()
        if self.fd is not None:
            raise RuntimeError("This writer points out a file")

        if key not in self.children:
            child = DatadirWriter((self.path / key), combine=self.combine)
            self.children[key] = child
            self.has_children = True

        retval = self.children[key]
        assert check_return_type(retval)
        return retval

    def __setitem__(self, key: str, value: str):
        """Write one ``"{key}{combine}{value}"`` line to this writer's file."""
        assert check_argument_types()
        if self.has_children:
            raise RuntimeError("This writer points out a directory")
        if key in self.keys:
            warnings.warn(f"Duplicated: {key}")

        if self.fd is None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.fd = self.path.open("w", encoding="utf-8")

        self.keys.add(key)
        self.fd.write(f"{key}{self.combine}{value}\n")

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def close(self):
        """Recursively close this writer and all of its children.

        While closing children, warns if two sibling files were not
        written with the same set of keys (a common data-prep mistake).
        """
        if self.has_children:
            prev_child: Optional["DatadirWriter"] = None
            for child in self.children.values():
                child.close()
                if prev_child is not None and prev_child.keys != child.keys:
                    warnings.warn(
                        f"Ids are mismatching between "
                        f"{prev_child.path} and {child.path}"
                    )
                prev_child = child

        elif self.fd is not None:
            self.fd.close()
