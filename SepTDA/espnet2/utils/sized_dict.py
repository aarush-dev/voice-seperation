"""A dict-like cache that tracks the approximate in-memory size of its contents.

`SizedDict` is used to cache decoded audio/features (e.g. in a Dataset)
while keeping an approximate running total of memory usage via
`get_size`, so callers can bound cache growth. Optionally backed by a
`multiprocessing.Manager` dict for sharing across DataLoader worker
processes.
"""

import collections
import sys
from typing import Any, Dict, Optional, Set

from torch import multiprocessing


def get_size(obj: Any, seen: Optional[Set[int]] = None) -> int:
    """Recursively finds size of objects

    Taken from https://github.com/bosswissam/pysize

    """

    size = sys.getsizeof(obj)
    if seen is None:
        seen = set()

    obj_id = id(obj)
    if obj_id in seen:
        return 0

    # Important mark as seen *before* entering recursion to gracefully handle
    # self-referential objects
    seen.add(obj_id)

    if isinstance(obj, dict):
        size += sum([get_size(v, seen) for v in obj.values()])
        size += sum([get_size(k, seen) for k in obj.keys()])
    elif hasattr(obj, "__dict__"):
        size += get_size(obj.__dict__, seen)
    elif isinstance(obj, (list, set, tuple)):
        size += sum([get_size(i, seen) for i in obj])

    return size


class SizedDict(collections.abc.MutableMapping):
    """A MutableMapping that maintains a running estimate of its total byte size.

    Behaves like a regular dict (via the `collections.abc.MutableMapping`
    mixin methods), but every `__setitem__`/`__delitem__` updates
    `self.size`, an approximate size in bytes of all stored keys/values
    as computed by `get_size`.
    """

    def __init__(self, shared: bool = False, data: Optional[Dict] = None):
        """Create the cache.

        Args:
            shared: If True, back the cache with a
                `multiprocessing.Manager` dict so it can be shared across
                worker processes.
            data: Initial key/value pairs to seed the cache with. Note
                these are not counted in `self.size` (only items set via
                `__setitem__` are tracked).
        """
        if data is None:
            data = {}

        if shared:
            # NOTE(kamo): Don't set manager as a field because Manager, which includes
            # weakref object, causes following error with method="spawn",
            # "TypeError: can't pickle weakref objects"
            self.cache = multiprocessing.Manager().dict(**data)
        else:
            self.manager = None
            self.cache = dict(**data)
        self.size = 0

    def __setitem__(self, key, value) -> None:
        """Store ``value`` under ``key``, updating the tracked total size."""
        if key in self.cache:
            self.size -= get_size(self.cache[key])
        else:
            self.size += sys.getsizeof(key)
        self.size += get_size(value)
        self.cache[key] = value

    def __getitem__(self, key):
        """Return the value stored under ``key``."""
        return self.cache[key]

    def __delitem__(self, key) -> None:
        """Remove ``key``, updating the tracked total size."""
        self.size -= get_size(self.cache[key])
        self.size -= sys.getsizeof(key)
        del self.cache[key]

    def __iter__(self):
        """Iterate over stored keys."""
        return iter(self.cache)

    def __contains__(self, key) -> bool:
        """Return whether ``key`` is stored in the cache."""
        return key in self.cache

    def __len__(self) -> int:
        """Return the number of stored items."""
        return len(self.cache)
