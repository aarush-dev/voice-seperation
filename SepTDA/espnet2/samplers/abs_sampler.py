"""Abstract base class for batch samplers used by the training data loaders.

A batch sampler yields tuples of sample keys (utterance ids) instead of
single indices, so that the ``torch.utils.data.DataLoader`` built on top of
it can fetch whole mini-batches at once. Concrete strategies (folded,
length, numel, sorted, unsorted) live in sibling modules in this package.
"""

from abc import ABC, abstractmethod
from typing import Iterator, Tuple

from torch.utils.data import Sampler


class AbsSampler(Sampler, ABC):
    """Base class for all batch samplers.

    Subclasses must precompute (or lazily determine) a sequence of
    mini-batches, where each mini-batch is a tuple of sample keys, and
    expose them via ``__len__`` and ``__iter__``.
    """

    @abstractmethod
    def __len__(self) -> int:
        """Return the number of mini-batches."""
        raise NotImplementedError

    @abstractmethod
    def __iter__(self) -> Iterator[Tuple[str, ...]]:
        """Iterate over mini-batches, each a tuple of sample keys."""
        raise NotImplementedError

    def generate(self, seed):
        """Materialize the sampler's mini-batches into a list.

        Args:
            seed: Unused by the base implementation; kept for interface
                compatibility with callers that seed batch generation.

        Returns:
            A list of mini-batches, each a tuple of sample keys.
        """
        return list(self)
