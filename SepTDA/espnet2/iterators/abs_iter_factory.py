"""Abstract base class for iterator factories.

An iterator factory is a lightweight, picklable object that knows how to
build a fresh iterator (typically wrapping a PyTorch ``DataLoader``) for a
given training epoch. The training loop (see ``espnet2/train/trainer.py``)
asks the factory for a new iterator at the start of every epoch instead of
holding on to a single iterator/DataLoader instance. This allows epoch-
dependent behavior such as reproducible per-epoch shuffling and, for the
chunk-based factory, resampling of chunk boundaries.

Concrete implementations (``SequenceIterFactory``, ``ChunkIterFactory``,
``MultipleIterFactory``) live alongside this module and are instantiated by
name from task configuration, so their public interface must stay stable.
"""

from abc import ABC, abstractmethod
from typing import Iterator, Optional


class AbsIterFactory(ABC):
    """Interface for classes that build a new data iterator per epoch."""

    @abstractmethod
    def build_iter(self, epoch: int, shuffle: Optional[bool] = None) -> Iterator:
        """Build an iterator for the given epoch.

        Args:
            epoch: The current epoch number (typically 1-indexed), used to
                seed any epoch-dependent shuffling.
            shuffle: Whether to shuffle the data. If ``None``, implementations
                fall back to their own configured default.

        Returns:
            An iterator yielding batches for the epoch.
        """
        raise NotImplementedError
