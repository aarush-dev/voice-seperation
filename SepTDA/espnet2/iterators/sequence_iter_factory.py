"""Standard per-epoch PyTorch ``DataLoader`` factory.

``SequenceIterFactory`` is the default iterator factory used by the training
loop: it wraps a batch sampler and dataset and produces a fresh
``torch.utils.data.DataLoader`` for each epoch, with epoch-dependent
reproducible shuffling and optional truncation/extension of the number of
iterations per epoch. ``ChunkIterFactory`` builds on top of this class to
draw one sample at a time before re-chunking it.
"""

import random
from functools import partial
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np
from torch.utils.data import DataLoader
from typeguard import check_argument_types

from espnet2.iterators.abs_iter_factory import AbsIterFactory
from espnet2.samplers.abs_sampler import AbsSampler


def worker_init_fn(worker_id: int, base_seed: int = 0) -> None:
    """Set random seed for each worker in DataLoader."""
    seed = base_seed + worker_id
    random.seed(seed)
    np.random.seed(seed)


class RawSampler(AbsSampler):
    """Wraps a plain sequence of batches so it satisfies the ``AbsSampler`` API."""

    def __init__(self, batches: Sequence[Sequence[Any]]) -> None:
        self.batches = batches

    def __len__(self) -> int:
        return len(self.batches)

    def __iter__(self):
        return iter(self.batches)

    def generate(self, seed: int) -> List[Sequence[Any]]:
        return list(self.batches)


class SequenceIterFactory(AbsIterFactory):
    """Build iterator for each epoch.

    This class simply creates pytorch DataLoader except for the following points:
    - The random seed is decided according to the number of epochs. This feature
      guarantees reproducibility when resuming from middle of training process.
    - Enable to restrict the number of samples for one epoch. This features
      controls the interval number between training and evaluation.

    """

    def __init__(
        self,
        dataset,
        batches: Union[AbsSampler, Sequence[Sequence[Any]]],
        num_iters_per_epoch: Optional[int] = None,
        seed: int = 0,
        shuffle: bool = False,
        num_workers: int = 0,
        collate_fn=None,
        pin_memory: bool = False,
    ) -> None:
        assert check_argument_types()

        if not isinstance(batches, AbsSampler):
            self.sampler = RawSampler(batches)
        else:
            self.sampler = batches

        self.dataset = dataset
        self.num_iters_per_epoch = num_iters_per_epoch
        self.shuffle = shuffle
        self.seed = seed
        self.num_workers = num_workers
        self.collate_fn = collate_fn
        # https://discuss.pytorch.org/t/what-is-the-disadvantage-of-using-pin-memory/1702
        self.pin_memory = pin_memory

    def build_iter(self, epoch: int, shuffle: Optional[bool] = None) -> DataLoader:
        """Build a ``DataLoader`` for the given epoch.

        Args:
            epoch: The current epoch number, used to seed batch generation
                and shuffling so that resuming training reproduces the same
                sequence of batches.
            shuffle: Whether to shuffle batches. Falls back to
                ``self.shuffle`` when ``None``.

        Returns:
            A ``DataLoader`` whose batch sampler yields the batches selected
            for this epoch.
        """
        if shuffle is None:
            shuffle = self.shuffle

        if self.num_iters_per_epoch is not None:
            batches = self._generate_fixed_size_batches(epoch, shuffle)
        else:
            batches = self.sampler.generate(epoch + self.seed)
            if shuffle:
                np.random.RandomState(epoch + self.seed).shuffle(batches)

        # For backward compatibility for pytorch DataLoader
        if self.collate_fn is not None:
            kwargs = dict(collate_fn=self.collate_fn)
        else:
            kwargs = {}

        return DataLoader(
            dataset=self.dataset,
            batch_sampler=batches,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            worker_init_fn=partial(worker_init_fn, base_seed=epoch + self.seed),
            **kwargs,
        )

    def _generate_fixed_size_batches(
        self, epoch: int, shuffle: bool
    ) -> List[Sequence[Any]]:
        """Select exactly ``num_iters_per_epoch`` batches for this epoch.

        The sampler is treated as an infinite, epoch-seeded stream of
        batches: this epoch's window of ``num_iters_per_epoch`` batches is
        cut out of that stream starting right after the previous epoch's
        window ended, wrapping around (and reshuffling) whenever the stream
        of a single sampler-epoch is exhausted. This keeps consecutive
        epochs' batches disjoint and reproducible regardless of how corpus
        size relates to ``num_iters_per_epoch``.

        Args:
            epoch: The current epoch number.
            shuffle: Whether to shuffle each generated pool of batches.

        Returns:
            A list of exactly ``self.num_iters_per_epoch`` batches.
        """
        num_batches_total = len(self.sampler)
        if self.num_iters_per_epoch < num_batches_total:
            return self._select_window_from_stream(epoch, shuffle, num_batches_total)
        else:
            return self._accumulate_across_sampler_epochs(
                epoch, shuffle, num_batches_total
            )

    def _select_window_from_stream(
        self, epoch: int, shuffle: bool, num_batches_total: int
    ) -> List[Sequence[Any]]:
        """Cut this epoch's batch window out of the sampler-epoch stream.

        Used when the corpus (``num_batches_total``) is larger than
        ``num_iters_per_epoch``, so the window normally fits within a single
        sampler-epoch's batches, but may straddle the boundary between two
        consecutive sampler-epochs.
        """
        real_epoch, offset = divmod(
            self.num_iters_per_epoch * epoch, num_batches_total
        )

        if offset >= self.num_iters_per_epoch:
            current_batches = self.sampler.generate(real_epoch + self.seed)
            if shuffle:
                np.random.RandomState(real_epoch + self.seed).shuffle(
                    current_batches
                )
            batches = current_batches[offset - self.num_iters_per_epoch : offset]
        else:
            prev_batches = self.sampler.generate(real_epoch - 1 + self.seed)
            current_batches = self.sampler.generate(real_epoch + self.seed)
            if shuffle:
                np.random.RandomState(real_epoch - 1 + self.seed).shuffle(
                    prev_batches
                )
                np.random.RandomState(real_epoch + self.seed).shuffle(
                    current_batches
                )
            batches = (
                prev_batches[offset - self.num_iters_per_epoch :]
                + current_batches[:offset]
            )
        return batches

    def _accumulate_across_sampler_epochs(
        self, epoch: int, shuffle: bool, num_batches_total: int
    ) -> List[Sequence[Any]]:
        """Accumulate batches over multiple sampler-epochs to fill the window.

        Used when ``num_iters_per_epoch`` exceeds the corpus size, so a
        single sampler-epoch does not contain enough batches and the window
        must draw from as many consecutive (re-generated) sampler-epochs as
        needed.
        """
        current_epoch, cursor = divmod(
            self.num_iters_per_epoch * (epoch - 1), num_batches_total
        )
        remaining = self.num_iters_per_epoch
        batches: List[Sequence[Any]] = []
        current_batches = self.sampler.generate(current_epoch + self.seed)
        if shuffle:
            np.random.RandomState(current_epoch + self.seed).shuffle(current_batches)
        while remaining > 0:
            selected = current_batches[cursor : cursor + remaining]
            batches += selected
            if cursor + remaining >= num_batches_total:
                current_epoch += 1
                cursor = 0
                current_batches = self.sampler.generate(current_epoch + self.seed)
                if shuffle:
                    np.random.RandomState(current_epoch + self.seed).shuffle(
                        current_batches
                    )
            else:
                cursor = cursor + remaining
            remaining -= len(selected)

        assert len(batches) == self.num_iters_per_epoch
        return batches
