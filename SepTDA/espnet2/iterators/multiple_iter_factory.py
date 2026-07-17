"""Iterator factory that chains together several underlying iter factories.

This is used when a single epoch's data is too large to build a sampler for
all at once (e.g. very large corpora split into shards). Each shard gets its
own build function that lazily constructs an ``AbsIterFactory``, and
``MultipleIterFactory`` builds and drains them one at a time, in an order
that can optionally be shuffled per epoch.
"""

import logging
from typing import Callable, Collection, Iterator, List, Optional

import numpy as np
from typeguard import check_argument_types

from espnet2.iterators.abs_iter_factory import AbsIterFactory


class MultipleIterFactory(AbsIterFactory):
    """Builds and concatenates the iterators of several sub-factories.

    Each entry in ``build_funcs`` is a zero-argument callable that, when
    invoked, constructs an ``AbsIterFactory`` (e.g. a ``SequenceIterFactory``
    or ``ChunkIterFactory``). ``build_iter`` builds each sub-factory in turn
    and yields all of its batches before moving to the next one, so callers
    see a single continuous stream of batches for the epoch.
    """

    def __init__(
        self,
        build_funcs: Collection[Callable[[], AbsIterFactory]],
        seed: int = 0,
        shuffle: bool = False,
    ) -> None:
        assert check_argument_types()
        self.build_funcs = list(build_funcs)
        self.seed = seed
        self.shuffle = shuffle

    def build_iter(self, epoch: int, shuffle: Optional[bool] = None) -> Iterator:
        """Build a chained iterator over all sub-factories for this epoch.

        Args:
            epoch: The current epoch number, used both to seed the ordering
                shuffle and forwarded to each sub-factory's ``build_iter``.
            shuffle: Whether to shuffle the order of sub-factories (and, by
                being forwarded, the batches within each one). Falls back to
                ``self.shuffle`` when ``None``.

        Yields:
            Batches from each sub-factory's iterator, in sequence.
        """
        if shuffle is None:
            shuffle = self.shuffle

        build_funcs: List[Callable[[], AbsIterFactory]] = list(self.build_funcs)

        if shuffle:
            np.random.RandomState(epoch + self.seed).shuffle(build_funcs)

        for index, build_func in enumerate(build_funcs):
            logging.info(f"Building {index}th iter-factory...")
            iter_factory = build_func()
            assert isinstance(iter_factory, AbsIterFactory), type(iter_factory)
            yield from iter_factory.build_iter(epoch, shuffle)
