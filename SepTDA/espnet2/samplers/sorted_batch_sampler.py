"""Batch sampler with a fixed batch size, globally sorted by length.

Unlike the variable-batch-size samplers (folded, length, numel),
``SortedBatchSampler`` always uses ``batch_size`` samples per mini-batch
(the last one or two may be slightly larger/smaller due to even splitting).
It differs from :class:`~espnet2.samplers.unsorted_batch_sampler.
UnsortedBatchSampler` in that samples are globally sorted by length before
being sliced into batches, which keeps sample lengths within a batch close
together and reduces padding waste.
"""

import logging
from typing import Dict, Iterator, List, Tuple

from typeguard import check_argument_types

from espnet2.fileio.read_text import load_num_sequence_text
from espnet2.samplers.abs_sampler import AbsSampler


class SortedBatchSampler(AbsSampler):
    """BatchSampler with sorted samples by length.

    All samples (keys from ``shape_file``) are sorted once by their first
    shape dimension (length), then evenly split into ``N = max(len(keys)
    // batch_size, 1)`` mini-batches, so every batch has a similar size
    close to ``batch_size``.

    Args:
        batch_size: Target number of samples per mini-batch.
        shape_file: Text file describing the length (and optionally
            further dimensions) of each sample, e.g. ``uttA 1000,80``.
        sort_in_batch: 'descending', 'ascending' or None.
        sort_batch:
    """

    def __init__(
        self,
        batch_size: int,
        shape_file: str,
        sort_in_batch: str = "descending",
        sort_batch: str = "ascending",
        drop_last: bool = False,
    ):
        assert check_argument_types()
        assert batch_size > 0
        self.batch_size = batch_size
        self.shape_file = shape_file
        self.sort_in_batch = sort_in_batch
        self.sort_batch = sort_batch
        self.drop_last = drop_last

        # utt2shape: (Length, ...)
        #    uttA 100,...
        #    uttB 201,...
        utt2shape = load_num_sequence_text(shape_file, loader_type="csv_int")
        keys = self._sort_keys(utt2shape, sort_in_batch)
        if len(keys) == 0:
            raise RuntimeError(f"0 lines found: {shape_file}")

        self.batch_list = self._split_into_batches(keys, batch_size)

        if len(self.batch_list) == 0:
            logging.warning(f"{shape_file} is empty")

        if sort_in_batch != sort_batch:
            if sort_batch not in ("ascending", "descending"):
                raise ValueError(
                    f"sort_batch must be ascending or descending: {sort_batch}"
                )
            self.batch_list.reverse()

        if len(self.batch_list) == 0:
            raise RuntimeError("0 batches")

    @staticmethod
    def _sort_keys(utt2shape: Dict[str, List[int]], sort_in_batch: str) -> List[str]:
        """Sort sample keys by length, ascending or descending."""
        if sort_in_batch == "descending":
            # Sort samples in descending order (required by RNN)
            return sorted(utt2shape, key=lambda k: -utt2shape[k][0])
        elif sort_in_batch == "ascending":
            # Sort samples in ascending order
            return sorted(utt2shape, key=lambda k: utt2shape[k][0])
        else:
            raise ValueError(
                f"sort_in_batch must be either one of "
                f"ascending, descending, or None: {sort_in_batch}"
            )

    def _split_into_batches(
        self, keys: List[str], batch_size: int
    ) -> List[Tuple[str, ...]]:
        """Split the sorted keys into evenly-sized mini-batches."""
        # Apply max(, 1) to avoid 0-batches
        n_batches = max(len(keys) // batch_size, 1)
        if not self.drop_last:
            # Split keys evenly as possible as. Note that If N != 1,
            # the these batches always have size of batch_size at minimum.
            return [
                keys[i * len(keys) // n_batches : (i + 1) * len(keys) // n_batches]
                for i in range(n_batches)
            ]
        else:
            return [
                tuple(keys[i * batch_size : (i + 1) * batch_size])
                for i in range(n_batches)
            ]

    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"N-batch={len(self)}, "
            f"batch_size={self.batch_size}, "
            f"shape_file={self.shape_file}, "
            f"sort_in_batch={self.sort_in_batch}, "
            f"sort_batch={self.sort_batch})"
        )

    def __len__(self) -> int:
        return len(self.batch_list)

    def __iter__(self) -> Iterator[Tuple[str, ...]]:
        return iter(self.batch_list)
