"""Batch sampler that targets a constant total length ("bins") per batch.

Compared to :class:`~espnet2.samplers.num_elements_batch_sampler.
NumElementsBatchSampler`, which sizes batches by total tensor element
count (length x feature dimensions), ``LengthBatchSampler`` sizes batches
purely by summed sequence length (optionally as-if padded to the batch's
max length), ignoring feature dimensionality. It differs from
:class:`~espnet2.samplers.folded_batch_sampler.FoldedBatchSampler`, which
derives batch size from a length/fold_length ratio instead of an explicit
bin budget, and from :class:`~espnet2.samplers.sorted_batch_sampler.
SortedBatchSampler` / :class:`~espnet2.samplers.unsorted_batch_sampler.
UnsortedBatchSampler`, which both use a fixed batch size.
"""

from typing import Dict, Iterator, List, Tuple, Union

from typeguard import check_argument_types

from espnet2.fileio.read_text import load_num_sequence_text
from espnet2.samplers.abs_sampler import AbsSampler


class LengthBatchSampler(AbsSampler):
    """Batch sampler with a variable batch size driven by a length budget.

    Samples are sorted by length (from ``shape_files[0]``) and then greedily
    accumulated into a mini-batch until the running bin count would exceed
    ``batch_bins``, at which point a new mini-batch starts. The bin count of
    a candidate batch is either:

    - ``padding=True``: ``batch_size * max_length`` per shape file (i.e. as
      if every sequence were padded to the longest one currently in the
      batch), summed over shape files; or
    - ``padding=False``: the sum of the individual sequence lengths.
    """

    def __init__(
        self,
        batch_bins: int,
        shape_files: Union[Tuple[str, ...], List[str]],
        min_batch_size: int = 1,
        sort_in_batch: str = "descending",
        sort_batch: str = "ascending",
        drop_last: bool = False,
        padding: bool = True,
    ):
        assert check_argument_types()
        assert batch_bins > 0
        if sort_batch != "ascending" and sort_batch != "descending":
            raise ValueError(
                f"sort_batch must be ascending or descending: {sort_batch}"
            )
        if sort_in_batch != "descending" and sort_in_batch != "ascending":
            raise ValueError(
                f"sort_in_batch must be ascending or descending: {sort_in_batch}"
            )

        self.batch_bins = batch_bins
        self.shape_files = shape_files
        self.sort_in_batch = sort_in_batch
        self.sort_batch = sort_batch
        self.drop_last = drop_last

        # utt2shape: (Length, ...)
        #    uttA 100,...
        #    uttB 201,...
        utt2shapes = [
            load_num_sequence_text(s, loader_type="csv_int") for s in shape_files
        ]

        first_utt2shape = utt2shapes[0]
        for shape_file, utt2shape in zip(shape_files, utt2shapes):
            if set(utt2shape) != set(first_utt2shape):
                raise RuntimeError(
                    f"keys are mismatched between {shape_file} != {shape_files[0]}"
                )

        # Sort samples in ascending order
        # (shape order should be like (Length, Dim))
        keys = sorted(first_utt2shape, key=lambda k: first_utt2shape[k][0])
        if len(keys) == 0:
            raise RuntimeError(f"0 lines found: {shape_files[0]}")

        batch_sizes = self._decide_batch_sizes(
            keys, utt2shapes, batch_bins, min_batch_size, padding
        )
        self._redistribute_undersized_last_batch(batch_sizes, min_batch_size)

        if not self.drop_last:
            # Bug check
            assert sum(batch_sizes) == len(keys), f"{sum(batch_sizes)} != {len(keys)}"

        self.batch_list = self._make_minibatches(keys, batch_sizes, sort_in_batch)

        if sort_batch == "ascending":
            pass
        elif sort_batch == "descending":
            self.batch_list.reverse()
        else:
            raise ValueError(
                f"sort_batch must be ascending or descending: {sort_batch}"
            )

    def _decide_batch_sizes(
        self,
        keys: List[str],
        utt2shapes: List[Dict[str, List[int]]],
        batch_bins: int,
        min_batch_size: int,
        padding: bool,
    ) -> List[int]:
        """Greedily accumulate keys into batches until the bin budget is hit."""
        batch_sizes: List[int] = []
        current_batch_keys: List[str] = []
        for key in keys:
            current_batch_keys.append(key)
            # shape: (Length, dim1, dim2, ...)
            if padding:
                # bins = bs x max_length
                bins = sum(
                    len(current_batch_keys) * shape[key][0] for shape in utt2shapes
                )
            else:
                # bins = sum of lengths
                bins = sum(
                    shape[k][0] for k in current_batch_keys for shape in utt2shapes
                )

            if bins > batch_bins and len(current_batch_keys) >= min_batch_size:
                batch_sizes.append(len(current_batch_keys))
                current_batch_keys = []
        else:
            if len(current_batch_keys) != 0 and (
                not self.drop_last or len(batch_sizes) == 0
            ):
                batch_sizes.append(len(current_batch_keys))

        if len(batch_sizes) == 0:
            # Maybe we can't reach here
            raise RuntimeError("0 batches")
        return batch_sizes

    @staticmethod
    def _redistribute_undersized_last_batch(
        batch_sizes: List[int], min_batch_size: int
    ) -> None:
        """If the trailing batch is too small, spread it over earlier batches."""
        if len(batch_sizes) > 1 and batch_sizes[-1] < min_batch_size:
            for i in range(batch_sizes.pop(-1)):
                batch_sizes[-(i % len(batch_sizes)) - 1] += 1

    @staticmethod
    def _make_minibatches(
        keys: List[str], batch_sizes: List[int], sort_in_batch: str
    ) -> List[Tuple[str, ...]]:
        """Slice ``keys`` into tuples according to ``batch_sizes``."""
        batch_list = []
        iter_bs = iter(batch_sizes)
        bs = next(iter_bs)
        minibatch_keys: List[str] = []
        for key in keys:
            minibatch_keys.append(key)
            if len(minibatch_keys) == bs:
                if sort_in_batch == "descending":
                    minibatch_keys.reverse()
                elif sort_in_batch == "ascending":
                    # Key are already sorted in ascending
                    pass
                else:
                    raise ValueError(
                        "sort_in_batch must be ascending"
                        f" or descending: {sort_in_batch}"
                    )
                batch_list.append(tuple(minibatch_keys))
                minibatch_keys = []
                try:
                    bs = next(iter_bs)
                except StopIteration:
                    break
        return batch_list

    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"N-batch={len(self)}, "
            f"batch_bins={self.batch_bins}, "
            f"sort_in_batch={self.sort_in_batch}, "
            f"sort_batch={self.sort_batch})"
        )

    def __len__(self) -> int:
        return len(self.batch_list)

    def __iter__(self) -> Iterator[Tuple[str, ...]]:
        return iter(self.batch_list)
