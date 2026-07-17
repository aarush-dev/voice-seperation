"""Batch sampler with a variable batch size based on a length "fold factor".

Unlike :class:`~espnet2.samplers.sorted_batch_sampler.SortedBatchSampler`
(fixed batch size, global sort) or
:class:`~espnet2.samplers.length_batch_sampler.LengthBatchSampler` /
:class:`~espnet2.samplers.num_elements_batch_sampler.NumElementsBatchSampler`
(batch size chosen to hit a target number of bins), ``FoldedBatchSampler``
shrinks the batch size as the longest sample in the current chunk grows,
following ``batch_size // (1 + L // fold_length)``. This keeps
(batch_size x length) roughly bounded without requiring an explicit bin
budget, which is convenient when memory usage scales with both the number
of sequences and their length (e.g. RNN/Transformer training).
"""

from typing import Dict, Iterator, List, Sequence, Tuple, Union

from typeguard import check_argument_types

from espnet2.fileio.read_text import load_num_sequence_text, read_2columns_text
from espnet2.samplers.abs_sampler import AbsSampler


def _has_speaker_duplicates(speaker_ids: List[str]) -> bool:
    """Return True if the same speaker id appears more than once in a mixture."""
    if len(speaker_ids) <= 1:
        return False
    return len(speaker_ids) != len(set(speaker_ids))


class FoldedBatchSampler(AbsSampler):
    """Batch sampler that "folds" the batch size down for longer samples.

    Samples are sorted by length (from ``shape_files[0]``) and then split
    into consecutive chunks. Within each chunk, the mini-batch size is
    computed from the longest sample in that chunk and the corresponding
    ``fold_lengths`` entry, so that batches of long sequences are smaller
    than batches of short sequences:

        factor = max(length_i // fold_length_i for each shape file i)
        batch_size_for_chunk = max(min_batch_size, batch_size // (1 + factor))

    See the module docstring for how this compares to the sibling
    samplers (sorted, length, numel, unsorted).
    """

    def __init__(
        self,
        batch_size: int,
        shape_files: Union[Tuple[str, ...], List[str]],
        fold_lengths: Sequence[int],
        min_batch_size: int = 1,
        sort_in_batch: str = "descending",
        sort_batch: str = "ascending",
        drop_last: bool = False,
        utt2category_file: str = None,
        remove_samples_with_speaker_overlap: bool = False,
    ):
        assert check_argument_types()
        assert batch_size > 0
        if sort_batch != "ascending" and sort_batch != "descending":
            raise ValueError(
                f"sort_batch must be ascending or descending: {sort_batch}"
            )
        if sort_in_batch != "descending" and sort_in_batch != "ascending":
            raise ValueError(
                f"sort_in_batch must be ascending or descending: {sort_in_batch}"
            )

        self.batch_size = batch_size
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
        self._check_keys_match(shape_files, utt2shapes, first_utt2shape)

        # Sort samples in ascending order
        # (shape order should be like (Length, Dim))
        keys = sorted(first_utt2shape, key=lambda k: first_utt2shape[k][0])

        if remove_samples_with_speaker_overlap:
            keys = self._drop_speaker_overlap(keys)

        if len(keys) == 0:
            raise RuntimeError(f"0 lines found: {shape_files[0]}")

        category2utt = self._group_by_category(
            keys, utt2category_file, first_utt2shape, shape_files
        )

        self.batch_list = []
        for category_keys in category2utt.values():
            batch_sizes = self._decide_batch_sizes(
                category_keys, utt2shapes, fold_lengths, batch_size, min_batch_size
            )
            self._redistribute_undersized_last_batch(batch_sizes, min_batch_size)

            if not self.drop_last:
                # Bug check
                assert sum(batch_sizes) == len(
                    category_keys
                ), f"{sum(batch_sizes)} != {len(category_keys)}"

            cur_batch_list = self._make_minibatches(
                category_keys, batch_sizes, sort_in_batch
            )

            if sort_batch == "ascending":
                pass
            elif sort_batch == "descending":
                cur_batch_list.reverse()
            else:
                raise ValueError(
                    f"sort_batch must be ascending or descending: {sort_batch}"
                )
            self.batch_list.extend(cur_batch_list)

    @staticmethod
    def _check_keys_match(
        shape_files: Union[Tuple[str, ...], List[str]],
        utt2shapes: List[Dict[str, List[int]]],
        first_utt2shape: Dict[str, List[int]],
    ) -> None:
        """Ensure every shape file describes exactly the same set of keys."""
        for shape_file, utt2shape in zip(shape_files, utt2shapes):
            if set(utt2shape) != set(first_utt2shape):
                raise RuntimeError(
                    f"keys are mismatched between {shape_file} != {shape_files[0]}"
                )

    @staticmethod
    def _drop_speaker_overlap(keys: List[str]) -> List[str]:
        """Filter out mixtures whose speaker ids repeat.

        The number of speakers in a mixture is encoded as the first
        character of its key, and the following underscore-separated
        fields (up to that count) are the speaker ids.
        """
        filtered_keys = []
        for key in keys:
            speaker_ids = key.split("_")[1 : (int(key[0]) + 1)]
            if not _has_speaker_duplicates(speaker_ids):
                filtered_keys.append(key)
        return filtered_keys

    @staticmethod
    def _group_by_category(
        keys: List[str],
        utt2category_file: str,
        first_utt2shape: Dict[str, List[int]],
        shape_files: Union[Tuple[str, ...], List[str]],
    ) -> Dict[str, List[str]]:
        """Group keys by category, or a single "default_category" bucket."""
        category2utt: Dict[str, List[str]] = {}
        if utt2category_file is not None:
            utt2category = read_2columns_text(utt2category_file)
            if set(utt2category) != set(first_utt2shape):
                raise RuntimeError(
                    "keys are mismatched between "
                    f"{utt2category_file} != {shape_files[0]}"
                )
            for key in keys:
                category2utt.setdefault(utt2category[key], []).append(key)
        else:
            category2utt["default_category"] = keys
        return category2utt

    def _decide_batch_sizes(
        self,
        category_keys: List[str],
        utt2shapes: List[Dict[str, List[int]]],
        fold_lengths: Sequence[int],
        batch_size: int,
        min_batch_size: int,
    ) -> List[int]:
        """Walk through ``category_keys`` deciding each chunk's batch size.

        The fold factor is recomputed at the start of each chunk from the
        (already length-sorted) key at the current offset, so batch size
        shrinks as sample length grows.
        """
        start = 0
        batch_sizes: List[int] = []
        while True:
            key = category_keys[start]
            factor = max(
                int(shape[key][0] / fold_length)
                for shape, fold_length in zip(utt2shapes, fold_lengths)
            )
            bs = max(min_batch_size, int(batch_size / (1 + factor)))
            if self.drop_last and start + bs > len(category_keys):
                # This if-block avoids 0-batches
                if len(self.batch_list) > 0:
                    break

            bs = min(len(category_keys) - start, bs)
            batch_sizes.append(bs)
            start += bs
            if start >= len(category_keys):
                break

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
                batch_sizes[-(i % len(batch_sizes)) - 2] += 1

    @staticmethod
    def _make_minibatches(
        category_keys: List[str], batch_sizes: List[int], sort_in_batch: str
    ) -> List[Tuple[str, ...]]:
        """Slice ``category_keys`` into tuples according to ``batch_sizes``."""
        cur_batch_list = []
        start = 0
        for bs in batch_sizes:
            assert len(category_keys) >= start + bs, "Bug"
            minibatch_keys = category_keys[start : start + bs]
            start += bs
            if sort_in_batch == "descending":
                minibatch_keys.reverse()
            elif sort_in_batch == "ascending":
                # Key are already sorted in ascending
                pass
            else:
                raise ValueError(
                    "sort_in_batch must be ascending or "
                    f"descending: {sort_in_batch}"
                )
            cur_batch_list.append(tuple(minibatch_keys))
        return cur_batch_list

    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"N-batch={len(self)}, "
            f"batch_size={self.batch_size}, "
            f"shape_files={self.shape_files}, "
            f"sort_in_batch={self.sort_in_batch}, "
            f"sort_batch={self.sort_batch})"
        )

    def __len__(self) -> int:
        return len(self.batch_list)

    def __iter__(self) -> Iterator[Tuple[str, ...]]:
        return iter(self.batch_list)
