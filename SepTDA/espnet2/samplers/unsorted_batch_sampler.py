"""Batch sampler with a fixed batch size and no length-based ordering.

The simplest sampler in this package: it does not require or use any
length information, unlike :class:`~espnet2.samplers.sorted_batch_sampler.
SortedBatchSampler` (fixed size, globally sorted) or the variable-size
samplers (folded, length, numel). Samples keep whatever order they appear
in ``key_file``, split (optionally per-category) into fixed-size chunks.
This makes it convenient for decoding or non-sequence tasks such as
classification, where batch composition doesn't need to be length-aware.
"""

import logging
from typing import Dict, Iterator, List, Tuple

from typeguard import check_argument_types

from espnet2.fileio.msvs_json import read_msvs_json_from_yaml
from espnet2.fileio.read_text import read_2columns_text
from espnet2.samplers.abs_sampler import AbsSampler


class UnsortedBatchSampler(AbsSampler):
    """BatchSampler with constant batch-size.

    Any sorting is not done in this class,
    so no length information is required,
    This class is convenient for decoding mode,
    or not seq2seq learning e.g. classification.

    Args:
        batch_size:
        key_file:
    """

    def __init__(
        self,
        batch_size: int,
        key_file: str,
        drop_last: bool = False,
        utt2category_file: str = None,
    ):
        assert check_argument_types()
        assert batch_size > 0
        self.batch_size = batch_size
        self.key_file = key_file
        self.drop_last = drop_last

        # utt2shape:
        #    uttA <anything is o.k>
        #    uttB <anything is o.k>
        if key_file.endswith(".yaml"):
            utt2any = read_msvs_json_from_yaml(key_file)
        else:
            utt2any = read_2columns_text(key_file)
        if len(utt2any) == 0:
            logging.warning(f"{key_file} is empty")
        # In this case the, the first column in only used
        keys = list(utt2any)
        if len(keys) == 0:
            raise RuntimeError(f"0 lines found: {key_file}")

        category2utt = self._group_by_category(keys, utt2category_file, key_file)

        self.batch_list = []
        for category_keys in category2utt.values():
            self.batch_list.extend(
                self._split_into_batches(category_keys, keys, batch_size)
            )

    @staticmethod
    def _group_by_category(
        keys: List[str], utt2category_file: str, key_file: str
    ) -> Dict[str, List[str]]:
        """Group keys by category, or a single "default_category" bucket."""
        category2utt: Dict[str, List[str]] = {}
        if utt2category_file is not None:
            utt2category = read_2columns_text(utt2category_file)
            if set(utt2category) != set(keys):
                raise RuntimeError(
                    f"keys are mismatched between {utt2category_file} != {key_file}"
                )
            for key, category in utt2category.items():
                category2utt.setdefault(category, []).append(key)
        else:
            category2utt["default_category"] = keys
        return category2utt

    def _split_into_batches(
        self, category_keys: List[str], keys: List[str], batch_size: int
    ) -> List[Tuple[str, ...]]:
        """Split one category's keys into fixed-size mini-batches."""
        # Apply max(, 1) to avoid 0-batches
        n_batches = max(len(category_keys) // batch_size, 1)
        if not self.drop_last:
            # Split keys evenly as possible as. Note that If N != 1,
            # the these batches always have size of batch_size at minimum.
            return [
                category_keys[
                    i * len(keys) // n_batches : (i + 1) * len(keys) // n_batches
                ]
                for i in range(n_batches)
            ]
        else:
            return [
                tuple(category_keys[i * batch_size : (i + 1) * batch_size])
                for i in range(n_batches)
            ]

    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"N-batch={len(self)}, "
            f"batch_size={self.batch_size}, "
            f"key_file={self.key_file}, "
        )

    def __len__(self) -> int:
        return len(self.batch_list)

    def __iter__(self) -> Iterator[Tuple[str, ...]]:
        return iter(self.batch_list)
