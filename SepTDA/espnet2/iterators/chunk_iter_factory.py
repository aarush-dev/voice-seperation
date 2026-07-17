"""Iterator factory that re-chunks whole-utterance samples into fixed-width chunks.

Many separation/enhancement tasks train on fixed-length chunks cut out of
variable-length utterances rather than on whole utterances. ``ChunkIterFactory``
wraps a per-sample ``SequenceIterFactory`` (batch size 1) so it can inspect
each utterance's length, slice it into one or more chunks of a (possibly
randomly chosen) width, and re-batch those chunks into fixed-size mini-batches
of ``batch_size`` before handing them to the training loop.
"""

import logging
import re
from collections import defaultdict
from typing import (
    Any,
    Dict,
    Iterator,
    List,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import numpy as np
import torch
from typeguard import check_argument_types

from espnet2.iterators.abs_iter_factory import AbsIterFactory
from espnet2.iterators.sequence_iter_factory import SequenceIterFactory
from espnet2.samplers.abs_sampler import AbsSampler

# Per-(category, chunk-width) caches of pending chunks/ids awaiting batching.
ChunkCache = Dict[str, List[torch.Tensor]]


class ChunkIterFactory(AbsIterFactory):
    """Creates chunks from a sequence

    Examples:
        >>> batches = [["id1"], ["id2"], ...]
        >>> batch_size = 128
        >>> chunk_length = 1000
        >>> iter_factory = ChunkIterFactory(dataset, batches, batch_size, chunk_length)
        >>> it = iter_factory.build_iter(epoch)
        >>> for ids, batch in it:
        ...     ...

    - The number of mini-batches are varied in each epochs and
      we can't get the number in advance
      because IterFactory doesn't be given to the length information.
    - Since the first reason, "num_iters_per_epoch" can't be implemented
      for this iterator. Instead of it, "num_samples_per_epoch" is implemented.

    Chunking strategy:
        Samples are drawn one at a time (via an internal per-sample
        ``SequenceIterFactory``). For each sample, a chunk width ``W`` is
        drawn randomly from ``chunk_lengths`` (candidates shorter than the
        sample), and the sample is split into overlapping chunks of width
        ``W`` with stride ``S = W * chunk_shift_ratio``, optionally starting
        at a random offset when ``shuffle`` is enabled. Chunks are cached
        per ``(utt2category, W)`` bucket, and whenever a bucket accumulates
        more than ``num_cache_chunks`` entries, it is drained into
        fixed-size mini-batches of ``batch_size``. Any remaining partial
        buckets are flushed at the end of the epoch. Marginal frames that
        don't fill a full chunk are discarded.

    """

    def __init__(
        self,
        dataset,
        batch_size: int,
        batches: Union[AbsSampler, Sequence[Sequence[Any]]],
        chunk_length: Union[int, str],
        chunk_shift_ratio: float = 0.5,
        num_cache_chunks: int = 1024,
        num_samples_per_epoch: int = None,
        seed: int = 0,
        shuffle: bool = False,
        num_workers: int = 0,
        collate_fn=None,
        pin_memory: bool = False,
        excluded_key_prefixes: Optional[List[str]] = None,
    ) -> None:
        assert check_argument_types()
        assert all(len(x) == 1 for x in batches), "batch-size must be 1"

        self.per_sample_iter_factory = SequenceIterFactory(
            dataset=dataset,
            batches=batches,
            num_iters_per_epoch=num_samples_per_epoch,
            seed=seed,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=collate_fn,
            pin_memory=pin_memory,
        )

        self.num_cache_chunks = max(num_cache_chunks, batch_size)
        self.chunk_lengths = self._parse_chunk_lengths(chunk_length)

        self.chunk_shift_ratio = chunk_shift_ratio
        self.batch_size = batch_size
        self.seed = seed
        self.shuffle = shuffle

        # keys that satisfy either condition below will be excluded from the length
        # consistency check:
        #  - exactly match one of the prefixes in `excluded_key_prefixes`
        #  - have one of the prefixes in `excluded_key_prefixes` and end with numbers
        self.excluded_key_pattern = (
            "(" + "[0-9]*)|(".join(excluded_key_prefixes) + "[0-9]*)"
            if excluded_key_prefixes
            else None
        )
        if self.excluded_key_pattern:
            logging.info(
                f"Data keys with the following patterns will be excluded from the "
                f"length consistency check:\n{self.excluded_key_pattern}"
            )

    @staticmethod
    def _parse_chunk_lengths(chunk_length: Union[int, str]) -> List[int]:
        """Parse the ``chunk_length`` constructor argument into a candidate list.

        Args:
            chunk_length: Either a single fixed chunk length, or a string
                specification such as ``"5,8"`` (discrete candidates) or
                ``"3-5"`` (inclusive range), or a comma-separated mix of both,
                e.g. ``"3-5,8"``.

        Returns:
            The list of candidate chunk lengths that a chunk width is chosen
            from for each sample.
        """
        if not isinstance(chunk_length, str):
            return [chunk_length]

        if len(chunk_length) == 0:
            raise ValueError("e.g. 5,8 or 3-5: but got empty string")

        chunk_lengths: List[int] = []
        for length_spec in chunk_length.split(","):
            try:
                bounds = list(map(int, length_spec.split("-")))
            except ValueError:
                raise ValueError(f"e.g. 5,8 or 3-5: but got {chunk_length}")

            if len(bounds) > 2:
                raise ValueError(f"e.g. 5,8 or 3-5: but got {chunk_length}")
            elif len(bounds) == 2:
                # Append all numbers between the range into the candidates
                chunk_lengths += list(range(bounds[0], bounds[1] + 1))
            else:
                chunk_lengths += [bounds[0]]
        return chunk_lengths

    def build_iter(
        self,
        epoch: int,
        shuffle: bool = None,
    ) -> Iterator[Tuple[List[str], Dict[str, torch.Tensor]]]:
        """Build the chunked-and-rebatched iterator for the given epoch.

        Args:
            epoch: The current epoch number, forwarded to the internal
                per-sample factory and used to seed chunk-selection
                randomness.
            shuffle: Whether to shuffle chunk order and randomize chunk
                start offsets. Falls back to ``self.shuffle`` when ``None``.

        Yields:
            ``(ids, batch)`` mini-batch tuples of size ``self.batch_size``,
            where ``batch`` maps each data key to a stacked tensor of
            chunks.
        """
        per_sample_loader = self.per_sample_iter_factory.build_iter(epoch, shuffle)

        if shuffle is None:
            shuffle = self.shuffle
        state = np.random.RandomState(epoch + self.seed)

        # NOTE(kamo):
        #   This iterator supports multiple chunk lengths and
        #   keep chunks for each lengths here until collecting specified numbers
        cache_chunks_dict: Dict[Any, Dict[int, ChunkCache]] = defaultdict(dict)
        cache_id_list_dict: Dict[Any, Dict[int, List[str]]] = defaultdict(dict)
        for ids, batch in per_sample_loader:
            # Must be per-sample-loader
            assert len(ids) == 1, f"Must be per-sample-loader: {len(ids)}"
            assert all(len(x) == 1 for x in batch.values())

            sequence_keys = self._get_sequence_keys(batch)
            # Remove lengths data and get the first sample
            batch = {k: v[0] for k, v in batch.items() if not k.endswith("_lengths")}
            id_ = ids[0]

            self._check_length_consistency(batch, sequence_keys)

            seq_len = len(batch[sequence_keys[0]])
            # Select chunk length
            chunk_length_candidates = [
                lg for lg in self.chunk_lengths if lg < seq_len
            ]
            if len(chunk_length_candidates) == 0:
                continue

            # Convert numpy array to number
            category = batch.get("utt2category", torch.zeros(1)).to(torch.int64)[0]

            chunk_width = int(state.choice(chunk_length_candidates, 1))
            cache_id_list = cache_id_list_dict[category].setdefault(chunk_width, [])
            cache_chunks = cache_chunks_dict[category].setdefault(chunk_width, {})

            self._extend_cache_with_chunks(
                cache_chunks=cache_chunks,
                cache_id_list=cache_id_list,
                batch=batch,
                sequence_keys=sequence_keys,
                id_=id_,
                seq_len=seq_len,
                chunk_width=chunk_width,
                shuffle=shuffle,
                state=state,
            )

            if len(cache_id_list) > self.num_cache_chunks:
                cache_id_list, cache_chunks = yield from self._generate_mini_batches(
                    cache_id_list,
                    cache_chunks,
                    shuffle,
                    state,
                )

            cache_id_list_dict[category][chunk_width] = cache_id_list
            cache_chunks_dict[category][chunk_width] = cache_chunks

        else:
            for category in cache_id_list_dict.keys():
                for chunk_width in cache_id_list_dict[category]:
                    cache_id_list = cache_id_list_dict[category].setdefault(
                        chunk_width, []
                    )
                    cache_chunks = cache_chunks_dict[category].setdefault(
                        chunk_width, {}
                    )

                    yield from self._generate_mini_batches(
                        cache_id_list,
                        cache_chunks,
                        shuffle,
                        state,
                    )

    @staticmethod
    def _get_sequence_keys(batch: Dict[str, Any]) -> List[str]:
        """Return the keys in ``batch`` that have a matching ``<key>_lengths`` entry."""
        return [key for key in batch if key + "_lengths" in batch]

    def _check_length_consistency(
        self, batch: Dict[str, torch.Tensor], sequence_keys: List[str]
    ) -> None:
        """Verify all sequence-valued entries in ``batch`` share the same length.

        Keys matching ``self.excluded_key_pattern`` are skipped, since those
        are known to legitimately have a different length (they are
        duplicated rather than chunked; see ``_extend_cache_with_chunks``).

        Raises:
            RuntimeError: If two sequence keys have inconsistent lengths.
        """
        reference_key = sequence_keys[0]
        for key in sequence_keys:
            if self.excluded_key_pattern is not None and re.fullmatch(
                self.excluded_key_pattern, key
            ):
                # ignore length inconsistency for `excluded_key_prefixes`
                continue
            if len(batch[key]) != len(batch[reference_key]):
                raise RuntimeError(
                    f"All sequences must has same length: "
                    f"{len(batch[key])} != {len(batch[reference_key])}"
                )

    def _extend_cache_with_chunks(
        self,
        cache_chunks: ChunkCache,
        cache_id_list: List[str],
        batch: Dict[str, torch.Tensor],
        sequence_keys: List[str],
        id_: str,
        seq_len: int,
        chunk_width: int,
        shuffle: bool,
        state: np.random.RandomState,
    ) -> None:
        """Split one sample into chunks and append them to the pending cache.

        Sequence-valued keys are sliced into ``num_chunks`` overlapping
        windows of width ``chunk_width`` (unless excluded via
        ``excluded_key_prefixes``, in which case the whole value is
        duplicated for each chunk instead). Non-sequence keys are simply
        duplicated for each chunk. Mutates ``cache_chunks`` and
        ``cache_id_list`` in place.
        """
        # Shift width to the next chunk
        chunk_stride = int(chunk_width * self.chunk_shift_ratio)
        # Number of chunks
        num_chunks = (seq_len - chunk_width) // chunk_stride + 1
        if shuffle:
            start_offset = state.randint(0, (seq_len - chunk_width) % chunk_stride + 1)
        else:
            start_offset = 0

        # Split a sequence into chunks.
        # Note that the marginal frames divided by chunk length are discarded
        for key, value in batch.items():
            if key not in cache_chunks:
                cache_chunks[key] = []
            if key in sequence_keys:
                # Shift chunks with overlapped length for data augmentation
                if self.excluded_key_pattern is not None and re.fullmatch(
                    self.excluded_key_pattern, key
                ):
                    for _ in range(num_chunks):
                        cache_chunks[key].append(value)
                else:
                    cache_chunks[key] += [
                        value[
                            start_offset
                            + i * chunk_stride : start_offset
                            + i * chunk_stride
                            + chunk_width
                        ]
                        for i in range(num_chunks)
                    ]
            else:
                # If not sequence, use whole data instead of chunk
                cache_chunks[key] += [value for _ in range(num_chunks)]
        cache_id_list += [id_ for _ in range(num_chunks)]

    def _generate_mini_batches(
        self,
        id_list: List[str],
        chunk_cache: ChunkCache,
        shuffle: bool,
        state: np.random.RandomState,
    ) -> Iterator[Tuple[List[str], Dict[str, torch.Tensor]]]:
        """Drain full mini-batches out of a pending chunk cache.

        Args:
            id_list: Utterance ids, one per pending chunk (parallel to the
                lists in ``chunk_cache``).
            chunk_cache: Maps each data key to the list of pending chunk
                tensors for that key.
            shuffle: Whether to shuffle chunks before batching.
            state: Random state used for the shuffle.

        Yields:
            ``(ids, batch)`` tuples of size ``self.batch_size``.

        Returns:
            The leftover ``(id_list, chunk_cache)`` that didn't fill a full
            mini-batch, so the caller can carry it over.
        """
        if shuffle:
            order = np.arange(0, len(id_list))
            state.shuffle(order)
            chunk_cache = {k: [v[i] for i in order] for k, v in chunk_cache.items()}
            id_list = [id_list[i] for i in order]

        mini_batch_size = self.batch_size
        while len(id_list) >= mini_batch_size:
            # Make mini-batch and yield
            yield (
                id_list[:mini_batch_size],
                {
                    k: torch.stack(v[:mini_batch_size], 0)
                    for k, v in chunk_cache.items()
                },
            )
            id_list = id_list[mini_batch_size:]
            chunk_cache = {k: v[mini_batch_size:] for k, v in chunk_cache.items()}

        return id_list, chunk_cache
