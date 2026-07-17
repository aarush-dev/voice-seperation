"""Collects feature-shape and mean/variance statistics for global MVN.

Runs a single pass over the train and valid iterators before actual
training, writing (per split) a `{feat}_shape` file for every batch key and
a `{feat}_stats.npz` file (`count`, `sum`, `sum_square`) per collected
feature. Downstream code combines these into global mean/variance
normalization statistics.
"""

import logging
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, MutableMapping, Optional, Tuple, Union

import numpy as np
import torch
from torch.nn.parallel import data_parallel
from torch.utils.data import DataLoader
from typeguard import check_argument_types

from espnet2.fileio.datadir_writer import DatadirWriter
from espnet2.fileio.npy_scp import NpyScpWriter
from espnet2.torch_utils.device_funcs import to_device
from espnet2.torch_utils.forward_adaptor import ForwardAdaptor
from espnet2.train.abs_espnet_model import AbsESPnetModel


def _write_shape_file(
    datadir_writer: DatadirWriter,
    keys: List[str],
    batch: Dict[str, torch.Tensor],
) -> None:
    """Record each sample's (length-trimmed) tensor shape as `{name}_shape`.

    Args:
        datadir_writer: Writer whose `[f"{name}_shape"][key] = ...` entries
            become lines in `output_dir/{mode}/{name}_shape`.
        keys: The utterance ids for this batch, aligned with `batch` tensors.
        batch: Batch tensors, keyed by feature name (plus `f"{name}_lengths"`
            length tensors).
    """
    for name in batch:
        if name.endswith("_lengths"):
            continue
        for i, (key, data) in enumerate(zip(keys, batch[name])):
            if f"{name}_lengths" in batch:
                length = int(batch[f"{name}_lengths"][i])
                data = data[:length]  # data: (Length, Dim, ...) trimmed to valid length
            datadir_writer[f"{name}_shape"][key] = ",".join(map(str, data.shape))


def _accumulate_feature_stats(
    feats: Dict[str, torch.Tensor],
    keys: List[str],
    mode: str,
    output_dir: Path,
    write_collected_feats: bool,
    sum_dict: MutableMapping[str, np.ndarray],
    sq_dict: MutableMapping[str, np.ndarray],
    count_dict: MutableMapping[str, int],
    npy_scp_writers: MutableMapping[Tuple[str, str], NpyScpWriter],
) -> None:
    """Accumulate per-feature sum/sum-of-squares/count over one batch.

    Args:
        feats: Output of `model.collect_feats(**batch)`: feature name ->
            tensor of shape `(B, Length, Dim, ...)`, plus optional
            `f"{name}_lengths"` tensors of shape `(B,)`.
        keys: Utterance ids for this batch, aligned with `feats` tensors.
        mode: `"train"` or `"valid"`, used to place optional npy output.
        output_dir: Root stats directory; npy files go under
            `output_dir/{mode}/collect_feats`.
        write_collected_feats: If True, also write each trimmed feature
            sequence to disk as an `.npy` file referenced by a `.scp`.
        sum_dict: Mutated in place: feature name -> running sum over time
            and batch, shape `(Dim, ...)`.
        sq_dict: Mutated in place: feature name -> running sum of squares,
            shape `(Dim, ...)`.
        count_dict: Mutated in place: feature name -> running element count.
        npy_scp_writers: Mutated in place: lazily-created `(feature, mode)`
            -> `NpyScpWriter`, reused across calls for the same split.
    """
    for key, value in feats.items():
        for i, (uttid, seq) in enumerate(zip(keys, value.cpu().numpy())):
            # Truncate zero-padding region
            if f"{key}_lengths" in feats:
                length = feats[f"{key}_lengths"][i]
                seq = seq[:length]  # seq: (Length, Dim, ...)
            else:
                seq = seq[None]  # seq: (Dim, ...) -> (1, Dim, ...)
            # Accumulate value, its square, and count
            sum_dict[key] += seq.sum(0)
            sq_dict[key] += (seq**2).sum(0)
            count_dict[key] += len(seq)

            # [Option] Write derived features as npy format file.
            if write_collected_feats:
                # Instantiate NpyScpWriter for the first iteration
                if (key, mode) not in npy_scp_writers:
                    stats_dir = output_dir / mode / "collect_feats"
                    npy_scp_writers[(key, mode)] = NpyScpWriter(
                        stats_dir / f"data_{key}", stats_dir / f"{key}.scp"
                    )
                # Save array as npy file
                npy_scp_writers[(key, mode)][uttid] = seq


@torch.no_grad()
def collect_stats(
    model: Union[AbsESPnetModel, None],
    train_iter: DataLoader and Iterable[Tuple[List[str], Dict[str, torch.Tensor]]],
    valid_iter: DataLoader and Iterable[Tuple[List[str], Dict[str, torch.Tensor]]],
    output_dir: Path,
    ngpu: Optional[int],
    log_interval: Optional[int],
    write_collected_feats: bool,
) -> None:
    """Perform on collect_stats mode.

    Running for deriving the shape information from data
    and gathering statistics.
    This method is used before executing train().

    Args:
        model: The model whose `collect_feats(**batch)` is used to derive
            features to accumulate statistics over. If None, only shape
            files are written and no `*_stats.npz` files are produced.
        train_iter: Yields `(keys, batch)` for the training split, where
            `keys` are utterance ids and `batch` maps feature name to tensor.
        valid_iter: Same as `train_iter`, for the validation split.
        output_dir: Root directory to write `train/` and `valid/` stats into.
        ngpu: Number of GPUs to run `collect_feats` on; a value > 1 uses
            `torch.nn.parallel.data_parallel`.
        log_interval: Log every this many iterations; if None, derived from
            the iterator length (or defaults to 100 if length is unknown).
        write_collected_feats: If True, additionally dump each sample's
            collected features to disk as `.npy` files with a `.scp` index.

    Side Effects:
        For each of "train" and "valid", writes under `output_dir/{mode}/`:
        `{name}_shape` per batch key, `{feat}_stats.npz` (count/sum/sum_square)
        per collected feature, `batch_keys`, `stats_keys`, and (if
        `write_collected_feats`) a `collect_feats/` directory of `.npy` data.
    """
    assert check_argument_types()

    npy_scp_writers = {}
    for itr, mode in zip([train_iter, valid_iter], ["train", "valid"]):
        if log_interval is None:
            try:
                log_interval = max(len(itr) // 20, 10)
            except TypeError:
                log_interval = 100

        sum_dict = defaultdict(lambda: 0)
        sq_dict = defaultdict(lambda: 0)
        count_dict = defaultdict(lambda: 0)

        with DatadirWriter(output_dir / mode) as datadir_writer:
            for iiter, (keys, batch) in enumerate(itr, 1):
                batch = to_device(batch, "cuda" if ngpu > 0 else "cpu")

                # 1. Write shape file
                _write_shape_file(datadir_writer, keys, batch)

                if model is not None:
                    # 2. Extract feats
                    if ngpu <= 1:
                        feats = model.collect_feats(**batch)
                    else:
                        # Note that data_parallel can parallelize only "forward()"
                        feats = data_parallel(
                            ForwardAdaptor(model, "collect_feats"),
                            (),
                            range(ngpu),
                            module_kwargs=batch,
                        )

                    # 3. Calculate sum and square sum, optionally dumping feats
                    _accumulate_feature_stats(
                        feats,
                        keys,
                        mode,
                        output_dir,
                        write_collected_feats,
                        sum_dict,
                        sq_dict,
                        count_dict,
                        npy_scp_writers,
                    )

                if iiter % log_interval == 0:
                    logging.info(f"Niter: {iiter}")

        for key in sum_dict:
            np.savez(
                output_dir / mode / f"{key}_stats.npz",
                count=count_dict[key],
                sum=sum_dict[key],
                sum_square=sq_dict[key],
            )

        # batch_keys and stats_keys are used by aggregate_stats_dirs.py
        with (output_dir / mode / "batch_keys").open("w", encoding="utf-8") as f:
            f.write(
                "\n".join(filter(lambda x: not x.endswith("_lengths"), batch)) + "\n"
            )
        with (output_dir / mode / "stats_keys").open("w", encoding="utf-8") as f:
            f.write("\n".join(sum_dict) + "\n")
