"""Averages checkpoint weights for final model export.

Provides two entry points:

- `average_nbest_models`: for each `(phase, metric)` criterion tracked by a
  `Reporter`, averages the N best epochs (by that criterion) and writes the
  result as `{phase}.{metric}.ave_{n}best.pth`, plus a `.ave.pth` symlink to
  the largest-N average.
- `average_selected_models`: averages an explicit, caller-chosen list of
  epoch checkpoints into a single `ave_epochs_{...}.pth` file.

Both operate on the `*.pth` state-dict checkpoints written per epoch during
training and produce new state-dict checkpoints with identical key sets and
tensor shapes/dtypes to the inputs; only tensor values change (via averaging).
"""

import logging
import warnings
from pathlib import Path
from typing import Collection, Dict, List, MutableMapping, Optional, Sequence, Union

import torch
from typeguard import check_argument_types

from espnet2.train.reporter import Reporter

StateDict = Dict[str, torch.Tensor]


def _normalize_nbest(nbest: Union[Collection[int], int]) -> List[int]:
    """Coerce the `nbest` argument into a non-empty list of ints.

    Falls back to `[1]` if given an empty collection, matching the original
    behavior of warning and defaulting to averaging just the single best model.
    """
    nbests = [nbest] if isinstance(nbest, int) else list(nbest)
    if len(nbests) == 0:
        warnings.warn("At least 1 nbest values are required")
        nbests = [1]
    return nbests


def _replace_symlink(link_path: Path, target_name: str) -> None:
    """(Re)create `link_path` as a symlink pointing at `target_name`.

    Removes any pre-existing file or symlink at `link_path` first, so this is
    safe to call repeatedly (e.g. once per training run).
    """
    if link_path.is_symlink() or link_path.exists():
        link_path.unlink()
    link_path.symlink_to(target_name)


def _average_checkpoints(
    epochs: Sequence[int],
    output_dir: Path,
    cache: MutableMapping[int, StateDict],
) -> StateDict:
    """Load and average the state dicts for `epochs`, using/populating `cache`.

    Integer-dtype tensors (e.g. `BatchNorm.num_batches_tracked`) are only
    accumulated, not divided by the count, matching the original averaging
    behavior for such buffers.

    Note:
        The returned dict is the same object as the first loaded checkpoint's
        state dict (no copy is made), and accumulation reassigns tensors
        in-place on that dict. Since `cache` stores that same dict object, a
        cached epoch's entry can end up holding averaged values rather than
        the raw checkpoint values if reused across multiple averaging calls.
        This mirrors pre-existing behavior and is preserved as-is.

    Args:
        epochs: Epoch numbers to average, in the order they contribute.
        output_dir: Directory containing `{epoch}epoch.pth` checkpoint files.
        cache: Maps epoch number -> loaded state dict, shared across calls to
            avoid re-reading a checkpoint already loaded for another criterion.

    Returns:
        The averaged state dict.
    """
    avg: Optional[StateDict] = None
    count = 0
    for epoch in epochs:
        count += 1
        if epoch not in cache:
            cache[epoch] = torch.load(
                output_dir / f"{epoch}epoch.pth",
                map_location="cpu",
            )
        states = cache[epoch]

        if avg is None:
            avg = states
        else:
            for key in avg:
                avg[key] = avg[key] + states[key]

    for key in avg:
        if str(avg[key].dtype).startswith("torch.int"):
            # Integer buffers (e.g. num_batches_tracked) are accumulated only.
            pass
        else:
            avg[key] = avg[key] / count

    return avg


@torch.no_grad()
def average_nbest_models(
    output_dir: Path,
    reporter: Reporter,
    best_model_criterion: Sequence[Sequence[str]],
    nbest: Union[Collection[int], int],
    suffix: Optional[str] = None,
) -> None:
    """Generate averaged model from n-best models

    Args:
        output_dir: The directory contains the model file for each epoch
        reporter: Reporter instance
        best_model_criterion: Give criterions to decide the best model.
            e.g. [("valid", "loss", "min"), ("train", "acc", "max")]
        nbest: Number of best model files to be averaged
        suffix: A suffix added to the averaged model file name
    """
    assert check_argument_types()
    nbests = _normalize_nbest(nbest)
    suffix = suffix + "." if suffix is not None else ""

    # 1. Get nbests: List[Tuple[str, str, List[Tuple[epoch, value]]]]
    nbest_epochs = [
        (phase, key, reporter.sort_epochs_and_values(phase, key, mode)[: max(nbests)])
        for phase, key, mode in best_model_criterion
        if reporter.has(phase, key)
    ]

    loaded_states: Dict[int, StateDict] = {}
    for phase, criterion, epoch_and_values in nbest_epochs:
        candidate_nbests = [i for i in nbests if i <= len(epoch_and_values)]
        if len(candidate_nbests) == 0:
            candidate_nbests = [1]

        for n in candidate_nbests:
            if n == 0:
                continue
            elif n == 1:
                # The averaged model is same as the best model
                best_epoch, _ = epoch_and_values[0]
                checkpoint_path = output_dir / f"{best_epoch}epoch.pth"
                ave_path = output_dir / f"{phase}.{criterion}.ave_1best.{suffix}pth"
                _replace_symlink(ave_path, checkpoint_path.name)
            else:
                ave_path = output_dir / f"{phase}.{criterion}.ave_{n}best.{suffix}pth"
                logging.info(
                    f"Averaging {n}best models: "
                    f'criterion="{phase}.{criterion}": {ave_path}'
                )

                epochs_to_average = [e for e, _ in epoch_and_values[:n]]
                avg = _average_checkpoints(
                    epochs_to_average, output_dir, loaded_states
                )

                # 2.b. Save the ave model and create a symlink
                torch.save(avg, ave_path)

        # 3. *.*.ave.pth is a symlink to the max ave model
        max_n_path = output_dir / (
            f"{phase}.{criterion}.ave_{max(candidate_nbests)}best.{suffix}pth"
        )
        ave_path = output_dir / f"{phase}.{criterion}.ave.{suffix}pth"
        _replace_symlink(ave_path, max_n_path.name)


@torch.no_grad()
def average_selected_models(
    output_dir: Path,
    epochs: Union[Collection[int], int],
    suffix: Optional[str] = None,
) -> None:
    """Generate an averaged model from an explicit list of epoch checkpoints.

    Args:
        output_dir: The directory contains the model file for each epoch
        epochs: Epoch numbers whose checkpoints should be averaged together.
        suffix: A suffix added to the averaged model file name

    Side Effects:
        Mutates `epochs` in place, replacing each element with its string
        form, and writes `ave_epochs_{...}.{suffix}pth` to `output_dir`.
    """
    assert check_argument_types()
    suffix = suffix + "." if suffix is not None else ""

    loaded_states: Dict[int, StateDict] = {}
    avg = _average_checkpoints(epochs, output_dir, loaded_states)

    # 2.b. Save the ave model and create a symlink
    for i in range(len(epochs)):
        epochs[i] = str(epochs[i])
    epoch_suffix = "_".join(epochs)
    ave_path = output_dir / f"ave_epochs_{epoch_suffix}.{suffix}pth"
    torch.save(avg, ave_path)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, nargs="*", required=True)
    args = parser.parse_args()
    average_selected_models(args.exp_dir, args.epochs)
