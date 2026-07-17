#!/usr/bin/env python3
"""CLI that merges multiple per-shard statistics directories (as produced by
running the ``collect_stats`` stage with several parallel jobs) into one.

For each of the "train"/"valid" splits, this concatenates the per-key shape
files (sorted by utterance key, since shard order is not guaranteed when
``num_workers >= 1``), sums the per-key ``*_stats.npz`` arrays across shards,
and concatenates any collected-feature SCP files.
"""
import argparse
import logging
import sys
from pathlib import Path
from typing import Iterable, List, Union

import numpy as np

from espnet.utils.cli_utils import get_commandline_args


def _merge_shape_files(
    input_dirs: List[Path], output_dir: Path, mode: str, key: str
) -> None:
    """Concatenate one ``<key>_shape`` file per shard, sorted by utterance key."""
    with (output_dir / mode / f"{key}_shape").open("w", encoding="utf-8") as fout:
        for idir in input_dirs:
            with (idir / mode / f"{key}_shape").open("r", encoding="utf-8") as fin:
                # Read to the last in order to sort keys
                # because the order can be changed if num_workers>=1
                lines = sorted(fin.readlines(), key=lambda x: x.split()[0])
                for line in lines:
                    fout.write(line)


def _sum_stats_npz(input_dirs: List[Path], output_dir: Path, mode: str, key: str) -> None:
    """Sum the ``<key>_stats.npz`` arrays across all shards and write the total."""
    sum_stats = None
    for idir in input_dirs:
        stats = np.load(idir / mode / f"{key}_stats.npz")
        if sum_stats is None:
            sum_stats = dict(**stats)
        else:
            for k in stats:
                sum_stats[k] += stats[k]
    np.savez(output_dir / mode / f"{key}_stats.npz", **sum_stats)


def _merge_collected_feats(
    input_dirs: List[Path], output_dir: Path, mode: str, key: str
) -> None:
    """Concatenate collected-feature SCPs across shards, if they exist (--write_collected_feats)."""
    relative_scp = Path(mode) / "collect_feats" / f"{key}.scp"
    if not (input_dirs[0] / relative_scp).exists():
        return
    (output_dir / relative_scp).parent.mkdir(parents=True, exist_ok=True)
    with (output_dir / relative_scp).open("w", encoding="utf-8") as fout:
        for idir in input_dirs:
            with (idir / relative_scp).open("r", encoding="utf-8") as fin:
                for line in fin:
                    fout.write(line)


def aggregate_stats_dirs(
    input_dir: Iterable[Union[str, Path]],
    output_dir: Union[str, Path],
    log_level: str,
    skip_sum_stats: bool,
) -> None:
    """Merge the given per-shard stats directories into a single ``output_dir``."""
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s (%(module)s:%(lineno)d) (levelname)s: %(message)s",
    )

    input_dirs = [Path(p) for p in input_dir]
    output_dir = Path(output_dir)

    for mode in ["train", "valid"]:
        with (input_dirs[0] / mode / "batch_keys").open("r", encoding="utf-8") as f:
            batch_keys = [line.strip() for line in f if line.strip() != ""]
        with (input_dirs[0] / mode / "stats_keys").open("r", encoding="utf-8") as f:
            stats_keys = [line.strip() for line in f if line.strip() != ""]
        (output_dir / mode).mkdir(parents=True, exist_ok=True)

        for key in batch_keys:
            _merge_shape_files(input_dirs, output_dir, mode, key)

        for key in stats_keys:
            if not skip_sum_stats:
                _sum_stats_npz(input_dirs, output_dir, mode, key)
            _merge_collected_feats(input_dirs, output_dir, mode, key)


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Aggregate statistics directories into one directory",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--log_level",
        type=lambda x: x.upper(),
        default="INFO",
        choices=("CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG", "NOTSET"),
        help="The verbose level of logging",
    )
    parser.add_argument(
        "--skip_sum_stats",
        default=False,
        action="store_true",
        help="Skip computing the sum of statistics.",
    )

    parser.add_argument("--input_dir", action="append", help="Input directories")
    parser.add_argument("--output_dir", required=True, help="Output directory")
    return parser


def main(cmd: Union[List[str], None] = None) -> None:
    print(get_commandline_args(), file=sys.stderr)
    parser = get_parser()
    args = parser.parse_args(cmd)
    kwargs = vars(args)
    aggregate_stats_dirs(**kwargs)


if __name__ == "__main__":
    main()
