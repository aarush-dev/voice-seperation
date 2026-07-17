#!/usr/bin/env python3
"""CLI entry point for training the joint target-speaker-extraction /
speech-separation (TSE+SS) model, delegating all argument parsing and the
training loop to ``TargetSpeakerExtractionAndEnhancementTask``.
"""
from typing import List, Optional

from espnet2.tasks.enh_tse_ss import TargetSpeakerExtractionAndEnhancementTask
from espnet2.utils import config_argparse


def get_parser() -> config_argparse.ArgumentParser:
    """Return the training argument parser defined by the TSE+SS task."""
    parser = TargetSpeakerExtractionAndEnhancementTask.get_parser()
    return parser


def main(cmd: Optional[List[str]] = None) -> None:
    r"""Target Speaker Extraction model training.

    Example:

        % python enh_tse_train.py asr --print_config --optim adadelta \
                > conf/train_enh.yaml
        % python enh_tse_train.py --config conf/train_enh.yaml
    """
    TargetSpeakerExtractionAndEnhancementTask.main(cmd=cmd)


if __name__ == "__main__":
    main()
