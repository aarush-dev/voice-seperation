"""Abstract interface for blind speech-separation (SS) frontends.

A separator takes the encoded mixture only (no enrollment/target-speaker
information) and returns one estimate per active speaker.
"""
from abc import ABC, abstractmethod
from collections import OrderedDict
from typing import Dict, Optional, Tuple

import torch


class AbsSeparator(torch.nn.Module, ABC):
    """Base class for modules that separate a mixture into per-speaker signals."""

    @abstractmethod
    def forward(
        self,
        input: torch.Tensor,
        ilens: torch.Tensor,
        additional: Optional[Dict] = None,
    ) -> Tuple[Tuple[torch.Tensor], torch.Tensor, OrderedDict]:
        """Separate an encoded mixture into per-speaker estimates.

        Args:
            input: (Batch, Frames, Freq) encoded mixture feature.
            ilens: (Batch,) valid frame lengths of ``input``.
            additional: optional extra conditioning inputs, model-specific.

        Returns:
            masked: tuple of per-speaker extracted features.
            ilens: (Batch,) output frame lengths.
            others: OrderedDict of auxiliary outputs (e.g. predicted masks).
        """
        raise NotImplementedError

    def forward_streaming(
        self,
        input_frame: torch.Tensor,
        buffer=None,
    ):
        """Process a single streaming frame; not all separators support this."""
        raise NotImplementedError

    @property
    @abstractmethod
    def num_spk(self):
        """Number of speakers this separator is configured to output."""
        raise NotImplementedError
