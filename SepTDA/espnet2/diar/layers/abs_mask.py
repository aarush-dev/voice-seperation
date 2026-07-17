"""Abstract interface for modules that turn a bottleneck feature into masks."""

from abc import ABC, abstractmethod
from collections import OrderedDict
from typing import Tuple

import torch


class AbsMask(torch.nn.Module, ABC):
    """Base class for "mask" modules used by TCN-style separators.

    A mask module takes the separator's bottleneck feature together with the
    (oracle or estimated) number of speakers, and produces one mask per
    speaker to apply to the encoder output.
    """

    @property
    @abstractmethod
    def max_num_spk(self) -> int:
        """Maximum number of speakers this module can produce masks for."""
        raise NotImplementedError

    @abstractmethod
    def forward(
        self,
        input: torch.Tensor,
        ilens: torch.Tensor,
        bottleneck_feat: torch.Tensor,
        num_spk: int,
    ) -> Tuple[Tuple[torch.Tensor, ...], torch.Tensor, OrderedDict]:
        """Apply per-speaker masks to ``input``.

        Args:
            input: Encoder output, shape (Batch, Time, Feat).
            ilens: Valid lengths per batch item, shape (Batch,).
            bottleneck_feat: Separator bottleneck feature,
                shape (Batch, Time, BottleneckDim).
            num_spk: Number of speakers to produce masks for.

        Returns:
            A tuple of ``(masked, ilens, others)`` where ``masked`` holds one
            masked tensor per speaker, and ``others`` carries auxiliary
            outputs such as the individual masks.
        """
        raise NotImplementedError
