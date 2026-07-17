"""Abstract base class for layers that support an inverse transform.

Front-end layers such as :class:`~espnet2.layers.stft.Stft` and
:class:`~espnet2.layers.global_mvn.GlobalMVN` implement this interface so
that the pipeline can map estimated features back to the original domain
(e.g. inverse STFT to reconstruct a waveform, or un-normalizing features).
"""
from abc import ABC, abstractmethod
from typing import Optional, Tuple

import torch


class InversibleInterface(ABC):
    """Mixin requiring an ``inverse()`` counterpart to ``forward()``."""

    @abstractmethod
    def inverse(
        self, input: torch.Tensor, input_lengths: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Invert the forward transform.

        Args:
            input: Tensor previously produced by ``forward()``.
            input_lengths: Valid length per batch element, (B,).

        Returns:
            output: Tensor mapped back to the pre-forward domain.
            output_lengths: Lengths corresponding to ``output``, (B,).
        """
        raise NotImplementedError
