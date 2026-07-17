"""Abstract base class for feature normalization layers.

Normalization layers (e.g. :class:`~espnet2.layers.global_mvn.GlobalMVN`,
:class:`~espnet2.layers.utterance_mvn.UtteranceMVN`) sit in the frontend of
the separation/ASR pipeline and rescale acoustic features (mean/variance
normalization) before they are fed to the encoder. Subclasses are looked up
by their registered config name (e.g. "global_mvn", "utterance_mvn") and
instantiated directly from YAML task configs.
"""
from abc import ABC, abstractmethod
from typing import Optional, Tuple

import torch


class AbsNormalize(torch.nn.Module, ABC):
    """Interface that all feature-normalization modules must implement."""

    @abstractmethod
    def forward(
        self, input: torch.Tensor, input_lengths: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Normalize ``input`` and return the (possibly updated) lengths.

        Args:
            input: Feature tensor, e.g. (B, T, D).
            input_lengths: Valid length per batch element, (B,).

        Returns:
            output: Normalized feature tensor, same shape as ``input``.
            output_lengths: Lengths corresponding to ``output``, (B,).
        """
        raise NotImplementedError
