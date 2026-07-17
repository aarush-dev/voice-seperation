"""Identity decoder: passes the waveform through unchanged."""
from typing import Tuple

import torch

from espnet2.enh.decoder.abs_decoder import AbsDecoder


class NullDecoder(AbsDecoder):
    """No-op decoder, return the same args."""

    def __init__(self):
        super().__init__()

    def forward(
        self, input: torch.Tensor, ilens: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward. The input should be the waveform already.

        Args:
            input (torch.Tensor): wav [Batch, sample]
            ilens (torch.Tensor): input lengths [Batch]
        """
        return input, ilens
