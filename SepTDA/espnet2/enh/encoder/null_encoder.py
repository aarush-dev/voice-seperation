"""Identity encoder: passes the waveform through unchanged."""
from typing import Tuple

import torch

from espnet2.enh.encoder.abs_encoder import AbsEncoder


class NullEncoder(AbsEncoder):
    """No-op encoder used when the separator operates directly on the waveform."""

    def __init__(self):
        super().__init__()

    @property
    def output_dim(self) -> int:
        return 1

    def forward(
        self, input: torch.Tensor, ilens: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return the input waveform and lengths unchanged.

        Args:
            input (torch.Tensor): mixed speech [Batch, sample]
            ilens (torch.Tensor): input lengths [Batch]
        """
        return input, ilens
