"""Frame-level label aggregation layer.

Used to downsample sample-level labels (e.g. speech-activity annotations)
onto the same time grid as STFT-domain features, so that per-frame labels
can be supervised alongside frame-level acoustic features. Framing and
padding are kept compatible with :class:`~espnet2.layers.stft.Stft` /
``torch.stft`` so that label frames line up with feature frames.
"""
from typing import Optional, Tuple

import torch
from typeguard import check_argument_types

from espnet.nets.pytorch_backend.nets_utils import make_pad_mask


class LabelAggregate(torch.nn.Module):
    """Aggregate sample-level labels into per-frame labels.

    Each output frame is set to 1 if more than half of the samples it
    covers are labeled 1, mirroring the framing/centering behaviour of
    ``torch.stft`` so that label frames align with STFT feature frames.

    Attributes:
        win_length: Number of samples per frame.
        hop_length: Number of samples between consecutive frame starts.
        center: Whether to pad the input so frames are centered on their
            hop position, matching ``torch.stft(center=True)``.
    """

    def __init__(
        self,
        win_length: int = 512,
        hop_length: int = 128,
        center: bool = True,
    ):
        assert check_argument_types()
        super().__init__()

        self.win_length = win_length
        self.hop_length = hop_length
        self.center = center

    def extra_repr(self) -> str:
        return (
            f"win_length={self.win_length}, "
            f"hop_length={self.hop_length}, "
            f"center={self.center}, "
        )

    def forward(
        self, input: torch.Tensor, ilens: torch.Tensor = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """LabelAggregate forward function.

        Args:
            input: (Batch, Nsamples, Label_dim)
            ilens: (Batch)
        Returns:
            output: (Batch, Frames, Label_dim)

        """
        bs = input.size(0)
        max_length = input.size(1)
        label_dim = input.size(2)

        # NOTE(jiatong):
        #   The default behaviour of label aggregation is compatible with
        #   torch.stft about framing and padding.

        # Step1: center padding
        if self.center:
            pad = self.win_length // 2
            max_length = max_length + 2 * pad
            input = torch.nn.functional.pad(input, (0, 0, pad, pad), "constant", 0)
            input[:, :pad, :] = input[:, pad : (2 * pad), :]
            input[:, (max_length - pad) : max_length, :] = input[
                :, (max_length - 2 * pad) : (max_length - pad), :
            ]
            nframe = (max_length - self.win_length) // self.hop_length + 1

        # Step2: framing
        # output: (Batch, Frames, win_length, Label_dim)
        output = input.as_strided(
            (bs, nframe, self.win_length, label_dim),
            (max_length * label_dim, self.hop_length * label_dim, label_dim, 1),
        )

        # Step3: aggregate label
        # output: (Batch, Frames, Label_dim), majority vote over each window
        output = torch.gt(output.sum(dim=2, keepdim=False), self.win_length // 2)
        output = output.float()

        # Step4: process lengths
        if ilens is not None:
            if self.center:
                pad = self.win_length // 2
                ilens = ilens + 2 * pad

            olens = (ilens - self.win_length) // self.hop_length + 1
            output.masked_fill_(make_pad_mask(olens, output, 1), 0.0)
        else:
            olens = None

        return output, olens
