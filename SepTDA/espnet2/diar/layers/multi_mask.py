# This is an implementation of the multiple 1x1 convolution layer architecture
# in https://arxiv.org/pdf/2203.17068.pdf
"""Multi-speaker mask estimation via a bank of per-speaker-count 1x1 convs."""

from collections import OrderedDict
from typing import List, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_complex.tensor import ComplexTensor

from espnet2.diar.layers.abs_mask import AbsMask


class MultiMask(AbsMask):
    """Multiple 1x1 convolution layer Module.

    This module corresponds to the final 1x1 conv block and non-linear
    function in TCNSeparator. It holds ``max_num_spk`` separate 1x1 conv
    blocks, one per possible speaker count; ``forward`` selects the block
    matching ``num_spk`` to produce that many masks, so a single model can
    handle a variable number of speakers.

    Shapes (as used in :meth:`forward`):
        M: batch size, K: sequence length, N: number of encoder filters,
        B: bottleneck dimension.
    """

    def __init__(
        self,
        input_dim: int,
        bottleneck_dim: int = 128,
        max_num_spk: int = 3,
        mask_nonlinear: str = "relu",
    ):
        """Build one 1x1 conv block per candidate speaker count.

        Args:
            input_dim: Number of filters in autoencoder.
            bottleneck_dim: Number of channels in bottleneck 1 * 1-conv block.
            max_num_spk: Number of mask_conv1x1 modules
                        (>= Max number of speakers in the dataset).
            mask_nonlinear: use which non-linear function to generate mask.
        """
        super().__init__()
        # Hyper-parameter
        self._max_num_spk = max_num_spk
        self.mask_nonlinear = mask_nonlinear
        # [M, B, K] -> [M, C*N, K]
        self.mask_conv1x1 = nn.ModuleList()
        for num_spk in range(1, max_num_spk + 1):
            self.mask_conv1x1.append(
                nn.Conv1d(bottleneck_dim, num_spk * input_dim, 1, bias=False)
            )

    @property
    def max_num_spk(self) -> int:
        return self._max_num_spk

    def forward(
        self,
        input: Union[torch.Tensor, ComplexTensor],
        ilens: torch.Tensor,
        bottleneck_feat: torch.Tensor,
        num_spk: int,
    ) -> Tuple[List[Union[torch.Tensor, ComplexTensor]], torch.Tensor, OrderedDict]:
        """Keep this API same with TasNet.

        Args:
            input: encoder output, shape (M, K, N). M is batch size.
            ilens: valid lengths, shape (M,).
            bottleneck_feat: separator bottleneck feature, shape (M, K, B).
            num_spk: number of speakers
                (Training: oracle,
                Inference: estimated by other module (e.g, EEND-EDA)).

        Returns:
            masked (List[Union(torch.Tensor, ComplexTensor)]): [(M, K, N), ...],
                one tensor per speaker.
            ilens (torch.Tensor): (M,)
            others: OrderedDict[
                'mask_spk1': torch.Tensor(Batch, Frames, Freq),
                'mask_spk2': torch.Tensor(Batch, Frames, Freq),
                ...
                'mask_spkn': torch.Tensor(Batch, Frames, Freq),
            ]

        """
        batch_size, seq_len, num_filters = input.size()
        bottleneck_feat = bottleneck_feat.transpose(1, 2)  # [M, B, K]
        score = self._compute_score(bottleneck_feat, num_spk, num_filters)
        # [M, num_spk*N, K] -> [M, num_spk, N, K]
        score = score.view(batch_size, num_spk, num_filters, seq_len)
        est_mask = self._apply_mask_nonlinear(score)

        masks = est_mask.transpose(2, 3)  # [M, num_spk, K, N]
        masks = masks.unbind(dim=1)  # List[M, K, N]

        masked = [input * mask for mask in masks]

        others = OrderedDict(
            zip([f"mask_spk{i + 1}" for i in range(len(masks))], masks)
        )

        return masked, ilens, others

    def _compute_score(
        self,
        bottleneck_feat: torch.Tensor,
        num_spk: int,
        num_filters: int,
    ) -> torch.Tensor:
        """Run the 1x1 conv block for ``num_spk``, shape [M, num_spk*N, K].

        The unused conv blocks (for speaker counts other than ``num_spk``)
        are still evaluated and added in with a zero factor, purely so that
        every parameter participates in the graph and distributed training
        (which requires all parameters to receive gradients) keeps working.
        """
        score = self.mask_conv1x1[num_spk - 1](bottleneck_feat)
        for spk_idx in range(self._max_num_spk):
            if spk_idx != num_spk - 1:
                unused_score = self.mask_conv1x1[spk_idx](bottleneck_feat)
                score += 0.0 * F.interpolate(
                    unused_score.transpose(1, 2),
                    size=num_spk * num_filters,
                ).transpose(1, 2)
        return score

    def _apply_mask_nonlinear(self, score: torch.Tensor) -> torch.Tensor:
        """Apply the configured non-linearity to turn scores into masks."""
        if self.mask_nonlinear == "softmax":
            return F.softmax(score, dim=1)
        elif self.mask_nonlinear == "relu":
            return F.relu(score)
        elif self.mask_nonlinear == "sigmoid":
            return torch.sigmoid(score)
        elif self.mask_nonlinear == "tanh":
            return torch.tanh(score)
        else:
            raise ValueError("Unsupported mask non-linear function")
