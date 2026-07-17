from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor


class TimeSegmentation(nn.Module):
    """Splits a sequence into overlapping segments and merges them back.

    Splits time-series features into overlapping chunks so that long
    sequences can be processed segment-by-segment (e.g. by a dual-path
    model), then reconstructs the original sequence from the processed
    segments via overlap-add.
    """

    def __init__(self, segment_size: int = 96):
        """Initialize the time segmentation module.

        Args:
            segment_size (int): size of each segment. Default is 96.
        """
        super().__init__()
        self.segment_size = segment_size

    def split_feature(self, x: Tensor) -> Tensor:
        """Split features into 50%-overlapping segments.

        Args:
            x (torch.Tensor): input features, shape (B, D, T)

        Returns:
            torch.Tensor: segmented features, shape (B, D, segment_size, n_chunks)
        """
        B, D, T = x.size()
        unfolded = torch.nn.functional.unfold(
            x.unsqueeze(-1),
            kernel_size=(self.segment_size, 1),
            padding=(self.segment_size, 0),
            stride=(self.segment_size // 2, 1),
        )
        return unfolded.reshape(B, D, self.segment_size, -1)

    def merge_feature(self, x: Tensor, length: Optional[int] = None) -> Tensor:
        """Merge overlapping segments back into a single sequence (overlap-add).

        Args:
            x (torch.Tensor): segmented features, shape (B, D, L, n_chunks)
            length (int, optional): target sequence length. If None, it is
                computed from the number of chunks and the hop size.

        Returns:
            torch.Tensor: merged features, shape (B, D, length)
        """
        B, D, L, n_chunks = x.size()
        hop_size = self.segment_size // 2
        if length is None:
            length = (n_chunks - 1) * hop_size + L
            padding = 0
        else:
            padding = (0, L)

        seq = x.reshape(B, D * L, n_chunks)
        x = torch.nn.functional.fold(
            seq,
            output_size=(1, length),
            kernel_size=(1, L),
            padding=padding,
            stride=(1, hop_size),
        )
        norm_mat = torch.nn.functional.fold(
            input=torch.ones_like(seq),
            output_size=(1, length),
            kernel_size=(1, L),
            padding=padding,
            stride=(1, hop_size),
        )

        x /= norm_mat

        return x.reshape(B, D, length)
