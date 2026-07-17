from typing import List

import torch
import torch.nn as nn
from torch import Tensor


class MulCatBlock(nn.Module):
    """The MulCat block.

    An LSTM whose output is gated (elementwise multiplied) by a second,
    independent "gate" LSTM running over the same input, following the
    dual-path RNN speech separation model used in e.g. FurcaNeXt / DPRNN
    variants with MulCat blocks.

    Args:
        input_size: int, dimension of the input feature.
            The input should have shape (batch, seq_len, input_size).
        hidden_size: int, dimension of the hidden state.
        dropout: float, the dropout rate in the LSTM layer. (Default: 0.0)
        bidirectional: bool, whether the RNN layers are bidirectional. (Default: True)
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        dropout: float = 0.0,
        bidirectional: bool = True,
    ):
        super().__init__()

        num_direction = int(bidirectional) + 1

        self.rnn = nn.LSTM(
            input_size,
            hidden_size,
            1,
            dropout=dropout,
            batch_first=True,
            bidirectional=bidirectional,
        )
        self.rnn_proj = nn.Linear(hidden_size * num_direction, input_size)

        self.gate_rnn = nn.LSTM(
            input_size,
            hidden_size,
            num_layers=1,
            batch_first=True,
            dropout=dropout,
            bidirectional=bidirectional,
        )
        self.gate_rnn_proj = nn.Linear(hidden_size * num_direction, input_size)

        self.block_projection = nn.Linear(input_size * 2, input_size)

    def forward(self, input: Tensor) -> Tensor:
        """Compute output after MulCatBlock.

        Args:
            input (torch.Tensor): The input feature.
                Tensor of shape (batch, time, feature_dim)

        Returns:
            (torch.Tensor): The output feature after MulCatBlock.
                Tensor of shape (batch, time, feature_dim)
        """
        orig_shape = input.shape
        # run rnn module
        rnn_output, _ = self.rnn(input)
        rnn_output = (
            self.rnn_proj(rnn_output.contiguous().view(-1, rnn_output.shape[2]))
            .view(orig_shape)
            .contiguous()
        )
        # run gate rnn module
        gate_rnn_output, _ = self.gate_rnn(input)
        gate_rnn_output = (
            self.gate_rnn_proj(
                gate_rnn_output.contiguous().view(-1, gate_rnn_output.shape[2])
            )
            .view(orig_shape)
            .contiguous()
        )
        # apply gated rnn
        gated_output = torch.mul(rnn_output, gate_rnn_output)
        # concatenate the input with rnn output
        gated_output = torch.cat([gated_output, input], 2)
        # linear projection to make the output shape the same as input
        gated_output = self.block_projection(
            gated_output.contiguous().view(-1, gated_output.shape[2])
        ).view(orig_shape)
        return gated_output


class DPMulCat(nn.Module):
    """Dual-path RNN module with MulCat blocks.

    Args:
        input_size: int, dimension of the input feature.
            The input should have shape (batch, seq_len, input_size).
        hidden_size: int, dimension of the hidden state.
        output_size: int, dimension of the output size.
        num_spk: int, the number of speakers in the output.
        dropout: float, the dropout rate in the LSTM layer. (Default: 0.0)
        bidirectional: bool, whether the RNN layers are bidirectional. (Default: True)
        num_layers: int, number of stacked MulCat blocks. (Default: 4)
        input_normalize: bool, whether to apply GroupNorm on the input Tensor.
            (Default: False)
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        num_spk: int,
        dropout: float = 0.0,
        num_layers: int = 4,
        bidirectional: bool = True,
        input_normalize: bool = False,
    ):
        super().__init__()

        self.rows_grnn = nn.ModuleList([])
        self.cols_grnn = nn.ModuleList([])
        self.rows_normalization = nn.ModuleList([])
        self.cols_normalization = nn.ModuleList([])

        # create the dual path pipeline
        for i in range(num_layers):
            self.rows_grnn.append(
                MulCatBlock(
                    input_size, hidden_size, dropout, bidirectional=bidirectional
                )
            )
            self.cols_grnn.append(
                MulCatBlock(
                    input_size, hidden_size, dropout, bidirectional=bidirectional
                )
            )
            if input_normalize:
                self.rows_normalization.append(nn.GroupNorm(1, input_size, eps=1e-8))
                self.cols_normalization.append(nn.GroupNorm(1, input_size, eps=1e-8))
            else:
                # used to disable normalization
                self.rows_normalization.append(nn.Identity())
                self.cols_normalization.append(nn.Identity())

        self.output = nn.Sequential(
            nn.PReLU(), nn.Conv2d(input_size, output_size * num_spk, 1)
        )

    def forward(self, input: Tensor) -> List[Tensor]:
        """Compute output after DPMulCat module.

        Args:
            input (torch.Tensor): The input feature.
                Tensor of shape (batch, N, dim1, dim2)
                Apply RNN on dim1 first and then dim2

        Returns:
            (list(torch.Tensor) or list(list(torch.Tensor))
                In training mode, the module returns output of each DPMulCat block.
                In eval mode, the module only returns output in the last block.
        """
        batch_size, _, d1, d2 = input.shape
        output = input  # (batch, N, d1, d2)
        output_all = []
        num_blocks = len(self.rows_grnn)
        for i in range(num_blocks):
            output = self._apply_rows(output, i, batch_size, d1, d2)
            output = self._apply_cols(output, i, batch_size, d1, d2)

            # if training mode, it returns the output Tensor from all layers.
            # Otherwise, it only returns the one from the last layer.
            if self.training or i == (num_blocks - 1):
                output_all.append(self.output(output))
        return output_all

    def _apply_rows(
        self, output: Tensor, i: int, batch_size: int, d1: int, d2: int
    ) -> Tensor:
        """Run the i-th MulCat block along dim1, with a residual connection."""
        row_input = (
            output.permute(0, 3, 2, 1).contiguous().view(batch_size * d2, d1, -1)
        )
        row_output = self.rows_grnn[i](row_input)
        row_output = (
            row_output.view(batch_size, d2, d1, -1).permute(0, 3, 2, 1).contiguous()
        )
        row_output = self.rows_normalization[i](row_output)
        return output + row_output

    def _apply_cols(
        self, output: Tensor, i: int, batch_size: int, d1: int, d2: int
    ) -> Tensor:
        """Run the i-th MulCat block along dim2, with a residual connection."""
        col_input = (
            output.permute(0, 2, 3, 1).contiguous().view(batch_size * d1, d2, -1)
        )
        col_output = self.cols_grnn[i](col_input)
        col_output = (
            col_output.view(batch_size, d1, d2, -1).permute(0, 3, 1, 2).contiguous()
        )
        col_output = self.cols_normalization[i](col_output).contiguous()
        return output + col_output
