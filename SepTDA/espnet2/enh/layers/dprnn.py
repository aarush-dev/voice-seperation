"""Dual-path RNN (DPRNN) and its TAC (transform-average-concatenate) variant.

Implementation of the model proposed in:
    Luo et al., "Dual-path RNN: efficient long sequence modeling for
    time-domain single-channel speech separation."

Ported from:
    https://github.com/yluo42/TAC/blob/master/utility/models.py
    (Licensed under CC BY-NC-SA 3.0 US.)

DPRNN alternates two RNN passes over a chunked feature map of shape
(B, N, dim1, dim2):
    - the "row"/intra-chunk pass runs along `dim1` (within a chunk),
      folding `dim2` (the chunk index) into the batch dimension;
    - the "col"/inter-chunk pass runs along `dim2` (across chunks),
      folding `dim1` into the batch dimension.
`DPRNN_TAC` additionally folds in a microphone-channel dimension and
inserts a TAC block after every dual-path layer to exchange information
across channels.

`split_feature`/`merge_feature` implement the 50%-overlap chunking used
to turn a (B, N, T) sequence into the (B, N, dim1, dim2) chunk map that
DPRNN consumes, and back.
"""

from typing import Optional, Tuple, Union

import torch
import torch.nn as nn


class SingleRNN(nn.Module):
    """A single RNN layer followed by a linear projection back to `input_size`.

    Args:
        rnn_type: one of 'RNN', 'LSTM', 'GRU'.
        input_size: feature dimension of the input, shape (batch, seq_len,
            input_size).
        hidden_size: dimension of the RNN hidden state.
        dropout: dropout ratio applied to the RNN output. Default is 0.
        bidirectional: whether the RNN is bidirectional. Default is False.
    """

    def __init__(
        self,
        rnn_type: str,
        input_size: int,
        hidden_size: int,
        dropout: float = 0,
        bidirectional: bool = False,
    ) -> None:
        super().__init__()

        rnn_type = rnn_type.upper()

        assert rnn_type in [
            "RNN",
            "LSTM",
            "GRU",
        ], f"Only support 'RNN', 'LSTM' and 'GRU', current type: {rnn_type}"

        self.rnn_type = rnn_type
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_direction = int(bidirectional) + 1

        self.rnn = getattr(nn, rnn_type)(
            input_size,
            hidden_size,
            1,
            batch_first=True,
            bidirectional=bidirectional,
        )

        self.dropout = nn.Dropout(p=dropout)

        # linear projection layer
        self.proj = nn.Linear(hidden_size * self.num_direction, input_size)

    def forward(
        self,
        input: torch.Tensor,
        state: Optional[Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]] = None,
    ) -> Tuple[torch.Tensor, Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]]:
        """Run the RNN and project its output back to `input_size`.

        Args:
            input: (batch, seq_len, input_size)
            state: initial RNN state (hidden, or (hidden, cell) for LSTM).

        Returns:
            output: (batch, seq_len, input_size)
            state: final RNN state, same structure as `state`.
        """
        rnn_output, state = self.rnn(input, state)
        rnn_output = self.dropout(rnn_output)
        rnn_output = self.proj(
            rnn_output.contiguous().view(-1, rnn_output.shape[2])
        ).view(input.shape)
        return rnn_output, state


class DPRNN(nn.Module):
    """Deep dual-path RNN.

    Alternates an intra-chunk ("row") RNN pass along `dim1` and an
    inter-chunk ("col") RNN pass along `dim2`, each followed by a residual
    GroupNorm, for `num_layers` layers. A final PReLU + 1x1 Conv2d projects
    the feature dimension to `output_size`.

    Args:
        rnn_type: one of 'RNN', 'LSTM', 'GRU'.
        input_size: feature dimension of the input, shape (batch, seq_len,
            input_size).
        hidden_size: dimension of the RNN hidden state.
        output_size: feature dimension of the output.
        dropout: dropout ratio. Default is 0.
        num_layers: number of stacked dual-path layers. Default is 1.
        bidirectional: whether the inter-chunk ("col") RNN is bidirectional.
            The intra-chunk ("row") RNN is always bidirectional. Default is
            True.
    """

    def __init__(
        self,
        rnn_type: str,
        input_size: int,
        hidden_size: int,
        output_size: int,
        dropout: float = 0,
        num_layers: int = 1,
        bidirectional: bool = True,
    ) -> None:
        super().__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size

        # dual-path RNN
        self.row_rnn = nn.ModuleList([])
        self.col_rnn = nn.ModuleList([])
        self.row_norm = nn.ModuleList([])
        self.col_norm = nn.ModuleList([])
        for _ in range(num_layers):
            self.row_rnn.append(
                SingleRNN(
                    rnn_type, input_size, hidden_size, dropout, bidirectional=True
                )
            )  # intra-segment RNN is always noncausal
            self.col_rnn.append(
                SingleRNN(
                    rnn_type,
                    input_size,
                    hidden_size,
                    dropout,
                    bidirectional=bidirectional,
                )
            )
            self.row_norm.append(nn.GroupNorm(1, input_size, eps=1e-8))
            # default is to use noncausal LayerNorm for inter-chunk RNN.
            # For causal setting change it to causal normalization accordingly.
            self.col_norm.append(nn.GroupNorm(1, input_size, eps=1e-8))

        # output layer
        self.output = nn.Sequential(nn.PReLU(), nn.Conv2d(input_size, output_size, 1))

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Apply the dual-path RNN stack.

        Args:
            input: (B, N, dim1, dim2), dim1 is the intra-chunk (segment)
                length and dim2 is the number of chunks.

        Returns:
            (B, output_size, dim1, dim2)
        """
        batch_size, _, dim1, dim2 = input.shape
        output = input
        for layer_idx in range(len(self.row_rnn)):
            output = self._intra_chunk_pass(output, layer_idx, batch_size, dim1, dim2)
            output = self._inter_chunk_pass(output, layer_idx, batch_size, dim1, dim2)

        return self.output(output)  # B, output_size, dim1, dim2

    def _intra_chunk_pass(
        self,
        output: torch.Tensor,
        layer_idx: int,
        batch_size: int,
        dim1: int,
        dim2: int,
    ) -> torch.Tensor:
        """Row pass: RNN along dim1, with dim2 folded into the batch dim."""
        row_input = (
            output.permute(0, 3, 2, 1).contiguous().view(batch_size * dim2, dim1, -1)
        )  # B*dim2, dim1, N
        row_output, _ = self.row_rnn[layer_idx](row_input)  # B*dim2, dim1, N
        row_output = (
            row_output.view(batch_size, dim2, dim1, -1).permute(0, 3, 2, 1).contiguous()
        )  # B, N, dim1, dim2
        row_output = self.row_norm[layer_idx](row_output)
        return output + row_output

    def _inter_chunk_pass(
        self,
        output: torch.Tensor,
        layer_idx: int,
        batch_size: int,
        dim1: int,
        dim2: int,
    ) -> torch.Tensor:
        """Col pass: RNN along dim2, with dim1 folded into the batch dim."""
        col_input = (
            output.permute(0, 2, 3, 1).contiguous().view(batch_size * dim1, dim2, -1)
        )  # B*dim1, dim2, N
        col_output, _ = self.col_rnn[layer_idx](col_input)  # B*dim1, dim2, N
        col_output = (
            col_output.view(batch_size, dim1, dim2, -1).permute(0, 3, 1, 2).contiguous()
        )  # B, N, dim1, dim2
        col_output = self.col_norm[layer_idx](col_output)
        return output + col_output


class DPRNN_TAC(nn.Module):
    """Deep dual-path RNN with TAC applied after each dual-path layer.

    Extends `DPRNN` to a multi-channel input (B, ch, N, dim1, dim2): the
    intra-chunk and inter-chunk RNN passes run per-channel (channel folded
    into the batch dim, as in `DPRNN`), and a transform-average-concatenate
    (TAC) block is applied afterwards to exchange information across
    microphone channels.

    Args:
        rnn_type: one of 'RNN', 'LSTM', 'GRU'.
        input_size: feature dimension of the input, shape (batch, seq_len,
            input_size).
        hidden_size: dimension of the RNN hidden state.
        output_size: feature dimension of the output.
        dropout: dropout ratio. Default is 0.
        num_layers: number of stacked dual-path+TAC layers. Default is 1.
        bidirectional: whether the inter-chunk ("col") RNN is bidirectional.
            Default is False.
    """

    def __init__(
        self,
        rnn_type: str,
        input_size: int,
        hidden_size: int,
        output_size: int,
        dropout: float = 0,
        num_layers: int = 1,
        bidirectional: bool = True,
    ) -> None:
        super(DPRNN_TAC, self).__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size

        # DPRNN + TAC for 3D input (ch, N, T)
        self.row_rnn = nn.ModuleList([])
        self.col_rnn = nn.ModuleList([])
        self.ch_transform = nn.ModuleList([])
        self.ch_average = nn.ModuleList([])
        self.ch_concat = nn.ModuleList([])

        self.row_norm = nn.ModuleList([])
        self.col_norm = nn.ModuleList([])
        self.ch_norm = nn.ModuleList([])

        for _ in range(num_layers):
            self.row_rnn.append(
                SingleRNN(
                    rnn_type, input_size, hidden_size, dropout, bidirectional=True
                )
            )  # intra-segment RNN is always noncausal
            self.col_rnn.append(
                SingleRNN(
                    rnn_type,
                    input_size,
                    hidden_size,
                    dropout,
                    bidirectional=bidirectional,
                )
            )
            self.ch_transform.append(
                nn.Sequential(nn.Linear(input_size, hidden_size * 3), nn.PReLU())
            )
            self.ch_average.append(
                nn.Sequential(nn.Linear(hidden_size * 3, hidden_size * 3), nn.PReLU())
            )
            self.ch_concat.append(
                nn.Sequential(nn.Linear(hidden_size * 6, input_size), nn.PReLU())
            )

            self.row_norm.append(nn.GroupNorm(1, input_size, eps=1e-8))
            # default is to use noncausal LayerNorm for
            # inter-chunk RNN and TAC modules.
            # For causal setting change them to causal normalization
            # techniques accordingly.
            self.col_norm.append(nn.GroupNorm(1, input_size, eps=1e-8))
            self.ch_norm.append(nn.GroupNorm(1, input_size, eps=1e-8))

        # output layer
        self.output = nn.Sequential(nn.PReLU(), nn.Conv2d(input_size, output_size, 1))

    def forward(self, input: torch.Tensor, num_mic: torch.Tensor) -> torch.Tensor:
        """Apply the dual-path RNN + TAC stack.

        Args:
            input: (B, ch, N, dim1, dim2) per-channel chunked features.
            num_mic: (B,) number of valid microphones per example. A value
                of 0 for an example means a fixed array geometry, i.e. all
                `ch` channels are valid for that example.

        Returns:
            (B*ch, output_size, dim1, dim2)
        """
        batch_size, ch, N, dim1, dim2 = input.shape
        output = input.view(batch_size * ch, N, dim1, dim2)
        for layer_idx in range(len(self.row_rnn)):
            output = self._intra_chunk_pass(output, layer_idx, batch_size, ch, dim1, dim2)
            output = self._inter_chunk_pass(output, layer_idx, batch_size, ch, dim1, dim2)
            output = self._cross_channel_tac(
                output, layer_idx, input.shape, num_mic, batch_size, ch, N, dim1, dim2
            )

        return self.output(output)  # B*ch, output_size, dim1, dim2

    def _intra_chunk_pass(
        self,
        output: torch.Tensor,
        layer_idx: int,
        batch_size: int,
        ch: int,
        dim1: int,
        dim2: int,
    ) -> torch.Tensor:
        """Row pass: RNN along dim1, with (channel, dim2) folded into batch."""
        row_input = (
            output.permute(0, 3, 2, 1)
            .contiguous()
            .view(batch_size * ch * dim2, dim1, -1)
        )  # B*ch*dim2, dim1, N
        row_output, _ = self.row_rnn[layer_idx](row_input)  # B*ch*dim2, dim1, N
        row_output = (
            row_output.view(batch_size * ch, dim2, dim1, -1)
            .permute(0, 3, 2, 1)
            .contiguous()
        )  # B*ch, N, dim1, dim2
        row_output = self.row_norm[layer_idx](row_output)
        return output + row_output  # B*ch, N, dim1, dim2

    def _inter_chunk_pass(
        self,
        output: torch.Tensor,
        layer_idx: int,
        batch_size: int,
        ch: int,
        dim1: int,
        dim2: int,
    ) -> torch.Tensor:
        """Col pass: RNN along dim2, with (channel, dim1) folded into batch."""
        col_input = (
            output.permute(0, 2, 3, 1)
            .contiguous()
            .view(batch_size * ch * dim1, dim2, -1)
        )  # B*ch*dim1, dim2, N
        col_output, _ = self.col_rnn[layer_idx](col_input)  # B*dim1, dim2, N
        col_output = (
            col_output.view(batch_size * ch, dim1, dim2, -1)
            .permute(0, 3, 1, 2)
            .contiguous()
        )  # B*ch, N, dim1, dim2
        col_output = self.col_norm[layer_idx](col_output)
        return output + col_output  # B*ch, N, dim1, dim2

    def _cross_channel_tac(
        self,
        output: torch.Tensor,
        layer_idx: int,
        input_shape: torch.Size,
        num_mic: torch.Tensor,
        batch_size: int,
        ch: int,
        N: int,
        dim1: int,
        dim2: int,
    ) -> torch.Tensor:
        """Transform-average-concatenate block for cross-channel communication.

        Each channel's features are transformed, averaged across the valid
        channels of each example, then concatenated back with the
        per-channel transform and projected to `input_size`.

        output: (B*ch, N, dim1, dim2)
        Returns: (B*ch, N, dim1, dim2)
        """
        ch_input = output.view(input_shape)  # B, ch, N, dim1, dim2
        ch_input = (
            ch_input.permute(0, 3, 4, 1, 2).contiguous().view(-1, N)
        )  # B*dim1*dim2*ch, N
        ch_output = self.ch_transform[layer_idx](ch_input).view(
            batch_size, dim1 * dim2, ch, -1
        )  # B, dim1*dim2, ch, H
        ch_mean = self._channel_mean(ch_output, num_mic, batch_size, dim1, dim2)
        ch_output = ch_output.view(
            batch_size * dim1 * dim2, ch, -1
        )  # B*dim1*dim2, ch, H
        ch_mean = (
            self.ch_average[layer_idx](ch_mean)
            .unsqueeze(1)
            .expand_as(ch_output)
            .contiguous()
        )  # B*dim1*dim2, ch, H
        ch_output = torch.cat([ch_output, ch_mean], 2)  # B*dim1*dim2, ch, 2H
        ch_output = self.ch_concat[layer_idx](
            ch_output.view(-1, ch_output.shape[-1])
        )  # B*dim1*dim2*ch, N
        ch_output = (
            ch_output.view(batch_size, dim1, dim2, ch, -1)
            .permute(0, 3, 4, 1, 2)
            .contiguous()
        )  # B, ch, N, dim1, dim2
        ch_output = self.ch_norm[layer_idx](
            ch_output.view(batch_size * ch, N, dim1, dim2)
        )  # B*ch, N, dim1, dim2
        return output + ch_output

    @staticmethod
    def _channel_mean(
        ch_output: torch.Tensor,
        num_mic: torch.Tensor,
        batch_size: int,
        dim1: int,
        dim2: int,
    ) -> torch.Tensor:
        """Average TAC features across the valid channels of each example.

        ch_output: (B, dim1*dim2, ch, H)
        Returns: (B*dim1*dim2, H)
        """
        if num_mic.max() == 0:
            # fixed geometry array: all channels are valid
            return ch_output.mean(2).view(batch_size * dim1 * dim2, -1)
        # variable geometry array: only average over the valid channels
        per_example_mean = [
            ch_output[b, :, : num_mic[b]].mean(1).unsqueeze(0)
            for b in range(batch_size)
        ]  # 1, dim1*dim2, H
        return torch.cat(per_example_mean, 0).view(batch_size * dim1 * dim2, -1)


def _pad_segment(input: torch.Tensor, segment_size: int) -> Tuple[torch.Tensor, int]:
    """Zero-pad a sequence so it can be split into 50%-overlapping segments.

    Pads the end so the sequence length is a multiple of `segment_size`,
    then pads `segment_size // 2` zeros on both ends so `split_feature` can
    extract full-length overlapping windows everywhere.

    Args:
        input: (B, N, T)
        segment_size: length of each segment.

    Returns:
        padded: (B, N, T'), T' >= T.
        rest: number of zero frames appended at the end (before the border
            padding), needed by `merge_feature` to trim the output back to
            length T.
    """
    batch_size, feat_dim, seq_len = input.shape
    segment_stride = segment_size // 2

    rest = segment_size - (segment_stride + seq_len % segment_size) % segment_size
    if rest > 0:
        pad = torch.zeros(
            batch_size, feat_dim, rest, dtype=input.dtype, device=input.device
        )
        input = torch.cat([input, pad], 2)

    pad_aux = torch.zeros(
        batch_size, feat_dim, segment_stride, dtype=input.dtype, device=input.device
    )
    input = torch.cat([pad_aux, input, pad_aux], 2)

    return input, rest


def split_feature(
    input: torch.Tensor, segment_size: int
) -> Tuple[torch.Tensor, int]:
    """Split a sequence into 50%-overlapping segments.

    Args:
        input: (B, N, T)
        segment_size: length K of each segment.

    Returns:
        segments: (B, N, K, S), S is the number of segments.
        rest: padding length to pass to `merge_feature`.
    """
    input, rest = _pad_segment(input, segment_size)
    batch_size, feat_dim, seq_len = input.shape
    segment_stride = segment_size // 2

    segments1 = (
        input[:, :, :-segment_stride]
        .contiguous()
        .view(batch_size, feat_dim, -1, segment_size)
    )
    segments2 = (
        input[:, :, segment_stride:]
        .contiguous()
        .view(batch_size, feat_dim, -1, segment_size)
    )
    segments = (
        torch.cat([segments1, segments2], 3)
        .view(batch_size, feat_dim, -1, segment_size)
        .transpose(2, 3)
    )

    return segments.contiguous(), rest


def merge_feature(input: torch.Tensor, rest: int) -> torch.Tensor:
    """Overlap-add segmented features back into a full sequence.

    Inverse of `split_feature`.

    Args:
        input: (B, N, K, S) segmented features (K: segment length,
            S: number of segments).
        rest: padding length returned by `split_feature`, trimmed from the
            end of the reconstructed sequence.

    Returns:
        (B, N, T)
    """
    batch_size, feat_dim, chunk_len, _ = input.shape
    segment_stride = chunk_len // 2
    input = (
        input.transpose(2, 3)
        .contiguous()
        .view(batch_size, feat_dim, -1, chunk_len * 2)
    )  # B, N, S, 2K

    input1 = (
        input[:, :, :, :chunk_len]
        .contiguous()
        .view(batch_size, feat_dim, -1)[:, :, segment_stride:]
    )
    input2 = (
        input[:, :, :, chunk_len:]
        .contiguous()
        .view(batch_size, feat_dim, -1)[:, :, :-segment_stride]
    )

    output = input1 + input2
    if rest > 0:
        output = output[:, :, :-rest]

    return output.contiguous()  # B, N, T
