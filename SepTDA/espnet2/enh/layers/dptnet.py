"""Dual-path transformer network (DPTNet).

Implementation of the model proposed in:
    J. Chen, Q. Mao, and D. Liu, "Dual-path transformer network: Direct
    context-aware modeling for end-to-end monaural speech separation,"
    in Proc. ISCA Interspeech, 2020, pp. 2642-2646.

Ported from https://github.com/ujscjj/DPTNet

Like DPRNN (see `dprnn.py`), DPTNet alternates two passes over a chunked
feature map of shape (B, N, dim1, dim2): an intra-chunk pass along `dim1`
(chunk index folded into batch) and an inter-chunk pass along `dim2`
(intra-chunk position folded into batch). Each pass here is an
"improved" Transformer layer (self-attention + RNN-based feed-forward)
rather than a plain RNN.
"""

from typing import Optional

import torch
import torch.nn as nn

from espnet2.enh.layers.tcn import choose_norm
from espnet.nets.pytorch_backend.nets_utils import get_activation


class ImprovedTransformerLayer(nn.Module):
    """Container module of the (improved) Transformer proposed in [1].

    Combines multi-head self-attention with an RNN-based feed-forward
    sublayer (instead of the usual position-wise MLP), each followed by a
    residual connection and normalization.

    Reference:
        Dual-path transformer network: Direct context-aware modeling for end-to-end
        monaural speech separation; Chen et al, Interspeech 2020.

    Args:
        rnn_type (str): select from 'RNN', 'LSTM' and 'GRU'.
        input_size (int): Dimension of the input feature.
        att_heads (int): Number of attention heads.
        hidden_size (int): Dimension of the hidden state.
        dropout (float): Dropout ratio. Default is 0.
        activation (str): activation function applied at the output of RNN.
        bidirectional (bool, optional): True for bidirectional Inter-Chunk RNN
            (Intra-Chunk is always bidirectional).
        norm (str, optional): Type of normalization to use.
    """

    def __init__(
        self,
        rnn_type: str,
        input_size: int,
        att_heads: int,
        hidden_size: int,
        dropout: float = 0.0,
        activation: str = "relu",
        bidirectional: bool = True,
        norm: str = "gLN",
    ) -> None:
        super().__init__()

        rnn_type = rnn_type.upper()
        assert rnn_type in [
            "RNN",
            "LSTM",
            "GRU",
        ], f"Only support 'RNN', 'LSTM' and 'GRU', current type: {rnn_type}"
        self.rnn_type = rnn_type

        self.att_heads = att_heads
        self.self_attn = nn.MultiheadAttention(input_size, att_heads, dropout=dropout)
        self.dropout = nn.Dropout(p=dropout)
        self.norm_attn = choose_norm(norm, input_size)

        self.rnn = getattr(nn, rnn_type)(
            input_size,
            hidden_size,
            1,
            batch_first=True,
            bidirectional=bidirectional,
        )

        activation_fn = get_activation(activation)
        hdim = 2 * hidden_size if bidirectional else hidden_size
        self.feed_forward = nn.Sequential(
            activation_fn, nn.Dropout(p=dropout), nn.Linear(hdim, input_size)
        )

        self.norm_ff = choose_norm(norm, input_size)

    def forward(
        self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Self-attention sublayer followed by an RNN feed-forward sublayer.

        Args:
            x: (batch, seq, input_size)
            attn_mask: optional attention mask passed to `nn.MultiheadAttention`.

        Returns:
            (batch, seq, input_size)
        """
        src = x.permute(1, 0, 2)  # (batch, seq, input_size) -> (seq, batch, input_size)
        attn_out = self.self_attn(src, src, src, attn_mask=attn_mask)[0].permute(1, 0, 2)
        out = self.dropout(attn_out) + x
        out = self.norm_attn(out.transpose(-1, -2)).transpose(-1, -2)

        ff_out = self.feed_forward(self.rnn(out)[0])
        out2 = self.dropout(ff_out) + out
        return self.norm_ff(out2.transpose(-1, -2)).transpose(-1, -2)


class DPTNet(nn.Module):
    """Dual-path transformer network.

    Alternates an intra-chunk Transformer pass along `dim1` and an
    inter-chunk Transformer pass along `dim2`, for `num_layers` layers. A
    final PReLU + 1x1 Conv2d projects the feature dimension to
    `output_size`.

    args:
        rnn_type (str): select from 'RNN', 'LSTM' and 'GRU'.
        input_size (int): dimension of the input feature.
            Input size must be a multiple of `att_heads`.
        hidden_size (int): dimension of the hidden state.
        output_size (int): dimension of the output size.
        att_heads (int): number of attention heads.
        dropout (float): dropout ratio. Default is 0.
        activation (str): activation function applied at the output of RNN.
        num_layers (int): number of stacked RNN layers. Default is 1.
        bidirectional (bool): whether the RNN layers are bidirectional. Default is True.
        norm_type (str): type of normalization to use after each inter- or
            intra-chunk Transformer block.
    """

    def __init__(
        self,
        rnn_type: str,
        input_size: int,
        hidden_size: int,
        output_size: int,
        att_heads: int = 4,
        dropout: float = 0,
        activation: str = "relu",
        num_layers: int = 1,
        bidirectional: bool = True,
        norm_type: str = "gLN",
    ) -> None:
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # dual-path transformer
        self.row_transformer = nn.ModuleList()
        self.col_transformer = nn.ModuleList()
        for _ in range(num_layers):
            self.row_transformer.append(
                ImprovedTransformerLayer(
                    rnn_type,
                    input_size,
                    att_heads,
                    hidden_size,
                    dropout=dropout,
                    activation=activation,
                    bidirectional=True,
                    norm=norm_type,
                )
            )  # intra-segment RNN is always noncausal
            self.col_transformer.append(
                ImprovedTransformerLayer(
                    rnn_type,
                    input_size,
                    att_heads,
                    hidden_size,
                    dropout=dropout,
                    activation=activation,
                    bidirectional=bidirectional,
                    norm=norm_type,
                )
            )

        # output layer
        self.output = nn.Sequential(nn.PReLU(), nn.Conv2d(input_size, output_size, 1))

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Apply the dual-path Transformer stack.

        Args:
            input: (B, N, dim1, dim2), dim1 is the intra-chunk (segment)
                length and dim2 is the number of chunks.

        Returns:
            (B, output_size, dim1, dim2)
        """
        output = input
        for layer_idx in range(len(self.row_transformer)):
            output = self._intra_chunk_process(output, layer_idx)
            output = self._inter_chunk_process(output, layer_idx)

        return self.output(output)  # B, output_size, dim1, dim2

    def _intra_chunk_process(self, x: torch.Tensor, layer_index: int) -> torch.Tensor:
        """Row pass: Transformer along dim1, with dim2 folded into batch."""
        batch, N, chunk_size, n_chunks = x.size()
        x = x.transpose(1, -1).reshape(batch * n_chunks, chunk_size, N)
        x = self.row_transformer[layer_index](x)
        x = x.reshape(batch, n_chunks, chunk_size, N).permute(0, 3, 2, 1)
        return x

    def _inter_chunk_process(self, x: torch.Tensor, layer_index: int) -> torch.Tensor:
        """Col pass: Transformer along dim2, with dim1 folded into batch."""
        batch, N, chunk_size, n_chunks = x.size()
        x = x.permute(0, 2, 3, 1).reshape(batch * chunk_size, n_chunks, N)
        x = self.col_transformer[layer_index](x)
        x = x.reshape(batch, chunk_size, n_chunks, N).permute(0, 3, 1, 2)
        return x
