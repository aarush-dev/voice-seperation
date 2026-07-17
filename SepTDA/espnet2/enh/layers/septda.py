"""SepTDA separation backbone (improved Sepformer blocks).

Reference:
    "Boosting Unknown-Number Speaker Separation with Transformer
    Decoder-Based Attractor" (SepTDA, ICASSP 2024).

This module implements the dual-path / triple-path separation blocks used
by SepTDA. Each block interleaves an LSTM with self-attention along
different axes of a chunked representation:
  - intra-chunk: attend within a chunk (across the ``k`` axis).
  - inter-chunk: attend across chunks (across the ``s`` axis).
  - inter-speaker ("triple" blocks only): attend across the speaker axis
    ``c``, used once attractors have split the mixture into per-speaker
    streams.
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from espnet.nets.pytorch_backend.nets_utils import get_activation
from espnet2.enh.layers.tcn import choose_norm


class SepTDABlock(nn.Module):
    """A wrapper for the improved Sepformer block used in SepTDA.

    Reference:
        BOOSTING UNKNOWN-NUMBER SPEAKER SEPARATION WITH TRANSFORMER
        DECODER-BASED ATTRACTOR

    Args:
        rope_intra_chunk: rotary embedding instance for the intra-chunk
            attention, or None.
        rope_inter_chunk: rotary embedding instance for the inter-chunk
            attention, or None.
        block_type: "dual" (intra + inter chunk) or "triple" (also adds an
            inter-speaker sub-block).
        hidden_dim: feature dimension ``d``.
        dropout: dropout probability used throughout the block.
        activation: activation function name for the feed-forward layers.
        norm_type: normalization type passed to ``choose_norm``.
        rnn_type: "RNN", "LSTM", "GRU", or None to disable the RNN.
        rnn_dim: RNN hidden size.
        bidirectional: whether the RNN is bidirectional.
        att_heads: number of self-attention heads.
        attention_dim: total dimension of the self-attention projections.
        flash_attention: whether to prefer PyTorch's flash attention kernel.
        expansion_factor: feed-forward hidden size multiplier.
    """

    def __init__(
        self,
        rope_intra_chunk,
        rope_inter_chunk,
        block_type: str,  # "dual" or "triple"
        hidden_dim: int,
        dropout: float,
        activation: str,
        norm_type: str,
        # rnn setup
        rnn_type: Optional[str],
        rnn_dim: Optional[int],
        bidirectional: bool,
        # self-attention setup
        att_heads: int,
        attention_dim: int,
        flash_attention: bool,
        # ffn setup
        expansion_factor: int,
    ):
        super().__init__()
        self.intra_chunk = LSTMAttentionBlock(
            rope=rope_intra_chunk,
            hidden_dim=hidden_dim,
            norm_type=norm_type,
            dropout=dropout,
            activation=activation,
            rnn_type=rnn_type,
            bidirectional=bidirectional,
            rnn_dim=rnn_dim,
            att_heads=att_heads,
            attention_dim=attention_dim,
            flash_attention=flash_attention,
            expansion_factor=expansion_factor,
        )
        self.inter_chunk = LSTMAttentionBlock(
            rope=rope_inter_chunk,
            hidden_dim=hidden_dim,
            norm_type=norm_type,
            dropout=dropout,
            activation=activation,
            rnn_type=rnn_type,
            bidirectional=bidirectional,
            rnn_dim=rnn_dim,
            att_heads=att_heads,
            attention_dim=attention_dim,
            flash_attention=flash_attention,
            expansion_factor=expansion_factor,
        )
        self.block_type = block_type
        if block_type == "triple":
            self.inter_spk = LSTMAttentionBlock(
                rope=None,
                hidden_dim=hidden_dim,
                norm_type=norm_type,
                dropout=dropout,
                activation=activation,
                rnn_type=None,
                bidirectional=False,
                rnn_dim=None,
                att_heads=att_heads,
                attention_dim=attention_dim,
                flash_attention=flash_attention,
                expansion_factor=expansion_factor,
            )
        self.norm_output = choose_norm(norm_type, hidden_dim)

    def _forward_dual(self, x: torch.Tensor) -> torch.Tensor:
        """intra-chunk -> inter-chunk, over a (B, k, s, d) tensor."""
        residual = x
        x = rearrange(x, "b k s d -> b s k d")
        x = self.intra_chunk(x)
        x = rearrange(x, "b s k d -> b k s d")
        x = self.inter_chunk(x)
        x = x + residual
        return self.norm_output(x)

    def _forward_triple(self, x: torch.Tensor) -> torch.Tensor:
        """intra-chunk -> inter-chunk -> inter-speaker, over (B, c, k, s, d)."""
        b, c, k, s, d = x.shape
        residual = x
        x = rearrange(x, "b c k s d -> b (c s) k d", c=c, s=s)
        x = self.intra_chunk(x)
        x = rearrange(x, "b (c s) k d -> b (c k) s d", c=c, s=s, k=k)
        x = self.inter_chunk(x)
        x = rearrange(x, "b (c k) s d -> b (k s) c d", c=c, k=k, s=s)
        x = self.inter_spk(x, flatten=True)
        x = rearrange(x, "b (k s) c d -> b c k s d", c=c, k=k, s=s)
        x = x + residual
        return self.norm_output(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """permute -> intra_chunk -> permute -> inter_chunk -> [-> inter_spk] -> norm_output.

        Args:
            x: (B, k, s, d) for "dual" blocks, or (B, c, k, s, d) for
                "triple" blocks, where ``k``/``s`` are the two chunking
                axes and ``c`` is the speaker axis.

        Returns:
            Tensor of the same shape as ``x``.
        """
        if self.block_type == "dual":
            assert len(x.shape) == 4, f"Expected 4D input, but got {len(x.shape)}D input"
            return self._forward_dual(x)
        elif self.block_type == "triple":
            assert len(x.shape) == 5, f"Expected 5D input, but got {len(x.shape)}D input"
            return self._forward_triple(x)
        raise ValueError(f"Unknown block_type: {self.block_type}")


class LSTMAttentionBlock(nn.Module):
    """A wrapper for the LSTM-attention block used in SepTDA.

    Reference:
        BOOSTING UNKNOWN-NUMBER SPEAKER SEPARATION WITH TRANSFORMER
        DECODER-BASED ATTRACTOR

    Args:
        rnn_type (str): select from 'RNN', 'LSTM' and 'GRU'.
        hidden_dim (int): Dimension of the input feature.
        att_heads (int): Number of attention heads.
        rnn_dim (int): Dimension of the hidden state.
        dropout (float): Dropout ratio. Default is 0.
        activation (str): activation function applied at the output of RNN.
        bidirectional (bool, optional): True for bidirectional Inter-Chunk RNN
            (Intra-Chunk is always bidirectional).
        norm_type (str, optional): Type of normalization to use.
    """

    def __init__(
        self,
        rope,
        # general params
        hidden_dim: int,
        norm_type: str,
        dropout: float,
        activation: str,
        # rnn related
        rnn_type: Optional[str],
        bidirectional: bool,
        rnn_dim: Optional[int],
        # attention related
        att_heads: int,
        attention_dim: int,
        flash_attention: bool,
        # feed forward related
        expansion_factor: int,
    ):
        super().__init__()

        assert rnn_type in [
            "RNN",
            "LSTM",
            "GRU",
            None,
        ], f"Only support 'RNN', 'LSTM' and 'GRU', current type: {rnn_type}"
        self.rnn_type = rnn_type
        if rnn_type is not None:
            self.rnn = getattr(nn, rnn_type)(
                hidden_dim,
                rnn_dim,
                1,
                batch_first=True,
                bidirectional=bidirectional,
            )
            hdim = 2 * rnn_dim if bidirectional else rnn_dim
            self.linear_rnn = nn.Linear(hdim, hidden_dim)
            self.norm_rnn = choose_norm(norm_type, hidden_dim)

        self.attn = MultiHeadSelfAttention(
            hidden_dim,
            attention_dim=attention_dim,
            n_heads=att_heads,
            rope=rope,
            dropout=dropout,
            flash_attention=flash_attention,
        )
        self.dropout = nn.Dropout(p=dropout)
        self.norm_attn = choose_norm(norm_type, hidden_dim)

        self.feed_forward = PositionalwiseFeedForward(
            d_ffn=hidden_dim * expansion_factor,
            input_size=hidden_dim,
            dropout=dropout,
            activation=activation,
        )
        self.norm_ff = choose_norm(norm_type, hidden_dim)

        self.norm_output = choose_norm(norm_type, hidden_dim)

    def _apply_rnn_sublayer(self, x: torch.Tensor) -> torch.Tensor:
        """Pre-norm RNN + linear projection + residual, shape (BM, T, C)."""
        if self.rnn_type is None:
            return x
        residual = x
        x = self.norm_rnn(x)
        x, _ = self.rnn(x)
        x = self.linear_rnn(x)
        x = self.dropout(x)
        return x + residual

    def _apply_attention_sublayer(self, x: torch.Tensor, flatten: bool) -> torch.Tensor:
        """Pre-norm self-attention + residual, shape (BM, T, C)."""
        residual = x
        x = self.norm_attn(x)
        x = self.attn(x, flatten=flatten)
        x = self.dropout(x)
        return x + residual

    def _apply_feed_forward_sublayer(self, x: torch.Tensor) -> torch.Tensor:
        """Pre-norm position-wise feed-forward + residual, shape (BM, T, C)."""
        residual = x
        x = self.norm_ff(x)
        x = self.feed_forward(x)
        x = self.dropout(x)
        return x + residual

    def forward(self, x: torch.Tensor, flatten: bool = False) -> torch.Tensor:
        """LSTM + Residual -> MultiheadAttention + Residual -> FFN + Residual -> Norm.

        Args:
            x: Tensor of shape (B, M, T, C), where ``M`` is folded into the
                batch dimension before the RNN/attention/FFN sub-layers and
                restored afterwards.
            flatten: forwarded to the self-attention sub-layer; see
                ``MultiHeadSelfAttention.forward``.

        Returns:
            Tensor of shape (B, M, T, C).
        """
        batch_size, num_groups, seq_len, channels = x.shape
        x = rearrange(x, "b m t c -> (b m) t c")

        x = self._apply_rnn_sublayer(x)
        x = self._apply_attention_sublayer(x, flatten)
        x = self._apply_feed_forward_sublayer(x)

        x = rearrange(x, "(b m) t c -> b m t c", b=batch_size, m=num_groups)
        return self.norm_output(x)


class PositionalwiseFeedForward(nn.Module):
    """The class implements the positional-wise feed forward module in
    “Attention Is All You Need”.

    Arguments
    ---------
    d_ffn: int
        Hidden layer size.
    input_shape : tuple, optional
        Expected shape of the input. Alternatively use ``hidden_dim``.
    hidden_dim : int, optional
        Expected size of the input. Alternatively use ``input_shape``.
    dropout: float, optional
        Dropout rate.
    activation: torch.nn.Module, optional
        activation functions to be applied (Recommendation: ReLU, GELU).

    Example
    -------
    >>> inputs = torch.rand([8, 60, 512])
    >>> net = PositionalwiseFeedForward(256, hidden_dim=inputs.shape[-1])
    >>> outputs = net(inputs)
    >>> outputs.shape
    torch.Size([8, 60, 512])
    """

    def __init__(
        self,
        d_ffn,
        input_shape=None,
        input_size=None,
        output_size=None,
        dropout=0.0,
        activation=nn.ReLU,
    ):
        super().__init__()

        if input_shape is None and input_size is None:
            raise ValueError("Expected one of input_shape or input_size")

        if input_size is None:
            input_size = input_shape[-1]
        if output_size is None:
            output_size = input_size
        self.ffn = nn.Sequential(
            nn.Linear(input_size, d_ffn),
            get_activation(activation),
            nn.Dropout(dropout),
            nn.Linear(d_ffn, output_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Applies PositionalwiseFeedForward to the input tensor x."""
        x = self.ffn(x)
        return x


class MultiHeadSelfAttention(nn.Module):
    """Scaled-dot-product multi-head self-attention with optional RoPE."""

    def __init__(
        self,
        emb_dim: int,
        attention_dim: int,
        n_heads: int = 8,
        dropout: float = 0.0,
        rope=None,
        flash_attention: bool = False,
    ):
        super().__init__()

        self.n_heads = n_heads
        self.dropout = dropout

        self.rope = rope
        self.qkv = nn.Linear(emb_dim, attention_dim * 3, bias=False)
        self.aggregate_heads = nn.Sequential(
            nn.Linear(attention_dim, emb_dim, bias=False), nn.Dropout(dropout)
        )

        if flash_attention:
            self.flash_attention_config = dict(
                enable_flash=True, enable_math=False, enable_mem_efficient=False
            )
        else:
            self.flash_attention_config = dict(
                enable_flash=False, enable_math=True, enable_mem_efficient=True
            )

    def _project_qkv(
        self, input: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Project ``input`` to per-head query/key/value.

        Returns tensors of shape (B, n_heads, T, D_h).
        """
        n_batch, seq_len = input.shape[:2]
        x = self.qkv(input).reshape(n_batch, seq_len, 3, self.n_heads, -1)
        x = x.movedim(-2, 1)  # (B, n_heads, T, 3, D_h)
        query, key, value = x[..., 0, :], x[..., 1, :], x[..., 2, :]
        return query, key, value

    @torch.cuda.amp.autocast(enabled=False)
    def _apply_rope(
        self, query: torch.Tensor, key: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        query = self.rope.rotate_queries_or_keys(query)
        key = self.rope.rotate_queries_or_keys(key)
        return query, key

    def _log_attention_failure(
        self,
        exc: Exception,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> None:
        print("scaled_dot_product_attention failed with exception:")
        print(exc)
        print("--- Debug info ---")
        print("query shape:", query.shape, "dtype:", query.dtype, "device:", query.device)
        print("key shape:", key.shape, "dtype:", key.dtype, "device:", key.device)
        print("value shape:", value.shape, "dtype:", value.dtype, "device:", value.device)
        print("Expected dtype: float16 or bfloat16, head_dim % 8 == 0")
        print("flash_attention_config:", self.flash_attention_config)

    def forward(self, input: torch.Tensor, flatten: bool = False) -> torch.Tensor:
        """Compute multi-head self-attention.

        Args:
            input: (B, T, D) input features.
            flatten: if True, flatten the batch and head axes together
                before calling scaled-dot-product attention (used when an
                extra leading axis, e.g. a speaker axis, has been folded
                into the batch dimension upstream).

        Returns:
            (B, T, D) attention output.
        """
        query, key, value = self._project_qkv(input)

        if self.rope is not None:
            query, key = self._apply_rope(query, key)

        if flatten:
            q_shape = query.shape  # (B, n_heads, T, D_h) before flattening
            query = query.flatten(0, 1)
            key = key.flatten(0, 1)
            value = value.flatten(0, 1)

        try:
            with torch.backends.cuda.sdp_kernel(**self.flash_attention_config):
                output = F.scaled_dot_product_attention(
                    query=query,
                    key=key,
                    value=value,
                    attn_mask=None,
                    dropout_p=self.dropout if self.training else 0.0,
                )
        except Exception as e:
            self._log_attention_failure(e, query, key, value)
            raise

        if flatten:
            output = output.view(q_shape)

        output = output.transpose(1, 2)  # (B, T, n_heads, D_h)
        output = output.reshape(output.shape[:2] + (-1,))
        return self.aggregate_heads(output)
