"""Transformer-Decoder-based Attractor (TDA) with Rotary Position Embedding.

This is the RoPE variant of the attractor decoder described in:
    "Boosting Unknown-Number Speaker Separation with Transformer
    Decoder-Based Attractor" (SepTDA, ICASSP 2024).

It mirrors ``espnet2.enh.layers.tda`` but replaces ``nn.MultiheadAttention``
with a custom scaled-dot-product attention that optionally applies rotary
position embeddings (RoPE) to queries and keys before attending.
"""

import copy
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from rotary_embedding_torch import RotaryEmbedding

from espnet.nets.pytorch_backend.nets_utils import get_activation


def _get_clones(module: nn.Module, n: int) -> nn.ModuleList:
    """Return a ``ModuleList`` of ``n`` independent deep copies of ``module``."""
    return nn.ModuleList([copy.deepcopy(module) for _ in range(n)])


class MultiHeadAttention(nn.Module):
    """Scaled-dot-product multi-head attention with optional RoPE.

    Supports both self-attention (``key_value_input=None``) and
    cross-attention (query from ``input``, key/value from
    ``key_value_input``).
    """

    def __init__(
        self,
        emb_dim: int,
        attention_dim: int,
        n_heads: int = 8,
        dropout: float = 0.0,
        rope: Optional[RotaryEmbedding] = None,
        flash_attention: bool = False,
    ):
        super().__init__()
        self.n_heads = n_heads
        self.dropout = dropout
        self.rope = rope

        # Query is projected from `input`; key/value from `key_value_input`.
        self.q_proj = nn.Linear(emb_dim, attention_dim, bias=False)
        self.kv_proj = nn.Linear(emb_dim, attention_dim * 2, bias=False)

        self.aggregate_heads = nn.Sequential(
            nn.Linear(attention_dim, emb_dim, bias=False),
            nn.Dropout(dropout),
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
        self, input: torch.Tensor, key_value_input: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Project inputs to per-head query/key/value tensors.

        ``key_value_input=None`` means self-attention (key/value are
        projected from ``input`` itself); otherwise this is cross-attention.

        Returns:
            query: (B, n_heads, T_q, D_h)
            key:   (B, n_heads, T_kv, D_h)
            value: (B, n_heads, T_kv, D_h)
        """
        if key_value_input is None:
            key_value_input = input

        batch_size, t_q, _ = input.shape
        t_kv = key_value_input.size(1)
        d_head = self.q_proj.out_features // self.n_heads

        query = (
            self.q_proj(input)
            .reshape(batch_size, t_q, self.n_heads, d_head)
            .transpose(1, 2)
        )  # (B, T_q, H, D_h) -> (B, H, T_q, D_h)

        kv = (
            self.kv_proj(key_value_input)
            .reshape(batch_size, t_kv, 2, self.n_heads, d_head)
            .permute(2, 0, 3, 1, 4)
        )  # (B, T_kv, 2, H, D_h) -> (2, B, H, T_kv, D_h)

        key, value = kv[0], kv[1]
        return query, key, value

    @torch.cuda.amp.autocast(enabled=False)
    def _apply_rope(
        self, query: torch.Tensor, key: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        query = self.rope.rotate_queries_or_keys(query)
        key = self.rope.rotate_queries_or_keys(key)
        return query, key

    def forward(
        self,
        input: torch.Tensor,
        key_value_input: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
        flatten: bool = False,
    ) -> torch.Tensor:
        """Compute multi-head attention.

        Args:
            input: (B, T_q, D) query source.
            key_value_input: (B, T_kv, D) key/value source, or None to
                self-attend on ``input``.
            attn_mask: optional attention mask broadcastable to
                (B, n_heads, T_q, T_kv).
            flatten: if True, flatten the batch and head axes together
                before calling scaled-dot-product attention (used when an
                extra leading axis, e.g. a speaker axis, has been folded
                into the batch dimension upstream).

        Returns:
            (B, T_q, D) attention output.
        """
        query, key, value = self._project_qkv(input, key_value_input)

        if self.rope is not None:
            query, key = self._apply_rope(query, key)

        if flatten:
            q_shape = query.shape  # (B, H, T_q, D_h) before flattening
            query = query.flatten(0, 1)
            key = key.flatten(0, 1)
            value = value.flatten(0, 1)

        try:
            with torch.backends.cuda.sdp_kernel(**self.flash_attention_config):
                attn_output = F.scaled_dot_product_attention(
                    query=query,
                    key=key,
                    value=value,
                    attn_mask=attn_mask,
                    dropout_p=self.dropout if self.training else 0.0,
                )
        except Exception as e:
            print("scaled_dot_product_attention error:", e)
            raise

        if flatten:
            attn_output = attn_output.view(q_shape)

        # (B, H, T_q, D_h) -> (B, T_q, H*D_h)
        out = attn_output.transpose(1, 2).reshape(
            attn_output.size(0), attn_output.size(2), -1
        )
        return self.aggregate_heads(out)


class TransformerDecoderAttractorLayer(nn.Module):
    """One Transformer decoder layer used to refine attractor embeddings.

    Same structure as ``espnet2.enh.layers.tda.TransformerDecoderAttractorLayer``,
    but attention is computed with ``MultiHeadAttention`` (RoPE-aware)
    instead of ``nn.MultiheadAttention``:
      1. Causal self-attention among attractors.
      2. Cross-attention from attractors to source chunk features ``src``.
      3. A position-wise feed-forward network.
    Each sub-layer uses a residual connection followed by ``LayerNorm``.
    """

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        dropout: float = 0.0,
        activation: str = "relu",
        rope: Optional[RotaryEmbedding] = None,
    ):
        super(TransformerDecoderAttractorLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, d_model, nhead, dropout=dropout, rope=rope)
        self.attn = MultiHeadAttention(d_model, d_model, nhead, dropout=dropout, rope=rope)

        # Position-wise feed-forward network.
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        if activation.lower() == "linear":
            self.activation = nn.Identity()
        else:
            self.activation = get_activation(activation)

    def __setstate__(self, state):
        if "activation" not in state:
            state["activation"] = torch.nn.functional.relu
        super(TransformerDecoderAttractorLayer, self).__setstate__(state)

    @staticmethod
    def _causal_mask(num_attractors: int, device: torch.device) -> torch.Tensor:
        """Boolean mask blocking attractor ``i`` from attending to ``j > i``.

        Returns a ``(num_attractors, num_attractors)`` mask that is ``True``
        (masked out) strictly above the diagonal.
        """
        return torch.triu(
            torch.ones(num_attractors, num_attractors, device=device, dtype=torch.bool),
            diagonal=1,
        )

    def _self_attend(self, attractor: torch.Tensor) -> torch.Tensor:
        """Causal self-attention among attractors, shape (B, C, H) -> (B, C, H)."""
        num_attractors = attractor.size(1)
        mask = self._causal_mask(num_attractors, attractor.device)
        attn_out = self.self_attn(attractor, attn_mask=mask)
        attractor = attractor + self.dropout3(attn_out)
        return self.norm3(attractor)

    def _cross_attend(self, attractor: torch.Tensor, src: torch.Tensor) -> torch.Tensor:
        """Cross-attention from attractors (B, C, H) to source chunks (B, T, H)."""
        attn_out = self.attn(attractor, key_value_input=src)
        attractor = attractor + self.dropout1(attn_out)
        return self.norm1(attractor)

    def _feed_forward(self, attractor: torch.Tensor) -> torch.Tensor:
        ff_out = self.linear2(self.dropout(self.activation(self.linear1(attractor))))
        attractor = attractor + self.dropout2(ff_out)
        return self.norm2(attractor)

    def forward(self, attractor: torch.Tensor, src: torch.Tensor) -> torch.Tensor:
        """Refine attractors with one decoder layer.

        Args:
            attractor: (B, C, H) current attractor embeddings, ``C`` speakers.
            src: (B, T, H) source chunk features to attend to.

        Returns:
            (B, C, H) updated attractor embeddings.
        """
        attractor = self._self_attend(attractor)
        attractor = self._cross_attend(attractor, src)
        attractor = self._feed_forward(attractor)
        return attractor


class TransformerDecoderAttractor(nn.Module):
    """A stack of ``TransformerDecoderAttractorLayer`` modules."""

    def __init__(
        self,
        decoder_layer: nn.Module,
        num_layers: int,
        norm: Optional[nn.Module] = None,
        return_intermediate: bool = False,
    ):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate

    def forward(self, attractor: torch.Tensor, src: torch.Tensor) -> torch.Tensor:
        """Run all decoder layers.

        Args:
            attractor: (B, C, H) attractor embeddings.
            src: (B, T, H) source chunk features.

        Returns:
            If ``return_intermediate`` is False: (B, C, H) final attractors.
            Otherwise: (num_layers, B, C, H) normalized outputs of every layer.
        """
        output = attractor
        intermediate = []

        for layer in self.layers:
            output = layer(output, src)
            if self.return_intermediate:
                intermediate.append(self.norm(output))

        if self.norm is not None:
            output = self.norm(output)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)

        if self.return_intermediate:
            return torch.stack(intermediate)

        return output


class RopeAttractorDecode(nn.Module):
    """Autoregressive Transformer-Decoder Attractor (TDA) head, RoPE variant.

    Identical decoding procedure to ``espnet2.enh.layers.tda.AttractorDecode``:
    attractors are generated one at a time, each step attending over the
    source chunk features with the previously generated attractors, and
    predicting an existence probability. Positional information is injected
    via rotary embeddings inside the attention layers instead of learned
    positional encodings.
    """

    def __init__(
        self,
        hidden_size: int,
        n_heads: int = 4,
        dim_feedforward: int = 2048,
        dropout: float = 0.5,
        activation: str = "relu",
        depth: int = 2,
        chunk_shuffle: bool = True,  # Only shuffle when the T is chunk_num
    ):
        super(RopeAttractorDecode, self).__init__()
        self.depth = depth
        self.n_heads = n_heads
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout
        self.activation = activation
        rope = RotaryEmbedding(hidden_size // n_heads)
        decoder_layer = TransformerDecoderAttractorLayer(
            hidden_size, n_heads, dim_feedforward, dropout, activation, rope
        )
        self.transformerdecoder = TransformerDecoderAttractor(decoder_layer, depth)
        self.attractor_existence_estimator = nn.Sequential(
            nn.Linear(hidden_size, 1), nn.Sigmoid()
        )
        self.chunk_shuffle = chunk_shuffle

    def _maybe_shuffle_chunks(
        self, input: torch.Tensor, num_chunks: int
    ) -> torch.Tensor:
        """Randomly permute the chunk axis (dim=1) if ``chunk_shuffle`` is set."""
        if not self.chunk_shuffle:
            return input
        return input[..., torch.randperm(num_chunks), :]

    def _decode_step(
        self, attractor_output: torch.Tensor, r_input: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run the decoder once and append the newest attractor to the sequence.

        Args:
            attractor_output: (B, j, H) attractors generated so far.
            r_input: (B, T, H) source chunk features.

        Returns:
            attractor_output: (B, j+1, H) with the new attractor appended.
            latest_hidden: (B, H) hidden state of the newest attractor.
            existence_probability: (B,) probability that this attractor is valid.
        """
        output = self.transformerdecoder(attractor_output, r_input)
        latest_hidden = output[..., -1, :]  # (B, H)
        existence_probability = self.attractor_existence_estimator(latest_hidden)
        attractor_output = torch.cat(
            [attractor_output, latest_hidden.unsqueeze(1)], dim=1
        )
        return attractor_output, latest_hidden, existence_probability[..., 0]

    def _decode_until_stop(
        self,
        r_input: torch.Tensor,
        batch_size: int,
        hidden_size: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate attractors one by one until existence probability <= 0.5."""
        assert batch_size == 1, "We don't support batched computation in inference"
        attractor_output = torch.zeros(
            (batch_size, 1, hidden_size), device=r_input.device, dtype=r_input.dtype
        )
        hidden_states: List[torch.Tensor] = []
        existence_probabilities: List[torch.Tensor] = []

        existence_probability = 1
        while existence_probability > 0.5:
            attractor_output, latest_hidden, existence_probability = self._decode_step(
                attractor_output, r_input
            )
            hidden_states.append(latest_hidden)
            existence_probabilities.append(existence_probability)

        outputs = torch.stack(hidden_states, dim=1)  # (B, J, H)
        existence_probabilities = torch.stack(existence_probabilities, dim=1)  # (B, J)
        return outputs, existence_probabilities

    def _decode_fixed_steps(
        self,
        r_input: torch.Tensor,
        batch_size: int,
        hidden_size: int,
        num_spk: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate exactly ``num_spk + 1`` attractors (teacher-forced training)."""
        attractor_output = torch.zeros(
            (batch_size, 1, hidden_size), device=r_input.device, dtype=r_input.dtype
        )
        hidden_states: List[torch.Tensor] = []
        existence_probabilities: List[torch.Tensor] = []

        for _ in range(num_spk + 1):
            attractor_output, latest_hidden, existence_probability = self._decode_step(
                attractor_output, r_input
            )
            hidden_states.append(latest_hidden)
            existence_probabilities.append(existence_probability)

        outputs = torch.stack(hidden_states, dim=1)  # (B, J, H)
        existence_probabilities = torch.stack(existence_probabilities, dim=1)  # (B, J)
        return outputs, existence_probabilities

    def forward(
        self, input: torch.Tensor, num_spk: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate attractors from chunk features.

        Args:
            input: (B, num_chunks, H) per-chunk features.
            num_spk: if given, generate ``num_spk + 1`` attractors (training,
                the extra one models the "no more speakers" stop signal).
                If None, generate attractors autoregressively until the
                existence probability drops below 0.5 (inference; requires
                B == 1).

        Returns:
            outputs: (B, J, H) generated attractor embeddings.
            existence_probabilities: (B, J) existence probability per attractor.
        """
        batch_size, num_chunks, hidden_size = input.shape
        r_input = self._maybe_shuffle_chunks(input, num_chunks)

        if num_spk is None:
            return self._decode_until_stop(r_input, batch_size, hidden_size)
        return self._decode_fixed_steps(r_input, batch_size, hidden_size, num_spk)
