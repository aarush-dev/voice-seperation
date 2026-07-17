"""Transformer-Decoder-based Attractor (TDA) module.

This implements the attractor generation mechanism from:
    "Boosting Unknown-Number Speaker Separation with Transformer
    Decoder-Based Attractor" (SepTDA, ICASSP 2024).

An attractor is a fixed-size embedding representing one speaker in the
mixture. Attractors are produced autoregressively: a Transformer decoder
attends over the separated chunk representations (``src``) and, at each
step, emits one new attractor plus a probability that this attractor is
the last valid one (i.e. that the speaker count has been exhausted).
"""

import copy
from typing import List, Optional, Tuple

import torch
import torch.nn as nn

from espnet.nets.pytorch_backend.nets_utils import get_activation


def _get_clones(module: nn.Module, n: int) -> nn.ModuleList:
    """Return a ``ModuleList`` of ``n`` independent deep copies of ``module``."""
    return nn.ModuleList([copy.deepcopy(module) for _ in range(n)])


class TransformerDecoderAttractorLayer(nn.Module):
    """One Transformer decoder layer used to refine attractor embeddings.

    Each layer performs, in order:
      1. Causal multi-head self-attention among attractors, so attractor
         ``i`` only sees attractors ``0..i`` (enforces the autoregressive,
         one-speaker-at-a-time generation order).
      2. Multi-head cross-attention from attractors to the source chunk
         features ``src``.
      3. A position-wise feed-forward network.

    Each sub-layer uses a residual connection followed by ``LayerNorm``
    (post-norm), matching the original Transformer decoder design.
    """

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        dropout: float = 0.0,
        activation: str = "relu",
    ):
        super(TransformerDecoderAttractorLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )
        self.attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )

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
        attn_out = self.self_attn(attractor, attractor, attractor, attn_mask=mask)[0]
        attractor = attractor + self.dropout3(attn_out)
        return self.norm3(attractor)

    def _cross_attend(
        self,
        attractor: torch.Tensor,
        src: torch.Tensor,
        src_key_padding_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Cross-attention from attractors (B, C, H) to source chunks (B, T, H)."""
        attn_out = self.attn(
            attractor, src, src, key_padding_mask=src_key_padding_mask
        )[0]
        attractor = attractor + self.dropout1(attn_out)
        return self.norm1(attractor)

    def _feed_forward(self, attractor: torch.Tensor) -> torch.Tensor:
        ff_out = self.linear2(self.dropout(self.activation(self.linear1(attractor))))
        attractor = attractor + self.dropout2(ff_out)
        return self.norm2(attractor)

    def forward(
        self,
        attractor: torch.Tensor,
        src: torch.Tensor,
        src_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Refine attractors with one decoder layer.

        Args:
            attractor: (B, C, H) current attractor embeddings, ``C`` speakers.
            src: (B, T, H) source chunk features to attend to.
            src_key_padding_mask: optional (B, T) padding mask for ``src``.

        Returns:
            (B, C, H) updated attractor embeddings.
        """
        attractor = self._self_attend(attractor)
        attractor = self._cross_attend(attractor, src, src_key_padding_mask)
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

    def forward(
        self,
        attractor: torch.Tensor,
        src: torch.Tensor,
        src_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Run all decoder layers.

        Args:
            attractor: (B, C, H) attractor embeddings.
            src: (B, T, H) source chunk features.
            src_key_padding_mask: optional (B, T) padding mask for ``src``.

        Returns:
            If ``return_intermediate`` is False: (B, C, H) final attractors.
            Otherwise: (num_layers, B, C, H) normalized outputs of every layer.
        """
        output = attractor
        intermediate = []

        for layer in self.layers:
            output = layer(output, src, src_key_padding_mask)
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


class AttractorDecode(nn.Module):
    """Autoregressive Transformer-Decoder Attractor (TDA) head.

    Given per-chunk features, this module generates attractors one at a
    time: at step ``j`` the decoder attends to all source chunks using the
    attractors produced so far (starting from a single zero vector) and
    emits attractor ``j`` together with a probability that speaker ``j``
    exists. Generation either runs for a fixed number of steps (training,
    when the number of speakers is known) or until the existence
    probability drops below 0.5 (inference).
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
        super(AttractorDecode, self).__init__()
        self.depth = depth
        self.n_heads = n_heads
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout
        self.activation = activation
        decoder_layer = TransformerDecoderAttractorLayer(
            hidden_size, n_heads, dim_feedforward, dropout, activation
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
