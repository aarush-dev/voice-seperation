"""DPTNet-based MUSE model with a Transformer-decoder attractor (TDA).

The implementation of the DPTNet-based MUSE model proposed in
***, in Proc. IEEE ASRU 2023.

This module wires a dual-path Transformer (see `dptnet.py`) together with a
mid-network attractor module that both estimates the number of active
speakers and produces one gating "attractor" vector per speaker
(`AttractorDecode`/`RopeAttractorDecode` from `tda.py`/`rope_tda.py`, a
Transformer-decoder-based attractor, as opposed to the LSTM
encoder-decoder attractor used by the sibling module `dptnet_eda.py`).

Speaker separation proceeds as:
    1. Run `i_tda_layer` dual-path Transformer layers over the mixture.
    2. Aggregate each chunk into a single vector (`SequenceAggregation`)
       and decode attractors + speaker-existence probabilities from the
       aggregated sequence (`self.tda`).
    3. Gate the shared feature map with each speaker's attractor,
       broadcasting the speaker axis into the batch dimension so the
       remaining dual-path layers process one speaker's stream at a time.

Note: unlike `DPTNet_EDA_Informed.forward`, `DPTNet_TDA_Informed.forward`
constructs `self.adapt_layer` (for enrollment-conditioned target-speaker
extraction) in `__init__` but never applies it in `forward` — the
`enroll_emb` argument is accepted for interface parity but is unused.
"""

from typing import Optional

import torch
import torch.nn as nn

from espnet2.enh.layers.adapt_layers import make_adapt_layer
from espnet2.enh.layers.dptnet import ImprovedTransformerLayer
from espnet2.enh.layers.rope_tda import RopeAttractorDecode
from espnet2.enh.layers.tda import AttractorDecode

attractors = {
    "tda": AttractorDecode,
    "ropetda": RopeAttractorDecode,
}


def _apply_intra_chunk_transformer(
    x: torch.Tensor, transformer_layer: nn.Module
) -> torch.Tensor:
    """Row pass shared by the dual-path blocks in this module.

    Runs `transformer_layer` along dim1 (chunk length), folding dim2
    (chunk index) into the batch dimension.

    x: (batch, N, chunk_size, n_chunks) -> (batch, N, chunk_size, n_chunks)
    """
    batch, N, chunk_size, n_chunks = x.size()
    x = x.transpose(1, -1).contiguous().view(batch * n_chunks, chunk_size, N)
    x = transformer_layer(x)
    x = x.reshape(batch, n_chunks, chunk_size, N).permute(0, 3, 2, 1)
    return x


def _apply_inter_chunk_transformer(
    x: torch.Tensor, transformer_layer: nn.Module
) -> torch.Tensor:
    """Col pass shared by the dual-path blocks in this module.

    Runs `transformer_layer` along dim2 (chunk index), folding dim1
    (chunk length) into the batch dimension.

    x: (batch, N, chunk_size, n_chunks) -> (batch, N, chunk_size, n_chunks)
    """
    batch, N, chunk_size, n_chunks = x.size()
    x = x.permute(0, 2, 3, 1).contiguous().view(batch * chunk_size, n_chunks, N)
    x = transformer_layer(x)
    x = x.view(batch, chunk_size, n_chunks, N).permute(0, 3, 1, 2)
    return x


class DPTNet_TDA_Informed(nn.Module):
    """Dual-path transformer network with a mid-network TDA attractor block.

    args:
        rnn_type (str): select from 'RNN', 'LSTM' and 'GRU'.
        input_size (int): dimension of the input feature.
            Input size must be a multiple of `att_heads`.
        hidden_size (int): dimension of the hidden state.
        output_size (int): unused; the output feature dimension is always
            `input_size` (kept for interface parity, see `self.output_size`).
        att_heads (int): number of attention heads.
        dropout (float): dropout ratio. Default is 0.
        activation (str): activation function applied at the output of RNN.
        num_layers (int): number of stacked dual-path Transformer layers.
        bidirectional (bool): whether the inter-chunk Transformer's RNN
            sublayer is bidirectional. Default is True.
        norm_type (str): type of normalization to use after each inter- or
            intra-chunk Transformer block.
        i_tda_layer (int, optional): index (0-based) of the dual-path layer
            after which the TDA attractor block is applied. If None, no
            attractor block is used.
        num_tda_modules (int): must be odd; kept for interface parity with
            the (unused) ensembling logic that used to live in this class.
        attractor (str): "tda" for a plain Transformer-decoder attractor, or
            "ropetda" for the rotary-position variant.
        i_adapt_layer (int, optional): index of the dual-path layer after
            which the enrollment adapt layer would be applied. Configured
            but not used by `forward` in this class.
        adapt_layer_type (str): "attn" or "attn_improved".
        adapt_enroll_dim (int): enrollment embedding dimension.
        adapt_attention_dim (int): attention dimension inside the adapt layer.
        adapt_hidden_dim (int): hidden dimension inside the adapt layer.
        adapt_softmax_temp (float): softmax temperature inside the adapt layer.
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
        i_tda_layer: int = 4,
        num_tda_modules: int = 1,
        attractor: str = "tda",  # "tda" or "ropetda"
        i_adapt_layer: int = 4,
        adapt_layer_type: str = "attn",
        adapt_enroll_dim: int = 64,
        adapt_attention_dim: int = 512,
        adapt_hidden_dim: int = 512,
        adapt_softmax_temp: float = 1,
    ) -> None:
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = input_size

        # dual-path transformer
        self.row_transformer = nn.ModuleList()
        self.col_transformer = nn.ModuleList()
        self.chan_transformer = nn.ModuleList()
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
        self.output = nn.Sequential(
            nn.PReLU(), nn.Conv2d(input_size, output_size, 1)
        )

        # tda related params
        self.i_tda_layer = i_tda_layer
        self.num_tda_modules = num_tda_modules
        assert (
            self.num_tda_modules % 2 == 1
        ), "number of tda modules should be odd number"
        if i_tda_layer is not None:
            self.sequence_aggregation = SequenceAggregation(input_size)
            self.tda = attractors[attractor](
                input_size,
                att_heads,
                hidden_size,
                dropout,
                activation,
            )

        # tse related params
        self.i_adapt_layer = i_adapt_layer
        if i_adapt_layer is not None:
            assert adapt_layer_type in ["attn", "attn_improved"]
            self.adapt_enroll_dim = adapt_enroll_dim
            self.adapt_layer_type = adapt_layer_type
            # set parameters
            adapt_layer_params = {
                "attention_dim": adapt_attention_dim,
                "hidden_dim": adapt_hidden_dim,
                "softmax_temp": adapt_softmax_temp,
                "is_dualpath_process": True,
            }
            # prepare additional processing block
            if adapt_layer_type == "attn_improved":
                self.conditioning_model = ConditionalDPTNet(
                    rnn_type=rnn_type,
                    input_size=input_size,
                    hidden_size=hidden_size,
                    output_size=input_size,
                    att_heads=att_heads,
                    dropout=dropout,
                    activation=activation,
                    num_layers=2,
                    bidirectional=bidirectional,
                    norm_type=norm_type,
                    enroll_size=input_size,
                    conditioning_size=512,
                )
                adapt_layer_type = "attn"
            # load speaker selection module
            self.adapt_layer = make_adapt_layer(
                adapt_layer_type,
                indim=input_size,
                enrolldim=adapt_enroll_dim,
                ninputs=1,
                adapt_layer_kwargs=adapt_layer_params,
            )

    def forward(
        self,
        input: torch.Tensor,
        enroll_emb: Optional[torch.Tensor],
        num_spk: Optional[int] = None,
    ):
        """Apply the dual-path Transformer stack and the TDA attractor block.

        Args:
            input: (B, N, dim1, dim2) chunked features. N is the feature
                dimension, dim1 is the intra-chunk (segment) length (K),
                dim2 is the number of chunks (S).
            enroll_emb: unused (see class docstring); accepted only for
                interface parity with `DPTNet_EDA_Informed`.
            num_spk: if given, `self.tda` decodes exactly `num_spk + 1`
                attractors (the last is a "stop" attractor used only to
                supervise the existence-probability loss); otherwise it
                decodes attractors one at a time until the existence
                probability drops below 0.5.

        Returns:
            output: (B, output_size, dim1, dim2) if `i_tda_layer` is None,
                else (B * J, output_size, dim1, dim2) where J is the number
                of speaker attractors used to gate the feature map.
            probabilities: (B, J + 1) speaker-existence probabilities from
                the TDA module, or None if `i_tda_layer` is None.
        """
        output = input
        batch, hidden_dim, dim1, dim2 = output.shape
        probabilities = None
        for layer_idx in range(len(self.row_transformer)):
            output = _apply_intra_chunk_transformer(
                output, self.row_transformer[layer_idx]
            )  # (b, N, chunk_size, n_chunks)
            output = _apply_inter_chunk_transformer(
                output, self.col_transformer[layer_idx]
            )  # (b, N, chunk_size, n_chunks)

            if layer_idx == self.i_tda_layer:
                output, probabilities = self._apply_attractor_gate(
                    output, hidden_dim, dim1, dim2, num_spk
                )

        output = self.output(output)  # B, output_size, dim1, dim2
        return output, probabilities

    def _apply_attractor_gate(
        self,
        output: torch.Tensor,
        hidden_dim: int,
        dim1: int,
        dim2: int,
        num_spk: Optional[int],
    ):
        """Decode attractors and gate the shared feature map with them.

        output: (B, N, dim1, dim2) -> ((B * J), N, dim1, dim2), where J is
        the number of speaker attractors kept (decoded attractors minus the
        trailing "stop" attractor).
        """
        aggregated_sequence = self.sequence_aggregation(
            output.transpose(-1, -3)
        )  # (B, N, chunk_size, n_chunks) -> (B, n_chunks, N)
        attractor_vectors, probabilities = self.tda(
            aggregated_sequence, num_spk=num_spk
        )
        gated = (
            output[..., None, :, :, :] * attractor_vectors[..., :-1, :, None, None]
        )  # (B, J, N, dim1, dim2)
        gated = gated.view(-1, hidden_dim, dim1, dim2)  # (B * J, N, dim1, dim2)
        return gated, probabilities


class ConditionalDPTNet(nn.Module):
    """Dual-path transformer network with FiLM enrollment conditioning.

    Applied before each dual-path layer, `self.film[i]` modulates the
    feature map with the enrollment embedding (FiLM: feature-wise linear
    modulation) before the intra-/inter-chunk Transformer passes. Unlike
    `DPTNet`, this class has no final projection layer: it is meant to be
    used as an additional conditioning block whose output stays in
    `input_size` feature dimension.

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
        # film related
        enroll_size: Optional[int] = None,
        conditioning_size: Optional[int] = None,
    ) -> None:
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        if enroll_size is None:
            enroll_size = input_size
        if conditioning_size is None:
            conditioning_size = input_size
        # dual-path transformer
        self.row_transformer = nn.ModuleList()
        self.col_transformer = nn.ModuleList()
        self.film = nn.ModuleList()
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
            self.film.append(
                FiLM(
                    input_size,
                    enroll_size,
                    conditioning_size,
                )
            )

    def forward(self, input: torch.Tensor, enroll_emb: torch.Tensor) -> torch.Tensor:
        """Apply FiLM conditioning followed by the dual-path Transformer stack.

        Args:
            input: (B, N, dim1, dim2) chunked features.
            enroll_emb: enrollment embedding broadcastable against `input`
                after `input` is transposed to put N last (see `FiLM`).

        Returns:
            (B, N, dim1, dim2)
        """
        output = input
        for layer_idx in range(len(self.row_transformer)):
            output = self.film[layer_idx](
                output.transpose(-1, -3), enroll_emb
            ).transpose(-1, -3)
            output = _apply_intra_chunk_transformer(
                output, self.row_transformer[layer_idx]
            )
            output = _apply_inter_chunk_transformer(
                output, self.col_transformer[layer_idx]
            )

        return output


class FiLM(nn.Module):
    """Feature-wise linear modulation, conditioned on an enrollment embedding."""

    def __init__(
        self,
        indim: int,
        enrolldim: int,
        filmdim: int,
        skip_connection: bool = True,
    ) -> None:
        super().__init__()
        self.linear1 = nn.Sequential(
            nn.Linear(indim, filmdim),
            nn.PReLU(),
        )
        self.film_gamma = nn.Linear(enrolldim, filmdim)
        self.film_beta = nn.Linear(enrolldim, filmdim)
        self.linear2 = nn.Linear(filmdim, indim)
        self.skip_connection = skip_connection

    def forward(self, input: torch.Tensor, enroll_emb: torch.Tensor) -> torch.Tensor:
        """Modulate `input` with an affine transform predicted from `enroll_emb`.

        Args:
            input: (..., indim)
            enroll_emb: (..., enrolldim), broadcastable against `input`.

        Returns:
            (..., indim)
        """
        gamma = self.film_gamma(enroll_emb)
        beta = self.film_beta(enroll_emb)
        output = self.linear1(input)
        output = output * gamma + beta
        output = self.linear2(output)
        if self.skip_connection:
            output = output + input
        return output


class SequenceAggregation(nn.Module):
    """Attention-pooling that collapses the intra-chunk axis to a vector.

    For each chunk, learns `r` attention distributions over the chunk's K
    positions, pools the (per-position) `hidden_size // r`-dim projected
    features with each distribution, and concatenates the r pooled vectors
    back to `hidden_size`.
    """

    def __init__(self, hidden_size: int, r: int = 4) -> None:
        super(SequenceAggregation, self).__init__()
        self.path1 = nn.Sequential(
            nn.Linear(hidden_size, r * hidden_size),
            nn.Tanh(),
            nn.Linear(r * hidden_size, r),
            nn.Softmax(dim=-1),
        )
        self.linear = nn.Linear(hidden_size, hidden_size // r)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Pool each chunk's K positions into a single hidden_size vector.

        Args:
            input: (B, S, K, hidden_size), S chunks of length K.

        Returns:
            (B, S, hidden_size)
        """
        batch_size, num_segments, segment_size, hidden_size = input.shape
        attn_weights = self.path1(input)  # (B, S, K, r)
        pooled = torch.matmul(
            attn_weights.transpose(-1, -2), self.linear(input)
        ).reshape(
            batch_size, num_segments, hidden_size
        )  # (B, S, r, hidden_size // r) -> (B, S, hidden_size)
        return pooled
