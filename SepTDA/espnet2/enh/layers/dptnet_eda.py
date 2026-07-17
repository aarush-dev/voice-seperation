"""DPTNet-based MUSE model with an LSTM encoder-decoder attractor (EDA).

The implementation of the DPTNet-based MUSE model proposed in
***, in Proc. IEEE ASRU 2023.

This module wires a dual-path Transformer (see `dptnet.py`) together with a
mid-network attractor module that both estimates the number of active
speakers and produces one gating "attractor" vector per speaker
(`EncoderDecoderAttractor`, an LSTM encoder-decoder attractor as used in
EEND-EDA-style speaker counting, as opposed to the Transformer-decoder
attractor used by the sibling module `dptnet_tda.py`).

Speaker separation proceeds as:
    1. Run `i_eda_layer` dual-path Transformer layers over the mixture.
    2. Aggregate each chunk into a single vector (`SequenceAggregation`)
       and decode attractors + speaker-existence probabilities from the
       aggregated sequence (`self.eda`).
    3. Gate the shared feature map with each speaker's attractor,
       broadcasting the speaker axis into the batch dimension so the
       remaining dual-path layers process one speaker's stream at a time.

Unlike `DPTNet_TDA_Informed.forward`, `DPTNet_EDA_Informed.forward` does
apply `self.adapt_layer` at `i_adapt_layer`: when an enrollment embedding
is given, it selects/adapts each speaker's stream using target-speaker
enrollment (target speaker extraction).
"""

from typing import Optional

import torch
import torch.nn as nn

from espnet2.enh.layers.adapt_layers import make_adapt_layer
from espnet2.enh.layers.dptnet import ImprovedTransformerLayer


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


class DPTNet_EDA_Informed(nn.Module):
    """Dual-path transformer network with a mid-network EDA attractor block.

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
        i_eda_layer (int, optional): index (0-based) of the dual-path layer
            after which the EDA attractor block is applied. If None, no
            attractor block is used.
        num_eda_modules (int): must be odd; kept for interface parity with
            the (unused) ensembling logic that used to live in this class.
        i_adapt_layer (int, optional): index of the dual-path layer after
            which the enrollment adapt layer (target speaker extraction) is
            applied, when an enrollment embedding is given.
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
        i_eda_layer: int = 4,
        num_eda_modules: int = 1,
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

        # eda related params
        self.i_eda_layer = i_eda_layer
        self.num_eda_modules = num_eda_modules
        assert (
            self.num_eda_modules % 2 == 1
        ), "number of EDA modules should be odd number"
        if i_eda_layer is not None:
            self.sequence_aggregation = SequenceAggregation(input_size)
            self.eda = EncoderDecoderAttractor(input_size)

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
        """Apply the dual-path Transformer stack, EDA attractor and TSE adaptation.

        Args:
            input: (B, N, dim1, dim2) chunked features. N is the feature
                dimension, dim1 is the intra-chunk (segment) length (K),
                dim2 is the number of chunks (S).
            enroll_emb: enrollment embedding used for target-speaker
                adaptation at `i_adapt_layer`, or None to skip it (in which
                case all attractor-gated speaker streams are kept).
            num_spk: if given, `self.eda` decodes exactly `num_spk + 1`
                attractors (the last is a "stop" attractor used only to
                supervise the existence-probability loss); otherwise it
                decodes attractors one at a time until the existence
                probability drops below 0.5.

        Returns:
            output: (B, output_size, dim1, dim2) if `i_eda_layer` is None,
                else (B_out, output_size, dim1, dim2), where B_out is
                B * J (J = number of speaker streams from EDA gating) if no
                target-speaker adaptation is applied, or B (one stream per
                enrollment) once `self.adapt_layer` has run.
            probabilities: (B, J + 1) speaker-existence probabilities from
                the EDA module, or None if `i_eda_layer` is None.
        """
        output = input
        batch, hidden_dim, dim1, dim2 = output.shape
        org_batch = batch
        is_tse = enroll_emb is not None
        probabilities = None
        for layer_idx in range(len(self.row_transformer)):
            output = _apply_intra_chunk_transformer(
                output, self.row_transformer[layer_idx]
            )
            output = _apply_inter_chunk_transformer(
                output, self.col_transformer[layer_idx]
            )

            if layer_idx == self.i_eda_layer:
                output, probabilities = self._apply_attractor_gate(
                    output, hidden_dim, dim1, dim2, num_spk
                )

            if layer_idx == self.i_adapt_layer and is_tse:
                output = self._apply_target_speaker_adaptation(
                    output, enroll_emb, org_batch, hidden_dim, dim1, dim2
                )

        output = self.output(output)  # B_out, output_size, dim1, dim2
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
        attractor_vectors, probabilities = self.eda(
            aggregated_sequence, num_spk=num_spk
        )
        gated = (
            output[..., None, :, :, :] * attractor_vectors[..., :-1, :, None, None]
        )  # (B, J, N, dim1, dim2)
        gated = gated.view(-1, hidden_dim, dim1, dim2)  # (B * J, N, dim1, dim2)
        return gated, probabilities

    def _apply_target_speaker_adaptation(
        self,
        output: torch.Tensor,
        enroll_emb: torch.Tensor,
        org_batch: int,
        hidden_dim: int,
        dim1: int,
        dim2: int,
    ) -> torch.Tensor:
        """Select/adapt the per-speaker streams using the enrollment embedding.

        output: (B_eff, N, dim1, dim2), B_eff = org_batch * (num speaker
            streams produced by the EDA gate, or 1 if EDA gating did not run
            before this layer).
        enroll_emb: permuted from (B, D, T, F) to (B, T, F, D) before use.

        Returns: (B_eff, N, dim1, dim2)
        """
        enroll_emb = enroll_emb.permute(0, 2, 3, 1)
        output = (
            output.view(org_batch, -1, hidden_dim, dim1, dim2)
            .permute(0, 1, 3, 4, 2)
        )
        output = self.adapt_layer(output, enroll_emb)
        output = output.permute(0, 3, 1, 2)
        if self.adapt_layer_type == "attn_improved":
            output = self.conditioning_model(
                output, enroll_emb.mean(dim=(1, 2), keepdim=True)
            )
        return output


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


class EncoderDecoderAttractor(nn.Module):
    """LSTM encoder-decoder attractor (EDA) for speaker counting.

    Encodes a sequence of chunk embeddings with an LSTM, then autoregressively
    decodes speaker attractor vectors from a zero input with a second LSTM,
    stopping (at inference time) once an "existence" probability estimated
    from the decoder output drops below 0.5. See e.g. Horiguchi et al.,
    "End-to-End Speaker Diarization for an Unknown Number of Speakers with
    Encoder-Decoder Based Attractors."
    """

    def __init__(
        self,
        hidden_size: int,
        chunk_shuffle: bool = True,  # only meaningful when the sequence axis is chunks
    ) -> None:
        super(EncoderDecoderAttractor, self).__init__()
        self.lstm_encoder = nn.LSTM(
            hidden_size,
            hidden_size,
            num_layers=1,
            dropout=0,
            bidirectional=False,
            batch_first=True,
        )
        self.lstm_decoder = nn.LSTM(
            hidden_size,
            hidden_size,
            num_layers=1,
            dropout=0,
            bidirectional=False,
            batch_first=True,
        )

        self.attractor_existence_estimator = nn.Sequential(
            nn.Linear(hidden_size, 1), nn.Sigmoid()
        )
        self.chunk_shuffle = chunk_shuffle

    def forward(self, input: torch.Tensor, num_spk: Optional[int] = None):
        """Encode a sequence of chunk embeddings into speaker attractors.

        Args:
            input: (B, C, H), C chunk-aggregated embeddings of dim H.
            num_spk: if given, decode exactly `num_spk + 1` attractors (the
                last one is the "stop" attractor, used only to supervise
                the existence-probability loss). If None, decode attractors
                one at a time until the existence probability drops below
                0.5 (inference-time only; batch size must be 1).

        Returns:
            outputs: (B, J, H) decoded attractor vectors.
            existence_probabilities: (B, J) existence probability per
                decoded attractor.
        """
        batch_size, num_chunks, hidden_size = input.shape
        encoder_input = input
        if self.chunk_shuffle:
            encoder_input = encoder_input[..., torch.randperm(num_chunks), :]
        _, state = self.lstm_encoder(encoder_input)  # state: (h, c), each [1, B, H]
        zero_input = input.new_zeros((batch_size, 1, hidden_size))

        if num_spk is None:
            return self._decode_until_stop(zero_input, state, batch_size)
        return self._decode_fixed_count(zero_input, state, num_spk)

    def _decode_until_stop(self, zero_input, state, batch_size):
        """Greedily decode attractors until the existence probability <= 0.5."""
        assert batch_size == 1, "We don't support batched computation in inference"
        outputs, existence_probabilities = [], []
        existence_probability = 1
        while existence_probability > 0.5:
            output, state = self.lstm_decoder(zero_input, state)  # (B, 1, H)
            existence_probability = self.attractor_existence_estimator(output)
            existence_probabilities.append(existence_probability[..., 0])
            outputs.append(output[..., 0, :])
        return (
            torch.stack(outputs, dim=1),
            torch.stack(existence_probabilities, dim=1),
        )

    def _decode_fixed_count(self, zero_input, state, num_spk: int):
        """Decode exactly `num_spk + 1` attractors (training, known speaker count)."""
        outputs, existence_probabilities = [], []
        for _ in range(num_spk + 1):
            output, state = self.lstm_decoder(zero_input, state)  # (B, 1, H)
            output = output[..., 0, :]
            outputs.append(output)
            existence_probability = self.attractor_existence_estimator(output)
            existence_probabilities.append(existence_probability[..., 0])
        return (
            torch.stack(outputs, dim=1),
            torch.stack(existence_probabilities, dim=1),
        )
