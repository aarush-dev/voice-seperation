"""SepTDA extractor/separator: dual-/triple-path Sepformer with a
Transformer-Decoder-Attractor (TDA) speaker-counting head.

Pipeline (see :meth:`SepformerTDAExtractor.forward`):

1. Project the encoded mixture to ``hidden_dim`` and split it into
   overlapping chunks (dual-path chunking), giving a (B, K, S, D) tensor
   where ``K`` is the chunk length and ``S`` the number of chunks.
2. A small stack of *dual-path* blocks (intra-/inter-chunk processing)
   refines the chunked features.
3. The dual-path output is overlap-added back to (B, T, D) and fed to a TDA
   attractor decoder, which autoregressively generates one attractor
   embedding per active speaker (plus a trailing "stop" attractor) together
   with an existence probability for each.
4. The per-speaker attractors condition the chunked features via FiLM,
   expanding the batch with a per-speaker channel axis ``C``.
5. A deeper stack of *triple-path* blocks (intra-chunk, inter-chunk, and
   inter-speaker processing) refines the FiLM-conditioned features. When
   ``multi_decode`` is enabled, every intermediate triple-path layer (except
   the last) also produces its own auxiliary time-domain estimate through a
   dedicated output head + decoder -- this is the "multi-decoder loss" deep
   supervision used by :mod:`espnet2.enh.espnet_model_tse_ss`.
6. The final triple-path output is projected back to the encoder's feature
   dimension and overlap-added to produce one estimate per speaker.

This extractor performs blind speaker counting + separation from the mixture
alone: ``input_aux``/``ilens_aux`` (the TSE enrollment features expected by
:class:`~espnet2.enh.extractor.abs_extractor.AbsExtractor`) are accepted for
interface compatibility but are not consumed by this implementation.
"""
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple, Union

import torch
from torch import nn
from torch_complex.tensor import ComplexTensor
from einops import rearrange
from rotary_embedding_torch import RotaryEmbedding

from espnet2.enh.extractor.abs_extractor import AbsExtractor
from espnet2.enh.separator.abs_separator import AbsSeparator
from espnet2.enh.layers.tcn import choose_norm
from espnet2.enh.decoder.conv_decoder import ConvDecoder
from espnet2.enh.layers.septda import SepTDABlock, PositionalwiseFeedForward
from espnet2.enh.layers.rope_tda import RopeAttractorDecode
from espnet2.enh.layers.tda import AttractorDecode

attractors = {
    "tda": AttractorDecode,
    "ropetda": RopeAttractorDecode,
}


class SepformerTDAExtractor(AbsExtractor, AbsSeparator):
    """Dual-/triple-path Sepformer separator with TDA speaker counting.

    See the module docstring for the overall pipeline. ``__init__``
    parameters follow the SepTDA paper's naming (De/D/K/N annotated inline).
    """

    def __init__(
        self,
        # general setup
        input_dim: int= 256, # De=256
        hidden_dim: int = 128, # D=128
        output_dim: int = None, # De=256
        num_spk: int = 5,
        activation: str = "gelu",
        norm_type: str = "gLN",
        dual_layers: int = 1, # a dual-path processing block
        triple_layers: int = 8, # triple-path blocks N is 8
        segment_size: int = 96, # chunks of length K = 96
        dropout: float = 0.0,
        # rnn setup
        rnn_type: str = "LSTM",
        bidirectional: bool = True,
        rnn_dim: int = 256, # the number of hidden units in BLSTM is set to be 256 in each direction
        # self-attention setup
        att_heads: int = 4, # 4 attention heads
        attention_dim: int = 128, # the attention dimension is set to be 128
        flash_attention=False,
        # ffn setup
        expansion_factor: int = 4, # an expansion factor of 4 for the feed-forward module.
        # tda setup
        film_skip_connection: bool = False,
        attractor_type="tda", # "eda" or "tda"
        # multi-decoding setup
        multi_decode: bool = False,
        kernel_size: int = 16,
        stride: int = 8,
    ):
        """SepTDA Separator

        Args:

        """
        super().__init__()

        self._num_spk = num_spk
        self.segment_size = segment_size
        self.linear = torch.nn.Linear(input_dim, hidden_dim) # input_dim -> hidden_dim
        self.output_dim = input_dim if output_dim is None else output_dim
        self.hidden_dim = hidden_dim
        self.rnn_dim = rnn_dim
        self.segment_size = segment_size

        # rope
        assert attention_dim % att_heads == 0, (attention_dim, att_heads)
        rope_intra_chunk = RotaryEmbedding(attention_dim // att_heads)
        rope_inter_chunk = RotaryEmbedding(attention_dim // att_heads)
        # dual-path processing blocks
        self.dual_path = nn.ModuleList()
        for i in range(dual_layers):
            self.dual_path.append(
                SepTDABlock(
                    rope_intra_chunk,
                    rope_inter_chunk,
                    block_type="dual",
                    hidden_dim=hidden_dim,
                    dropout=dropout,
                    activation=activation,
                    norm_type=norm_type,
                    rnn_type=rnn_type,
                    rnn_dim=rnn_dim,
                    bidirectional=bidirectional,

                    att_heads=att_heads,
                    attention_dim=attention_dim,
                    flash_attention=flash_attention,

                    expansion_factor=expansion_factor,
                )
            )
        # triple-path processing blocks
        self.triple_path = nn.ModuleList()
        for i in range(triple_layers):
            self.triple_path.append(
                SepTDABlock(
                    rope_intra_chunk,
                    rope_inter_chunk,
                    block_type="triple",
                    hidden_dim=hidden_dim,
                    dropout=dropout,
                    activation=activation,
                    norm_type=norm_type,
                    rnn_type=rnn_type,
                    rnn_dim=rnn_dim,
                    bidirectional=bidirectional,

                    att_heads=att_heads,
                    attention_dim=attention_dim,
                    flash_attention=flash_attention,

                    expansion_factor=expansion_factor,
                )
            )

        # output layer
        self.norm_output = choose_norm(norm_type, hidden_dim)
        self.ffn_output = PositionalwiseFeedForward(
            d_ffn=hidden_dim*expansion_factor,
            input_size=hidden_dim,
            output_size=self.output_dim,
            dropout=dropout,
            activation=activation,
        )
        self.multi_decode = multi_decode
        if self.multi_decode:
            self.aux_norm_output = nn.ModuleList()
            self.aux_ffn_output = nn.ModuleList()
            self.aux_decoder = nn.ModuleList()
            for i in range(triple_layers - 1):
                self.aux_norm_output.append(choose_norm(norm_type, hidden_dim))
                self.aux_ffn_output.append(
                    PositionalwiseFeedForward(
                        d_ffn=hidden_dim*expansion_factor,
                        input_size=hidden_dim,
                        output_size=self.output_dim,
                        dropout=dropout,
                        activation=activation,
                    )
                )
                self.aux_decoder.append(
                    ConvDecoder(
                        channel=self.output_dim,
                        kernel_size=kernel_size,
                        stride=stride,
                    )
                )

        # tda related params
        if attractor_type == "tda" or attractor_type == "ropetda":
            self.tda = attractors[attractor_type](hidden_dim, att_heads, 384, dropout, activation, chunk_shuffle=False)
        self.film = FiLM(
                hidden_dim,
                hidden_dim,
                hidden_dim,
                skip_connection=film_skip_connection,
            )

    def forward(
        self,
        input: torch.Tensor,
        ilens: torch.Tensor,
        additional: Optional[Dict] = None,
        input_aux: torch.Tensor = None,
        ilens_aux: torch.Tensor = None,
        suffix_tag: str = "",
        num_spk: int = None,
        task: str = None,
        speech_lengths: torch.Tensor = None,
    ) -> Tuple[List[Union[torch.Tensor, ComplexTensor]], torch.Tensor, OrderedDict]:
        """Separate the encoded mixture into per-speaker features.

        Args:
            input (torch.Tensor or ComplexTensor): Encoded mixture feature,
                (B, T, D_enc).
            ilens (torch.Tensor): input lengths, (Batch,).
            additional (Dict or None): unused by this model.
            input_aux, ilens_aux: unused by this model (see module docstring).
            suffix_tag: unused by this model.
            num_spk: number of active speakers; controls how many attractors
                the TDA head generates (``num_spk + 1`` including the "stop"
                attractor).
            task: unused by this model.
            speech_lengths: (Batch,) waveform lengths, forwarded to the
                auxiliary per-layer decoders when ``multi_decode`` is set.

        Returns:
            output (torch.Tensor): (num_spk, B, T, D_enc) per-speaker
                separated features, indexable like a list of ``num_spk``
                (B, T, D_enc) tensors.
            ilens (torch.Tensor): (B,), unchanged from the input.
            others (OrderedDict): ``existance_probability`` (B, num_spk + 1)
                attractor existence probabilities when a TDA head is
                configured; ``aux_speech_pre``, a list of per-layer
                auxiliary waveform estimates (each a list of ``num_spk``
                (B, T) tensors), when ``multi_decode`` is enabled.
        """
        chunked, batch_size, num_frames = self._project_and_chunk(input)

        dual_path_output = self._run_dual_path(chunked)
        attractor_embeddings, existence_probabilities = self._decode_attractors(
            dual_path_output, num_frames, num_spk
        )

        # Drop the trailing "stop" attractor before conditioning the features.
        output = self.film(dual_path_output, attractor_embeddings[..., :-1, :])
        output, aux_speech_pre = self._run_triple_path(
            output, num_frames, batch_size, speech_lengths
        )

        output = self._project_and_overlap_add(output, num_frames, batch_size)

        others = OrderedDict()
        if existence_probabilities is not None:
            others["existance_probability"] = existence_probabilities
        if self.multi_decode:
            others["aux_speech_pre"] = aux_speech_pre

        return output, ilens, others

    def _project_and_chunk(
        self, input: torch.Tensor
    ) -> Tuple[torch.Tensor, int, int]:
        """Project to ``hidden_dim`` and split into overlapping chunks.

        Args:
            input: (B, T, D_enc) encoded mixture feature.

        Returns:
            chunked: (B, K, S, D) hidden features split into chunks of
                length ``K = segment_size`` with 50% overlap (``S`` chunks).
            batch_size: B.
            num_frames: T, needed later to overlap-add back to T frames.
        """
        feature = self.linear(input)  # (B, T, D_enc) -> (B, T, D)
        batch_size, num_frames, _ = feature.shape
        feature = feature.transpose(1, 2)  # (B, D, T)
        chunked = self.split_feature(feature)  # (B, D, K, S)
        chunked = rearrange(chunked, "b d k s -> b k s d")
        return chunked, batch_size, num_frames

    def _run_dual_path(self, chunked: torch.Tensor) -> torch.Tensor:
        """Apply the dual-path (intra-/inter-chunk) processing blocks.

        Args:
            chunked: (B, K, S, D) chunked hidden features.

        Returns:
            (B, K, S, D) refined features, same shape as the input.
        """
        output = chunked
        for block in self.dual_path:
            output = block(output)
        return output

    def _decode_attractors(
        self,
        dual_path_output: torch.Tensor,
        num_frames: int,
        num_spk: Optional[int],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Overlap-add the dual-path output and decode speaker attractors.

        Args:
            dual_path_output: (B, K, S, D) dual-path features.
            num_frames: T, target length for the overlap-add.
            num_spk: number of active speakers (see :class:`RopeAttractorDecode`).

        Returns:
            attractor_embeddings: (B, num_spk + 1, D) per-speaker attractor
                embeddings, the last one modeling the "stop" signal.
            existence_probabilities: (B, num_spk + 1) or None.
        """
        overlap_output = self.merge_feature(
            rearrange(dual_path_output, "b k s d -> b d k s"), length=num_frames
        )
        overlap_output = rearrange(overlap_output, "b d t -> b t d")  # (B, T, D)
        attractor_embeddings, existence_probabilities = self.tda(
            overlap_output, num_spk
        )
        del overlap_output  # free memory before the (larger) triple-path stage
        return attractor_embeddings, existence_probabilities

    def _run_triple_path(
        self,
        output: torch.Tensor,
        num_frames: int,
        batch_size: int,
        speech_lengths: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, List[List[torch.Tensor]]]:
        """Apply the FiLM-conditioned triple-path blocks.

        Args:
            output: (B, C, K, S, D) FiLM-conditioned chunked features, with
                ``C`` the per-speaker channel axis introduced by FiLM.
            num_frames: T, target length for the auxiliary overlap-adds.
            batch_size: B.
            speech_lengths: (Batch,) waveform lengths for the auxiliary
                decoders (only used when ``multi_decode`` is set).

        Returns:
            output: (B, C, K, S, D) final triple-path features.
            aux_speech_pre: one entry per intermediate triple-path layer
                (empty unless ``multi_decode``), each a list of ``C``
                (B, T) waveform estimates -- the multi-decoder deep
                supervision targets.
        """
        aux_speech_pre = []
        num_layers = len(self.triple_path)
        for i, block in enumerate(self.triple_path):
            output = block(output)  # (B, C, K, S, D)
            if self.multi_decode and i < num_layers - 1:
                aux_speech_pre.append(
                    self._decode_auxiliary_layer(
                        output, num_frames, batch_size, speech_lengths, i
                    )
                )
        return output, aux_speech_pre

    def _decode_auxiliary_layer(
        self,
        output: torch.Tensor,
        num_frames: int,
        batch_size: int,
        speech_lengths: torch.Tensor,
        layer_idx: int,
    ) -> List[torch.Tensor]:
        """Decode one intermediate triple-path layer's auxiliary estimate.

        Args:
            output: (B, C, K, S, D) triple-path features after layer ``layer_idx``.
            num_frames: T, target length for the overlap-add.
            batch_size: B.
            speech_lengths: (Batch,) waveform lengths passed to the decoder.
            layer_idx: index into ``aux_norm_output``/``aux_ffn_output``/``aux_decoder``.

        Returns:
            List of ``C`` (B, T) waveform estimates for this layer.
        """
        aux_output = self.aux_ffn_output[layer_idx](
            self.aux_norm_output[layer_idx](output)
        )
        aux_output = rearrange(aux_output, "b c k s d -> (b c) d k s")
        aux_output = self.merge_feature(aux_output, length=num_frames)
        aux_output = rearrange(aux_output, "(b c) d t -> c b t d", b=batch_size)
        return [
            self.aux_decoder[layer_idx](ps.to(torch.float32), speech_lengths)[0]
            for ps in aux_output
        ]

    def _project_and_overlap_add(
        self, output: torch.Tensor, num_frames: int, batch_size: int
    ) -> torch.Tensor:
        """Project the final triple-path features to ``output_dim`` and
        overlap-add the chunks back into full-length per-speaker signals.

        Args:
            output: (B, C, K, S, D) final triple-path features.
            num_frames: T, target length for the overlap-add.
            batch_size: B.

        Returns:
            (C, B, T, D_enc) per-speaker encoded-domain estimates.
        """
        output = self.ffn_output(self.norm_output(output))  # (B, C, K, S, D_enc)
        output = rearrange(output, "b c k s d -> (b c) d k s")
        output = self.merge_feature(output, length=num_frames)  # (B*C, D_enc, T)
        return rearrange(output, "(b c) d t -> c b t d", b=batch_size)

    def split_feature(self, x: torch.Tensor) -> torch.Tensor:
        """Split (B, D, T) features into overlapping chunks (B, D, K, S).

        Chunks have length ``segment_size`` (K) with 50% overlap, using
        ``unfold`` with matching padding so that :meth:`merge_feature` can
        exactly invert this operation via overlap-add.
        """
        B, D, T = x.size()
        unfolded = torch.nn.functional.unfold(
            x.unsqueeze(-1),
            kernel_size=(self.segment_size, 1),
            padding=(self.segment_size, 0),
            stride=(self.segment_size // 2, 1),
        )
        return unfolded.reshape(B, D, self.segment_size, -1)

    def merge_feature(
        self, x: torch.Tensor, length: Optional[int] = None
    ) -> torch.Tensor:
        """Overlap-add chunked (B, D, K, S) features back to (B, D, length).

        Inverse of :meth:`split_feature`: normalizes by the overlap count so
        that regions covered by multiple chunks are averaged rather than
        summed.
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

    @property
    def num_spk(self) -> int:
        """Maximum number of speakers this extractor is configured for."""
        return self._num_spk


class FiLM(nn.Module):
    """Feature-wise Linear Modulation, conditioning chunked features on
    per-speaker attractor embeddings.

    Broadcasts one (gamma, beta) affine transform per speaker ``C`` over
    every (T, F) chunk position, expanding the input's batch dimension into
    a (B, C) grid.
    """

    def __init__(
        self,
        indim: int,
        enrolldim: int,
        filmdim: int,
        skip_connection: bool = False,
    ):
        super().__init__()
        self.linear1 = nn.Sequential(
            nn.Linear(indim, filmdim),
            nn.PReLU(),
        )
        self.film_gamma = nn.Linear(enrolldim, filmdim)
        self.film_beta = nn.Linear(enrolldim, filmdim)
        self.linear2 = nn.Linear(filmdim, indim)
        self.skip_connection = skip_connection

    def forward(
        self, input: torch.Tensor, enroll_emb: torch.Tensor
    ) -> torch.Tensor:
        """Apply per-speaker FiLM conditioning.

        Args:
            input: (B, T, F, D) chunked features (T = chunk length K,
                F = number of chunks S).
            enroll_emb: (B, C, D) per-speaker attractor embeddings.

        Returns:
            (B, C, T, F, D) conditioned features, one copy per speaker ``C``.
        """
        gamma = self.film_gamma(enroll_emb)  # (B, C, D)
        beta = self.film_beta(enroll_emb)  # (B, C, D)

        output = self.linear1(input)  # (B, T, F, D)

        # Rearrange for FiLM broadcasting
        output = rearrange(output, 'b t f d -> b 1 t f d')       # (B,1,T,F,D)
        gamma = rearrange(gamma, 'b c d -> b c 1 1 d')           # (B,C,1,1,D)
        beta = rearrange(beta, 'b c d -> b c 1 1 d')             # (B,C,1,1,D)

        output = gamma * output + beta                           # (B,C,T,F,D)

        output = self.linear2(output)                            # (B,C,T,F,D)

        if self.skip_connection:
            residual = rearrange(input, 'b t f d -> b 1 t f d')  # (B,1,T,F,D)
            output = output + residual                           # (B,C,T,F,D)

        return output
