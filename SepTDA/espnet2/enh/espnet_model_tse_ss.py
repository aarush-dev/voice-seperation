"""ESPnet model that jointly supports speech separation (SS) and target
speaker extraction (TSE), as used by the SepTDA recipes.

The model wraps an ``encoder -> extractor -> decoder`` pipeline (mirroring
the other ESPnet enhancement models) around an :class:`AbsExtractor` that can
operate in three modes, selected by ``task``:

* ``"enh"``: blind speech separation. The extractor is given only the
  mixture and outputs one estimate per speaker; no enrollment audio is used.
* ``"tse"``: target speaker extraction. The extractor is additionally given
  an enrollment (auxiliary) utterance for one speaker and outputs a single
  estimate for that speaker.
* ``"enh_tse"``: both heads are trained jointly. The extractor produces
  ``num_spk`` blind-separation outputs *and* one extra TSE output
  conditioned on a randomly chosen speaker's enrollment; the loss is the
  average of the "enh" loss (against all speakers) and the "tse" loss
  (against the chosen speaker only).

Extractors such as :class:`~espnet2.enh.extractor.septda_extractor.
SepformerTDAExtractor` additionally support a "multi-decoder loss": besides
the final separated signals, a handful of intermediate network layers each
produce their own auxiliary estimate (via extra output layers + decoders).
When present (``others["aux_speech_pre"]``), the main separation loss is
averaged together with the losses computed on every auxiliary estimate, which
provides a deep-supervision training signal.

Speaker-counting is trained separately: a TDA/RopeTDA attractor decoder can
emit an "existence probability" per generated attractor
(``others["existance_probability"]``), supervised with a BCE loss against
the true number of active speakers.

Shapes referenced throughout this module:
    B: batch size.
    T: number of raw audio samples (time domain).
    L: number of encoded frames (``T`` downsampled by the encoder).
    N: encoder/decoder feature dimension.
    S: number of speakers for a given utterance (``<= num_spk``).
"""
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from typeguard import check_argument_types

from espnet2.enh.decoder.abs_decoder import AbsDecoder
from espnet2.enh.encoder.abs_encoder import AbsEncoder
from espnet2.enh.extractor.abs_extractor import AbsExtractor
from espnet2.enh.layers.complex_utils import is_complex
from espnet2.enh.loss.criterions.tf_domain import FrequencyDomainLoss
from espnet2.enh.loss.criterions.time_domain import TimeDomainLoss
from espnet2.enh.loss.wrappers.abs_wrapper import AbsLossWrapper
from espnet2.torch_utils.device_funcs import force_gatherable
from espnet2.train.abs_espnet_model import AbsESPnetModel

import logging

logfmt = "%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s"
logging.basicConfig(level=logging.INFO, format=logfmt)


def normalization(
    speech_mix: torch.Tensor,
    speech_ref: Optional[List[torch.Tensor]] = None,
    eps: float = 1e-8,
) -> Union[
    Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    Tuple[torch.Tensor, List[torch.Tensor], torch.Tensor, torch.Tensor],
]:
    """Per-utterance mean/std standardization (CMVN over the time axis).

    Args:
        speech_mix: (Batch, samples) mixture waveform.
        speech_ref: optional list of (Batch, samples) reference waveforms to
            normalize with the *mixture's* mean/std (so mixture and
            references stay on a consistent scale).
        eps: numerical stability floor added to the standard deviation.

    Returns:
        If ``speech_ref`` is None: ``(normalized_speech_mix, mean, std)``.
        Otherwise: ``(normalized_speech_mix, normalized_speech_ref, mean, std)``.
        ``mean``/``std`` have shape (Batch, 1) and can be used to undo the
        normalization later via ``x * std + mean``.
    """
    mean = speech_mix.mean(dim=-1, keepdim=True)
    std = speech_mix.std(dim=-1, keepdim=True)
    speech_mix = (speech_mix - mean) / (std + eps)
    if speech_ref is None:
        return speech_mix, mean, std
    speech_ref = [(ref - mean) / (std + eps) for ref in speech_ref]
    return speech_mix, speech_ref, mean, std


def merge_two_dicts(
    stats_dicts: List[Dict[str, torch.Tensor]]
) -> Dict[str, torch.Tensor]:
    """Merge per-subtask stats dicts, averaging values for shared keys.

    Used to combine the "enh" and "tse" stats dicts produced for the
    ``task == "enh_tse"`` setting into a single stats dict: keys that appear
    in both are averaged, keys unique to one subtask are kept as-is.
    """
    merged = stats_dicts[0]
    if len(stats_dicts) == 1:
        return merged
    for key, value in stats_dicts[1].items():
        if key in merged:
            merged[key] = (merged[key] + value) / 2
        else:
            merged[key] = value
    return merged


class ESPnetExtractionEnhancementModel(AbsESPnetModel):
    """Joint target-speaker-extraction / speech-separation frontend model.

    See the module docstring for the ``task`` modes and the multi-decoder /
    speaker-counting losses. ``forward()`` computes the training loss;
    ``forward_enhance()`` runs only the enhancement network (encoder ->
    extractor -> decoder) and can also be used at inference time.
    """

    def __init__(
        self,
        encoder: AbsEncoder,
        extractor: AbsExtractor,
        decoder: AbsDecoder,
        loss_wrappers: List[AbsLossWrapper],
        num_spk: int = 1,
        share_encoder: bool = True,
        task: str = "enh_tse",
        normalization: bool = True,
    ):
        assert check_argument_types()

        super().__init__()

        self.encoder = encoder
        self.extractor = extractor
        self.decoder = decoder
        # Whether to share encoder for both mixture and enrollment
        self.share_encoder = share_encoder
        self.num_spk = num_spk
        self.normalization = normalization

        self.loss_wrappers = loss_wrappers
        names = [w.criterion.name for w in self.loss_wrappers]
        if len(set(names)) != len(names):
            raise ValueError(
                "Duplicated loss names are not allowed: {}".format(names)
            )
        for w in self.loss_wrappers:
            if getattr(w.criterion, "is_noise_loss", False):
                raise ValueError("is_noise_loss=True is not supported")
            elif getattr(w.criterion, "is_dereverb_loss", False):
                raise ValueError("is_dereverb_loss=True is not supported")

        # for multi-channel signal
        self.ref_channel = getattr(self.extractor, "ref_channel", -1)

        assert task in ["tse", "enh", "enh_tse"]
        self.task = task
        print(f"Task is {self.task}")
        print(f"Number of speakers {self.num_spk}")

    def forward(
        self,
        speech_mix: torch.Tensor,
        speech_mix_lengths: torch.Tensor = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
        """Encode + extract/separate + decode, then compute the training loss.

        Args:
            speech_mix: (Batch, samples) or (Batch, samples, channels).
            speech_mix_lengths: (Batch,), default None for chunk iterator,
                because the chunk-iterator does not have the speech lengths
                returned. See espnet2/iterators/chunk_iter_factory.py.
            kwargs: expected to contain ``speech_ref1``, ``speech_ref2``, ...
                (Batch, samples) reference signals for each speaker; when
                ``"tse" in self.task``, also ``enroll_ref1``, ``enroll_ref2``,
                ... (Batch, samples_aux) enrollment audio per speaker;
                optionally ``noise_ref1``, ... and ``dereverb_ref1``, ...
                for beamforming-style frontends. "utt_id" may also be
                present and is ignored.

        Returns:
            loss, stats, weight -- as required by :class:`AbsESPnetModel`.
        """
        batch_size = speech_mix.shape[0]

        speech_ref = self._gather_speech_refs(kwargs)
        # (Batch, num_speaker, samples) or (Batch, num_speaker, samples, channels)
        speech_ref = torch.stack(speech_ref, dim=1)

        enroll_ref: Optional[List[torch.Tensor]] = None
        enroll_ref_lengths: Optional[List[torch.Tensor]] = None
        if "tse" in self.task:
            enroll_ref, enroll_ref_lengths = self._gather_enrollment_refs(
                kwargs, batch_size
            )

        noise_ref = self._gather_noise_refs(kwargs)
        dereverb_speech_ref = self._gather_dereverb_refs(kwargs)

        speech_lengths = (
            speech_mix_lengths
            if speech_mix_lengths is not None
            else torch.ones(batch_size).int().fill_(speech_mix.shape[1])
        )
        assert speech_lengths.dim() == 1, speech_lengths.shape
        # Check that batch_size is unified
        assert (
            speech_mix.shape[0] == speech_ref.shape[0] == speech_lengths.shape[0]
        ), (
            speech_mix.shape,
            speech_ref.shape,
            speech_lengths.shape,
        )
        if "tse" in self.task:
            for aux in enroll_ref:
                assert aux.shape[0] == speech_mix.shape[0], (
                    aux.shape,
                    speech_mix.shape,
                )

        # for data-parallel: trim every tensor to the length actually used
        max_len = speech_lengths.max()
        speech_ref = speech_ref[..., :max_len].unbind(dim=1)
        if noise_ref is not None:
            noise_ref = noise_ref[..., :max_len].unbind(dim=1)
        if dereverb_speech_ref is not None:
            dereverb_speech_ref = dereverb_speech_ref[..., :max_len].unbind(dim=1)

        speech_mix = speech_mix[:, :max_len]
        if "tse" in self.task:
            enroll_ref = [
                enroll_ref[spk][:, : enroll_ref_lengths[spk].max()]
                for spk in range(len(enroll_ref))
            ]
            assert len(speech_ref) == len(enroll_ref), (
                len(speech_ref),
                len(enroll_ref),
            )

        if "num_spk" in kwargs:
            num_spk = int(kwargs["num_spk"][0].cpu().numpy())
        else:
            num_spk = len(speech_ref)

        enroll_ref_tmp, enroll_ref_lengths_tmp, spk_idx = (
            self._select_enrollment_and_spk_idx(
                speech_ref, enroll_ref, enroll_ref_lengths
            )
        )

        speech_pre, feature_mix, feature_pre, others = self.forward_enhance(
            speech_mix,
            speech_lengths,
            enroll_ref_tmp,
            enroll_ref_lengths_tmp,
            self.task,
            num_spk=num_spk,
            apply_normalization=self.normalization,
        )

        # ["enh"] or ["tse"] or ["enh", "tse"]
        task_list = self.task.split("_")
        loss = 0
        stats_per_subtask = []
        weight = torch.Tensor([0]).to(torch.int64).to(speech_ref[0].device)
        for subtask in task_list:
            (
                speech_ref_sub,
                speech_pre_sub,
                feature_mix_sub,
                feature_pre_sub,
            ) = self._select_subtask_outputs(
                subtask, spk_idx, speech_ref, speech_pre, feature_mix, feature_pre
            )

            l, s, w, _ = self.forward_loss(
                speech_pre_sub,
                speech_lengths,
                feature_mix_sub,
                feature_pre_sub,
                others,
                speech_ref_sub,
                noise_ref,
                dereverb_speech_ref,
                num_spk=num_spk,
                is_tse=(subtask == "tse"),
            )
            loss += l / len(task_list)
            stats_per_subtask.append(s)
            weight += w
        stats = merge_two_dicts(stats_per_subtask)
        return loss, stats, weight

    def forward_enhance(
        self,
        speech_mix: torch.Tensor,
        speech_lengths: torch.Tensor,
        enroll_ref: Optional[List[torch.Tensor]],
        enroll_ref_lengths: Optional[List[torch.Tensor]],
        task: str,
        num_spk: int = None,
        apply_normalization: bool = False,
    ) -> Tuple[
        Optional[List[torch.Tensor]], torch.Tensor, List[torch.Tensor], Dict
    ]:
        """Run encoder -> extractor -> decoder to produce separated waveforms.

        Args:
            speech_mix: (Batch, T) mixture waveform.
            speech_lengths: (Batch,) valid lengths of ``speech_mix``.
            enroll_ref: list with one (Batch, T_aux) enrollment waveform,
                required when ``"tse" in task``, otherwise None.
            enroll_ref_lengths: matching valid lengths for ``enroll_ref``.
            task: one of "enh", "tse", "enh_tse" (see module docstring).
            num_spk: number of active speakers, forwarded to the extractor
                (e.g. to control how many attractors are decoded).
            apply_normalization: whether to standardize the mixture (and
                enrollment) before encoding, undoing it on the decoded output.

        Returns:
            speech_pre: list of (Batch, T) separated/extracted waveforms, one
                per output source (``num_spk`` for "enh", 1 for "tse",
                ``num_spk + 1`` for "enh_tse" with the TSE output last), or
                None if the extractor does not predict a time-domain signal.
            feature_mix: (Batch, L, N) encoded mixture features.
            feature_pre: list of (Batch, L, N) separated features in the
                encoder's domain (pre-decoder).
            others: extractor side-outputs, e.g. ``existance_probability``
                (attractor counting) and ``aux_speech_pre`` (multi-decoder
                auxiliary estimates), used by ``forward_loss``.
        """
        mean, std = None, None
        if apply_normalization:
            speech_mix, mean, std = normalization(speech_mix)  # (B, T_original)
            if enroll_ref is not None:
                enroll_ref = [
                    normalization(enroll_ref[spk])[0]
                    for spk in range(len(enroll_ref))
                ]
        with torch.cuda.amp.autocast(enabled=False):
            feature_mix, flens = self.encoder(
                speech_mix.to(torch.float32), speech_lengths
            )  # (B, L, N)

        feature_aux, flens_aux = self._encode_enrollment(
            enroll_ref, enroll_ref_lengths, task
        )

        feature_pre, _, others = self.extractor(
            feature_mix,
            flens,
            feature_aux,
            flens_aux,
            suffix_tag="_spk1",
            num_spk=num_spk,
            task=task,
            speech_lengths=speech_lengths,
        )
        if task == "tse":
            feature_pre = [feature_pre]

        speech_pre = self._decode_predictions(feature_pre, speech_lengths, others)

        if apply_normalization and speech_pre is not None:
            speech_pre = [(pre + mean) * std for pre in speech_pre]
            if "aux_speech_pre" in others:
                others["aux_speech_pre"] = [
                    [(pre + mean) * std for pre in aux_speech_pre]
                    for aux_speech_pre in others["aux_speech_pre"]
                ]

        return speech_pre, feature_mix, feature_pre, others

    def forward_loss(
        self,
        speech_pre: Optional[List[torch.Tensor]],
        speech_lengths: torch.Tensor,
        feature_mix: torch.Tensor,
        feature_pre: List[torch.Tensor],
        others: Dict,
        speech_ref: List[torch.Tensor],
        noise_ref: torch.Tensor = None,
        dereverb_speech_ref: torch.Tensor = None,
        num_spk: int = None,
        is_tse: bool = True,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor, Optional[Dict]]:
        """Compute the loss for one subtask ("tse" or "enh") and its stats.

        Dispatches to :meth:`_compute_tse_loss` or :meth:`_compute_enh_loss`
        depending on ``is_tse``, then finalizes the result (records the
        overall "loss" stat and makes it DataParallel-gatherable).
        """
        if is_tse:
            loss, stats, perm = self._compute_tse_loss(
                speech_pre,
                speech_lengths,
                feature_mix,
                feature_pre,
                others,
                speech_ref,
                num_spk,
            )
        else:
            loss, stats, perm = self._compute_enh_loss(
                speech_pre,
                speech_lengths,
                feature_mix,
                feature_pre,
                others,
                speech_ref,
                noise_ref,
                dereverb_speech_ref,
                num_spk,
            )

        if self.training and isinstance(loss, float):
            raise AttributeError(
                "At least one criterion must satisfy: only_for_test=False"
            )
        stats["loss"] = loss.detach()

        # force_gatherable: to-device and to-tensor if scalar for DataParallel
        batch_size = speech_ref[0].shape[0]
        loss, stats, weight = force_gatherable(
            (loss, stats, batch_size), loss.device
        )
        return loss, stats, weight, perm

    # ------------------------------------------------------------------
    # forward(): reference/enrollment gathering helpers
    # ------------------------------------------------------------------

    def _gather_speech_refs(self, kwargs: Dict) -> List[torch.Tensor]:
        """Collect ``speech_ref1..speech_refN`` from kwargs into a list.

        Drops dummy length-1 placeholder tensors (used to pad batches with
        a varying number of active speakers) and removes the corresponding
        now-unused keys from ``kwargs`` in place.
        """
        assert (
            "speech_ref1" in kwargs
        ), "At least 1 reference signal input is required."
        speech_ref = [
            kwargs.get(
                f"speech_ref{spk + 1}", torch.zeros_like(kwargs["speech_ref1"])
            )
            for spk in range(self.num_spk)
            if f"speech_ref{spk + 1}" in kwargs
        ]
        speech_ref = [s for s in speech_ref if s.shape[-1] > 1]
        for s in range(len(speech_ref), self.num_spk):
            kwargs.pop(f"speech_ref{s + 1}", None)
        return speech_ref

    def _gather_enrollment_refs(
        self, kwargs: Dict, batch_size: int
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """Collect ``enroll_ref1..enroll_refN`` (+ their lengths) from kwargs.

        Mirrors :meth:`_gather_speech_refs`: drops dummy length-1 entries and
        pops the corresponding keys from ``kwargs`` before computing lengths,
        so ``enroll_ref_lengths`` lines up index-for-index with ``enroll_ref``.
        """
        assert (
            "enroll_ref1" in kwargs
        ), "At least 1 enrollment signal is required."
        enroll_ref = [
            kwargs[f"enroll_ref{spk + 1}"]
            for spk in range(self.num_spk)
            if f"enroll_ref{spk + 1}" in kwargs
        ]
        enroll_ref = [s for s in enroll_ref if s.shape[-1] > 1]
        for s in range(len(enroll_ref), self.num_spk):
            kwargs.pop(f"enroll_ref{s + 1}", None)
        enroll_ref_lengths = [
            kwargs.get(
                f"enroll_ref{spk + 1}_lengths",
                torch.ones(batch_size).int().fill_(enroll_ref[spk].size(1)),
            )
            for spk in range(self.num_spk)
            if f"enroll_ref{spk + 1}" in kwargs
        ]
        return enroll_ref, enroll_ref_lengths

    def _gather_noise_refs(self, kwargs: Dict) -> Optional[torch.Tensor]:
        """Stack optional ``noise_ref1..noise_refN`` (beamforming frontends)."""
        if "noise_ref1" not in kwargs:
            return None
        noise_ref = [
            kwargs[f"noise_ref{n + 1}"] for n in range(self.num_noise_type)
        ]
        # (Batch, num_noise_type, samples) or (Batch, num_noise_type, samples, channels)
        return torch.stack(noise_ref, dim=1)

    def _gather_dereverb_refs(self, kwargs: Dict) -> Optional[torch.Tensor]:
        """Stack optional ``dereverb_ref1..dereverb_refN`` (WPE frontends)."""
        if "dereverb_ref1" not in kwargs:
            return None
        dereverb_speech_ref = [
            kwargs[f"dereverb_ref{n + 1}"]
            for n in range(self.num_spk)
            if f"dereverb_ref{n + 1}" in kwargs
        ]
        assert len(dereverb_speech_ref) in (1, self.num_spk), len(
            dereverb_speech_ref
        )
        # (Batch, N, samples) or (Batch, N, samples, channels)
        return torch.stack(dereverb_speech_ref, dim=1)

    def _select_enrollment_and_spk_idx(
        self,
        speech_ref: Tuple[torch.Tensor, ...],
        enroll_ref: Optional[List[torch.Tensor]],
        enroll_ref_lengths: Optional[List[torch.Tensor]],
    ) -> Tuple[Optional[List[torch.Tensor]], Optional[List[torch.Tensor]], Optional[int]]:
        """Pick which speaker's enrollment drives the TSE branch this step.

        For ``task in ("tse", "enh_tse")``, one speaker index is sampled
        uniformly at random during training (fixed to speaker 0 during
        evaluation) and its enrollment audio is selected. For ``task ==
        "enh"`` there is no TSE branch, so both outputs are None.
        """
        if self.task == "enh":
            return None, None, None
        spk_idx = np.random.randint(0, len(speech_ref)) if self.training else 0
        return [enroll_ref[spk_idx]], [enroll_ref_lengths[spk_idx]], spk_idx

    def _select_subtask_outputs(
        self,
        subtask: str,
        spk_idx: Optional[int],
        speech_ref: Tuple[torch.Tensor, ...],
        speech_pre: Optional[List[torch.Tensor]],
        feature_mix: torch.Tensor,
        feature_pre: List[torch.Tensor],
    ) -> Tuple[
        List[torch.Tensor],
        List[torch.Tensor],
        Union[torch.Tensor, List[torch.Tensor]],
        List[torch.Tensor],
    ]:
        """Slice the joint enhancement outputs down to one subtask's outputs.

        When ``self.task == "enh_tse"``, ``forward_enhance`` produces
        ``num_spk`` blind-separation outputs followed by one TSE output for
        the enrolled speaker (``spk_idx``); this picks out the first
        ``num_spk`` outputs for the "enh" subtask, or the last output (paired
        with ``speech_ref[spk_idx]``) for the "tse" subtask. For a
        single-task model (``self.task in ("enh", "tse")``), all outputs
        already belong to that one subtask.
        """
        if self.task == "enh_tse" and subtask == "enh":
            return speech_ref, speech_pre[:-1], feature_mix[:-1], feature_pre[:-1]
        if self.task == "enh_tse" and subtask == "tse":
            return (
                [speech_ref[spk_idx]],
                [speech_pre[-1]],
                [feature_mix[-1]],
                [feature_pre[-1]],
            )
        if self.task == "tse":
            return [speech_ref[spk_idx]], speech_pre, feature_mix, feature_pre
        return speech_ref, speech_pre, feature_mix, feature_pre

    # ------------------------------------------------------------------
    # forward_enhance(): encode/decode helpers
    # ------------------------------------------------------------------

    def _encode_enrollment(
        self,
        enroll_ref: Optional[List[torch.Tensor]],
        enroll_ref_lengths: Optional[List[torch.Tensor]],
        task: str,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Encode the enrollment waveform for the extractor's TSE branch.

        If ``share_encoder`` is True, the enrollment is passed through the
        same encoder used for the mixture; otherwise the raw (or
        pre-computed embedding) enrollment tensor is forwarded as-is.
        """
        if enroll_ref is None:
            return None, None
        assert "tse" in task, "Something wrong in forward_enhance"
        if self.share_encoder:
            feature_aux, flens_aux = zip(
                *[
                    self.encoder(enroll_ref[spk], enroll_ref_lengths[spk])
                    for spk in range(len(enroll_ref))
                ]
            )
            assert len(feature_aux) == 1
            feature_aux, flens_aux = feature_aux[0], flens_aux[0]
        else:
            feature_aux, flens_aux = enroll_ref, enroll_ref_lengths
        return feature_aux, flens_aux

    def _decode_predictions(
        self,
        feature_pre: Optional[List[torch.Tensor]],
        speech_lengths: torch.Tensor,
        others: Dict,
    ) -> Optional[List[torch.Tensor]]:
        """Decode the extractor's per-source features back to waveforms.

        Also decodes the multi-decoder auxiliary features in
        ``others["aux_batch"]`` (if present) into ``others["aux_speech_pre"]``.
        Some extractors (e.g. mask-only beamformers during training) do not
        predict any time-domain signal, in which case ``feature_pre`` is None
        and this returns None.
        """
        if feature_pre is None:
            # some models (e.g. neural beamformer trained with mask loss)
            # do not predict time-domain signal in the training stage
            return None
        with torch.cuda.amp.autocast(enabled=False):
            if is_complex(feature_pre[0]):
                speech_pre = [
                    self.decoder(ps, speech_lengths)[0] for ps in feature_pre
                ]
                if "aux_batch" in others:
                    others["aux_speech_pre"] = [
                        [
                            self.decoder(ps, speech_lengths)[0]
                            for ps in aux_feature_pre
                        ]
                        for aux_feature_pre in others["aux_batch"]
                    ]
            else:
                speech_pre = [
                    self.decoder(ps.to(torch.float32), speech_lengths)[0]
                    for ps in feature_pre
                ]
        return speech_pre

    # ------------------------------------------------------------------
    # forward_loss(): "tse" subtask
    # ------------------------------------------------------------------

    def _compute_tse_loss(
        self,
        speech_pre: List[torch.Tensor],
        speech_lengths: torch.Tensor,
        feature_mix: torch.Tensor,
        feature_pre: List[torch.Tensor],
        others: Dict,
        speech_ref: List[torch.Tensor],
        num_spk: int,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], Optional[Dict]]:
        """Compute the TSE loss: single-source, no permutation ambiguity."""
        assert (
            len(speech_pre) == len(speech_ref) == 1
        ), f"TSE should output only 1 source, {len(speech_pre)} {len(speech_ref)}"

        loss = 0.0
        stats: Dict[str, torch.Tensor] = {}
        wrapper_out: Dict = {}
        perm = None

        for loss_wrapper in self.loss_wrappers:
            criterion = loss_wrapper.criterion
            if getattr(criterion, "only_for_test", False) and self.training:
                continue

            if isinstance(criterion, TimeDomainLoss):
                assert speech_pre is not None
                sref, spre = self._align_ref_pre_channels(
                    speech_ref, speech_pre, ch_dim=2, force_1ch=True
                )
                crit_loss, crit_stats, wrapper_out = loss_wrapper(
                    sref, spre, {**others, **wrapper_out}
                )
            elif isinstance(criterion, FrequencyDomainLoss):
                sref, spre = self._align_ref_pre_channels(
                    speech_ref, speech_pre, ch_dim=2, force_1ch=False
                )
                if criterion.compute_on_mask:
                    tf_ref, tf_pre = self._get_speech_masks(
                        criterion,
                        feature_mix,
                        None,
                        speech_ref,
                        speech_pre,
                        speech_lengths,
                        others,
                    )
                else:
                    tf_ref = [
                        self.encoder(sr, speech_lengths)[0] for sr in sref
                    ]
                    tf_pre = [
                        self.encoder(sp, speech_lengths)[0] for sp in spre
                    ]
                crit_loss, crit_stats, wrapper_out = loss_wrapper(
                    tf_ref, tf_pre, {**others, **wrapper_out}
                )
            else:
                raise NotImplementedError(
                    "Unsupported loss type: %s" % str(criterion)
                )

            loss += crit_loss * loss_wrapper.weight
            stats.update(crit_stats)
            if perm is None and "perm" in wrapper_out:
                perm = wrapper_out["perm"]

        self._record_per_numspk_loss(stats, "tse", num_spk, loss)
        return loss, stats, perm

    # ------------------------------------------------------------------
    # forward_loss(): "enh" subtask
    # ------------------------------------------------------------------

    def _compute_enh_loss(
        self,
        speech_pre: Optional[List[torch.Tensor]],
        speech_lengths: torch.Tensor,
        feature_mix: torch.Tensor,
        feature_pre: List[torch.Tensor],
        others: Dict,
        speech_ref: List[torch.Tensor],
        noise_ref: Optional[torch.Tensor],
        dereverb_speech_ref: Optional[torch.Tensor],
        num_spk: int,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], Optional[Dict]]:
        """Compute the blind-separation ("enh") loss plus auxiliary losses.

        Runs every loss wrapper (each handling its own permutation search),
        then adds the attractor-counting BCE loss and the multi-decoder
        speaker-count BCE loss when the extractor produced the corresponding
        side-outputs.
        """
        loss = 0.0
        stats: Dict[str, torch.Tensor] = {}
        perm = None

        self._decode_noise_and_dereverb_targets(
            others, noise_ref, dereverb_speech_ref, speech_lengths
        )

        for loss_wrapper in self.loss_wrappers:
            criterion = loss_wrapper.criterion
            if getattr(criterion, "only_for_test", False) and self.training:
                continue
            weighted_loss, crit_stats, wrapper_out = self._compute_enh_criterion_loss(
                loss_wrapper,
                feature_mix,
                speech_ref,
                speech_pre,
                noise_ref,
                dereverb_speech_ref,
                others,
                speech_lengths,
                num_spk,
            )
            loss += weighted_loss
            stats.update(crit_stats)
            if perm is None and "perm" in wrapper_out:
                perm = wrapper_out["perm"]

        self._record_per_numspk_loss(stats, "enh", num_spk, loss)

        attractor_loss, attractor_stats = self._compute_attractor_counting_loss(
            others, num_spk
        )
        if attractor_loss is not None:
            loss = loss + attractor_loss
            stats.update(attractor_stats)

        counting_loss, counting_stats = self._compute_multidecoder_counting_loss(
            others, num_spk
        )
        if counting_loss is not None:
            loss = loss + counting_loss
            stats.update(counting_stats)

        return loss, stats, perm

    def _decode_noise_and_dereverb_targets(
        self,
        others: Dict,
        noise_ref: Optional[torch.Tensor],
        dereverb_speech_ref: Optional[torch.Tensor],
        speech_lengths: torch.Tensor,
    ) -> None:
        """Decode extractor-predicted noise/dereverb features to waveforms.

        Mutates ``others`` in place. Used by beamforming-style frontends
        that also predict a noise and/or dereverberated-speech estimate;
        no-op for extractors (like SepTDA) that don't produce these.
        """
        if getattr(self.extractor, "predict_noise", False):
            assert "noise1" in others, others.keys()
        if noise_ref is not None and "noise1" in others:
            for n in range(self.num_noise_type):
                key = "noise{}".format(n + 1)
                others[key] = self.decoder(others[key], speech_lengths)[0]

        if getattr(self.extractor, "predict_dereverb", False):
            assert "dereverb1" in others, others.keys()
        if dereverb_speech_ref is not None and "dereverb1" in others:
            for spk in range(self.num_spk):
                key = "dereverb{}".format(spk + 1)
                if key in others:
                    others[key] = self.decoder(others[key], speech_lengths)[0]

    def _compute_enh_criterion_loss(
        self,
        loss_wrapper: AbsLossWrapper,
        feature_mix: torch.Tensor,
        speech_ref: List[torch.Tensor],
        speech_pre: Optional[List[torch.Tensor]],
        noise_ref: Optional[torch.Tensor],
        dereverb_speech_ref: Optional[torch.Tensor],
        others: Dict,
        speech_lengths: torch.Tensor,
        num_spk: int,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], Dict]:
        """Evaluate one loss wrapper for the "enh" subtask.

        Handles the three possible targets (speech / noise / dereverberated
        speech, chosen from the criterion's ``is_noise_loss`` /
        ``is_dereverb_loss`` flags), both time- and frequency-domain
        criterion types, and -- when SepTDA's multi-decoder auxiliary
        estimates (``others["aux_speech_pre"]``) are present -- averages the
        main loss together with the loss computed on every auxiliary
        estimate (deep supervision).

        Returns the loss already multiplied by ``loss_wrapper.weight``,
        together with its stats dict and the wrapper's raw output dict
        (which may carry a ``"perm"`` permutation).
        """
        criterion = loss_wrapper.criterion
        stats: Dict[str, torch.Tensor] = {}
        wrapper_out: Dict = {}

        if getattr(criterion, "is_noise_loss", False):
            if noise_ref is None:
                raise ValueError(
                    "No noise reference for training!\n"
                    'Please specify "--use_noise_ref true" in run.sh'
                )
            signal_ref = noise_ref
            signal_pre = [
                others["noise{}".format(n + 1)]
                for n in range(self.num_noise_type)
            ]
        elif getattr(criterion, "is_dereverb_loss", False):
            if dereverb_speech_ref is None:
                raise ValueError(
                    "No dereverberated reference for training!\n"
                    'Please specify "--use_dereverb_ref true" in run.sh'
                )
            signal_ref = dereverb_speech_ref
            signal_pre = [
                others["dereverb{}".format(n + 1)]
                for n in range(self.num_noise_type)
                if "dereverb{}".format(n + 1) in others
            ]
            if len(signal_pre) == 0:
                signal_pre = None
        else:
            signal_ref = speech_ref
            signal_pre = speech_pre

        aux_losses = None
        if isinstance(criterion, TimeDomainLoss):
            assert signal_pre is not None
            sref, spre = self._align_ref_pre_channels(
                signal_ref, signal_pre, ch_dim=2, force_1ch=True
            )
            crit_loss, crit_stats, wrapper_out = loss_wrapper(
                sref, spre, {**others, **wrapper_out}
            )
            aux_wrapper_out = wrapper_out if crit_loss <= -10 else {}
            if "aux_speech_pre" in others:
                aux_losses = []
                for aux_speech_pre in others["aux_speech_pre"]:
                    aux_sref, aux_spre = self._align_ref_pre_channels(
                        signal_ref, aux_speech_pre, ch_dim=2, force_1ch=True
                    )
                    aux_loss, _, aux_wrapper_out = loss_wrapper(
                        aux_sref, aux_spre, {**others, **aux_wrapper_out}
                    )
                    if aux_loss > -10:
                        aux_wrapper_out = {}
                    aux_losses.append(aux_loss)
        elif isinstance(criterion, FrequencyDomainLoss):
            sref, spre = self._align_ref_pre_channels(
                signal_ref, signal_pre, ch_dim=2, force_1ch=False
            )
            if criterion.compute_on_mask:
                if getattr(criterion, "is_noise_loss", False):
                    tf_ref, tf_pre = self._get_noise_masks(
                        criterion,
                        feature_mix,
                        speech_ref,
                        signal_ref,
                        signal_pre,
                        speech_lengths,
                        others,
                    )
                elif getattr(criterion, "is_dereverb_loss", False):
                    tf_ref, tf_pre = self._get_dereverb_masks(
                        criterion,
                        feature_mix,
                        noise_ref,
                        signal_ref,
                        signal_pre,
                        speech_lengths,
                        others,
                    )
                else:
                    tf_ref, tf_pre = self._get_speech_masks(
                        criterion,
                        feature_mix,
                        noise_ref,
                        signal_ref,
                        signal_pre,
                        speech_lengths,
                        others,
                    )
            else:
                tf_ref = [self.encoder(sr, speech_lengths)[0] for sr in sref]
                # for models like SVoice that output multiple lists of
                # separated signals
                pre_is_multi_list = isinstance(spre[0], (list, tuple))
                if pre_is_multi_list:
                    tf_pre = [
                        [self.encoder(sp, speech_lengths)[0] for sp in ps]
                        for ps in spre
                    ]
                else:
                    tf_pre = [
                        self.encoder(sp, speech_lengths)[0] for sp in spre
                    ]
            crit_loss, crit_stats, wrapper_out = loss_wrapper(
                tf_ref, tf_pre, {**others, **wrapper_out}
            )
        else:
            raise NotImplementedError(
                "Unsupported loss type: %s" % str(criterion)
            )

        if "aux_speech_pre" in others:
            self._record_per_numspk_loss(stats, "main", num_spk, crit_loss)
            for aux_idx, aux_loss in enumerate(aux_losses):
                self._record_per_numspk_loss(
                    stats, f"aux_{aux_idx + 1}", num_spk, aux_loss
                )
            crit_loss = (crit_loss + sum(aux_losses)) / (len(aux_losses) + 1)

        stats.update(crit_stats)
        return crit_loss * loss_wrapper.weight, stats, wrapper_out

    def _record_per_numspk_loss(
        self,
        stats: Dict[str, torch.Tensor],
        prefix: str,
        num_spk: int,
        value: torch.Tensor,
    ) -> None:
        """Log ``value`` under ``f"{prefix}_{num_spk}spk_loss"``.

        Also writes ``nan`` for every other speaker count the model
        supports, so that per-speaker-count loss curves stay aligned across
        batches with a different active ``num_spk``.
        """
        for n in range(1, self.num_spk + 1):
            stats[f"{prefix}_{n}spk_loss"] = (
                value.clone().detach() if n == num_spk else torch.nan
            )

    def _compute_attractor_counting_loss(
        self, others: Dict, num_spk: int
    ) -> Tuple[Optional[torch.Tensor], Dict[str, torch.Tensor]]:
        """BCE loss supervising the TDA/RopeTDA attractor existence probabilities.

        ``others["existance_probability"]`` has ``num_spk + 1`` entries per
        utterance: the first ``num_spk`` should predict "speaker present"
        (target 1) and the last models the "no more speakers" stop signal
        (target 0). Returns ``(None, {})`` if the extractor did not produce
        this side-output.
        """
        if "existance_probability" not in others:
            return None, {}
        stats: Dict[str, torch.Tensor] = {}
        with torch.cuda.amp.autocast(enabled=False):
            existence_prob = others["existance_probability"].to(torch.float32)
            bce = torch.nn.BCELoss(reduction="none")
            exist, non_exist = (
                existence_prob[..., :num_spk],
                existence_prob[..., num_spk],
            )
            bce_loss_exist = bce(exist, torch.ones_like(exist)).sum(dim=-1)
            bce_loss_non_exist = bce(non_exist, torch.zeros_like(non_exist))
            bce_loss = (
                (bce_loss_exist + bce_loss_non_exist) / (num_spk + 1)
            ).mean()
            stats["attractor_loss"] = bce_loss.detach()
            stats["attractor_loss_exist"] = bce_loss_exist.mean().detach()
            stats["attractor_loss_nonexist"] = bce_loss_non_exist.mean().detach()
            stats["attractor_exist_prob"] = exist.mean().detach()
            stats["attractor_nonexist_prob"] = non_exist.mean().detach()
        return bce_loss, stats

    def _compute_multidecoder_counting_loss(
        self, others: Dict, num_spk: int
    ) -> Tuple[Optional[torch.Tensor], Dict[str, torch.Tensor]]:
        """BCE loss for extractors that classify the speaker count directly.

        Expects ``others["spkest_probability"]`` (a distribution over
        ``self.extractor.num_spk_list``) and supervises it with a one-hot
        target for the true ``num_spk``. Returns ``(None, {})`` if the
        extractor did not produce this side-output.
        """
        if "spkest_probability" not in others:
            return None, {}
        stats: Dict[str, torch.Tensor] = {}
        bce = torch.nn.BCELoss(reduction="none")
        with torch.cuda.amp.autocast(enabled=False):
            est = others["spkest_probability"].to(torch.float32)
            idx = self.extractor.num_spk_list.index(num_spk)
            ref = (
                torch.nn.functional.one_hot(
                    torch.tensor([idx]), num_classes=est.shape[-1]
                )
                .tile((est.shape[0], 1))
                .to(torch.float32)
                .to(est.device)
            )
            bce_loss = bce(est, ref).mean()
            stats["bce_loss"] = bce_loss.detach()
            stats[f"{num_spk}-spk probability"] = est[:, idx].detach().mean()
        return bce_loss, stats

    # ------------------------------------------------------------------
    # Shared loss utilities
    # ------------------------------------------------------------------

    def _align_ref_pre_channels(
        self,
        ref: Optional[List[torch.Tensor]],
        pre: Optional[List[torch.Tensor]],
        ch_dim: int = 2,
        force_1ch: bool = False,
    ) -> Tuple[Optional[List[torch.Tensor]], Optional[List[torch.Tensor]]]:
        """Reconcile channel dimensions between references and predictions.

        Time-domain signals are expected as lists of (Batch, T) or
        (Batch, T, Channel) tensors. If one side is multi-channel and the
        other single-channel, the multi-channel side is reduced to
        ``self.ref_channel`` via ``index_select`` + ``squeeze``. When both
        sides are multi-channel and ``force_1ch`` is set (used for
        time-domain criteria, which only support single-channel targets),
        both are reduced to the reference channel.
        """
        if ref is None or pre is None:
            return ref, pre
        index = ref[0].new_tensor(self.ref_channel, dtype=torch.long)

        # for models like SVoice that output multiple lists of separated signals
        pre_is_multi_list = isinstance(pre[0], (list, tuple))
        pre_dim = pre[0][0].dim() if pre_is_multi_list else pre[0].dim()

        if ref[0].dim() > pre_dim:
            # multi-channel reference and single-channel output
            ref = [r.index_select(ch_dim, index).squeeze(ch_dim) for r in ref]
        elif ref[0].dim() < pre_dim:
            # single-channel reference and multi-channel output
            if pre_is_multi_list:
                pre = [
                    p.index_select(ch_dim, index).squeeze(ch_dim)
                    for plist in pre
                    for p in plist
                ]
            else:
                pre = [
                    p.index_select(ch_dim, index).squeeze(ch_dim) for p in pre
                ]
        elif ref[0].dim() == pre_dim == 3 and force_1ch:
            # multi-channel reference and output
            ref = [r.index_select(ch_dim, index).squeeze(ch_dim) for r in ref]
            if pre_is_multi_list:
                pre = [
                    p.index_select(ch_dim, index).squeeze(ch_dim)
                    for plist in pre
                    for p in plist
                ]
            else:
                pre = [
                    p.index_select(ch_dim, index).squeeze(ch_dim) for p in pre
                ]
        return ref, pre

    def _get_speech_masks(
        self,
        criterion: FrequencyDomainLoss,
        feature_mix: torch.Tensor,
        noise_ref: Optional[List[torch.Tensor]],
        speech_ref: List[torch.Tensor],
        speech_pre: List[torch.Tensor],
        ilens: torch.Tensor,
        others: Dict,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """Build reference/predicted TF masks for a mask-based frequency loss.

        Uses the extractor's own predicted masks (``others["mask_spk*"]``)
        when available instead of re-deriving them from ``speech_pre``.
        """
        if noise_ref is not None:
            noise_spec = self.encoder(sum(noise_ref), ilens)[0]
        else:
            noise_spec = None
        masks_ref = criterion.create_mask_label(
            feature_mix,
            [self.encoder(sr, ilens)[0] for sr in speech_ref],
            noise_spec=noise_spec,
        )
        if "mask_spk1" in others:
            masks_pre = [
                others["mask_spk{}".format(spk + 1)]
                for spk in range(self.num_spk)
            ]
        else:
            masks_pre = criterion.create_mask_label(
                feature_mix,
                [self.encoder(sp, ilens)[0] for sp in speech_pre],
                noise_spec=noise_spec,
            )
        return masks_ref, masks_pre

    def collect_feats(
        self,
        speech_mix: torch.Tensor,
        speech_mix_lengths: torch.Tensor,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """Return the (trimmed) mixture as the model's "feature" for stats."""
        speech_mix = speech_mix[:, : speech_mix_lengths.max()]
        feats, feats_lengths = speech_mix, speech_mix_lengths
        return {"feats": feats, "feats_lengths": feats_lengths}
