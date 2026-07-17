"""Enhancement model module.

Ties together the frontend pipeline used for speech enhancement/separation
training:

    waveform --[encoder]--> features --[separator]--> per-speaker features
             --[decoder]--> per-speaker waveforms

and computes the training loss by handing the (reference, estimate) pairs to
one or more configured loss wrappers (see ``espnet2.enh.loss.wrappers``),
which in turn call per-pair loss criterions (see
``espnet2.enh.loss.criterions``). Some loss wrappers (PIT, MixIT) search over
speaker permutations; the winning permutation is threaded through
``others["perm"]`` so that later loss wrappers/criteria in the same forward
pass can reuse it instead of re-solving the assignment.
"""
from typing import Dict, List, Optional, OrderedDict, Tuple

import torch
from typeguard import check_argument_types

from espnet2.diar.layers.abs_mask import AbsMask
from espnet2.enh.decoder.abs_decoder import AbsDecoder
from espnet2.enh.encoder.abs_encoder import AbsEncoder
from espnet2.enh.loss.criterions.tf_domain import FrequencyDomainLoss
from espnet2.enh.loss.criterions.time_domain import TimeDomainLoss
from espnet2.enh.loss.wrappers.abs_wrapper import AbsLossWrapper
from espnet2.enh.separator.abs_separator import AbsSeparator
from espnet2.torch_utils.device_funcs import force_gatherable
from espnet2.train.abs_espnet_model import AbsESPnetModel


class ESPnetEnhancementModel(AbsESPnetModel):
    """Speech enhancement or separation Frontend model"""

    def __init__(
        self,
        encoder: AbsEncoder,
        separator: AbsSeparator,
        decoder: AbsDecoder,
        mask_module: Optional[AbsMask],
        loss_wrappers: List[AbsLossWrapper],
        stft_consistency: bool = False,
        loss_type: str = "mask_mse",
        mask_type: Optional[str] = None,
    ):
        assert check_argument_types()

        super().__init__()

        self.encoder = encoder
        self.separator = separator
        self.decoder = decoder
        self.mask_module = mask_module
        self.num_spk = separator.num_spk
        self.num_noise_type = getattr(self.separator, "num_noise_type", 1)

        self.loss_wrappers = loss_wrappers
        names = [w.criterion.name for w in self.loss_wrappers]
        if len(set(names)) != len(names):
            raise ValueError("Duplicated loss names are not allowed: {}".format(names))

        # get mask type for TF-domain models
        # (only used when loss_type="mask_*") (deprecated, keep for compatibility)
        self.mask_type = mask_type.upper() if mask_type else None

        # get loss type for model training (deprecated, keep for compatibility)
        self.loss_type = loss_type

        # whether to compute the TF-domain loss while enforcing STFT consistency
        # (deprecated, keep for compatibility)
        # NOTE: STFT consistency is now always used for frequency-domain spectrum losses
        self.stft_consistency = stft_consistency

        # for multi-channel signal
        self.ref_channel = getattr(self.separator, "ref_channel", None)
        if self.ref_channel is None:
            self.ref_channel = 0

    def forward(
        self,
        speech_mix: torch.Tensor,
        speech_mix_lengths: torch.Tensor = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
        """Frontend + Encoder + Decoder + Calc loss

        Args:
            speech_mix: (Batch, samples) or (Batch, samples, channels)
            speech_ref: (Batch, num_speaker, samples)
                        or (Batch, num_speaker, samples, channels)
            speech_mix_lengths: (Batch,), default None for chunk interator,
                            because the chunk-iterator does not have the
                            speech_lengths returned. see in
                            espnet2/iterators/chunk_iter_factory.py
            kwargs: "utt_id" is among the input.
        """
        speech_ref = self._gather_speech_ref(kwargs)
        noise_ref = self._gather_noise_ref(kwargs)
        dereverb_speech_ref = self._gather_dereverb_ref(kwargs)

        batch_size = speech_mix.shape[0]
        speech_lengths = (
            speech_mix_lengths
            if speech_mix_lengths is not None
            else torch.ones(batch_size).int().fill_(speech_mix.shape[1])
        )
        assert speech_lengths.dim() == 1, speech_lengths.shape
        # Check that batch_size is unified
        assert speech_mix.shape[0] == speech_ref.shape[0] == speech_lengths.shape[0], (
            speech_mix.shape,
            speech_ref.shape,
            speech_lengths.shape,
        )

        # for data-parallel
        max_len = speech_lengths.max()
        speech_ref = speech_ref[..., :max_len].unbind(dim=1)
        if noise_ref is not None:
            noise_ref = noise_ref[..., :max_len].unbind(dim=1)
        if dereverb_speech_ref is not None:
            dereverb_speech_ref = dereverb_speech_ref[..., :max_len]
            dereverb_speech_ref = dereverb_speech_ref.unbind(dim=1)

        additional = {}
        speech_mix = speech_mix[:, :max_len]

        # model forward
        speech_pre, feature_mix, feature_pre, others = self.forward_enhance(
            speech_mix, speech_lengths, additional
        )

        # loss computation
        loss, stats, weight, perm = self.forward_loss(
            speech_pre,
            speech_lengths,
            feature_mix,
            feature_pre,
            others,
            speech_ref,
            noise_ref,
            dereverb_speech_ref,
        )
        return loss, stats, weight

    def _gather_speech_ref(self, kwargs: Dict) -> torch.Tensor:
        """Stack the per-speaker ``speech_refN`` kwargs into one tensor.

        Args:
            kwargs: forward() kwargs, expected to contain at least
                ``speech_ref1`` and, ideally, one ``speech_refN`` per speaker.
                Missing speakers (beyond what was actually provided) are
                padded with zeros so that all utterances in a batch line up
                to ``self.num_spk`` speakers.
        Returns:
            (Batch, num_spk, samples[, channels])
        """
        assert "speech_ref1" in kwargs, "At least 1 reference signal input is required."
        speech_ref = [
            kwargs.get(
                f"speech_ref{spk + 1}",
                torch.zeros_like(kwargs["speech_ref1"]),
            )
            for spk in range(self.num_spk)
        ]
        return torch.stack(speech_ref, dim=1)

    def _gather_noise_ref(self, kwargs: Dict) -> Optional[torch.Tensor]:
        """Stack the per-noise-type ``noise_refN`` kwargs, if present.

        Noise references are optional and only used when training
        beamforming-based frontend models that also predict a noise output.

        Returns:
            (Batch, num_noise_type, samples[, channels]) or ``None``.
        """
        if "noise_ref1" not in kwargs:
            return None
        noise_ref = [
            kwargs["noise_ref{}".format(n + 1)] for n in range(self.num_noise_type)
        ]
        return torch.stack(noise_ref, dim=1)

    def _gather_dereverb_ref(self, kwargs: Dict) -> Optional[torch.Tensor]:
        """Stack the per-speaker ``dereverb_refN`` kwargs, if present.

        Dereverberated (but still noisy) references are optional and only
        used for frontend models with a WPE-style dereverberation stage.
        Either a single shared dereverb reference or one per speaker is
        accepted.

        Returns:
            (Batch, N, samples[, channels]) or ``None``, where ``N`` is 1 or
            ``self.num_spk``.
        """
        if "dereverb_ref1" not in kwargs:
            return None
        dereverb_speech_ref = [
            kwargs["dereverb_ref{}".format(n + 1)]
            for n in range(self.num_spk)
            if "dereverb_ref{}".format(n + 1) in kwargs
        ]
        assert len(dereverb_speech_ref) in (1, self.num_spk), len(
            dereverb_speech_ref
        )
        return torch.stack(dereverb_speech_ref, dim=1)

    def forward_enhance(
        self,
        speech_mix: torch.Tensor,
        speech_lengths: torch.Tensor,
        additional: Optional[Dict] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict]:
        """Run encoder -> separator (+ optional mask module) -> decoder.

        Args:
            speech_mix: (Batch, samples[, channels]) mixture waveform.
            speech_lengths: (Batch,) valid length of each mixture.
            additional: extra separator inputs (e.g. ``num_spk`` for
                variable-speaker-count models using ``mask_module``).

        Returns:
            speech_pre: List[Tensor(Batch, samples)] per-speaker waveforms
                decoded from the separator's output features, or ``None``
                for models that only predict masks/features during training
                (e.g. a neural beamformer trained with a mask loss).
            feature_mix: encoder output for the mixture.
            feature_pre: separator output features (pre-decoder), or
                ``None`` if the separator did not produce a per-speaker
                estimate (e.g. no ``num_spk`` was given to ``mask_module``).
            others: dict of auxiliary separator outputs (e.g. estimated
                noise/dereverb features, masks, bottleneck features).
        """
        feature_mix, flens = self.encoder(speech_mix, speech_lengths)
        if self.mask_module is None:
            feature_pre, flens, others = self.separator(feature_mix, flens, additional)
        else:
            # Obtain bottleneck_feats from separator.
            # This is used for the input of diarization module in "enh + diar" task
            bottleneck_feats, bottleneck_feats_lengths = self.separator(
                feature_mix, flens
            )
            if additional.get("num_spk") is not None:
                feature_pre, flens, others = self.mask_module(
                    feature_mix, flens, bottleneck_feats, additional["num_spk"]
                )
                others["bottleneck_feats"] = bottleneck_feats
                others["bottleneck_feats_lengths"] = bottleneck_feats_lengths
            else:
                feature_pre = None
                others = {
                    "bottleneck_feats": bottleneck_feats,
                    "bottleneck_feats_lengths": bottleneck_feats_lengths,
                }
        if feature_pre is not None:
            # for models like SVoice that output multiple lists of separated signals
            pre_is_multi_list = isinstance(feature_pre[0], (list, tuple))
            if pre_is_multi_list:
                speech_pre = [
                    [self.decoder(p, speech_lengths)[0] for p in ps]
                    for ps in feature_pre
                ]
            else:
                speech_pre = [self.decoder(ps, speech_lengths)[0] for ps in feature_pre]
        else:
            # some models (e.g. neural beamformer trained with mask loss)
            # do not predict time-domain signal in the training stage
            speech_pre = None
        return speech_pre, feature_mix, feature_pre, others

    def forward_loss(
        self,
        speech_pre: torch.Tensor,
        speech_lengths: torch.Tensor,
        feature_mix: torch.Tensor,
        feature_pre: torch.Tensor,
        others: OrderedDict,
        speech_ref: torch.Tensor,
        noise_ref: torch.Tensor = None,
        dereverb_speech_ref: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
        """Decode auxiliary outputs and accumulate the weighted multi-task loss.

        Iterates over ``self.loss_wrappers``, and for each one:

        1. picks the (reference, estimate) pair the underlying criterion
           targets -- the main speech estimate, or (if the criterion is
           flagged ``is_noise_loss``/``is_dereverb_loss``) the auxiliary
           noise/dereverb estimate;
        2. dispatches to time-domain or frequency-domain handling depending
           on the criterion's type, since frequency-domain criterions may
           additionally need mask labels/estimates rather than raw spectra;
        3. accumulates ``loss += weight * wrapper_loss`` and merges the
           returned stats.

        Because loss wrappers may perform a permutation search (PIT/MixIT),
        each wrapper receives the ``others`` dict returned by the *previous*
        wrapper merged on top of the separator's ``others`` -- this is how a
        permutation chosen by an earlier wrapper (e.g. for the main speech
        loss) becomes visible to a later wrapper via ``others["perm"]``
        (typically combined with ``independent_perm=False`` on that later
        wrapper). The first "perm" seen across all wrappers is returned to
        the caller.

        Returns:
            loss: scalar Tensor, the total weighted loss.
            stats: dict of scalar Tensors for logging (includes ``"loss"``).
            weight: batch size, used by ``force_gatherable`` for DataParallel.
            perm: permutation chosen by the first wrapper that produced one,
                or ``None`` if no wrapper performed a permutation search.
        """
        others = self._decode_auxiliary_signals(
            others, speech_lengths, noise_ref, dereverb_speech_ref
        )

        loss = 0.0
        stats = {}
        wrapper_others = {}
        perm = None
        for loss_wrapper in self.loss_wrappers:
            criterion = loss_wrapper.criterion
            if getattr(criterion, "only_for_test", False) and self.training:
                continue

            signal_ref, signal_pre = self._select_loss_targets(
                criterion, others, speech_ref, speech_pre, noise_ref, dereverb_speech_ref
            )

            if isinstance(criterion, TimeDomainLoss):
                assert signal_pre is not None
                sref, spre = self._align_ref_pre_channels(
                    signal_ref, signal_pre, ch_dim=2, force_1ch=True
                )
                # for the time domain criterions
                l, s, wrapper_others = loss_wrapper(
                    sref, spre, {**others, **wrapper_others}
                )
            elif isinstance(criterion, FrequencyDomainLoss):
                sref, spre = self._align_ref_pre_channels(
                    signal_ref, signal_pre, ch_dim=2, force_1ch=False
                )
                tf_ref, tf_pre = self._compute_tf_domain_targets(
                    criterion,
                    feature_mix,
                    noise_ref,
                    speech_ref,
                    signal_ref,
                    signal_pre,
                    sref,
                    spre,
                    speech_lengths,
                    others,
                )
                # for the time-frequency domain criterions
                l, s, wrapper_others = loss_wrapper(
                    tf_ref, tf_pre, {**others, **wrapper_others}
                )
            else:
                raise NotImplementedError("Unsupported loss type: %s" % str(criterion))

            loss += l * loss_wrapper.weight
            stats.update(s)

            if perm is None and "perm" in wrapper_others:
                perm = wrapper_others["perm"]

        if self.training and isinstance(loss, float):
            raise AttributeError(
                "At least one criterion must satisfy: only_for_test=False"
            )
        stats["loss"] = loss.detach()

        # force_gatherable: to-device and to-tensor if scalar for DataParallel
        batch_size = speech_ref[0].shape[0]
        loss, stats, weight = force_gatherable((loss, stats, batch_size), loss.device)
        return loss, stats, weight, perm

    def _decode_auxiliary_signals(
        self,
        others: Dict,
        speech_lengths: torch.Tensor,
        noise_ref: Optional[torch.Tensor],
        dereverb_speech_ref: Optional[torch.Tensor],
    ) -> Dict:
        """Decode the separator's raw noise/dereverb *features* into waveforms.

        The separator may expose auxiliary per-type features under
        ``others["noiseN"]`` / ``others["dereverbN"]``; if the corresponding
        references were provided (so a loss can actually be computed against
        them), those entries are decoded in place to waveforms so that
        time-domain criterions can consume them directly.
        """
        if getattr(self.separator, "predict_noise", False):
            assert "noise1" in others, others.keys()
        if noise_ref is not None and "noise1" in others:
            for n in range(self.num_noise_type):
                key = "noise{}".format(n + 1)
                others[key] = self.decoder(others[key], speech_lengths)[0]

        if getattr(self.separator, "predict_dereverb", False):
            assert "dereverb1" in others, others.keys()
        if dereverb_speech_ref is not None and "dereverb1" in others:
            for spk in range(self.num_spk):
                key = "dereverb{}".format(spk + 1)
                if key in others:
                    others[key] = self.decoder(others[key], speech_lengths)[0]
        return others

    def _select_loss_targets(
        self,
        criterion,
        others: Dict,
        speech_ref,
        speech_pre,
        noise_ref: Optional[torch.Tensor],
        dereverb_speech_ref: Optional[torch.Tensor],
    ):
        """Pick which (reference, estimate) pair a given criterion is scored on.

        Most criterions target the main speech estimate, but a criterion may
        be flagged ``is_noise_loss`` or ``is_dereverb_loss`` (mutually
        exclusive, see :class:`TimeDomainLoss`/`FrequencyDomainLoss`) to
        instead be scored against the estimated noise or dereverberated
        signal produced alongside the main separation output.
        """
        if getattr(criterion, "is_noise_loss", False):
            if noise_ref is None:
                raise ValueError(
                    "No noise reference for training!\n"
                    'Please specify "--use_noise_ref true" in run.sh'
                )
            signal_ref = noise_ref
            signal_pre = [
                others["noise{}".format(n + 1)] for n in range(self.num_noise_type)
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
        return signal_ref, signal_pre

    def _compute_tf_domain_targets(
        self,
        criterion: FrequencyDomainLoss,
        feature_mix: torch.Tensor,
        noise_ref: Optional[torch.Tensor],
        speech_ref,
        signal_ref,
        signal_pre,
        sref,
        spre,
        speech_lengths: torch.Tensor,
        others: Dict,
    ):
        """Compute the (reference, estimate) pair a frequency-domain criterion sees.

        If ``criterion.compute_on_mask`` is set, both sides are oracle/estimated
        *masks* (built via ``criterion.create_mask_label``, dispatched by
        whether the criterion targets noise, dereverb, or speech), computed
        from the original, channel-unaligned ``signal_ref``/``signal_pre``
        (mask creation itself is channel-agnostic: it always encodes each
        signal from scratch). Otherwise, both sides are raw encoder spectra
        of the already channel-aligned ``sref``/``spre`` time-domain signals.
        """
        if criterion.compute_on_mask:
            if getattr(criterion, "is_noise_loss", False):
                return self._get_noise_masks(
                    criterion,
                    feature_mix,
                    speech_ref,
                    signal_ref,
                    signal_pre,
                    speech_lengths,
                    others,
                )
            elif getattr(criterion, "is_dereverb_loss", False):
                return self._get_dereverb_masks(
                    criterion,
                    feature_mix,
                    noise_ref,
                    signal_ref,
                    signal_pre,
                    speech_lengths,
                    others,
                )
            else:
                return self._get_speech_masks(
                    criterion,
                    feature_mix,
                    noise_ref,
                    signal_ref,
                    signal_pre,
                    speech_lengths,
                    others,
                )

        # compute on spectrum
        tf_ref = [self.encoder(sr, speech_lengths)[0] for sr in sref]
        # for models like SVoice that output multiple lists of separated signals
        pre_is_multi_list = isinstance(spre[0], (list, tuple))
        if pre_is_multi_list:
            tf_pre = [
                [self.encoder(sp, speech_lengths)[0] for sp in ps] for ps in spre
            ]
        else:
            tf_pre = [self.encoder(sp, speech_lengths)[0] for sp in spre]
        return tf_ref, tf_pre

    def _align_ref_pre_channels(self, ref, pre, ch_dim=2, force_1ch=False):
        """Reconcile channel-count mismatches between references and estimates.

        References and estimates may disagree in whether they carry a
        channel dimension at all (e.g. a multi-channel reference vs. a
        single-channel beamformed estimate, or vice versa): whichever side
        has *more* channels is reduced to ``self.ref_channel`` so both sides
        end up with the same number of dimensions before being handed to a
        criterion. When ``force_1ch=True`` and both sides already agree on
        being multi-channel (3-D), both are still collapsed to a single
        reference channel -- used for time-domain criterions, which are not
        defined for multi-channel signals.

        Args:
            ref: List[Tensor], each (Batch, samples[, channels]).
            pre: List[Tensor] or List[List[Tensor]] (for multi-list models
                like SVoice), each leaf tensor (Batch, samples[, channels]).
            ch_dim: dimension index of the channel axis when present.
            force_1ch: also collapse the channel axis when ref and pre are
                both already multi-channel with matching dims.
        Returns:
            (ref, pre) with channel dims reconciled; unchanged if either
            side is ``None``.
        """
        if ref is None or pre is None:
            return ref, pre
        # NOTE: input must be a list of time-domain signals
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
                pre = [p.index_select(ch_dim, index).squeeze(ch_dim) for p in pre]
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
                pre = [p.index_select(ch_dim, index).squeeze(ch_dim) for p in pre]
        return ref, pre

    def _get_noise_masks(
        self, criterion, feature_mix, speech_ref, noise_ref, noise_pre, ilens, others
    ):
        """Build oracle/estimated masks for a noise-targeted mask loss.

        The oracle mask treats the summed clean speech as the "interference"
        relative to each noise reference (see
        ``FrequencyDomainLoss.create_mask_label``'s ``noise_spec`` argument).
        """
        speech_spec = self.encoder(sum(speech_ref), ilens)[0]
        masks_ref = criterion.create_mask_label(
            feature_mix,
            [self.encoder(nr, ilens)[0] for nr in noise_ref],
            noise_spec=speech_spec,
        )
        if "mask_noise1" in others:
            masks_pre = [
                others["mask_noise{}".format(n + 1)] for n in range(self.num_noise_type)
            ]
        else:
            assert len(noise_pre) == len(noise_ref), (len(noise_pre), len(noise_ref))
            masks_pre = criterion.create_mask_label(
                feature_mix,
                [self.encoder(np, ilens)[0] for np in noise_pre],
                noise_spec=speech_spec,
            )
        return masks_ref, masks_pre

    def _get_dereverb_masks(
        self, criterion, feat_mix, noise_ref, dereverb_ref, dereverb_pre, ilens, others
    ):
        """Build oracle/estimated masks for a dereverberation-targeted mask loss."""
        if noise_ref is not None:
            noise_spec = self.encoder(sum(noise_ref), ilens)[0]
        else:
            noise_spec = None
        masks_ref = criterion.create_mask_label(
            feat_mix,
            [self.encoder(dr, ilens)[0] for dr in dereverb_ref],
            noise_spec=noise_spec,
        )
        if "mask_dereverb1" in others:
            masks_pre = [
                others["mask_dereverb{}".format(spk + 1)]
                for spk in range(self.num_spk)
                if "mask_dereverb{}".format(spk + 1) in others
            ]
            assert len(masks_pre) == len(masks_ref), (len(masks_pre), len(masks_ref))
        else:
            assert len(dereverb_pre) == len(dereverb_ref), (
                len(dereverb_pre),
                len(dereverb_ref),
            )
            masks_pre = criterion.create_mask_label(
                feat_mix,
                [self.encoder(dp, ilens)[0] for dp in dereverb_pre],
                noise_spec=noise_spec,
            )
        return masks_ref, masks_pre

    def _get_speech_masks(
        self, criterion, feature_mix, noise_ref, speech_ref, speech_pre, ilens, others
    ):
        """Build oracle/estimated masks for the main per-speaker mask loss."""
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
                others["mask_spk{}".format(spk + 1)] for spk in range(self.num_spk)
            ]
        else:
            masks_pre = criterion.create_mask_label(
                feature_mix,
                [self.encoder(sp, ilens)[0] for sp in speech_pre],
                noise_spec=noise_spec,
            )
        return masks_ref, masks_pre

    @staticmethod
    def sort_by_perm(nn_output, perm):
        """Sort the input list of tensors by the specified permutation.

        Args:
            nn_output: List[torch.Tensor(Batch, ...)], len(nn_output) == num_spk
            perm: (Batch, num_spk) or List[torch.Tensor(num_spk)]
        Returns:
            nn_output_new: List[torch.Tensor(Batch, ...)]
        """
        if len(nn_output) == 1:
            return nn_output
        # (Batch, num_spk, ...)
        nn_output = torch.stack(nn_output, dim=1)
        if not isinstance(perm, torch.Tensor):
            # perm is a list or tuple
            perm = torch.stack(perm, dim=0)
        assert nn_output.size(1) == perm.size(1), (nn_output.shape, perm.shape)
        diff_dim = nn_output.dim() - perm.dim()
        if diff_dim > 0:
            perm = perm.view(*perm.shape, *[1 for _ in range(diff_dim)]).expand_as(
                nn_output
            )
        return torch.gather(nn_output, 1, perm).unbind(dim=1)

    def collect_feats(
        self, speech_mix: torch.Tensor, speech_mix_lengths: torch.Tensor, **kwargs
    ) -> Dict[str, torch.Tensor]:
        # for data-parallel
        speech_mix = speech_mix[:, : speech_mix_lengths.max()]

        feats, feats_lengths = speech_mix, speech_mix_lengths
        return {"feats": feats, "feats_lengths": feats_lengths}
