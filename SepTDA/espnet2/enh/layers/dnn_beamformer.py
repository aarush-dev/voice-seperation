"""DNN mask-based neural beamformer module.

Wraps a `MaskEstimator` (predicting per-speaker and, optionally, noise
time-frequency masks) together with one of the classical beamformer designs
implemented in `beamformer.py` / `beamformer_th.py` (MVDR, MPDR, WPD, MWF,
GEV, LCMV, ...). The masks are used to estimate spatial covariance matrices
("PSDs"), which are then fed into the closed-form beamformer solution to
compute a linear filter that is applied to the multi-channel STFT input.

See `beamformer.py`'s module docstring for the beamformer math and the
`psd`/`Rf`/shape notation used throughout.
"""
import logging
from typing import List, Optional, Tuple, Union

import torch
from packaging.version import parse as V
from torch.nn import functional as F
from torch_complex.tensor import ComplexTensor

import espnet2.enh.layers.beamformer as bf_v1
import espnet2.enh.layers.beamformer_th as bf_v2
from espnet2.enh.layers.complex_utils import stack, to_double, to_float
from espnet2.enh.layers.mask_estimator import MaskEstimator

is_torch_1_9_plus = V(torch.__version__) >= V("1.9.0")
is_torch_1_12_1_plus = V(torch.__version__) >= V("1.12.1")

ComplexOrRealTensor = Union[torch.Tensor, ComplexTensor]


BEAMFORMER_TYPES = (
    # Minimum Variance Distortionless Response beamformer
    "mvdr",  # RTF-based formula
    "mvdr_souden",  # Souden's solution
    # Minimum Power Distortionless Response beamformer
    "mpdr",  # RTF-based formula
    "mpdr_souden",  # Souden's solution
    # weighted MPDR beamformer
    "wmpdr",  # RTF-based formula
    "wmpdr_souden",  # Souden's solution
    # Weighted Power minimization Distortionless response beamformer
    "wpd",  # RTF-based formula
    "wpd_souden",  # Souden's solution
    # Multi-channel Wiener Filter (MWF) and weighted MWF
    "mwf",
    "wmwf",
    # Speech Distortion Weighted (SDW) MWF
    "sdw_mwf",
    # Rank-1 MWF
    "r1mwf",
    # Linearly Constrained Minimum Variance beamformer
    "lcmv",
    # Linearly Constrained Minimum Power beamformer
    "lcmp",
    # weighted Linearly Constrained Minimum Power beamformer
    "wlcmp",
    # Generalized Eigenvalue beamformer
    "gev",
    "gev_ban",  # with blind analytic normalization (BAN) post-filtering
    # time-frequency-bin-wise switching (TFS) MVDR beamformer
    "mvdr_tfs",
    "mvdr_tfs_souden",
)


class DNN_Beamformer(torch.nn.Module):
    """DNN mask based Beamformer.

    Citation:
        Multichannel End-to-end Speech Recognition; T. Ochiai et al., 2017;
        http://proceedings.mlr.press/v70/ochiai17a/ochiai17a.pdf

    """

    def __init__(
        self,
        bidim,
        btype: str = "blstmp",
        blayers: int = 3,
        bunits: int = 300,
        bprojs: int = 320,
        num_spk: int = 1,
        use_noise_mask: bool = True,
        nonlinear: str = "sigmoid",
        dropout_rate: float = 0.0,
        badim: int = 320,
        ref_channel: int = -1,
        beamformer_type: str = "mvdr_souden",
        rtf_iterations: int = 2,
        mwf_mu: float = 1.0,
        eps: float = 1e-6,
        diagonal_loading: bool = True,
        diag_eps: float = 1e-7,
        mask_flooring: bool = False,
        flooring_thres: float = 1e-6,
        use_torch_solver: bool = True,
        # False to use old APIs; True to use torchaudio-based new APIs
        use_torchaudio_api: bool = False,
        # only for WPD beamformer
        btaps: int = 5,
        bdelay: int = 3,
    ):
        super().__init__()
        bnmask = num_spk + 1 if use_noise_mask else num_spk
        self.mask = MaskEstimator(
            btype,
            bidim,
            blayers,
            bunits,
            bprojs,
            dropout_rate,
            nmask=bnmask,
            nonlinear=nonlinear,
        )
        self.ref = (
            AttentionReference(bidim, badim, eps=eps) if ref_channel < 0 else None
        )
        self.ref_channel = ref_channel

        self.use_noise_mask = use_noise_mask
        assert num_spk >= 1, num_spk
        self.num_spk = num_spk
        self.nmask = bnmask

        if beamformer_type not in BEAMFORMER_TYPES:
            raise ValueError("Not supporting beamformer_type=%s" % beamformer_type)
        if (
            beamformer_type == "mvdr_souden" or not beamformer_type.endswith("_souden")
        ) and not use_noise_mask:
            if num_spk == 1:
                logging.warning(
                    "Initializing %s beamformer without noise mask "
                    "estimator (single-speaker case)" % beamformer_type.upper()
                )
                logging.warning(
                    "(1 - speech_mask) will be used for estimating noise "
                    "PSD in %s beamformer!" % beamformer_type.upper()
                )
            else:
                logging.warning(
                    "Initializing %s beamformer without noise mask "
                    "estimator (multi-speaker case)" % beamformer_type.upper()
                )
                logging.warning(
                    "Interference speech masks will be used for estimating "
                    "noise PSD in %s beamformer!" % beamformer_type.upper()
                )

        self.beamformer_type = beamformer_type
        if not beamformer_type.endswith("_souden"):
            assert rtf_iterations >= 2, rtf_iterations
        # number of iterations in power method for estimating the RTF
        self.rtf_iterations = rtf_iterations
        # noise suppression weight in SDW-MWF
        self.mwf_mu = mwf_mu

        assert btaps >= 0 and bdelay >= 0, (btaps, bdelay)
        self.btaps = btaps
        self.bdelay = bdelay if self.btaps > 0 else 1
        self.eps = eps
        self.diagonal_loading = diagonal_loading
        self.diag_eps = diag_eps
        self.mask_flooring = mask_flooring
        self.flooring_thres = flooring_thres
        self.use_torch_solver = use_torch_solver
        if not use_torch_solver:
            logging.warning(
                "The `use_torch_solver` argument has been deprecated. "
                "Now it will always be true in DNN_Beamformer"
            )

        if use_torchaudio_api and is_torch_1_12_1_plus:
            self.bf_func = bf_v2
        else:
            self.bf_func = bf_v1

    def _split_masks(
        self, masks: List[torch.Tensor]
    ) -> Tuple[Union[torch.Tensor, List[torch.Tensor]], Optional[torch.Tensor]]:
        """Split the raw mask-estimator output into speech mask(s) and noise mask.

        Args:
            masks: `self.nmask` masks predicted by `self.mask`/`oracle_masks`,
                each (B, F, C, T).

        Returns:
            mask_speech: a single mask (single-speaker case) or list of masks
                (multi-speaker case), each (B, F, C, T).
            mask_noise: (B, F, C, T), or None if no noise mask is available
                (single-speaker, `use_noise_mask=False` case: the noise mask
                is left as None and callers fall back to `1 - mask_speech`
                inside `prepare_beamformer_stats`).
        """
        if self.num_spk == 1:
            if self.use_noise_mask:
                mask_speech, mask_noise = masks
            else:
                mask_speech = masks[0]
                mask_noise = 1 - mask_speech
            return mask_speech, mask_noise
        else:
            if self.use_noise_mask:
                mask_speech = list(masks[:-1])
                mask_noise = masks[-1]
            else:
                mask_speech = list(masks)
                mask_noise = None
            return mask_speech, mask_noise

    def _forward_single_speaker(
        self,
        data: ComplexOrRealTensor,
        data_d: ComplexOrRealTensor,
        ilens: torch.LongTensor,
        mask_speech: torch.Tensor,
        mask_noise: Optional[torch.Tensor],
        powers: Optional[List[torch.Tensor]],
    ) -> Tuple[ComplexOrRealTensor, ComplexOrRealTensor]:
        """Compute the beamformed output for the single-speaker case.

        Args:
            data: (B, F, C, T), original precision.
            data_d: `data` in double precision.
            ilens: (B,)
            mask_speech: (B, F, C, T)
            mask_noise: (B, F, C, T) or None
            powers: optional precomputed per-speaker power, used by
                wMPDR/WPD; see `prepare_beamformer_stats`.

        Returns:
            enhanced: (B, F, T)
            ws: beamforming vector(s), (B, F, C) or (B, F, (btaps+1)*C).
        """
        if self.beamformer_type in ("lcmv", "lcmp", "wlcmp"):
            raise NotImplementedError("Single source is not supported yet")
        beamformer_stats = self.bf_func.prepare_beamformer_stats(
            data_d,
            [mask_speech],
            mask_noise,
            powers=powers,
            beamformer_type=self.beamformer_type,
            bdelay=self.bdelay,
            btaps=self.btaps,
            eps=self.eps,
        )

        if self.beamformer_type in ("mvdr", "mpdr", "wmpdr", "wpd"):
            enhanced, ws = self.apply_beamforming(
                data,
                ilens,
                beamformer_stats["psd_n"],
                beamformer_stats["psd_speech"],
                psd_distortion=beamformer_stats["psd_distortion"],
            )
        elif (
            self.beamformer_type.endswith("_souden")
            or self.beamformer_type == "mwf"
            or self.beamformer_type == "wmwf"
            or self.beamformer_type == "sdw_mwf"
            or self.beamformer_type == "r1mwf"
            or self.beamformer_type.startswith("gev")
        ):
            enhanced, ws = self.apply_beamforming(
                data,
                ilens,
                beamformer_stats["psd_n"],
                beamformer_stats["psd_speech"],
            )
        else:
            raise ValueError(
                "Not supporting beamformer_type={}".format(self.beamformer_type)
            )
        return enhanced, ws

    def _forward_multi_speaker(
        self,
        data: ComplexOrRealTensor,
        data_d: ComplexOrRealTensor,
        ilens: torch.LongTensor,
        mask_speech: List[torch.Tensor],
        mask_noise: Optional[torch.Tensor],
        powers: Optional[List[torch.Tensor]],
    ) -> Tuple[List[ComplexOrRealTensor], List[ComplexOrRealTensor]]:
        """Compute the beamformed output for each speaker (multi-speaker case).

        Args:
            data: (B, F, C, T), original precision.
            data_d: `data` in double precision.
            ilens: (B,)
            mask_speech: per-speaker masks, each (B, F, C, T).
            mask_noise: (B, F, C, T) or None.
            powers: optional precomputed per-speaker power.

        Returns:
            enhanced: per-speaker enhanced signals, each (B, F, T).
            ws: per-speaker beamforming vectors.
        """
        beamformer_stats = self.bf_func.prepare_beamformer_stats(
            data_d,
            mask_speech,
            mask_noise,
            powers=powers,
            beamformer_type=self.beamformer_type,
            bdelay=self.bdelay,
            btaps=self.btaps,
            eps=self.eps,
        )
        rtf_mat = None
        if self.beamformer_type in ("lcmv", "lcmp", "wlcmp"):
            rtf_mat = self.bf_func.get_rtf_matrix(
                beamformer_stats["psd_speech"],
                beamformer_stats["psd_distortion"],
                diagonal_loading=self.diagonal_loading,
                ref_channel=self.ref_channel,
                rtf_iterations=self.rtf_iterations,
                diag_eps=self.diag_eps,
            )

        enhanced, ws = [], []
        for i in range(self.num_spk):
            # treat all other speakers' psd_speech as noises
            if self.beamformer_type in ("mvdr", "mvdr_tfs", "wmpdr", "wpd"):
                enh, w = self.apply_beamforming(
                    data,
                    ilens,
                    beamformer_stats["psd_n"][i],
                    beamformer_stats["psd_speech"][i],
                    psd_distortion=beamformer_stats["psd_distortion"][i],
                )
            elif self.beamformer_type in (
                "mvdr_souden",
                "mvdr_tfs_souden",
                "wmpdr_souden",
                "wpd_souden",
                "wmwf",
                "sdw_mwf",
                "r1mwf",
                "gev",
                "gev_ban",
            ):
                enh, w = self.apply_beamforming(
                    data,
                    ilens,
                    beamformer_stats["psd_n"][i],
                    beamformer_stats["psd_speech"][i],
                )
            elif self.beamformer_type == "mpdr":
                enh, w = self.apply_beamforming(
                    data,
                    ilens,
                    beamformer_stats["psd_n"],
                    beamformer_stats["psd_speech"][i],
                    psd_distortion=beamformer_stats["psd_distortion"][i],
                )
            elif self.beamformer_type in ("mpdr_souden", "mwf"):
                enh, w = self.apply_beamforming(
                    data,
                    ilens,
                    beamformer_stats["psd_n"],
                    beamformer_stats["psd_speech"][i],
                )
            elif self.beamformer_type == "lcmp":
                enh, w = self.apply_beamforming(
                    data,
                    ilens,
                    beamformer_stats["psd_n"],
                    beamformer_stats["psd_speech"][i],
                    rtf_mat=rtf_mat,
                    spk=i,
                )
            elif self.beamformer_type in ("lcmv", "wlcmp"):
                enh, w = self.apply_beamforming(
                    data,
                    ilens,
                    beamformer_stats["psd_n"][i],
                    beamformer_stats["psd_speech"][i],
                    rtf_mat=rtf_mat,
                    spk=i,
                )
            else:
                raise ValueError(
                    "Not supporting beamformer_type={}".format(self.beamformer_type)
                )

            enhanced.append(enh)
            ws.append(w)
        return enhanced, ws

    def forward(
        self,
        data: ComplexOrRealTensor,
        ilens: torch.LongTensor,
        powers: Optional[List[torch.Tensor]] = None,
        oracle_masks: Optional[List[torch.Tensor]] = None,
    ) -> Tuple[ComplexOrRealTensor, torch.LongTensor, torch.Tensor]:
        """DNN_Beamformer forward function.

        Notation:
            B: Batch
            C: Channel
            T: Time or Sequence length
            F: Freq

        Args:
            data (torch.complex64/ComplexTensor): (B, T, C, F)
            ilens (torch.Tensor): (B,)
            powers (List[torch.Tensor] or None): used for wMPDR or WPD (B, F, T)
            oracle_masks (List[torch.Tensor] or None): oracle masks (B, F, C, T)
                if not None, oracle_masks will be used instead of self.mask
        Returns:
            enhanced (torch.complex64/ComplexTensor): (B, T, F)
            ilens (torch.Tensor): (B,)
            masks (torch.Tensor): (B, T, C, F)
        """
        data = data.permute(0, 3, 2, 1)  # (B, T, C, F) -> (B, F, C, T)
        data_d = to_double(data)

        if oracle_masks is not None:
            masks = oracle_masks  # [(B, F, C, T)]
        else:
            masks, _ = self.mask(data, ilens)
        assert self.nmask == len(masks), len(masks)
        # floor masks to increase numerical stability
        if self.mask_flooring:
            masks = [torch.clamp(m, min=self.flooring_thres) for m in masks]

        mask_speech, mask_noise = self._split_masks(masks)

        if self.num_spk == 1:
            enhanced, ws = self._forward_single_speaker(
                data, data_d, ilens, mask_speech, mask_noise, powers
            )
            enhanced = enhanced.transpose(-1, -2)  # (..., F, T) -> (..., T, F)
        else:
            enhanced, ws = self._forward_multi_speaker(
                data, data_d, ilens, mask_speech, mask_noise, powers
            )
            enhanced = [enh.transpose(-1, -2) for enh in enhanced]

        masks = [m.transpose(-1, -3) for m in masks]  # (..., F, C, T) -> (..., T, C, F)
        return enhanced, ilens, masks

    def apply_beamforming(
        self,
        data: ComplexOrRealTensor,
        ilens: torch.LongTensor,
        psd_n: ComplexOrRealTensor,
        psd_speech: ComplexOrRealTensor,
        psd_distortion: Optional[ComplexOrRealTensor] = None,
        rtf_mat: Optional[ComplexOrRealTensor] = None,
        spk: int = 0,
    ) -> Tuple[ComplexOrRealTensor, ComplexOrRealTensor]:
        """Beamforming with the provided statistics.

        Args:
            data (torch.complex64/ComplexTensor): (B, F, C, T)
            ilens (torch.Tensor): (B,)
            psd_n (torch.complex64/ComplexTensor):
                Noise covariance matrix for MVDR (B, F, C, C)
                Observation covariance matrix for MPDR/wMPDR (B, F, C, C)
                Stacked observation covariance for WPD (B,F,(btaps+1)*C,(btaps+1)*C)
            psd_speech (torch.complex64/ComplexTensor):
                Speech covariance matrix (B, F, C, C)
            psd_distortion (torch.complex64/ComplexTensor):
                Noise covariance matrix (B, F, C, C)
            rtf_mat (torch.complex64/ComplexTensor):
                RTF matrix (B, F, C, num_spk)
            spk (int): speaker index
        Return:
            enhanced (torch.complex64/ComplexTensor): (B, F, T)
            ws (torch.complex64/ComplexTensor): (B, F) or (B, F, (btaps+1)*C)
        """
        u = self._get_reference_vector(data, ilens, psd_speech)

        if self.beamformer_type in ("mvdr", "mpdr", "wmpdr"):
            ws = self.bf_func.get_mvdr_vector_with_rtf(
                to_double(psd_n),
                to_double(psd_speech),
                to_double(psd_distortion),
                iterations=self.rtf_iterations,
                reference_vector=u,
                diagonal_loading=self.diagonal_loading,
                diag_eps=self.diag_eps,
            )
            enhanced = self.bf_func.apply_beamforming_vector(ws, to_double(data))
        elif self.beamformer_type == "mvdr_tfs":
            enhanced, ws = self._apply_tfs_beamforming(
                data, psd_n, psd_speech, psd_distortion, u, souden=False
            )
        elif self.beamformer_type in (
            "mpdr_souden",
            "mvdr_souden",
            "wmpdr_souden",
        ):
            ws = self.bf_func.get_mvdr_vector(
                to_double(psd_speech),
                to_double(psd_n),
                u,
                diagonal_loading=self.diagonal_loading,
                diag_eps=self.diag_eps,
            )
            enhanced = self.bf_func.apply_beamforming_vector(ws, to_double(data))
        elif self.beamformer_type == "mvdr_tfs_souden":
            enhanced, ws = self._apply_tfs_beamforming(
                data, psd_n, psd_speech, psd_distortion, u, souden=True
            )
        elif self.beamformer_type == "wpd":
            ws = self.bf_func.get_WPD_filter_with_rtf(
                to_double(psd_n),
                to_double(psd_speech),
                to_double(psd_distortion),
                iterations=self.rtf_iterations,
                reference_vector=u,
                diagonal_loading=self.diagonal_loading,
                diag_eps=self.diag_eps,
            )
            enhanced = self.bf_func.perform_WPD_filtering(
                ws, to_double(data), self.bdelay, self.btaps
            )
        elif self.beamformer_type == "wpd_souden":
            ws = self.bf_func.get_WPD_filter_v2(
                to_double(psd_speech),
                to_double(psd_n),
                u,
                diagonal_loading=self.diagonal_loading,
                diag_eps=self.diag_eps,
            )
            enhanced = self.bf_func.perform_WPD_filtering(
                ws, to_double(data), self.bdelay, self.btaps
            )
        elif self.beamformer_type in ("mwf", "wmwf"):
            ws = self.bf_func.get_mwf_vector(
                to_double(psd_speech),
                to_double(psd_n),
                u,
                diagonal_loading=self.diagonal_loading,
                diag_eps=self.diag_eps,
            )
            enhanced = self.bf_func.apply_beamforming_vector(ws, to_double(data))
        elif self.beamformer_type == "sdw_mwf":
            ws = self.bf_func.get_sdw_mwf_vector(
                to_double(psd_speech),
                to_double(psd_n),
                u,
                denoising_weight=self.mwf_mu,
                diagonal_loading=self.diagonal_loading,
                diag_eps=self.diag_eps,
            )
            enhanced = self.bf_func.apply_beamforming_vector(ws, to_double(data))
        elif self.beamformer_type == "r1mwf":
            ws = self.bf_func.get_rank1_mwf_vector(
                to_double(psd_speech),
                to_double(psd_n),
                u,
                denoising_weight=self.mwf_mu,
                diagonal_loading=self.diagonal_loading,
                diag_eps=self.diag_eps,
            )
            enhanced = self.bf_func.apply_beamforming_vector(ws, to_double(data))
        elif self.beamformer_type in ("lcmp", "wlcmp", "lcmv"):
            ws = self.bf_func.get_lcmv_vector_with_rtf(
                to_double(psd_n),
                to_double(rtf_mat),
                reference_vector=spk,
                diagonal_loading=self.diagonal_loading,
                diag_eps=self.diag_eps,
            )
            enhanced = self.bf_func.apply_beamforming_vector(ws, to_double(data))
        elif self.beamformer_type.startswith("gev"):
            ws = self.bf_func.get_gev_vector(
                to_double(psd_n),
                to_double(psd_speech),
                mode="power",
                diagonal_loading=self.diagonal_loading,
                diag_eps=self.diag_eps,
            )
            enhanced = self.bf_func.apply_beamforming_vector(ws, to_double(data))
            if self.beamformer_type == "gev_ban":
                gain = self.bf_func.blind_analytic_normalization(ws, to_double(psd_n))
                enhanced = enhanced * gain.unsqueeze(-1)
        else:
            raise ValueError(
                "Not supporting beamformer_type={}".format(self.beamformer_type)
            )

        return enhanced.to(dtype=data.dtype), ws.to(dtype=data.dtype)

    def _get_reference_vector(
        self,
        data: ComplexOrRealTensor,
        ilens: torch.LongTensor,
        psd_speech: ComplexOrRealTensor,
    ) -> Union[torch.Tensor, int]:
        """Determine the reference microphone/vector `u` used by the beamformer.

        Either a learned attention-based soft reference (`self.ref`, when
        `ref_channel < 0`), a fixed one-hot vector for Souden-form
        beamformers, or the raw channel index for RTF-based beamformers
        (which index directly into the RTF rather than needing a one-hot
        vector).
        """
        if self.ref_channel < 0:
            u, _ = self.ref(psd_speech.to(dtype=data.dtype), ilens)
            return u.double()
        if self.beamformer_type.endswith("_souden"):
            # (optional) Create onehot vector for fixed reference microphone
            u = torch.zeros(
                *(data.size()[:-3] + (data.size(-2),)),
                device=data.device,
                dtype=torch.double
            )
            u[..., self.ref_channel].fill_(1)
            return u
        # for simplifying computation in RTF-based beamforming
        return self.ref_channel

    def _apply_tfs_beamforming(
        self,
        data: ComplexOrRealTensor,
        psd_n: List[ComplexOrRealTensor],
        psd_speech: ComplexOrRealTensor,
        psd_distortion: Optional[ComplexOrRealTensor],
        u: Union[torch.Tensor, int],
        souden: bool,
    ) -> Tuple[ComplexOrRealTensor, ComplexOrRealTensor]:
        """Time-frequency-bin-wise switching (TFS) MVDR beamforming.

        Computes one MVDR beamformer per candidate interferer PSD in the
        `psd_n` list, applies each to `data`, and per time-frequency bin
        keeps the candidate with the smallest output magnitude (the
        assumption being that suppressing the true interferer yields the
        lowest-energy output).

        Args:
            psd_n: candidate noise PSDs, one per interfering speaker.
            souden: if True, use Souden-form MVDR (`get_mvdr_vector`);
                otherwise use the RTF-form MVDR (`get_mvdr_vector_with_rtf`).

        Returns:
            enhanced: (B, F, T)
            ws: stacked beamforming vectors used for each candidate, (n, B, F, C).
        """
        assert isinstance(psd_n, (list, tuple))
        if souden:
            ws = [
                self.bf_func.get_mvdr_vector(
                    to_double(psd_speech),
                    to_double(psd_n_i),
                    u,
                    diagonal_loading=self.diagonal_loading,
                    diag_eps=self.diag_eps,
                )
                for psd_n_i in psd_n
            ]
        else:
            ws = [
                self.bf_func.get_mvdr_vector_with_rtf(
                    to_double(psd_n_i),
                    to_double(psd_speech),
                    to_double(psd_distortion),
                    iterations=self.rtf_iterations,
                    reference_vector=u,
                    diagonal_loading=self.diagonal_loading,
                    diag_eps=self.diag_eps,
                )
                for psd_n_i in psd_n
            ]
        enhanced = stack(
            [self.bf_func.apply_beamforming_vector(w, to_double(data)) for w in ws]
        )
        with torch.no_grad():
            index = enhanced.abs().argmin(dim=0, keepdims=True)
        enhanced = enhanced.gather(0, index).squeeze(0)
        ws = stack(ws, dim=0)
        return enhanced, ws

    def predict_mask(
        self, data: ComplexOrRealTensor, ilens: torch.LongTensor
    ) -> Tuple[Tuple[torch.Tensor, ...], torch.LongTensor]:
        """Predict masks for beamforming.

        Args:
            data (torch.complex64/ComplexTensor): (B, T, C, F), double precision
            ilens (torch.Tensor): (B,)
        Returns:
            masks (torch.Tensor): (B, T, C, F)
            ilens (torch.Tensor): (B,)
        """
        masks, _ = self.mask(to_float(data.permute(0, 3, 2, 1)), ilens)
        # (B, F, C, T) -> (B, T, C, F)
        masks = [m.transpose(-1, -3) for m in masks]
        return masks, ilens


class AttentionReference(torch.nn.Module):
    """Attention-based soft reference-microphone selector.

    Given the speech PSD, predicts a soft weighting `u` over channels (an
    attention distribution) to use as the reference vector for beamformer
    designs that need one, instead of hard-selecting a fixed channel.
    """

    def __init__(self, bidim: int, att_dim: int, eps: float = 1e-6):
        super().__init__()
        self.mlp_psd = torch.nn.Linear(bidim, att_dim)
        self.gvec = torch.nn.Linear(att_dim, 1)
        self.eps = eps

    def forward(
        self,
        psd_in: ComplexOrRealTensor,
        ilens: torch.LongTensor,
        scaling: float = 2.0,
    ) -> Tuple[torch.Tensor, torch.LongTensor]:
        """Attention-based reference forward function.

        Args:
            psd_in (torch.complex64/ComplexTensor): (B, F, C, C)
            ilens (torch.Tensor): (B,)
            scaling (float):
        Returns:
            u (torch.Tensor): (B, C)
            ilens (torch.Tensor): (B,)
        """
        B, _, C = psd_in.size()[:3]
        assert psd_in.size(2) == psd_in.size(3), psd_in.size()
        # psd_in: (B, F, C, C), zero out the diagonal (self-channel term)
        psd = psd_in.masked_fill(
            torch.eye(C, dtype=torch.bool, device=psd_in.device).type(torch.bool), 0
        )
        # psd: (B, F, C, C) -> (B, C, F), averaged over the other C - 1 channels
        psd = (psd.sum(dim=-1) / (C - 1)).transpose(-1, -2)

        # Calculate amplitude
        psd_feat = (psd.real**2 + psd.imag**2 + self.eps) ** 0.5

        # (B, C, F) -> (B, C, F2)
        mlp_psd = self.mlp_psd(psd_feat)
        # (B, C, F2) -> (B, C, 1) -> (B, C)
        e = self.gvec(torch.tanh(mlp_psd)).squeeze(-1)
        u = F.softmax(scaling * e, dim=-1)
        return u, ilens
