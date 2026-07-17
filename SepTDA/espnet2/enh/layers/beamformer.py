"""Frequency-domain multi-channel beamformers (MVDR/MPDR/MWF/GEV/WPD family).

All beamformers here operate per time-frequency bin on a spatial covariance
formulation: given a (possibly power-weighted) speech covariance matrix
``psd_speech`` and a noise/observation covariance matrix ``psd_n``, they solve
for a linear filter ``w`` (the "beamforming vector") that is applied to the
multi-channel STFT signal to produce a single-channel enhanced output,
``enhanced[..., t] = w^H @ mix[..., :, t]`` (see `apply_beamforming_vector`).

Supported filter designs (see the corresponding ``get_*_vector`` function):
    * MVDR / MPDR / wMPDR (`get_mvdr_vector`, `get_mvdr_vector_with_rtf`):
      Minimum Variance Distortionless Response, in Souden's closed form or
      the relative-transfer-function (RTF) form.
    * MWF / SDW-MWF / R1-MWF (`get_mwf_vector`, `get_sdw_mwf_vector`,
      `get_rank1_mwf_vector`): Multi-channel Wiener Filter variants that
      trade off noise suppression against speech distortion.
    * LCMV (`get_lcmv_vector_with_rtf`): Linearly Constrained Minimum
      Variance beamformer for multiple simultaneous speakers.
    * GEV (`get_gev_vector`): Generalized Eigenvalue beamformer.
    * WPD (`get_WPD_filter`, `get_WPD_filter_v2`, `get_WPD_filter_with_rtf`):
      Weighted Power minimization Distortionless response convolutional
      beamformer, which jointly performs dereverberation (via a WPE-like
      stacked spatio-temporal covariance) and denoising/beamforming.

Notation used throughout:
    ``B``: batch, ``F``: frequency bins, ``C``: microphone channels,
    ``T``: time frames, ``btaps``/``bdelay``: WPD filter order and
    prediction delay (analogous to WPE's `taps`/`delay`).
    ``psd`` denotes a spatial covariance ("power spectral density") matrix,
    shape (..., C, C); ``Rf`` denotes the stacked spatio-temporal covariance
    used by WPD, shape (..., (btaps+1)*C, (btaps+1)*C).

This module supports both the third-party ``ComplexTensor`` and native
``torch`` complex tensor backends via `espnet2.enh.layers.complex_utils`; see
`beamformer_th.py` for a torchaudio-native alternative implementation of
(mostly) the same functions.
"""
from typing import Dict, List, Optional, Sequence, Union

import torch
from packaging.version import parse as V
from torch_complex import functional as FC
from torch_complex.tensor import ComplexTensor

from espnet2.enh.layers.complex_utils import (
    cat,
    complex_norm,
    einsum,
    inverse,
    is_complex,
    is_torch_complex_tensor,
    matmul,
    reverse,
    solve,
    to_double,
)

is_torch_1_9_plus = V(torch.__version__) >= V("1.9.0")
EPS = torch.finfo(torch.double).eps

ComplexOrRealTensor = Union[torch.Tensor, ComplexTensor]


def _as_double_mask_list(
    masks_speech: Union[ComplexOrRealTensor, Sequence[ComplexOrRealTensor]],
) -> List[torch.Tensor]:
    """Normalize `masks_speech` into a list of double-precision mask tensors."""
    if isinstance(masks_speech, (list, tuple)):
        return [to_double(m) for m in masks_speech]
    return [to_double(masks_speech)]


def _needs_power_weighting(beamformer_type: str) -> bool:
    """True if `beamformer_type` weights its statistics by inverse signal power."""
    return (
        beamformer_type.startswith("wmpdr")
        or beamformer_type.startswith("wpd")
        or beamformer_type == "wlcmp"
        or beamformer_type == "wmwf"
    )


def _compute_inverse_powers(
    signal: ComplexOrRealTensor,
    masks_speech: List[torch.Tensor],
    powers: Optional[List[torch.Tensor]],
    eps: float,
) -> List[torch.Tensor]:
    """Compute per-speaker inverse power weighting, (..., T) each.

    Args:
        signal: (..., F, C, T)
        masks_speech: per-speaker masks, each (..., F, C, T).
        powers: optional precomputed per-speaker power, each (..., F, T); if
            None, it is estimated as the mask-weighted channel-averaged power
            of `signal`.
        eps: floor applied to power before inversion.
    """
    if powers is None:
        power_input = signal.real**2 + signal.imag**2  # (..., F, C, T)
        # Averaging along the channel axis: (..., C, T) -> (..., T)
        powers = [(power_input * m).mean(dim=-2) for m in masks_speech]
    else:
        assert len(powers) == len(masks_speech), (len(powers), len(masks_speech))
    return [1 / torch.clamp(p, min=eps) for p in powers]


def _compute_rtf_style_psd_noise(
    signal: ComplexOrRealTensor,
    psd_speeches: List[ComplexOrRealTensor],
    mask_noise: Optional[torch.Tensor],
    beamformer_type: str,
    num_spk: int,
):
    """Compute the noise/interference PSD(s) used by RTF-based beamformers.

    For a single speaker, this is simply the noise-mask PSD. For multiple
    speakers, the "noise" for speaker `i` is the background noise PSD (if a
    noise mask is available) plus the other speakers' speech PSDs -- except
    for `beamformer_type` starting with "mvdr_tfs", where the per-speaker
    interfering PSDs are kept as a list rather than summed (time-frequency
    switching selects among them later).

    Returns None if `beamformer_type` does not use this RTF-style noise PSD
    (e.g. Souden-form beamformers other than "mvdr_souden").
    """
    if not (
        beamformer_type == "mvdr_souden"
        or beamformer_type == "sdw_mwf"
        or beamformer_type == "r1mwf"
        or beamformer_type.startswith("mvdr_tfs")
        or not beamformer_type.endswith("_souden")
    ):
        return None

    if mask_noise is not None:
        psd_bg = get_power_spectral_density_matrix(signal, to_double(mask_noise))
    if num_spk == 1:
        assert mask_noise is not None
        psd_noise = psd_bg
    else:
        psd_noise = []
        for i in range(num_spk):
            if beamformer_type.startswith("mvdr_tfs"):
                # NOTE: psd_noise is a list only for this beamformer
                psd_noise_i = [psd for j, psd in enumerate(psd_speeches) if j != i]
            else:
                psd_sum = sum(psd for j, psd in enumerate(psd_speeches) if j != i)
                psd_noise_i = psd_bg + psd_sum if mask_noise is not None else psd_sum
            psd_noise.append(psd_noise_i)
    return psd_noise


def _select_psd_n(
    beamformer_type: str,
    signal: ComplexOrRealTensor,
    inverse_powers: Optional[List[torch.Tensor]],
    psd_noise,
    bdelay: int,
    btaps: int,
):
    """Pick/derive the noise or observation PSD `psd_n` fed to the beamformer.

    Different beamformer families need different `psd_n`: RTF-based
    beamformers use the noise PSD computed by `_compute_rtf_style_psd_noise`;
    MPDR-family beamformers use the raw observation PSD; weighted (w*)
    variants use a power-weighted observation PSD; WPD variants use the
    stacked spatio-temporal covariance from `get_covariances`.

    Returns:
        (psd_n, psd_noise): `psd_noise` is returned back (and possibly
        collapsed from a list of per-speaker lists into per-speaker sums,
        for "mvdr_tfs") since callers report it as "psd_distortion".
    """
    if beamformer_type in (
        "mvdr",
        "mvdr_souden",
        "mvdr_tfs_souden",
        "sdw_mwf",
        "r1mwf",
        "lcmv",
        "gev",
        "gev_ban",
    ):
        psd_n = psd_noise
    elif beamformer_type == "mvdr_tfs":
        psd_n = psd_noise
        psd_noise = [sum(psd_noise_i) for psd_noise_i in psd_noise]
    elif beamformer_type in ("mpdr", "mpdr_souden", "lcmp", "mwf"):
        # observation PSD: (..., C1, T) x (..., C2, T) -> (..., C1, C2)
        psd_n = einsum("...ct,...et->...ce", signal, signal.conj())
    elif beamformer_type in ("wmpdr", "wmpdr_souden", "wlcmp", "wmwf"):
        psd_n = [
            einsum(
                "...ct,...et->...ce",
                signal * inv_p[..., None, :],
                signal.conj(),
            )
            for inv_p in inverse_powers
        ]
    elif beamformer_type in ("wpd", "wpd_souden"):
        psd_n = [
            get_covariances(signal, inv_p, bdelay, btaps, get_vector=False)
            for inv_p in inverse_powers
        ]
    return psd_n, psd_noise


def prepare_beamformer_stats(
    signal: ComplexOrRealTensor,
    masks_speech: Union[ComplexOrRealTensor, Sequence[ComplexOrRealTensor]],
    mask_noise: Optional[torch.Tensor],
    powers: Optional[List[torch.Tensor]] = None,
    beamformer_type: str = "mvdr",
    bdelay: int = 3,
    btaps: int = 5,
    eps: float = 1e-6,
) -> Dict[str, object]:
    """Prepare necessary statistics for constructing the specified beamformer.

    Args:
        signal (torch.complex64/ComplexTensor): (..., F, C, T)
        masks_speech (List[torch.Tensor]): (..., F, C, T) masks for all speech sources
        mask_noise (torch.Tensor): (..., F, C, T) noise mask
        powers (List[torch.Tensor]): powers for all speech sources (..., F, T)
                                     used for wMPDR or WPD beamformers
        beamformer_type (str): one of the pre-defined beamformer types
        bdelay (int): delay factor, used for WPD beamformser
        btaps (int): number of filter taps, used for WPD beamformser
        eps (torch.Tensor): tiny constant
    Returns:
        beamformer_stats (dict): a dictionary containing all necessary statistics
            e.g. "psd_n", "psd_speech", "psd_distortion"
            Note:
            * When `masks_speech` is a tensor or a single-element list, all returned
              statistics are tensors;
            * When `masks_speech` is a multi-element list, some returned statistics
              can be a list, e.g., "psd_n" for MVDR, "psd_speech" and "psd_distortion".

    """
    from espnet2.enh.layers.dnn_beamformer import BEAMFORMER_TYPES

    assert beamformer_type in BEAMFORMER_TYPES, "%s is not supported yet"

    masks_speech = _as_double_mask_list(masks_speech)
    num_spk = len(masks_speech)

    inverse_powers = None
    if _needs_power_weighting(beamformer_type):
        inverse_powers = _compute_inverse_powers(signal, masks_speech, powers, eps)

    psd_speeches = [get_power_spectral_density_matrix(signal, m) for m in masks_speech]
    psd_noise = _compute_rtf_style_psd_noise(
        signal, psd_speeches, mask_noise, beamformer_type, num_spk
    )
    psd_n, psd_noise = _select_psd_n(
        beamformer_type, signal, inverse_powers, psd_noise, bdelay, btaps
    )

    if num_spk == 1:
        psd_speeches = psd_speeches[0]
        if isinstance(psd_n, (list, tuple)):
            psd_n = psd_n[0]

    if beamformer_type in (
        "mvdr",
        "mpdr",
        "wmpdr",
        "wpd",
        "lcmp",
        "wlcmp",
        "lcmv",
        "mvdr_tfs",
    ):
        return {"psd_n": psd_n, "psd_speech": psd_speeches, "psd_distortion": psd_noise}
    elif (
        beamformer_type.endswith("_souden")
        or beamformer_type.startswith("gev")
        or beamformer_type == "mwf"
        or beamformer_type == "wmwf"
        or beamformer_type == "sdw_mwf"
        or beamformer_type == "r1mwf"
    ):
        return {"psd_n": psd_n, "psd_speech": psd_speeches}


def get_power_spectral_density_matrix(
    xs: ComplexOrRealTensor,
    mask: torch.Tensor,
    normalization: bool = True,
    reduction: str = "mean",
    eps: float = 1e-15,
) -> ComplexOrRealTensor:
    """Return cross-channel power spectral density (PSD) matrix

    Args:
        xs (torch.complex64/ComplexTensor): (..., F, C, T)
        reduction (str): "mean" or "median"
        mask (torch.Tensor): (..., F, C, T)
        normalization (bool):
        eps (float):
    Returns
        psd (torch.complex64/ComplexTensor): (..., F, C, C)

    """
    if reduction == "mean":
        # Averaging mask along C: (..., C, T) -> (..., 1, T)
        mask = mask.mean(dim=-2, keepdim=True)
    elif reduction == "median":
        mask = mask.median(dim=-2, keepdim=True)
    else:
        raise ValueError("Unknown reduction mode: %s" % reduction)

    # Normalized mask along T: (..., T)
    if normalization:
        # If assuming the tensor is padded with zero, the summation along
        # the time axis is same regardless of the padding length.
        mask = mask / (mask.sum(dim=-1, keepdim=True) + eps)

    # outer product: (..., C_1, T) x (..., C_2, T) -> (..., C, C_2)
    psd = einsum("...ct,...et->...ce", xs * mask, xs.conj())

    return psd


def get_rtf(
    psd_speech: ComplexOrRealTensor,
    psd_noise: ComplexOrRealTensor,
    mode: str = "power",
    reference_vector: Union[int, torch.Tensor] = 0,
    iterations: int = 3,
) -> ComplexOrRealTensor:
    """Calculate the relative transfer function (RTF)

    Algorithm of power method:
        1) rtf = reference_vector
        2) for i in range(iterations):
             rtf = (psd_noise^-1 @ psd_speech) @ rtf
             rtf = rtf / ||rtf||_2  # this normalization can be skipped
        3) rtf = psd_noise @ rtf
        4) rtf = rtf / rtf[..., ref_channel, :]
    Note: 4) Normalization at the reference channel is not performed here.

    Args:
        psd_speech (torch.complex64/ComplexTensor):
            speech covariance matrix (..., F, C, C)
        psd_noise (torch.complex64/ComplexTensor):
            noise covariance matrix (..., F, C, C)
        mode (str): one of ("power", "evd")
            "power": power method
            "evd": eigenvalue decomposition
        reference_vector (torch.Tensor or int): (..., C) or scalar
        iterations (int): number of iterations in power method
    Returns:
        rtf (torch.complex64/ComplexTensor): (..., F, C, 1)
    """
    if mode == "power":
        phi = solve(psd_speech, psd_noise)  # psd_noise^-1 @ psd_speech
        rtf = (
            phi[..., reference_vector, None]
            if isinstance(reference_vector, int)
            else matmul(phi, reference_vector[..., None, :, None])
        )
        for _ in range(iterations - 2):
            rtf = matmul(phi, rtf)
            # rtf = rtf / complex_norm(rtf, dim=-1, keepdim=True)
        rtf = matmul(psd_speech, rtf)
    elif mode == "evd":
        assert (
            is_torch_1_9_plus
            and is_torch_complex_tensor(psd_speech)
            and is_torch_complex_tensor(psd_noise)
        )
        e_vec = generalized_eigenvalue_decomposition(psd_speech, psd_noise)[1]
        rtf = matmul(psd_noise, e_vec[..., -1, None])
    else:
        raise ValueError("Unknown mode: %s" % mode)
    return rtf


def get_mvdr_vector(
    psd_s: ComplexOrRealTensor,
    psd_n: ComplexOrRealTensor,
    reference_vector: torch.Tensor,
    diagonal_loading: bool = True,
    diag_eps: float = 1e-7,
    eps: float = 1e-8,
) -> ComplexOrRealTensor:
    """Return the MVDR (Minimum Variance Distortionless Response) vector:

        h = (Npsd^-1 @ Spsd) / (Tr(Npsd^-1 @ Spsd)) @ u

    Reference:
        On optimal frequency-domain multichannel linear filtering
        for noise reduction; M. Souden et al., 2010;
        https://ieeexplore.ieee.org/document/5089420

    Args:
        psd_s (torch.complex64/ComplexTensor):
            speech covariance matrix (..., F, C, C)
        psd_n (torch.complex64/ComplexTensor):
            observation/noise covariance matrix (..., F, C, C)
        reference_vector (torch.Tensor): (..., C)
        diagonal_loading (bool): Whether to add a tiny term to the diagonal of psd_n
        diag_eps (float):
        eps (float):
    Returns:
        beamform_vector (torch.complex64/ComplexTensor): (..., F, C)
    """  # noqa: D400
    if diagonal_loading:
        psd_n = tik_reg(psd_n, reg=diag_eps, eps=eps)

    numerator = solve(psd_s, psd_n)  # psd_n^-1 @ psd_s
    # NOTE (wangyou): until PyTorch 1.9.0, torch.trace does not
    # support bacth processing. Use FC.trace() as fallback.
    # ws: (..., C, C) / (...,) -> (..., C, C)
    ws = numerator / (FC.trace(numerator)[..., None, None] + eps)
    # h: (..., F, C_1, C_2) x (..., C_2) -> (..., F, C_1)
    beamform_vector = einsum("...fec,...c->...fe", ws, reference_vector)
    return beamform_vector


def get_mvdr_vector_with_rtf(
    psd_n: ComplexOrRealTensor,
    psd_speech: ComplexOrRealTensor,
    psd_noise: ComplexOrRealTensor,
    iterations: int = 3,
    reference_vector: Union[int, torch.Tensor, None] = None,
    diagonal_loading: bool = True,
    diag_eps: float = 1e-7,
    eps: float = 1e-8,
) -> ComplexOrRealTensor:
    """Return the MVDR (Minimum Variance Distortionless Response) vector
        calculated with RTF:

        h = (Npsd^-1 @ rtf) / (rtf^H @ Npsd^-1 @ rtf)

    Reference:
        On optimal frequency-domain multichannel linear filtering
        for noise reduction; M. Souden et al., 2010;
        https://ieeexplore.ieee.org/document/5089420

    Args:
        psd_n (torch.complex64/ComplexTensor):
            observation/noise covariance matrix (..., F, C, C)
        psd_speech (torch.complex64/ComplexTensor):
            speech covariance matrix (..., F, C, C)
        psd_noise (torch.complex64/ComplexTensor):
            noise covariance matrix (..., F, C, C)
        iterations (int): number of iterations in power method
        reference_vector (torch.Tensor or int): (..., C) or scalar
        diagonal_loading (bool): Whether to add a tiny term to the diagonal of psd_n
        diag_eps (float):
        eps (float):
    Returns:
        beamform_vector (torch.complex64/ComplexTensor): (..., F, C)
    """  # noqa: H405, D205, D400
    if diagonal_loading:
        psd_noise = tik_reg(psd_noise, reg=diag_eps, eps=eps)

    rtf = get_rtf(  # (B, F, C, 1)
        psd_speech,
        psd_noise,
        mode="power",
        reference_vector=reference_vector,
        iterations=iterations,
    )

    # numerator: (..., C_1, C_2) x (..., C_2, 1) -> (..., C_1)
    numerator = solve(rtf, psd_n).squeeze(-1)
    denominator = einsum("...d,...d->...", rtf.squeeze(-1).conj(), numerator)
    if reference_vector is not None:
        if isinstance(reference_vector, int):
            scale = rtf.squeeze(-1)[..., reference_vector, None].conj()
        else:
            scale = (rtf.squeeze(-1).conj() * reference_vector[..., None, :]).sum(
                dim=-1, keepdim=True
            )
        beamforming_vector = numerator * scale / (denominator.real.unsqueeze(-1) + eps)
    else:
        beamforming_vector = numerator / (denominator.real.unsqueeze(-1) + eps)
    return beamforming_vector


def apply_beamforming_vector(
    beamform_vector: ComplexOrRealTensor,
    mix: ComplexOrRealTensor,
) -> ComplexOrRealTensor:
    """Apply a beamforming filter: ``w^H @ mix``.

    Args:
        beamform_vector: (..., C)
        mix: (..., C, T)

    Returns:
        enhanced: (..., T)
    """
    es = einsum("...c,...ct->...t", beamform_vector.conj(), mix)
    return es


def get_mwf_vector(
    psd_s: ComplexOrRealTensor,
    psd_n: ComplexOrRealTensor,
    reference_vector: Union[torch.Tensor, int],
    diagonal_loading: bool = True,
    diag_eps: float = 1e-7,
    eps: float = 1e-8,
) -> ComplexOrRealTensor:
    """Return the MWF (Minimum Multi-channel Wiener Filter) vector:

        h = (Npsd^-1 @ Spsd) @ u

    Args:
        psd_s (torch.complex64/ComplexTensor):
            speech covariance matrix (..., F, C, C)
        psd_n (torch.complex64/ComplexTensor):
            power-normalized observation covariance matrix (..., F, C, C)
        reference_vector (torch.Tensor or int): (..., C) or scalar
        diagonal_loading (bool): Whether to add a tiny term to the diagonal of psd_n
        diag_eps (float):
        eps (float):
    Returns:
        beamform_vector (torch.complex64/ComplexTensor): (..., F, C)
    """  # noqa: D400
    if diagonal_loading:
        psd_n = tik_reg(psd_n, reg=diag_eps, eps=eps)

    ws = solve(psd_s, psd_n)  # psd_n^-1 @ psd_s
    # h: (..., F, C_1, C_2) x (..., C_2) -> (..., F, C_1)
    if isinstance(reference_vector, int):
        beamform_vector = ws[..., reference_vector]
    else:
        beamform_vector = einsum("...fec,...c->...fe", ws, reference_vector)
    return beamform_vector


def get_sdw_mwf_vector(
    psd_speech: ComplexOrRealTensor,
    psd_noise: ComplexOrRealTensor,
    reference_vector: Union[torch.Tensor, int],
    denoising_weight: float = 1.0,
    approx_low_rank_psd_speech: bool = False,
    iterations: int = 3,
    diagonal_loading: bool = True,
    diag_eps: float = 1e-7,
    eps: float = 1e-8,
) -> ComplexOrRealTensor:
    """Return the SDW-MWF (Speech Distortion Weighted Multi-channel Wiener Filter) vector

        h = (Spsd + mu * Npsd)^-1 @ Spsd @ u

    Reference:
        [1] Spatially pre-processed speech distortion weighted multi-channel Wiener
        filtering for noise reduction; A. Spriet et al, 2004
        https://dl.acm.org/doi/abs/10.1016/j.sigpro.2004.07.028
        [2] Rank-1 constrained multichannel Wiener filter for speech recognition in
        noisy environments; Z. Wang et al, 2018
        https://hal.inria.fr/hal-01634449/document
        [3] Low-rank approximation based multichannel Wiener filter algorithms for
        noise reduction with application in cochlear implants; R. Serizel, 2014
        https://ieeexplore.ieee.org/document/6730918

    Args:
        psd_speech (torch.complex64/ComplexTensor):
            speech covariance matrix (..., F, C, C)
        psd_noise (torch.complex64/ComplexTensor):
            noise covariance matrix (..., F, C, C)
        reference_vector (torch.Tensor or int): (..., C) or scalar
        denoising_weight (float): a trade-off parameter between noise reduction and
            speech distortion.
            A larger value leads to more noise reduction at the expense of more speech
            distortion.
            The plain MWF is obtained with `denoising_weight = 1` (by default).
        approx_low_rank_psd_speech (bool): whether to replace original input psd_speech
            with its low-rank approximation as in [2]
        iterations (int): number of iterations in power method, only used when
            `approx_low_rank_psd_speech = True`
        diagonal_loading (bool): Whether to add a tiny term to the diagonal of psd_n
        diag_eps (float):
        eps (float):
    Returns:
        beamform_vector (torch.complex64/ComplexTensor): (..., F, C)
    """  # noqa: H405, D205, D400, E501
    if approx_low_rank_psd_speech:
        if diagonal_loading:
            psd_noise = tik_reg(psd_noise, reg=diag_eps, eps=eps)

        recon_vec = get_rtf(  # (B, F, C, 1)
            psd_speech,
            psd_noise,
            mode="power",
            iterations=iterations,
            reference_vector=reference_vector,
        )
        # Eq. (25) in Ref[2]
        psd_speech_r1 = matmul(recon_vec, recon_vec.conj().transpose(-1, -2))
        sigma_speech = FC.trace(psd_speech) / (FC.trace(psd_speech_r1) + eps)
        psd_speech_r1 = psd_speech_r1 * sigma_speech[..., None, None]
        # c.f. Eq. (62) in Ref[3]
        psd_speech = psd_speech_r1

    psd_n = psd_speech + denoising_weight * psd_noise
    if diagonal_loading:
        psd_n = tik_reg(psd_n, reg=diag_eps, eps=eps)

    ws = solve(psd_speech, psd_n)  # psd_n^-1 @ psd_speech

    if isinstance(reference_vector, int):
        beamform_vector = ws[..., reference_vector]
    else:
        beamform_vector = einsum("...fec,...c->...fe", ws, reference_vector)
    return beamform_vector


def get_rank1_mwf_vector(
    psd_speech: ComplexOrRealTensor,
    psd_noise: ComplexOrRealTensor,
    reference_vector: Union[torch.Tensor, int],
    denoising_weight: float = 1.0,
    approx_low_rank_psd_speech: bool = False,
    iterations: int = 3,
    diagonal_loading: bool = True,
    diag_eps: float = 1e-7,
    eps: float = 1e-8,
) -> ComplexOrRealTensor:
    """Return the R1-MWF (Rank-1 Multi-channel Wiener Filter) vector

        h = (Npsd^-1 @ Spsd) / (mu + Tr(Npsd^-1 @ Spsd)) @ u

    Reference:
        [1] Rank-1 constrained multichannel Wiener filter for speech recognition in
        noisy environments; Z. Wang et al, 2018
        https://hal.inria.fr/hal-01634449/document
        [2] Low-rank approximation based multichannel Wiener filter algorithms for
        noise reduction with application in cochlear implants; R. Serizel, 2014
        https://ieeexplore.ieee.org/document/6730918

    Args:
        psd_speech (torch.complex64/ComplexTensor):
            speech covariance matrix (..., F, C, C)
        psd_noise (torch.complex64/ComplexTensor):
            noise covariance matrix (..., F, C, C)
        reference_vector (torch.Tensor or int): (..., C) or scalar
        denoising_weight (float): a trade-off parameter between noise reduction and
            speech distortion.
            A larger value leads to more noise reduction at the expense of more speech
            distortion.
            When `denoising_weight = 0`, it corresponds to MVDR beamformer.
        approx_low_rank_psd_speech (bool): whether to replace original input psd_speech
            with its low-rank approximation as in [1]
        iterations (int): number of iterations in power method, only used when
            `approx_low_rank_psd_speech = True`
        diagonal_loading (bool): Whether to add a tiny term to the diagonal of psd_n
        diag_eps (float):
        eps (float):
    Returns:
        beamform_vector (torch.complex64/ComplexTensor): (..., F, C)
    """  # noqa: H405, D205, D400
    if approx_low_rank_psd_speech:
        if diagonal_loading:
            psd_noise = tik_reg(psd_noise, reg=diag_eps, eps=eps)

        recon_vec = get_rtf(  # (B, F, C, 1)
            psd_speech,
            psd_noise,
            mode="power",
            iterations=iterations,
            reference_vector=reference_vector,
        )
        # Eq. (25) in Ref[1]
        psd_speech_r1 = matmul(recon_vec, recon_vec.conj().transpose(-1, -2))
        sigma_speech = FC.trace(psd_speech) / (FC.trace(psd_speech_r1) + eps)
        psd_speech_r1 = psd_speech_r1 * sigma_speech[..., None, None]
        # c.f. Eq. (62) in Ref[2]
        psd_speech = psd_speech_r1
    elif diagonal_loading:
        psd_noise = tik_reg(psd_noise, reg=diag_eps, eps=eps)

    numerator = solve(psd_speech, psd_noise)  # psd_noise^-1 @ psd_speech

    # NOTE (wangyou): until PyTorch 1.9.0, torch.trace does not
    # support bacth processing. Use FC.trace() as fallback.
    # ws: (..., C, C) / (...,) -> (..., C, C)
    ws = numerator / (denoising_weight + FC.trace(numerator)[..., None, None] + eps)

    # h: (..., F, C_1, C_2) x (..., C_2) -> (..., F, C_1)
    if isinstance(reference_vector, int):
        beamform_vector = ws[..., reference_vector]
    else:
        beamform_vector = einsum("...fec,...c->...fe", ws, reference_vector)
    return beamform_vector


def get_rtf_matrix(
    psd_speeches: List[ComplexOrRealTensor],
    psd_noises: List[ComplexOrRealTensor],
    diagonal_loading: bool = True,
    ref_channel: int = 0,
    rtf_iterations: int = 3,
    diag_eps: float = 1e-7,
    eps: float = 1e-8,
) -> ComplexOrRealTensor:
    """Calculate the RTF matrix with each column the relative transfer function
    of the corresponding source.

    Args:
        psd_speeches: per-speaker speech covariance matrices, each (..., F, C, C).
        psd_noises: per-speaker noise covariance matrices, each (..., F, C, C).

    Returns:
        rtf_mat: (..., F, C, num_spk), reference-channel-normalized.
    """  # noqa: H405
    assert isinstance(psd_speeches, list) and isinstance(psd_noises, list)
    rtf_mat = cat(
        [
            get_rtf(
                psd_speeches[spk],
                tik_reg(psd_n, reg=diag_eps, eps=eps) if diagonal_loading else psd_n,
                mode="power",
                reference_vector=ref_channel,
                iterations=rtf_iterations,
            )
            for spk, psd_n in enumerate(psd_noises)
        ],
        dim=-1,
    )
    # normalize at the reference channel
    return rtf_mat / rtf_mat[..., ref_channel, None, :]


def get_lcmv_vector_with_rtf(
    psd_n: ComplexOrRealTensor,
    rtf_mat: ComplexOrRealTensor,
    reference_vector: Union[int, torch.Tensor, None] = None,
    diagonal_loading: bool = True,
    diag_eps: float = 1e-7,
    eps: float = 1e-8,
) -> ComplexOrRealTensor:
    """Return the LCMV (Linearly Constrained Minimum Variance) vector
        calculated with RTF:

        h = (Npsd^-1 @ rtf_mat) @ (rtf_mat^H @ Npsd^-1 @ rtf_mat)^-1 @ p

    Reference:
        H. L. Van Trees, “Optimum array processing: Part IV of detection, estimation,
        and modulation theory,” John Wiley & Sons, 2004. (Chapter 6.7)

    Args:
        psd_n (torch.complex64/ComplexTensor):
            observation/noise covariance matrix (..., F, C, C)
        rtf_mat (torch.complex64/ComplexTensor):
            RTF matrix (..., F, C, num_spk)
        reference_vector (torch.Tensor or int): (..., num_spk) or scalar
        diagonal_loading (bool): Whether to add a tiny term to the diagonal of psd_n
        diag_eps (float):
        eps (float):
    Returns:
        beamform_vector (torch.complex64/ComplexTensor): (..., F, C)
    """  # noqa: H405, D205, D400
    if diagonal_loading:
        psd_n = tik_reg(psd_n, reg=diag_eps, eps=eps)

    # numerator: (..., C_1, C_2) x (..., C_2, num_spk) -> (..., C_1, num_spk)
    numerator = solve(rtf_mat, psd_n)
    denominator = matmul(rtf_mat.conj().transpose(-1, -2), numerator)
    if isinstance(reference_vector, int):
        ws = inverse(denominator)[..., reference_vector, None]
    else:
        ws = solve(reference_vector, denominator)
    beamforming_vector = matmul(numerator, ws).squeeze(-1)
    return beamforming_vector


def generalized_eigenvalue_decomposition(
    a: torch.Tensor, b: torch.Tensor, eps: float = 1e-6
):
    """Solves the generalized eigenvalue decomposition through Cholesky decomposition.

    ported from https://github.com/asteroid-team/asteroid/blob/master/asteroid/dsp/beamforming.py#L464

    a @ e_vec = e_val * b @ e_vec
    |
    |   Cholesky decomposition on `b`:
    |       b = L @ L^H, where `L` is a lower triangular matrix
    |
    |   Let C = L^-1 @ a @ L^-H, it is Hermitian.
    |
    => C @ y = lambda * y
    => e_vec = L^-H @ y

    Reference: https://www.netlib.org/lapack/lug/node54.html

    Args:
        a: A complex Hermitian or real symmetric matrix whose eigenvalues and
            eigenvectors will be computed. (..., C, C)
        b: A complex Hermitian or real symmetric definite positive matrix. (..., C, C)
    Returns:
        e_val: generalized eigenvalues (ascending order)
        e_vec: generalized eigenvectors
    """  # noqa: H405, E501
    try:
        cholesky = torch.linalg.cholesky(b)
    except RuntimeError:
        b = tik_reg(b, reg=eps, eps=eps)
        cholesky = torch.linalg.cholesky(b)
    inv_cholesky = cholesky.inverse()
    # Compute C matrix L⁻1 a L^-H
    cmat = inv_cholesky @ a @ inv_cholesky.conj().transpose(-1, -2)
    # Performing the eigenvalue decomposition
    e_val, e_vec = torch.linalg.eigh(cmat)
    # Collecting the eigenvectors
    e_vec = torch.matmul(inv_cholesky.conj().transpose(-1, -2), e_vec)
    return e_val, e_vec


def gev_phase_correction(vector: ComplexOrRealTensor) -> ComplexOrRealTensor:
    """Phase correction to reduce distortions due to phase inconsistencies.

    ported from https://github.com/fgnt/nn-gev/blob/master/fgnt/beamforming.py#L169

    Args:
        vector: Beamforming vector with shape (B, F, C)
    Returns:
        w: Phase corrected beamforming vectors (B, F, C)
    """
    B, F, C = vector.shape
    correction = torch.empty_like(vector.real)
    for f in range(F):
        correction[:, f, :] = torch.exp(
            (vector[:, f, :] * vector[:, f - 1, :].conj())
            .sum(dim=-1, keepdim=True)
            .angle()
        )
    if isinstance(vector, ComplexTensor):
        correction = ComplexTensor(torch.cos(correction), -torch.sin(correction))
    else:
        correction = torch.exp(-1j * correction)
    return vector * correction


def blind_analytic_normalization(
    ws: ComplexOrRealTensor, psd_noise: ComplexOrRealTensor, eps: float = 1e-8
) -> torch.Tensor:
    """Blind analytic normalization (BAN) for post-filtering

    Args:
        ws (torch.complex64/ComplexTensor): beamformer vector (..., F, C)
        psd_noise (torch.complex64/ComplexTensor): noise PSD matrix (..., F, C, C)
        eps (float)
    Returns:
        ws_ban (torch.complex64/ComplexTensor): normalized beamformer vector (..., F)
    """
    C2 = psd_noise.size(-1) ** 2
    denominator = einsum("...c,...ce,...e->...", ws.conj(), psd_noise, ws)
    numerator = einsum(
        "...c,...ce,...eo,...o->...", ws.conj(), psd_noise, psd_noise, ws
    )
    gain = (numerator + eps).sqrt() / (denominator * C2 + eps)
    return gain


def get_gev_vector(
    psd_noise: ComplexOrRealTensor,
    psd_speech: ComplexOrRealTensor,
    mode: str = "power",
    reference_vector: Union[int, torch.Tensor] = 0,
    iterations: int = 3,
    diagonal_loading: bool = True,
    diag_eps: float = 1e-7,
    eps: float = 1e-8,
) -> ComplexOrRealTensor:
    """Return the generalized eigenvalue (GEV) beamformer vector:

        psd_speech @ h = lambda * psd_noise @ h

    Reference:
        Blind acoustic beamforming based on generalized eigenvalue decomposition;
        E. Warsitz and R. Haeb-Umbach, 2007.

    Args:
        psd_noise (torch.complex64/ComplexTensor):
            noise covariance matrix (..., F, C, C)
        psd_speech (torch.complex64/ComplexTensor):
            speech covariance matrix (..., F, C, C)
        mode (str): one of ("power", "evd")
            "power": power method
            "evd": eigenvalue decomposition (only for torch builtin complex tensors)
        reference_vector (torch.Tensor or int): (..., C) or scalar
        iterations (int): number of iterations in power method
        diagonal_loading (bool): Whether to add a tiny term to the diagonal of psd_n
        diag_eps (float):
        eps (float):
    Returns:
        beamform_vector (torch.complex64/ComplexTensor): (..., F, C)
    """  # noqa: H405, D205, D400
    if diagonal_loading:
        psd_noise = tik_reg(psd_noise, reg=diag_eps, eps=eps)

    if mode == "power":
        phi = solve(psd_speech, psd_noise)  # psd_noise^-1 @ psd_speech
        e_vec = (
            phi[..., reference_vector, None]
            if isinstance(reference_vector, int)
            else matmul(phi, reference_vector[..., None, :, None])
        )
        for _ in range(iterations - 1):
            e_vec = matmul(phi, e_vec)
            # e_vec = e_vec / complex_norm(e_vec, dim=-1, keepdim=True)
        e_vec = e_vec.squeeze(-1)
    elif mode == "evd":
        assert (
            is_torch_1_9_plus
            and is_torch_complex_tensor(psd_speech)
            and is_torch_complex_tensor(psd_noise)
        )
        # e_vec = generalized_eigenvalue_decomposition(psd_speech, psd_noise)[1][...,-1]
        e_vec = psd_noise.new_zeros(psd_noise.shape[:-1])
        for f in range(psd_noise.shape[-3]):
            try:
                e_vec[..., f, :] = generalized_eigenvalue_decomposition(
                    psd_speech[..., f, :, :], psd_noise[..., f, :, :]
                )[1][..., -1]
            except RuntimeError:
                # port from github.com/fgnt/nn-gev/blob/master/fgnt/beamforming.py#L106
                print(
                    "GEV beamformer: LinAlg error for frequency {}".format(f),
                    flush=True,
                )
                C = psd_noise.size(-1)
                e_vec[..., f, :] = (
                    psd_noise.new_ones(e_vec[..., f, :].shape)
                    / FC.trace(psd_noise[..., f, :, :])
                    * C
                )
    else:
        raise ValueError("Unknown mode: %s" % mode)

    beamforming_vector = e_vec / complex_norm(e_vec, dim=-1, keepdim=True)
    beamforming_vector = gev_phase_correction(beamforming_vector)
    return beamforming_vector


def signal_framing(
    signal: ComplexOrRealTensor,
    frame_length: int,
    frame_step: int,
    bdelay: int,
    do_padding: bool = False,
    pad_value: int = 0,
    indices: Optional[List] = None,
) -> ComplexOrRealTensor:
    """Expand `signal` into several frames, with each frame of length `frame_length`.

    Unlike `wpe.signal_framing`, each output frame here is made of the
    `frame_length - 1` immediately preceding samples plus one extra sample
    delayed by `bdelay` -- the layout used to build the WPD stacked
    spatio-temporal covariance in `get_covariances`.

    Args:
        signal : (..., T)
        frame_length:   length of each segment
        frame_step:     step for selecting frames
        bdelay:         delay for WPD
        do_padding:     whether or not to pad the input signal at the beginning
                          of the time dimension
        pad_value:      value to fill in the padding

    Returns:
        torch.Tensor:
            if do_padding: (..., T, frame_length)
            else:          (..., T - bdelay - frame_length + 2, frame_length)
    """
    if isinstance(signal, ComplexTensor):
        complex_wrapper = ComplexTensor
        pad_func = FC.pad
    elif is_torch_complex_tensor(signal):
        complex_wrapper = torch.complex
        pad_func = torch.nn.functional.pad
    else:
        pad_func = torch.nn.functional.pad

    frame_length2 = frame_length - 1
    # pad to the right at the last dimension of `signal` (time dimension)
    if do_padding:
        # (..., T) --> (..., T + bdelay + frame_length - 2)
        signal = pad_func(
            signal, (bdelay + frame_length2 - 1, 0), "constant", pad_value
        )
        do_padding = False

    if indices is None:
        # [[ 0, 1, ..., frame_length2 - 1,              frame_length2 - 1 + bdelay ],
        #  [ 1, 2, ..., frame_length2,                  frame_length2 + bdelay     ],
        #  [ 2, 3, ..., frame_length2 + 1,              frame_length2 + 1 + bdelay ],
        #  ...
        #  [ T-bdelay-frame_length2, ..., T-1-bdelay,   T-1 ]]
        indices = [
            [*range(i, i + frame_length2), i + frame_length2 + bdelay - 1]
            for i in range(0, signal.shape[-1] - frame_length2 - bdelay + 1, frame_step)
        ]

    if is_complex(signal):
        real = signal_framing(
            signal.real,
            frame_length,
            frame_step,
            bdelay,
            do_padding,
            pad_value,
            indices,
        )
        imag = signal_framing(
            signal.imag,
            frame_length,
            frame_step,
            bdelay,
            do_padding,
            pad_value,
            indices,
        )
        return complex_wrapper(real, imag)
    else:
        # (..., T - bdelay - frame_length + 2, frame_length)
        signal = signal[..., indices]
        return signal


def get_covariances(
    Y: ComplexOrRealTensor,
    inverse_power: torch.Tensor,
    bdelay: int,
    btaps: int,
    get_vector: bool = False,
) -> ComplexOrRealTensor:
    """Calculates the power normalized spatio-temporal covariance
        matrix of the framed signal.

    Args:
        Y : Complex STFT signal with shape (B, F, C, T)
        inverse_power : Weighting factor with shape (B, F, T)

    Returns:
        Correlation matrix: (B, F, (btaps+1) * C, (btaps+1) * C)
        Correlation vector: (B, F, btaps + 1, C, C)
    """  # noqa: H405, D205, D400, D401
    assert inverse_power.dim() == 3, inverse_power.dim()
    assert inverse_power.size(0) == Y.size(0), (inverse_power.size(0), Y.size(0))

    Bs, Fdim, C, T = Y.shape

    # (B, F, C, T - bdelay - btaps + 1, btaps + 1)
    Psi = signal_framing(Y, btaps + 1, 1, bdelay, do_padding=False)[
        ..., : T - bdelay - btaps + 1, :
    ]
    # Reverse along btaps-axis:
    # [tau, tau-bdelay, tau-bdelay-1, ..., tau-bdelay-frame_length+1]
    Psi = reverse(Psi, dim=-1)
    Psi_norm = Psi * inverse_power[..., None, bdelay + btaps - 1 :, None]

    # let T' = T - bdelay - btaps + 1
    # (B, F, C, T', btaps + 1) x (B, F, C, T', btaps + 1)
    #  -> (B, F, btaps + 1, C, btaps + 1, C)
    covariance_matrix = einsum("bfdtk,bfetl->bfkdle", Psi, Psi_norm.conj())

    # (B, F, btaps + 1, C, btaps + 1, C)
    #   -> (B, F, (btaps + 1) * C, (btaps + 1) * C)
    covariance_matrix = covariance_matrix.view(
        Bs, Fdim, (btaps + 1) * C, (btaps + 1) * C
    )

    if get_vector:
        # (B, F, C, T', btaps + 1) x (B, F, C, T')
        #    --> (B, F, btaps +1, C, C)
        covariance_vector = einsum(
            "bfdtk,bfet->bfked", Psi_norm, Y[..., bdelay + btaps - 1 :].conj()
        )
        return covariance_matrix, covariance_vector
    else:
        return covariance_matrix


def get_WPD_filter(
    Phi: ComplexOrRealTensor,
    Rf: ComplexOrRealTensor,
    reference_vector: torch.Tensor,
    diagonal_loading: bool = True,
    diag_eps: float = 1e-7,
    eps: float = 1e-8,
) -> ComplexOrRealTensor:
    """Return the WPD vector.

        WPD is the Weighted Power minimization Distortionless response
        convolutional beamformer. As follows:

        h = (Rf^-1 @ Phi_{xx}) / tr[(Rf^-1) @ Phi_{xx}] @ u

    Reference:
        T. Nakatani and K. Kinoshita, "A Unified Convolutional Beamformer
        for Simultaneous Denoising and Dereverberation," in IEEE Signal
        Processing Letters, vol. 26, no. 6, pp. 903-907, June 2019, doi:
        10.1109/LSP.2019.2911179.
        https://ieeexplore.ieee.org/document/8691481

    Args:
        Phi (torch.complex64/ComplexTensor): (B, F, (btaps+1) * C, (btaps+1) * C)
            is the PSD of zero-padded speech [x^T(t,f) 0 ... 0]^T.
        Rf (torch.complex64/ComplexTensor): (B, F, (btaps+1) * C, (btaps+1) * C)
            is the power normalized spatio-temporal covariance matrix.
        reference_vector (torch.Tensor): (B, (btaps+1) * C)
            is the reference_vector.
        diagonal_loading (bool): Whether to add a tiny term to the diagonal of psd_n
        diag_eps (float):
        eps (float):

    Returns:
        filter_matrix (torch.complex64/ComplexTensor): (B, F, (btaps + 1) * C)
    """
    if diagonal_loading:
        Rf = tik_reg(Rf, reg=diag_eps, eps=eps)

    # numerator: (..., C_1, C_2) x (..., C_2, C_3) -> (..., C_1, C_3)
    numerator = solve(Phi, Rf)  # Rf^-1 @ Phi
    # NOTE (wangyou): until PyTorch 1.9.0, torch.trace does not
    # support bacth processing. Use FC.trace() as fallback.
    # ws: (..., C, C) / (...,) -> (..., C, C)
    ws = numerator / (FC.trace(numerator)[..., None, None] + eps)
    # h: (..., F, C_1, C_2) x (..., C_2) -> (..., F, C_1)
    beamform_vector = einsum("...fec,...c->...fe", ws, reference_vector)
    # (B, F, (btaps + 1) * C)
    return beamform_vector


def get_WPD_filter_v2(
    Phi: ComplexOrRealTensor,
    Rf: ComplexOrRealTensor,
    reference_vector: torch.Tensor,
    diagonal_loading: bool = True,
    diag_eps: float = 1e-7,
    eps: float = 1e-8,
) -> ComplexOrRealTensor:
    """Return the WPD vector (v2).

       This implementation is more efficient than `get_WPD_filter` as
        it skips unnecessary computation with zeros.

    Args:
        Phi (torch.complex64/ComplexTensor): (B, F, C, C)
            is speech PSD.
        Rf (torch.complex64/ComplexTensor): (B, F, (btaps+1) * C, (btaps+1) * C)
            is the power normalized spatio-temporal covariance matrix.
        reference_vector (torch.Tensor): (B, C)
            is the reference_vector.
        diagonal_loading (bool):
            Whether to add a tiny term to the diagonal of psd_n
        diag_eps (float):
        eps (float):

    Returns:
        filter_matrix (torch.complex64/ComplexTensor): (B, F, (btaps+1) * C)
    """
    C = reference_vector.shape[-1]
    if diagonal_loading:
        Rf = tik_reg(Rf, reg=diag_eps, eps=eps)
    inv_Rf = inverse(Rf)
    # (B, F, (btaps+1) * C, C)
    inv_Rf_pruned = inv_Rf[..., :C]
    # numerator: (..., C_1, C_2) x (..., C_2, C_3) -> (..., C_1, C_3)
    numerator = matmul(inv_Rf_pruned, Phi)
    # NOTE (wangyou): until PyTorch 1.9.0, torch.trace does not
    # support bacth processing. Use FC.trace() as fallback.
    # ws: (..., (btaps+1) * C, C) / (...,) -> (..., (btaps+1) * C, C)
    ws = numerator / (FC.trace(numerator[..., :C, :])[..., None, None] + eps)
    # h: (..., F, C_1, C_2) x (..., C_2) -> (..., F, C_1)
    beamform_vector = einsum("...fec,...c->...fe", ws, reference_vector)
    # (B, F, (btaps+1) * C)
    return beamform_vector


def get_WPD_filter_with_rtf(
    psd_observed_bar: ComplexOrRealTensor,
    psd_speech: ComplexOrRealTensor,
    psd_noise: ComplexOrRealTensor,
    iterations: int = 3,
    reference_vector: Union[int, torch.Tensor, None] = None,
    diagonal_loading: bool = True,
    diag_eps: float = 1e-7,
    eps: float = 1e-15,
) -> ComplexOrRealTensor:
    """Return the WPD vector calculated with RTF.

        WPD is the Weighted Power minimization Distortionless response
        convolutional beamformer. As follows:

        h = (Rf^-1 @ vbar) / (vbar^H @ R^-1 @ vbar)

    Reference:
        T. Nakatani and K. Kinoshita, "A Unified Convolutional Beamformer
        for Simultaneous Denoising and Dereverberation," in IEEE Signal
        Processing Letters, vol. 26, no. 6, pp. 903-907, June 2019, doi:
        10.1109/LSP.2019.2911179.
        https://ieeexplore.ieee.org/document/8691481

    Args:
        psd_observed_bar (torch.complex64/ComplexTensor):
            stacked observation covariance matrix
        psd_speech (torch.complex64/ComplexTensor):
            speech covariance matrix (..., F, C, C)
        psd_noise (torch.complex64/ComplexTensor):
            noise covariance matrix (..., F, C, C)
        iterations (int): number of iterations in power method
        reference_vector (torch.Tensor or int): (..., C) or scalar
        diagonal_loading (bool):
            Whether to add a tiny term to the diagonal of psd_n
        diag_eps (float):
        eps (float):
    Returns:
        beamform_vector (torch.complex64/ComplexTensor)r: (..., F, C)
    """
    if isinstance(psd_speech, ComplexTensor):
        pad_func = FC.pad
    elif is_torch_complex_tensor(psd_speech):
        pad_func = torch.nn.functional.pad
    else:
        raise ValueError(
            "Please update your PyTorch version to 1.9+ for complex support."
        )

    C = psd_noise.size(-1)
    if diagonal_loading:
        psd_noise = tik_reg(psd_noise, reg=diag_eps, eps=eps)

    rtf = get_rtf(  # (B, F, C, 1)
        psd_speech,
        psd_noise,
        mode="power",
        reference_vector=reference_vector,
        iterations=iterations,
    )

    # (B, F, (K+1)*C, 1)
    rtf = pad_func(rtf, (0, 0, 0, psd_observed_bar.shape[-1] - C), "constant", 0)
    # numerator: (..., C_1, C_2) x (..., C_2, 1) -> (..., C_1)
    numerator = solve(rtf, psd_observed_bar).squeeze(-1)
    denominator = einsum("...d,...d->...", rtf.squeeze(-1).conj(), numerator)
    if reference_vector is not None:
        if isinstance(reference_vector, int):
            scale = rtf.squeeze(-1)[..., reference_vector, None].conj()
        else:
            scale = (
                rtf.squeeze(-1)[:, :, :C].conj() * reference_vector[..., None, :]
            ).sum(dim=-1, keepdim=True)
        beamforming_vector = numerator * scale / (denominator.real.unsqueeze(-1) + eps)
    else:
        beamforming_vector = numerator / (denominator.real.unsqueeze(-1) + eps)
    return beamforming_vector


def perform_WPD_filtering(
    filter_matrix: ComplexOrRealTensor,
    Y: ComplexOrRealTensor,
    bdelay: int,
    btaps: int,
) -> ComplexOrRealTensor:
    """Perform WPD filtering.

    Args:
        filter_matrix: Filter matrix (B, F, (btaps + 1) * C)
        Y : Complex STFT signal with shape (B, F, C, T)

    Returns:
        enhanced (torch.complex64/ComplexTensor): (B, F, T)
    """
    # (B, F, C, T) --> (B, F, C, T, btaps + 1)
    Ytilde = signal_framing(Y, btaps + 1, 1, bdelay, do_padding=True, pad_value=0)
    Ytilde = reverse(Ytilde, dim=-1)

    Bs, Fdim, C, T = Y.shape
    # --> (B, F, T, btaps + 1, C) --> (B, F, T, (btaps + 1) * C)
    Ytilde = Ytilde.permute(0, 1, 3, 4, 2).contiguous().view(Bs, Fdim, T, -1)
    # (B, F, T, 1)
    enhanced = einsum("...tc,...c->...t", Ytilde, filter_matrix.conj())
    return enhanced


def tik_reg(
    mat: ComplexOrRealTensor, reg: float = 1e-8, eps: float = 1e-8
) -> ComplexOrRealTensor:
    """Perform Tikhonov regularization (only modifying real part).

    Diagonally loads `mat` by `reg` times its trace (plus a floor `eps`, to
    stay stable even when `mat` is all-zero), improving conditioning before
    matrix inversion/solve.

    Args:
        mat (torch.complex64/ComplexTensor): input matrix (..., C, C)
        reg (float): regularization factor
        eps (float)
    Returns:
        ret (torch.complex64/ComplexTensor): regularized matrix (..., C, C)
    """
    C = mat.size(-1)
    eye = torch.eye(C, dtype=mat.dtype, device=mat.device)
    shape = [1 for _ in range(mat.dim() - 2)] + [C, C]
    eye = eye.view(*shape).repeat(*mat.shape[:-2], 1, 1)
    with torch.no_grad():
        epsilon = FC.trace(mat).real[..., None, None] * reg
        # in case that correlation_matrix is all-zero
        epsilon = epsilon + eps
    mat = mat + epsilon * eye
    return mat
