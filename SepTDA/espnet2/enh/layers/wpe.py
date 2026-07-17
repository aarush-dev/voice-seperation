"""Weighted Prediction Error (WPE) dereverberation.

Ported from https://github.com/fgnt/nara_wpe (many functions aren't
extensively tested in this port).

WPE removes late reverberation from a multi-channel STFT signal by modeling
each channel as an autoregressive process driven by a "dry" (dereverberated)
signal: the reverberant tail at time `t` is predicted as a linear combination
of `taps` earlier frames (with a `delay`-frame guard interval so the direct
sound and early reflections are not touched), and that predicted tail is
subtracted off. The prediction filter is the closed-form linear MMSE
(least-squares) solution under a time-varying power weighting, and is
refined over a few outer `iterations` since the weighting itself depends on
the (a priori unknown) dereverberated signal power.

Shapes below use `F` for frequency bins, `C` for microphone channels, `T` for
time frames, and `taps`/`delay` for the WPE filter order and prediction
delay (in STFT frames).
"""
from typing import Tuple, Union

import torch
import torch.nn.functional as F
import torch_complex.functional as FC
from packaging.version import parse as V
from torch_complex.tensor import ComplexTensor

from espnet2.enh.layers.complex_utils import einsum, matmul, reverse

is_torch_1_9_plus = V(torch.__version__) >= V("1.9.0")


def signal_framing(
    signal: Union[torch.Tensor, ComplexTensor],
    frame_length: int,
    frame_step: int,
    pad_value=0,
) -> Union[torch.Tensor, ComplexTensor]:
    """Slice `signal` into overlapping frames along its last axis.

    Args:
        signal: (..., T)
        frame_length: number of samples per frame (`W` below).
        frame_step: hop size between consecutive frames, in samples.
        pad_value: value used to zero-pad the tail of `signal`.

    Returns:
        (..., T', W), where `T'` is the number of frames produced.
    """
    if isinstance(signal, ComplexTensor):
        real = signal_framing(signal.real, frame_length, frame_step, pad_value)
        imag = signal_framing(signal.imag, frame_length, frame_step, pad_value)
        return ComplexTensor(real, imag)
    elif is_torch_1_9_plus and torch.is_complex(signal):
        real = signal_framing(signal.real, frame_length, frame_step, pad_value)
        imag = signal_framing(signal.imag, frame_length, frame_step, pad_value)
        return torch.complex(real, imag)

    signal = F.pad(signal, (0, frame_length - 1), "constant", pad_value)
    indices = sum(
        [
            list(range(i, i + frame_length))
            for i in range(0, signal.size(-1) - frame_length + 1, frame_step)
        ],
        [],
    )

    signal = signal[..., indices].view(*signal.size()[:-1], -1, frame_length)
    return signal


def get_power(signal: Union[torch.Tensor, ComplexTensor], dim=-2) -> torch.Tensor:
    """Compute per-time-frame signal power, averaged over channels.

    Args:
        signal: single-frequency complex STFT signal, (F, C, T).
        dim: the channel axis to average over.

    Returns:
        power: (F, T)
    """
    power = signal.real**2 + signal.imag**2
    power = power.mean(dim=dim)
    return power


def get_correlations(
    Y: Union[torch.Tensor, ComplexTensor],
    inverse_power: torch.Tensor,
    taps: int,
    delay: int,
) -> Tuple[Union[torch.Tensor, ComplexTensor], Union[torch.Tensor, ComplexTensor]]:
    """Compute power-weighted spatio-temporal correlations for the WPE filter.

    Builds the normal-equation statistics ``R`` (correlation matrix) and
    ``r`` (correlation vector) whose solution ``R^{-1} r`` gives the WPE
    prediction filter (see `get_filter_matrix_conj`).

    Args:
        Y: complex-valued STFT signal, (F, C, T).
        inverse_power: per-time-frame weighting factor, (F, T). Typically the
            reciprocal of the estimated dereverberated signal power, so that
            low-power (noise-dominated) frames contribute less to the fit.
        taps: number of prediction filter taps.
        delay: prediction guard interval, in frames.

    Returns:
        correlation_matrix: (F, taps*C, taps*C), i.e. `R` reshaped so the
            (tap, channel) axes are flattened together.
        correlation_vector: (F, taps, C, C), i.e. `r`.
    """
    assert inverse_power.dim() == 2, inverse_power.dim()
    assert inverse_power.size(0) == Y.size(0), (inverse_power.size(0), Y.size(0))

    F, C, T = Y.size()

    # Y: (F, C, T) -> Psi: (F, C, T, taps)
    Psi = signal_framing(Y, frame_length=taps, frame_step=1)[
        ..., : T - delay - taps + 1, :
    ]
    # Reverse along taps-axis so index 0 is the most delayed tap.
    Psi = reverse(Psi, dim=-1)
    Psi_conj_norm = Psi.conj() * inverse_power[..., None, delay + taps - 1 :, None]

    # (F, C, T, taps) x (F, C, T, taps) -> (F, taps, C, taps, C)
    correlation_matrix = einsum("fdtk,fetl->fkdle", Psi_conj_norm, Psi)
    # (F, taps, C, taps, C) -> (F, taps * C, taps * C)
    correlation_matrix = correlation_matrix.reshape(F, taps * C, taps * C)

    # (F, C, T, taps) x (F, C, T) -> (F, taps, C, C)
    correlation_vector = einsum(
        "fdtk,fet->fked", Psi_conj_norm, Y[..., delay + taps - 1 :]
    )

    return correlation_matrix, correlation_vector


def get_filter_matrix_conj(
    correlation_matrix: Union[torch.Tensor, ComplexTensor],
    correlation_vector: Union[torch.Tensor, ComplexTensor],
    eps: float = 1e-10,
) -> Union[torch.Tensor, ComplexTensor]:
    """Solve the WPE normal equations for the (conjugated) prediction filter.

    Computes ``filter = R^{-1} r`` (per frequency bin), where `R` is
    diagonally loaded by `eps` for numerical stability before inversion.

    Args:
        correlation_matrix: `R`, (F, taps*C, taps*C).
        correlation_vector: `r`, (F, taps, C, C).
        eps: diagonal loading added to `correlation_matrix` before inversion.

    Returns:
        filter_matrix_conj: (F, taps, C, C).
    """
    F, taps, C, _ = correlation_vector.size()

    # (F, taps, C1, C2) -> (F, C1, taps, C2) -> (F, C1, taps * C2)
    correlation_vector = (
        correlation_vector.permute(0, 2, 1, 3).contiguous().view(F, C, taps * C)
    )

    eye = torch.eye(
        correlation_matrix.size(-1),
        dtype=correlation_matrix.dtype,
        device=correlation_matrix.device,
    )
    shape = (
        tuple(1 for _ in range(correlation_matrix.dim() - 2))
        + correlation_matrix.shape[-2:]
    )
    eye = eye.view(*shape)
    correlation_matrix += eps * eye

    inv_correlation_matrix = correlation_matrix.inverse()
    # (F, C, taps, C) x (F, taps * C, taps * C) -> (F, C, taps * C)
    stacked_filter_conj = matmul(
        correlation_vector, inv_correlation_matrix.transpose(-1, -2)
    )

    # (F, C1, taps * C2) -> (F, C1, taps, C2) -> (F, taps, C2, C1)
    filter_matrix_conj = stacked_filter_conj.view(F, C, taps, C).permute(0, 2, 3, 1)
    return filter_matrix_conj


def perform_filter_operation(
    Y: Union[torch.Tensor, ComplexTensor],
    filter_matrix_conj: Union[torch.Tensor, ComplexTensor],
    taps: int,
    delay: int,
) -> Union[torch.Tensor, ComplexTensor]:
    """Predict the reverberant tail with the WPE filter and subtract it off.

    Args:
        Y: complex-valued STFT signal, (F, C, T).
        filter_matrix_conj: WPE prediction filter, (F, taps, C, C).
        taps: number of prediction filter taps.
        delay: prediction guard interval, in frames.

    Returns:
        dereverberated signal: (F, C, T).
    """
    if isinstance(Y, ComplexTensor):
        complex_module = FC
        pad_func = FC.pad
    elif is_torch_1_9_plus and torch.is_complex(Y):
        complex_module = torch
        pad_func = F.pad
    else:
        raise ValueError(
            "Please update your PyTorch version to 1.9+ for complex support."
        )

    T = Y.size(-1)
    # Y_tilde: (taps, F, C, T), the `taps` delayed-and-shifted copies of Y
    # used as regressors for the linear predictor.
    Y_tilde = complex_module.stack(
        [
            pad_func(Y[:, :, : T - delay - i], (delay + i, 0), mode="constant", value=0)
            for i in range(taps)
        ],
        dim=0,
    )
    reverb_tail = complex_module.einsum("fpde,pfdt->fet", (filter_matrix_conj, Y_tilde))
    return Y - reverb_tail


def wpe_one_iteration(
    Y: Union[torch.Tensor, ComplexTensor],
    power: torch.Tensor,
    taps: int = 10,
    delay: int = 3,
    eps: float = 1e-10,
    inverse_power: bool = True,
) -> Union[torch.Tensor, ComplexTensor]:
    """Run a single WPE update: estimate the prediction filter and apply it.

    Args:
        Y: complex-valued STFT signal, (..., C, T).
        power: per-time-frame power estimate used to weight the correlations,
            (..., T). Combined with any leading batch/frequency dims of `Y`.
        taps: number of prediction filter taps.
        delay: prediction guard interval, in frames.
        eps: floor applied to `power` (or diagonal loading, see
            `get_filter_matrix_conj`) for numerical stability.
        inverse_power: if True, `power` is treated as signal power and
            inverted (so quiet frames get high weight); if False, `power` is
            used directly as the weighting factor.

    Returns:
        enhanced: (..., C, T)
    """
    assert Y.size()[:-2] == power.size()[:-1]
    batch_freq_size = Y.size()[:-2]
    Y = Y.view(-1, *Y.size()[-2:])
    power = power.view(-1, power.size()[-1])

    if inverse_power:
        inverse_power = 1 / torch.clamp(power, min=eps)
    else:
        inverse_power = power

    correlation_matrix, correlation_vector = get_correlations(
        Y, inverse_power, taps, delay
    )
    filter_matrix_conj = get_filter_matrix_conj(correlation_matrix, correlation_vector)
    enhanced = perform_filter_operation(Y, filter_matrix_conj, taps, delay)

    enhanced = enhanced.view(*batch_freq_size, *Y.size()[-2:])
    return enhanced


def wpe(
    Y: Union[torch.Tensor, ComplexTensor], taps=10, delay=3, iterations=3
) -> Union[torch.Tensor, ComplexTensor]:
    """Dereverberate a signal with iterative WPE.

    Alternates between (1) estimating the dereverberated-signal power from
    the current `enhanced` estimate and (2) re-solving and applying the WPE
    prediction filter against the original `Y`, which converges to a
    maximum-likelihood estimate of the dereverberated signal.

    Args:
        Y: complex-valued STFT signal, (F, C, T).
        taps: number of prediction filter taps.
        delay: prediction guard interval, in frames.
        iterations: number of outer power/filter re-estimation steps.

    Returns:
        enhanced: (F, C, T)
    """
    enhanced = Y
    for _ in range(iterations):
        power = get_power(enhanced)
        enhanced = wpe_one_iteration(Y, power, taps=taps, delay=delay)
    return enhanced
