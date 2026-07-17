"""Loudness normalization utilities built on ITU-R BS.1770 LUFS metering.

These helpers rescale audio pairs (e.g. a target/mixture pair used for
speech separation training) to match or randomly perturb their integrated
loudness (LUFS), as measured by an external meter object such as
`pyloudnorm.Meter`. Also includes dB<->linear conversions and magnitude
spectrogram normalization used elsewhere in the pipeline.
"""

import random
from typing import Tuple

import numpy as np
import torch


def linear2db(x: np.ndarray, eps: float = 1e-5, scale: float = 20) -> np.ndarray:
    """Convert a linear-amplitude value to decibels: ``scale * log10(x + eps)``."""
    return scale * np.log10(x + eps)


def db2linear(x: np.ndarray, eps: float = 1e-5, scale: float = 20) -> np.ndarray:
    """Convert a decibel value back to linear amplitude: ``10**(x/scale) - eps``."""
    return 10 ** (x / scale) - eps


def normalize_mag_spec(S: torch.Tensor, min_level_db: float = -100.0) -> torch.Tensor:
    """Rescale a dB-scale magnitude spectrogram ``S`` into [0, 1].

    Args:
        S: Magnitude spectrogram in dB, any shape.
        min_level_db: dB value mapped to 0.0.

    Returns:
        Normalized spectrogram, same shape as ``S``, clamped to [0, 1].
    """
    return torch.clamp((S - min_level_db) / -min_level_db, min=0.0, max=1.0)


def denormalize_mag_spec(S: torch.Tensor, min_level_db: float = -100.0) -> torch.Tensor:
    """Invert `normalize_mag_spec`, mapping a [0, 1]-normalized spectrogram back to dB.

    Args:
        S: Normalized spectrogram in [0, 1] (values outside are clamped).
        min_level_db: dB value that 0.0 maps back to.

    Returns:
        Spectrogram in dB, same shape as ``S``.
    """
    return torch.clamp(S, min=0.0, max=1.0) * -min_level_db + min_level_db


def loudness_match_and_norm(
    audio1: np.ndarray, audio2: np.ndarray, meter
) -> Tuple[np.ndarray, np.ndarray]:
    """Scale ``audio2`` so its integrated loudness matches ``audio1``.

    Args:
        audio1: Reference waveform.
        audio2: Waveform to be rescaled to match ``audio1``'s loudness.
        meter: A loudness meter exposing ``integrated_loudness(audio) -> LUFS``.

    Returns:
        ``(audio1, audio2)``, with ``audio2`` rescaled unless either
        input is silent (infinite negative LUFS), in which case both are
        returned unchanged.
    """
    lufs_1 = meter.integrated_loudness(audio1)
    lufs_2 = meter.integrated_loudness(audio2)

    if np.isinf(lufs_1) or np.isinf(lufs_2):
        return audio1, audio2
    else:
        audio2 = audio2 * db2linear(lufs_1 - lufs_2)

        return audio1, audio2


def loudness_normal_match_and_norm(
    audio1: np.ndarray, audio2: np.ndarray, meter
) -> Tuple[np.ndarray, np.ndarray]:
    """Scale ``audio2`` to a loudness drawn from N(lufs_1, 6.0) around ``audio1``.

    Args:
        audio1: Reference waveform (its LUFS is the target distribution's mean).
        audio2: Waveform to be rescaled.
        meter: A loudness meter exposing ``integrated_loudness(audio) -> LUFS``.

    Returns:
        ``(audio1, audio2)``, with ``audio2`` rescaled unless either
        input is silent, in which case both are returned unchanged.
    """
    lufs_1 = meter.integrated_loudness(audio1)
    lufs_2 = meter.integrated_loudness(audio2)

    if np.isinf(lufs_1) or np.isinf(lufs_2):
        return audio1, audio2
    else:
        target_lufs = random.normalvariate(lufs_1, 6.0)
        audio2 = audio2 * db2linear(target_lufs - lufs_2)

        return audio1, audio2


def loudness_normal_match_and_norm_output_louder_first(
    audio1: np.ndarray, audio2: np.ndarray, meter
) -> Tuple[np.ndarray, np.ndarray]:
    """Scale ``audio2`` so ``audio1`` tends to be louder, by a random margin.

    Draws the target LUFS for ``audio2`` from N(lufs_1 - 6.0, 2.0), i.e.
    ``audio1`` is expected to end up roughly 6 dB louder than ``audio2``.

    Args:
        audio1: Reference waveform, intended to be the louder of the pair.
        audio2: Waveform to be rescaled to be quieter than ``audio1``.
        meter: A loudness meter exposing ``integrated_loudness(audio) -> LUFS``.

    Returns:
        ``(audio1, audio2)``, with ``audio2`` rescaled unless either
        input is silent, in which case both are returned unchanged.
    """
    lufs_1 = meter.integrated_loudness(audio1)
    lufs_2 = meter.integrated_loudness(audio2)

    if np.isinf(lufs_1) or np.isinf(lufs_2):
        return audio1, audio2
    else:
        target_lufs = random.normalvariate(
            lufs_1 - 6.0, 2.0
        )  # we want audio1 to be louder than audio2 about target_lufs_diff
        audio2 = audio2 * db2linear(target_lufs - lufs_2)

        return audio1, audio2


def loudnorm(
    audio: np.ndarray, target_lufs: float, meter, eps: float = 1e-5
) -> Tuple[np.ndarray, float]:
    """Rescale ``audio`` to a target integrated loudness.

    Args:
        audio: Waveform to normalize.
        target_lufs: Desired integrated loudness, in LUFS.
        meter: A loudness meter exposing ``integrated_loudness(audio) -> LUFS``.
        eps: Passed through to `db2linear` for the gain conversion.

    Returns:
        A tuple ``(audio, adjusted_gain_db)``. If ``audio`` is silent
        (infinite negative LUFS), returns ``(audio, 0.0)`` unchanged.
    """
    lufs = meter.integrated_loudness(audio)
    if np.isinf(lufs):
        return audio, 0.0
    else:
        adjusted_gain = target_lufs - lufs
        audio = audio * db2linear(adjusted_gain, eps)

        return audio, adjusted_gain
