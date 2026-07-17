# Copyright (c) 2023 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Waveform-domain data augmentation utilities for speech separation training.

Provides randomized formant/pitch shifting (via Praat through parselmouth),
time stretching (via audiomentations), and parametric-EQ perturbation (via
pedalboard), plus a top-level dispatcher (`wav_manipulation`) that applies a
requested combination of these augmentations to a waveform.

The exact sequence and distribution of `random`/`np.random` calls in these
functions is part of their reproducibility contract under a fixed seed and
must not be reordered.
"""

import math
import random
from typing import Tuple

import numpy as np
import parselmouth
import torch

from audiomentations import TimeStretch

from pedalboard import (
    Pedalboard,
    HighShelfFilter,
    LowShelfFilter,
    PeakFilter,
    PitchShift,
)

PRAAT_CHANGEGENDER_PITCHMEDIAN_DEFAULT = 0.0
PRAAT_CHANGEGENDER_FORMANTSHIFTRATIO_DEFAULT = 1.0
PRAAT_CHANGEGENDER_PITCHSHIFTRATIO_DEFAULT = 1.0
PRAAT_CHANGEGENDER_PITCHRANGERATIO_DEFAULT = 1.0
PRAAT_CHANGEGENDER_DURATIONFACTOR_DEFAULT = 1.0


def change_pitch(sound: parselmouth.Sound, factor: float) -> parselmouth.Sound:
    """Shift the pitch contour of a sound by a fixed semitone offset.

    Extracts the pitch tier via Praat's Manipulation object, shifts every
    frequency in it by ``factor`` semitones, reattaches the shifted tier,
    and resynthesizes the audio with overlap-add.

    Args:
        sound: Source sound.
        factor: Pitch shift in semitones (positive raises pitch).

    Returns:
        The resynthesized sound with shifted pitch.
    """
    manipulation = parselmouth.praat.call(sound, "To Manipulation", 0.01, 75, 600)

    pitch_tier = parselmouth.praat.call(manipulation, "Extract pitch tier")

    # Arguments : Time range (s), Time range (s), Frequency shift, Unit
    parselmouth.praat.call(pitch_tier, "Shift frequencies", 0, 1000, factor, "semitones")

    parselmouth.praat.call([pitch_tier, manipulation], "Replace pitch tier")
    return parselmouth.praat.call(manipulation, "Get resynthesis (overlap-add)")


def wav_to_Sound(wav, sr: int) -> parselmouth.Sound:
    """Convert a waveform to a parselmouth.Sound object

    Args:
        wav (np.ndarray/torch.Tensor): waveform of shape (n_channels, n_samples)
        sr (int, optional): sampling rate.

    Returns:
        parselmouth.Sound: a parselmouth.Sound object
    """
    assert wav.shape == (1, len(wav[0])), "wav must be of shape (1, n_samples)"
    sound = None
    if isinstance(wav, np.ndarray):
        sound = parselmouth.Sound(wav[0], sampling_frequency=sr)
    elif isinstance(wav, torch.Tensor):
        sound = parselmouth.Sound(wav[0].numpy(), sampling_frequency=sr)
    assert sound is not None, "wav must be either np.ndarray or torch.Tensor"
    return sound


def get_pitch_median(wav, sr: int):
    """Get the median pitch of a waveform

    Args:
        wav (np.ndarray/torch.Tensor): waveform of shape (n_channels, n_samples)
        sr (int, optional): sampling rate.

    Returns:
        parselmouth.Pitch, float: a parselmouth.Pitch object and the median pitch
    """
    if not isinstance(wav, parselmouth.Sound):
        sound = wav_to_Sound(wav, sr)
    else:
        sound = wav
    pitch_median = PRAAT_CHANGEGENDER_PITCHMEDIAN_DEFAULT

    # To Pitch: Time step(s)(standard value: 0.0), Pitch floor (Hz)(standard value: 75), Pitch ceiling (Hz)(standard value: 600.0)
    pitch = parselmouth.praat.call(sound, "To Pitch", 0.8 / 75, 75, 600)
    # Get quantile: From time (s), To time (s), Quantile(0.5 is then the 50% quantile, i.e., the median), Units (Hertz or Bark)
    pitch_median = parselmouth.praat.call(pitch, "Get quantile", 0.0, 0.0, 0.5, "Hertz")

    return pitch, pitch_median


def change_gender(
    sound,
    pitch=None,
    formant_shift_ratio: float = PRAAT_CHANGEGENDER_FORMANTSHIFTRATIO_DEFAULT,
    new_pitch_median: float = PRAAT_CHANGEGENDER_PITCHMEDIAN_DEFAULT,
    pitch_range_ratio: float = PRAAT_CHANGEGENDER_PITCHRANGERATIO_DEFAULT,
    duration_factor: float = PRAAT_CHANGEGENDER_DURATIONFACTOR_DEFAULT,
) -> parselmouth.Sound:
    """Invoke change gender function in praat

    Args:
        sound (parselmouth.Sound): a parselmouth.Sound object
        pitch (parselmouth.Pitch, optional): a parselmouth.Pitch object. Defaults to None.
        formant_shift_ratio (float, optional): formant shift ratio. A value of 1.0 means no change. Greater than 1.0 means higher pitch. Less than 1.0 means lower pitch.
        new_pitch_median (float, optional): new pitch median.
        pitch_range_ratio (float, optional): pitch range ratio. A value of 1.0 means no change. Greater than 1.0 means higher pitch range. Less than 1.0 means lower pitch range.
        duration_factor (float, optional): duration factor. A value of 1.0 means no change. Greater than 1.0 means longer duration. Less than 1.0 means shorter duration.

    Returns:
        parselmouth.Sound: a parselmouth.Sound object
    """
    if pitch is None:
        new_sound = parselmouth.praat.call(
            sound,
            "Change gender",
            75,
            600,
            formant_shift_ratio,
            new_pitch_median,
            pitch_range_ratio,
            duration_factor,
        )
    else:
        new_sound = parselmouth.praat.call(
            (sound, pitch),
            "Change gender",
            formant_shift_ratio,
            new_pitch_median,
            pitch_range_ratio,
            duration_factor,
        )
    return new_sound


def apply_formant_and_pitch_shift(
    sound: parselmouth.Sound,
    formant_shift_ratio: float = PRAAT_CHANGEGENDER_FORMANTSHIFTRATIO_DEFAULT,
    pitch_shift_ratio: float = PRAAT_CHANGEGENDER_PITCHSHIFTRATIO_DEFAULT,
    pitch_range_ratio: float = PRAAT_CHANGEGENDER_PITCHRANGERATIO_DEFAULT,
    duration_factor: float = PRAAT_CHANGEGENDER_DURATIONFACTOR_DEFAULT,
) -> parselmouth.Sound:
    """use Praat "Changer gender" command to manipulate pitch and formant
    "Change gender": Praat -> Sound Object -> Convert -> Change gender
    refer to Help of Praat for more details
    # https://github.com/YannickJadoul/Parselmouth/issues/25#issuecomment-608632887 might help
    """
    pitch = None
    new_pitch_median = PRAAT_CHANGEGENDER_PITCHMEDIAN_DEFAULT
    if pitch_shift_ratio != 1.0:
        pitch, pitch_median = get_pitch_median(sound, sound.sampling_frequency)
        new_pitch_median = pitch_median * pitch_shift_ratio

        # refer to https://github.com/praat/praat/issues/1926#issuecomment-974909408
        pitch_minimum = parselmouth.praat.call(
            pitch, "Get minimum", 0.0, 0.0, "Hertz", "Parabolic"
        )
        new_median = pitch_median * pitch_shift_ratio
        scaled_minimum = pitch_minimum * pitch_shift_ratio
        result_minimum = new_median + (scaled_minimum - new_median) * pitch_range_ratio
        if result_minimum < 0:
            new_pitch_median = PRAAT_CHANGEGENDER_PITCHMEDIAN_DEFAULT
            pitch_range_ratio = PRAAT_CHANGEGENDER_PITCHRANGERATIO_DEFAULT

        if math.isnan(new_pitch_median):
            new_pitch_median = PRAAT_CHANGEGENDER_PITCHMEDIAN_DEFAULT
            pitch_range_ratio = PRAAT_CHANGEGENDER_PITCHRANGERATIO_DEFAULT

    new_sound = change_gender(
        sound,
        pitch,
        formant_shift_ratio,
        new_pitch_median,
        pitch_range_ratio,
        duration_factor,
    )
    return new_sound


def pedalboard_equalizer(wav: np.ndarray, sr: int) -> np.ndarray:
    """Apply a randomized 10-band parametric EQ using pedalboard.

    Builds a low-shelf filter, eight peaking filters, and a high-shelf
    filter with randomly drawn center frequencies, Q values, and gains,
    then filters the waveform through them.

    Args:
        wav: Waveform, shape (n_samples,) or (n_channels, n_samples).
        sr: Sampling rate in Hz.

    Returns:
        The equalized waveform, same shape as ``wav``.
    """
    board = Pedalboard()

    q_min = 2
    q_max = 5

    num_filters = 10
    key_freqs = [random.uniform(1, 12000) for _ in range(num_filters)]
    q_values = [
        power_ratio(random.uniform(0, 1), q_min, q_max) for _ in range(num_filters)
    ]
    gains = [random.uniform(-12, 12) for _ in range(num_filters)]
    # low-shelving filter
    board.append(
        LowShelfFilter(
            cutoff_frequency_hz=key_freqs[0], gain_db=gains[0], q=q_values[0]
        )
    )
    # peaking filters
    for i in range(1, 9):
        board.append(
            PeakFilter(
                cutoff_frequency_hz=key_freqs[i], gain_db=gains[i], q=q_values[i]
            )
        )
    # high-shelving filter
    board.append(
        HighShelfFilter(
            cutoff_frequency_hz=key_freqs[9], gain_db=gains[9], q=q_values[9]
        )
    )

    # Apply the pedalboard to the audio
    processed_audio = board(wav, sr)
    return processed_audio


def power_ratio(r: float, a: float, b: float) -> float:
    """Interpolate geometrically between ``a`` and ``b``.

    Args:
        r: Interpolation position in [0, 1] (0 -> ``a``, 1 -> ``b``).
        a: Lower bound.
        b: Upper bound.

    Returns:
        ``a * (b / a) ** r``.
    """
    return a * math.pow((b / a), r)


def audiomentations_time_stretch(wav: np.ndarray, sr: int) -> np.ndarray:
    """Randomly time-stretch a waveform using audiomentations.

    Args:
        wav: Waveform, shape (n_samples,).
        sr: Sampling rate in Hz.

    Returns:
        Time-stretched waveform (length may differ from the input since
        ``leave_length_unchanged=False``).
    """
    transform = TimeStretch(
        min_rate=0.8, max_rate=1.25, leave_length_unchanged=False, p=1.0
    )
    augmented_wav = transform(wav, sample_rate=sr)
    return augmented_wav


def formant_and_pitch_shift(
    sound: parselmouth.Sound, fs: bool, ps: bool
) -> parselmouth.Sound:
    """Apply exactly one of formant shift (via Praat) or pitch shift (via pedalboard).

    Args:
        sound: Source sound.
        fs: If True, apply a random formant shift.
        ps: If True, apply a random pitch shift.

    Returns:
        The shifted sound.

    Raises:
        AssertionError: If ``fs`` and ``ps`` are not mutually exclusive
            (exactly one of them must be True).
    """
    formant_shift_ratio = PRAAT_CHANGEGENDER_FORMANTSHIFTRATIO_DEFAULT
    pitch_shift_ratio = PRAAT_CHANGEGENDER_PITCHSHIFTRATIO_DEFAULT
    pitch_range_ratio = PRAAT_CHANGEGENDER_PITCHRANGERATIO_DEFAULT

    assert fs != ps, "fs, ps are mutually exclusive"

    if fs:
        formant_shift_ratio = random.uniform(1.0, 1.4)
        use_reciprocal = random.uniform(-1, 1) > 0
        if use_reciprocal:
            formant_shift_ratio = 1.0 / formant_shift_ratio
        # only use praat to change formant
        new_sound = apply_formant_and_pitch_shift(
            sound,
            formant_shift_ratio=formant_shift_ratio,
        )
        return new_sound

    if ps:
        board = Pedalboard()
        board.append(PitchShift(random.uniform(-12, 12)))
        wav_numpy = sound.values
        wav_numpy = board(wav_numpy, sound.sampling_frequency)
        # use pedalboard to change pitch
        new_sound = parselmouth.Sound(
            wav_numpy, sampling_frequency=sound.sampling_frequency
        )
        return new_sound


def _trim_or_pad_to_length(output: np.ndarray, target_length: int) -> np.ndarray:
    """Trim or zero-pad a 1-D array on the right to exactly ``target_length``.

    Args:
        output: Array of shape (n_samples,).
        target_length: Desired length after trimming/padding.

    Returns:
        Array of shape (target_length,).
    """
    if output.shape[0] >= target_length:
        return output[:target_length]
    return np.pad(output, (0, target_length - output.shape[0]))


def change_pitch_and_formant_random(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    """Apply a randomized pitch shift and formant/gender shift to audio.

    Args:
        audio: Waveform, shape (1, n_samples).
        sample_rate: Sampling rate in Hz.

    Returns:
        Perturbed waveform, shape (n_samples,), same length as the input.
    """
    sound = parselmouth.Sound(audio, sampling_frequency=sample_rate)
    original_size = sound.values.shape[1]  # shape of sound.values is [1, audio_length]
    pitch_shift_ratio = random.uniform(-0.2, 0.2)  # -15 to +15 cents
    pitch_shift_ratio = random.choice([-12, 0, 12]) + pitch_shift_ratio

    sound = change_pitch(sound, pitch_shift_ratio)  # -15 to +15 cents

    formant_shift_ratio = random.uniform(1, 1.4)
    formant_shift_ratio = random.choice([formant_shift_ratio, 1 / formant_shift_ratio])

    sound = change_gender(
        sound, None, formant_shift_ratio, 0.0, 1, max(0.7, random.normalvariate(1.0, 0.05))
    )

    output = sound.values[0]  # shape of sound.values is [1, audio_length]
    return _trim_or_pad_to_length(output, original_size)


def change_pitch_and_formant(
    audio: np.ndarray,
    sample_rate: int,
    pitch_shift_ratio: float,
    formant_shift_ratio: float,
    pitch_range_ratio: float,
    time_stretch_ratio: float,
) -> np.ndarray:
    """Apply an explicitly-specified pitch shift and formant/gender shift.

    Args:
        audio: Waveform, shape (1, n_samples).
        sample_rate: Sampling rate in Hz.
        pitch_shift_ratio: Pitch shift in semitones, passed to `change_pitch`.
        formant_shift_ratio: Formant shift ratio, passed to `change_gender`.
        pitch_range_ratio: Pitch range ratio, passed to `change_gender`.
        time_stretch_ratio: Duration factor, passed to `change_gender`.

    Returns:
        Perturbed waveform, shape (n_samples,), same length as the input.
    """
    sound = parselmouth.Sound(audio, sampling_frequency=sample_rate)
    original_size = sound.values.shape[1]  # shape of sound.values is [1, audio_length]

    sound = change_pitch(sound, pitch_shift_ratio)  # -15 to +15 cents
    sound = change_gender(
        sound, None, formant_shift_ratio, 0.0, pitch_range_ratio, time_stretch_ratio
    )

    output = sound.values[0]  # shape of sound.values is [1, audio_length]
    return _trim_or_pad_to_length(output, original_size)


def _resolve_augmentation_flags(
    aug_type: str,
    formant_shift: bool,
    pitch_shift: bool,
    time_stretch: bool,
    equalizer: bool,
) -> Tuple[bool, bool, bool, bool]:
    """Validate `aug_type`/flag combination and fold `aug_type` into flags.

    Args:
        aug_type: Either "None" or exactly one of "formant_shift",
            "pitch_shift", "time_stretch", "equalizer".
        formant_shift: Explicit formant-shift flag.
        pitch_shift: Explicit pitch-shift flag.
        time_stretch: Explicit time-stretch flag.
        equalizer: Explicit equalizer flag.

    Returns:
        The (formant_shift, pitch_shift, time_stretch, equalizer) flags
        with the one named by `aug_type` (if any) set to True.

    Raises:
        AssertionError: If `aug_type` is invalid, or if `aug_type` is
            given while any explicit flag is also True.
    """
    assert aug_type == "None" or aug_type in [
        "formant_shift",
        "pitch_shift",
        "time_stretch",
        "equalizer",
    ], "aug_type must be one of formant_shift, pitch_shift, time_stretch, equalizer"

    assert aug_type == "None" or (
        formant_shift == False
        and pitch_shift == False
        and time_stretch == False
        and equalizer == False
    ), "if aug_type is specified, other argument must be False"

    if aug_type != "None":
        if aug_type == "formant_shift":
            formant_shift = True
        if aug_type == "pitch_shift":
            pitch_shift = True
        if aug_type == "equalizer":
            equalizer = True
        if aug_type == "time_stretch":
            time_stretch = True

    return formant_shift, pitch_shift, time_stretch, equalizer


def wav_manipulation(
    wav: torch.Tensor,
    sr: int,
    aug_type: str = "None",
    formant_shift: bool = False,
    pitch_shift: bool = False,
    time_stretch: bool = False,
    equalizer: bool = False,
) -> torch.Tensor:
    """Apply one or more waveform augmentations, selected by flag or `aug_type`.

    Either pass `aug_type` alone (with all boolean flags left False), or
    set the boolean flags directly to combine multiple augmentations;
    `formant_shift` and `pitch_shift` are mutually exclusive with each
    other in the underlying Praat-based shift.

    Args:
        wav: Waveform tensor, shape (1, n_samples).
        sr: Sampling rate in Hz.
        aug_type: Convenience selector; "None" or one of "formant_shift",
            "pitch_shift", "time_stretch", "equalizer".
        formant_shift: Apply a random formant shift.
        pitch_shift: Apply a random pitch shift.
        time_stretch: Apply a random time stretch.
        equalizer: Apply a random 10-band parametric EQ.

    Returns:
        Augmented waveform tensor, shape (1, n_samples).

    Raises:
        AssertionError: See `_resolve_augmentation_flags`.
    """
    formant_shift, pitch_shift, time_stretch, equalizer = _resolve_augmentation_flags(
        aug_type, formant_shift, pitch_shift, time_stretch, equalizer
    )

    wav_numpy = wav.numpy()

    if equalizer:
        wav_numpy = pedalboard_equalizer(wav_numpy, sr)

    if time_stretch:
        wav_numpy = audiomentations_time_stretch(wav_numpy, sr)

    sound = wav_to_Sound(wav_numpy, sr)

    if formant_shift or pitch_shift:
        sound = formant_and_pitch_shift(sound, formant_shift, pitch_shift)

    wav = torch.from_numpy(sound.values).float()
    # shape (1, n_samples)
    return wav