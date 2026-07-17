"""Preprocessor module.

A "preprocessor" is a callable ``(uid, data) -> data`` applied per-utterance
by :class:`espnet2.train.dataset.ESPnetDataset` / ``IterableESPnetDataset``
right after the raw arrays are loaded from disk, and before batching. Each
preprocessor in this module handles a different task's data: text
tokenization, on-the-fly speech augmentation (RIR convolution, additive
noise, dynamic mixing), enrollment loading for target-speaker extraction,
score/label alignment for singing voice synthesis, etc. ``train=True``
enables augmentation that should not run at inference/validation time.
"""
import json
import logging
import random
import re
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Collection, Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
import scipy.signal
import soundfile
from typeguard import check_argument_types, check_return_type

from espnet2.text.build_tokenizer import build_tokenizer
from espnet2.text.cleaner import TextCleaner
from espnet2.text.token_id_converter import TokenIDConverter
from espnet2.text.whisper_token_id_converter import (
    OpenAIWhisperTokenIDConverter,
)


class AbsPreprocessor(ABC):
    """Base class for all per-utterance preprocessors.

    Subclasses implement :meth:`__call__`, mapping the raw
    ``{data_name: array_or_str}`` dict loaded for one utterance to the dict
    of ``np.ndarray``\\ s that the model's ``forward()`` expects.
    """

    def __init__(self, train: bool):
        self.train = train

    @abstractmethod
    def __call__(
        self, uid: str, data: Dict[str, Union[str, np.ndarray]]
    ) -> Dict[str, np.ndarray]:
        raise NotImplementedError


def framing(
    x: np.ndarray,
    frame_length: int = 512,
    frame_shift: int = 256,
    centered: bool = True,
    padded: bool = True,
) -> np.ndarray:
    """Split the last axis of ``x`` into overlapping frames.

    Args:
        x: Input array; framing is applied along the last axis.
        frame_length: Number of samples per frame.
        frame_shift: Number of samples between successive frame starts.
        centered: If True, zero-pad ``frame_length // 2`` on both sides
            first (so frame ``i`` is centered on sample ``i * frame_shift``).
        padded: If True, zero-pad the end so the last frame is complete.

    Returns:
        A view of ``x`` with shape ``(..., n_frames, frame_length)``.
    """
    if x.size == 0:
        raise ValueError("Input array size is zero")
    if frame_length < 1:
        raise ValueError("frame_length must be a positive integer")
    if frame_length > x.shape[-1]:
        raise ValueError("frame_length is greater than input length")
    if 0 >= frame_shift:
        raise ValueError("frame_shift must be greater than 0")

    if centered:
        pad_shape = [(0, 0) for _ in range(x.ndim - 1)] + [
            (frame_length // 2, frame_length // 2)
        ]
        x = np.pad(x, pad_shape, mode="constant", constant_values=0)

    if padded:
        # Pad to integer number of windowed segments
        # I.e make x.shape[-1] = frame_length + (nseg-1)*nstep,
        #  with integer nseg
        nadd = (-(x.shape[-1] - frame_length) % frame_shift) % frame_length
        pad_shape = [(0, 0) for _ in range(x.ndim - 1)] + [(0, nadd)]
        x = np.pad(x, pad_shape, mode="constant", constant_values=0)

    # Created strided array of data segments
    if frame_length == 1 and frame_length == frame_shift:
        result = x[..., None]
    else:
        shape = x.shape[:-1] + (
            (x.shape[-1] - frame_length) // frame_shift + 1,
            frame_length,
        )
        strides = x.strides[:-1] + (frame_shift * x.strides[-1], x.strides[-1])
        result = np.lib.stride_tricks.as_strided(
            x, shape=shape, strides=strides
        )
    return result


def detect_non_silence(
    x: np.ndarray,
    threshold: float = 0.01,
    frame_length: int = 1024,
    frame_shift: int = 512,
    window: str = "boxcar",
) -> np.ndarray:
    """Power based voice activity detection.

    Args:
        x: (Channel, Time)
    >>> x = np.random.randn(1000)
    >>> detect = detect_non_silence(x)
    >>> assert x.shape == detect.shape
    >>> assert detect.dtype == np.bool
    """
    if x.shape[-1] < frame_length:
        return np.full(x.shape, fill_value=True, dtype=np.bool)

    if x.dtype.kind == "i":
        x = x.astype(np.float64)
    # framed_w: (C, T, F)
    framed_w = framing(
        x,
        frame_length=frame_length,
        frame_shift=frame_shift,
        centered=False,
        padded=True,
    )
    framed_w *= scipy.signal.get_window(window, frame_length).astype(
        framed_w.dtype
    )
    # power: (C, T)
    power = (framed_w**2).mean(axis=-1)
    # mean_power: (C, 1)
    mean_power = np.mean(power, axis=-1, keepdims=True)
    if np.all(mean_power == 0):
        return np.full(x.shape, fill_value=True, dtype=np.bool)
    # detect_frames: (C, T)
    detect_frames = power / mean_power > threshold
    # detects: (C, T, F)
    detects = np.broadcast_to(
        detect_frames[..., None], detect_frames.shape + (frame_shift,)
    )
    # detects: (C, TF)
    detects = detects.reshape(*detect_frames.shape[:-1], -1)
    # detects: (C, TF)
    return np.pad(
        detects,
        [(0, 0)] * (x.ndim - 1) + [(0, x.shape[-1] - detects.shape[-1])],
        mode="edge",
    )


class CommonPreprocessor(AbsPreprocessor):
    """Preprocessor shared by most ASR/TTS-style tasks.

    Handles two independent pieces of work:

    * Speech augmentation (train-time only): convolve with a randomly
      sampled RIR and/or mix in additive noise, then peak-normalize.
    * Text tokenization: clean -> tokenize -> convert tokens to integer
      ids, for ``text_name`` and any ``aux_task_names``.
    """

    def __init__(
        self,
        train: bool,
        token_type: str = None,
        token_list: Union[Path, str, Iterable[str]] = None,
        bpemodel: Union[Path, str, Iterable[str]] = None,
        text_cleaner: Collection[str] = None,
        g2p_type: str = None,
        unk_symbol: str = "<unk>",
        space_symbol: str = "<space>",
        non_linguistic_symbols: Union[Path, str, Iterable[str]] = None,
        delimiter: str = None,
        rir_scp: str = None,
        rir_apply_prob: float = 1.0,
        noise_scp: str = None,
        noise_apply_prob: float = 1.0,
        noise_db_range: str = "3_10",
        short_noise_thres: float = 0.5,
        aux_task_names: Collection[str] = None,
        speech_volume_normalize: float = None,
        speech_name: str = "speech",
        text_name: str = "text",
        fs: int = 0,
        nonsplit_symbol: Iterable[str] = None,
    ):
        super().__init__(train)
        self.train = train
        self.speech_name = speech_name
        self.text_name = text_name
        self.speech_volume_normalize = speech_volume_normalize
        self.rir_apply_prob = rir_apply_prob
        self.noise_apply_prob = noise_apply_prob
        self.short_noise_thres = short_noise_thres
        self.aux_task_names = aux_task_names

        if token_type is not None:
            if token_list is None:
                raise ValueError(
                    "token_list is required if token_type is not None"
                )
            self.text_cleaner = TextCleaner(text_cleaner)

            self.tokenizer = build_tokenizer(
                token_type=token_type,
                bpemodel=bpemodel,
                delimiter=delimiter,
                space_symbol=space_symbol,
                non_linguistic_symbols=non_linguistic_symbols,
                g2p_type=g2p_type,
                nonsplit_symbol=nonsplit_symbol,
            )
            if bpemodel not in ["whisper_en", "whisper_multilingual"]:
                self.token_id_converter = TokenIDConverter(
                    token_list=token_list,
                    unk_symbol=unk_symbol,
                )
            else:
                self.token_id_converter = OpenAIWhisperTokenIDConverter(
                    model_type=bpemodel
                )
        else:
            self.text_cleaner = None
            self.tokenizer = None
            self.token_id_converter = None

        if train and rir_scp is not None:
            self.rirs = []
            with open(rir_scp, "r", encoding="utf-8") as f:
                for line in f:
                    sps = line.strip().split(None, 1)
                    if len(sps) == 1:
                        self.rirs.append(sps[0])
                    else:
                        self.rirs.append(sps[1])
        else:
            self.rirs = None

        if train and noise_scp is not None:
            self.noises = []
            with open(noise_scp, "r", encoding="utf-8") as f:
                for line in f:
                    sps = line.strip().split(None, 1)
                    if len(sps) == 1:
                        self.noises.append(sps[0])
                    else:
                        self.noises.append(sps[1])
            sps = noise_db_range.split("_")
            if len(sps) == 1:
                self.noise_db_low = self.noise_db_high = float(sps[0])
            elif len(sps) == 2:
                self.noise_db_low, self.noise_db_high = float(sps[0]), float(
                    sps[1]
                )
            else:
                raise ValueError(
                    "Format error: '{noise_db_range}' e.g. -3_4 -> [-3db,4db]"
                )
        else:
            self.noises = None

    def _convolve_rir(
        self, speech: np.ndarray, power: float
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Convolve ``speech`` with a randomly chosen RIR, preserving its power.

        Args:
            speech: (Nmic, Time)
            power: Reference power (of ``speech`` on non-silent frames) to
                rescale the convolved signal back to.

        Returns:
            (convolved_speech, rir) -- ``rir`` is None if no RIR was applied.
        """
        rir_path = np.random.choice(self.rirs)
        rir = None
        if rir_path is not None:
            rir, _ = soundfile.read(rir_path, dtype=np.float64, always_2d=True)

            # rir: (Nmic, Time)
            rir = rir.T

            # speech: (Nmic, Time)
            # Note that this operation doesn't change the signal length
            speech = scipy.signal.convolve(speech, rir, mode="full")[
                :, : speech.shape[1]
            ]
            # Reverse mean power to the original power
            power2 = (speech[detect_non_silence(speech)] ** 2).mean()
            speech = np.sqrt(power / max(power2, 1e-10)) * speech
        return speech, rir

    def _add_noise(
        self, speech: np.ndarray, power: float
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Mix in a randomly chosen, randomly gained noise clip.

        Args:
            speech: (Nmic, Time)
            power: Reference power of ``speech``, used to scale the noise
                to a randomly sampled SNR in ``[noise_db_low, noise_db_high]``.

        Returns:
            (noisy_speech, noise) -- ``noise`` is None if no noise was applied.
        """
        nsamples = speech.shape[1]
        noise_path = np.random.choice(self.noises)
        noise = None
        if noise_path is not None:
            noise_db = np.random.uniform(self.noise_db_low, self.noise_db_high)
            with soundfile.SoundFile(noise_path) as f:
                if f.frames == nsamples:
                    noise = f.read(dtype=np.float64, always_2d=True)
                elif f.frames < nsamples:
                    if f.frames / nsamples < self.short_noise_thres:
                        logging.warning(
                            f"Noise ({f.frames}) is much shorter than "
                            f"speech ({nsamples}) in dynamic mixing"
                        )
                    offset = np.random.randint(0, nsamples - f.frames)
                    # noise: (Time, Nmic)
                    noise = f.read(dtype=np.float64, always_2d=True)
                    # Repeat noise
                    noise = np.pad(
                        noise,
                        [(offset, nsamples - f.frames - offset), (0, 0)],
                        mode="wrap",
                    )
                else:
                    offset = np.random.randint(0, f.frames - nsamples)
                    f.seek(offset)
                    # noise: (Time, Nmic)
                    noise = f.read(nsamples, dtype=np.float64, always_2d=True)
                    if len(noise) != nsamples:
                        raise RuntimeError(f"Something wrong: {noise_path}")
            # noise: (Nmic, Time)
            noise = noise.T

            noise_power = (noise**2).mean()
            scale = (
                10 ** (-noise_db / 20)
                * np.sqrt(power)
                / np.sqrt(max(noise_power, 1e-10))
            )
            speech = speech + scale * noise
        return speech, noise

    def _speech_process(
        self, data: Dict[str, Union[str, np.ndarray]]
    ) -> Dict[str, Union[str, np.ndarray]]:
        """Apply RIR/noise augmentation (train only) and volume normalization."""
        assert check_argument_types()
        if self.speech_name in data:
            if self.train and (
                self.rirs is not None or self.noises is not None
            ):
                speech = data[self.speech_name]

                # speech: (Nmic, Time)
                if speech.ndim == 1:
                    speech = speech[None, :]
                else:
                    speech = speech.T
                # Calc power on non silence region
                power = (speech[detect_non_silence(speech)] ** 2).mean()

                # 1. Convolve RIR
                if (
                    self.rirs is not None
                    and self.rir_apply_prob >= np.random.random()
                ):
                    speech, _ = self._convolve_rir(speech, power)

                # 2. Add Noise
                if (
                    self.noises is not None
                    and self.noise_apply_prob >= np.random.random()
                ):
                    speech, _ = self._add_noise(speech, power)

                speech = speech.T
                ma = np.max(np.abs(speech))
                if ma > 1.0:
                    speech /= ma
                data[self.speech_name] = speech

            if self.speech_volume_normalize is not None:
                speech = data[self.speech_name]
                ma = np.max(np.abs(speech))
                data[self.speech_name] = (
                    speech * self.speech_volume_normalize / ma
                )
        assert check_return_type(data)
        return data

    def _text_process(
        self, data: Dict[str, Union[str, np.ndarray]]
    ) -> Dict[str, np.ndarray]:
        """Clean, tokenize, and convert ``text_name``/``aux_task_names`` fields to ids."""
        if self.text_name in data and self.tokenizer is not None:
            text = data[self.text_name]
            if isinstance(text, np.ndarray):
                return data
            text = self.text_cleaner(text)
            tokens = self.tokenizer.text2tokens(text)
            text_ints = self.token_id_converter.tokens2ids(tokens)
            data[self.text_name] = np.array(text_ints, dtype=np.int64)
        if self.aux_task_names is not None and self.tokenizer is not None:
            for name in self.aux_task_names:
                if name in data:
                    text = data[name]
                    text = self.text_cleaner(text)
                    tokens = self.tokenizer.text2tokens(text)
                    text_ints = self.token_id_converter.tokens2ids(tokens)
                    data[name] = np.array(text_ints, dtype=np.int64)
        assert check_return_type(data)
        return data

    def __call__(
        self, uid: str, data: Dict[str, Union[str, np.ndarray]]
    ) -> Dict[str, np.ndarray]:
        """Augment speech (train-time) and tokenize text for one utterance."""
        assert check_argument_types()

        data = self._speech_process(data)
        data = self._text_process(data)
        return data


class SLUPreprocessor(CommonPreprocessor):
    """CommonPreprocessor variant that additionally tokenizes a "transcript" field."""

    def __init__(
        self,
        train: bool,
        token_type: str = None,
        token_list: Union[Path, str, Iterable[str]] = None,
        transcript_token_list: Union[Path, str, Iterable[str]] = None,
        bpemodel: Union[Path, str, Iterable[str]] = None,
        text_cleaner: Collection[str] = None,
        g2p_type: str = None,
        unk_symbol: str = "<unk>",
        space_symbol: str = "<space>",
        non_linguistic_symbols: Union[Path, str, Iterable[str]] = None,
        delimiter: str = None,
        rir_scp: str = None,
        rir_apply_prob: float = 1.0,
        noise_scp: str = None,
        noise_apply_prob: float = 1.0,
        noise_db_range: str = "3_10",
        short_noise_thres: float = 0.5,
        speech_volume_normalize: float = None,
        speech_name: str = "speech",
        text_name: str = "text",
    ):
        super().__init__(
            train=train,
            token_type=token_type,
            token_list=token_list,
            bpemodel=bpemodel,
            text_cleaner=text_cleaner,
            g2p_type=g2p_type,
            unk_symbol=unk_symbol,
            space_symbol=space_symbol,
            non_linguistic_symbols=non_linguistic_symbols,
            delimiter=delimiter,
            rir_scp=rir_scp,
            rir_apply_prob=rir_apply_prob,
            noise_scp=noise_scp,
            noise_apply_prob=noise_apply_prob,
            noise_db_range=noise_db_range,
            short_noise_thres=short_noise_thres,
            speech_volume_normalize=speech_volume_normalize,
            speech_name=speech_name,
            text_name=text_name,
        )
        if transcript_token_list is not None:
            print("using transcript")
            self.transcript_tokenizer = build_tokenizer(
                token_type="word",
                bpemodel=bpemodel,
                delimiter=delimiter,
                space_symbol=space_symbol,
                non_linguistic_symbols=non_linguistic_symbols,
                g2p_type=g2p_type,
            )
            self.transcript_token_id_converter = TokenIDConverter(
                token_list=transcript_token_list,
                unk_symbol=unk_symbol,
            )
        else:
            self.transcript_tokenizer = None
            self.transcript_token_id_converter = None

    def _text_process(
        self, data: Dict[str, Union[str, np.ndarray]]
    ) -> Dict[str, np.ndarray]:
        """Tokenize ``text_name`` with the main tokenizer and "transcript" with
        the (word-level) transcript tokenizer."""
        if self.text_name in data and self.tokenizer is not None:
            text = data[self.text_name]
            text = self.text_cleaner(text)
            tokens = self.tokenizer.text2tokens(text)
            text_ints = self.token_id_converter.tokens2ids(tokens)
            data[self.text_name] = np.array(text_ints, dtype=np.int64)
        if "transcript" in data and self.tokenizer is not None:
            text = data["transcript"]
            text = self.text_cleaner(text)
            tokens = self.transcript_tokenizer.text2tokens(text)
            text_ints = self.transcript_token_id_converter.tokens2ids(tokens)
            data["transcript"] = np.array(text_ints, dtype=np.int64)
        assert check_return_type(data)
        return data


class CommonPreprocessor_multi(CommonPreprocessor):
    """CommonPreprocessor variant supporting multiple text fields (e.g. multi-speaker)."""

    def __init__(
        self,
        train: bool,
        token_type: str = None,
        token_list: Union[Path, str, Iterable[str]] = None,
        bpemodel: Union[Path, str, Iterable[str]] = None,
        text_cleaner: Collection[str] = None,
        g2p_type: str = None,
        unk_symbol: str = "<unk>",
        space_symbol: str = "<space>",
        non_linguistic_symbols: Union[Path, str, Iterable[str]] = None,
        delimiter: str = None,
        rir_scp: str = None,
        rir_apply_prob: float = 1.0,
        noise_scp: str = None,
        noise_apply_prob: float = 1.0,
        noise_db_range: str = "3_10",
        short_noise_thres: float = 0.5,
        aux_task_names: Collection[str] = None,
        speech_volume_normalize: float = None,
        speech_name: str = "speech",
        text_name: List[str] = ["text"],
        fs: int = 0,
        speaker_change_symbol: Iterable[str] = None,
    ):
        super().__init__(
            train=train,
            token_type=token_type,
            token_list=token_list,
            bpemodel=bpemodel,
            text_cleaner=text_cleaner,
            g2p_type=g2p_type,
            unk_symbol=unk_symbol,
            space_symbol=space_symbol,
            non_linguistic_symbols=non_linguistic_symbols,
            delimiter=delimiter,
            rir_scp=rir_scp,
            rir_apply_prob=rir_apply_prob,
            noise_scp=noise_scp,
            noise_apply_prob=noise_apply_prob,
            noise_db_range=noise_db_range,
            short_noise_thres=short_noise_thres,
            aux_task_names=aux_task_names,
            speech_volume_normalize=speech_volume_normalize,
            speech_name=speech_name,
            fs=fs,
            nonsplit_symbol=speaker_change_symbol,
        )
        if isinstance(text_name, str):
            self.text_name = [text_name]
        else:
            self.text_name = text_name

        self.speaker_change_symbol = speaker_change_symbol
        if speaker_change_symbol is not None:
            assert (
                len(self.text_name) == 1
            ), "SOT model with speaker_change_symbol only support single text input."

    def _text_process(
        self, data: Dict[str, Union[str, np.ndarray]]
    ) -> Dict[str, np.ndarray]:
        """Tokenize every field in ``self.text_name`` plus any aux-task fields."""
        for text_n in self.text_name:
            if text_n in data and self.tokenizer is not None:
                text = data[text_n]
                text = self.text_cleaner(text)
                tokens = self.tokenizer.text2tokens(text)
                text_ints = self.token_id_converter.tokens2ids(tokens)
                data[text_n] = np.array(text_ints, dtype=np.int64)
        if self.aux_task_names is not None and self.tokenizer is not None:
            for name in self.aux_task_names:
                if name in data:
                    text = data[name]
                    text = self.text_cleaner(text)
                    tokens = self.tokenizer.text2tokens(text)
                    text_ints = self.token_id_converter.tokens2ids(tokens)
                    data[name] = np.array(text_ints, dtype=np.int64)
        assert check_return_type(data)
        return data

    def __call__(
        self, uid: str, data: Dict[str, Union[str, np.ndarray]]
    ) -> Dict[str, np.ndarray]:
        """Augment speech (train-time) and tokenize all text fields."""
        assert check_argument_types()

        data = self._speech_process(data)
        data = self._text_process(data)
        return data


class MutliTokenizerCommonPreprocessor(CommonPreprocessor):
    """CommonPreprocessor variant using a separate tokenizer per text field."""

    def __init__(
        self,
        train: bool,
        token_type: List[str] = [None],
        token_list: List[Union[Path, str, Iterable[str]]] = [None],
        bpemodel: List[Union[Path, str, Iterable[str]]] = [None],
        text_cleaner: Collection[str] = None,
        g2p_type: str = None,
        unk_symbol: str = "<unk>",
        space_symbol: str = "<space>",
        non_linguistic_symbols: Union[Path, str, Iterable[str]] = None,
        delimiter: str = None,
        rir_scp: str = None,
        rir_apply_prob: float = 1.0,
        noise_scp: str = None,
        noise_apply_prob: float = 1.0,
        noise_db_range: str = "3_10",
        short_noise_thres: float = 0.5,
        speech_volume_normalize: float = None,
        speech_name: str = "speech",
        text_name: List[str] = ["text"],
    ):
        # TODO(jiatong): sync with Kamo and Jing on interface for preprocessor
        super().__init__(
            train=train,
            token_type=token_type[0],
            token_list=token_list[0],
            bpemodel=bpemodel[0],
            text_cleaner=text_cleaner,
            g2p_type=g2p_type,
            unk_symbol=unk_symbol,
            space_symbol=space_symbol,
            non_linguistic_symbols=non_linguistic_symbols,
            delimiter=delimiter,
            speech_name=speech_name,
            text_name=text_name[0],
            rir_scp=rir_scp,
            rir_apply_prob=rir_apply_prob,
            noise_scp=noise_scp,
            noise_apply_prob=noise_apply_prob,
            noise_db_range=noise_db_range,
            short_noise_thres=short_noise_thres,
            speech_volume_normalize=speech_volume_normalize,
        )

        assert (
            len(token_type)
            == len(token_list)
            == len(bpemodel)
            == len(text_name)
        ), "token_type, token_list, bpemodel, or processing text_name mismatched"
        self.num_tokenizer = len(token_type)
        self.tokenizer = []
        self.token_id_converter = []

        for i in range(self.num_tokenizer):
            if token_type[i] is not None:
                if token_list[i] is None:
                    raise ValueError(
                        "token_list is required if token_type is not None"
                    )

                self.tokenizer.append(
                    build_tokenizer(
                        token_type=token_type[i],
                        bpemodel=bpemodel[i],
                        delimiter=delimiter,
                        space_symbol=space_symbol,
                        non_linguistic_symbols=non_linguistic_symbols,
                        g2p_type=g2p_type,
                    )
                )
                self.token_id_converter.append(
                    TokenIDConverter(
                        token_list=token_list[i],
                        unk_symbol=unk_symbol,
                    )
                )
            else:
                self.tokenizer.append(None)
                self.token_id_converter.append(None)

        self.text_cleaner = TextCleaner(text_cleaner)
        self.text_name = (
            text_name  # override the text_name from CommonPreprocessor
        )

    def _text_process(
        self, data: Dict[str, Union[str, np.ndarray]]
    ) -> Dict[str, np.ndarray]:
        """Tokenize each ``self.text_name[i]`` field with its own tokenizer."""
        for i in range(self.num_tokenizer):
            text_name = self.text_name[i]
            if text_name in data and self.tokenizer[i] is not None:
                text = data[text_name]
                text = self.text_cleaner(text)
                tokens = self.tokenizer[i].text2tokens(text)
                text_ints = self.token_id_converter[i].tokens2ids(tokens)
                data[text_name] = np.array(text_ints, dtype=np.int64)
        assert check_return_type(data)
        return data


class DynamicMixingPreprocessor(AbsPreprocessor):
    """Builds a fresh random speech mixture from single-speaker sources per call.

    On each ``__call__`` (train only), picks ``ref_num - 1`` additional
    single-speaker utterances (different speakers, when possible), applies a
    random per-source gain, and sums them with the utterance's own source to
    produce ``speech_mix`` plus ``speech_ref{1..ref_num}``.
    """

    def __init__(
        self,
        train: bool,
        source_scp: str = None,
        ref_num: int = 2,
        dynamic_mixing_gain_db: float = 0.0,
        speech_name: str = "speech_mix",
        speech_ref_name_prefix: str = "speech_ref",
        mixture_source_name: str = None,
        utt2spk: str = None,
    ):
        super().__init__(train)
        self.source_scp = source_scp
        self.ref_num = ref_num
        self.dynamic_mixing_gain_db = dynamic_mixing_gain_db
        self.speech_name = speech_name
        self.speech_ref_name_prefix = speech_ref_name_prefix
        # mixture_source_name: the key to select source utterances from dataloader
        if mixture_source_name is None:
            self.mixture_source_name = f"{speech_ref_name_prefix}1"
        else:
            self.mixture_source_name = mixture_source_name

        self.sources = {}
        assert (
            source_scp is not None
        ), f"Please pass `source_scp` to {type(self).__name__}"
        with open(source_scp, "r", encoding="utf-8") as f:
            for line in f:
                sps = line.strip().split(None, 1)
                assert len(sps) == 2
                self.sources[sps[0]] = sps[1]

        self.utt2spk = {}
        if utt2spk is None:
            # if utt2spk is not provided, create a dummy utt2spk with uid.
            for key in self.sources.keys():
                self.utt2spk[key] = key
        else:
            with open(utt2spk, "r", encoding="utf-8") as f:
                for line in f:
                    sps = line.strip().split(None, 1)
                    assert len(sps) == 2
                    self.utt2spk[sps[0]] = sps[1]

            for key in self.sources.keys():
                assert key in self.utt2spk

        self.source_keys = list(self.sources.keys())

    def _pick_source_utterances_(self, uid: str) -> List[str]:
        """Pick ``ref_num - 1`` reference-source uids, preferring distinct speakers."""
        source_keys = [uid]

        spk_ids = [self.utt2spk[uid]]

        retry_cnt = 0
        while len(source_keys) < self.ref_num:
            picked = random.choice(self.source_keys)
            spk_id = self.utt2spk[picked]

            # make one utterance or one speaker only appears once in mixing.
            if (picked not in source_keys) and (spk_id not in spk_ids):
                source_keys.append(picked)
            else:
                retry_cnt += 1
                if retry_cnt > 10:
                    source_keys.append(picked)
                    logging.warning(
                        "Can not find speech source from different speaker "
                        f"for {retry_cnt} times."
                        "There may be problems with training data. "
                        "Please check the utt2spk file."
                    )

        return source_keys[1:]

    def _read_source_(self, key: str, speech_length: int) -> np.ndarray:
        """Read the source audio for ``key``, padding/cropping to ``speech_length``."""
        source, _ = soundfile.read(
            self.sources[key],
            dtype=np.float32,
            always_2d=False,
        )

        if speech_length > source.shape[0]:
            pad = speech_length - source.shape[0]
            source = np.pad(source, (0, pad), "reflect")
        else:
            source = source[0:speech_length]

        assert speech_length == source.shape[0]

        return source

    def _mix_speech_(
        self, uid: str, data: Dict[str, Union[str, np.ndarray]]
    ) -> Dict[str, Union[str, np.ndarray]]:
        """Sample sources, apply random per-source gain, and sum into a mixture."""
        # pick sources
        source_keys = self._pick_source_utterances_(uid)

        # load audios
        speech_length = data[self.mixture_source_name].shape[0]
        ref_audios = [
            self._read_source_(key, speech_length) for key in source_keys
        ]
        ref_audios = [data[self.mixture_source_name]] + ref_audios

        # apply random gain to speech sources

        gain_in_db = [
            random.uniform(
                -self.dynamic_mixing_gain_db, self.dynamic_mixing_gain_db
            )
            for i in range(len(ref_audios))
        ]
        gain = [10 ** (g_db / 20.0) for g_db in gain_in_db]

        ref_audios = [ref * g for ref, g in zip(ref_audios, gain)]

        speech_mix = np.sum(np.array(ref_audios), axis=0)

        for i, ref in enumerate(ref_audios):
            data[f"{self.speech_ref_name_prefix}{i+1}"] = ref
        data[self.speech_name] = speech_mix

        return data

    def __call__(
        self, uid: str, data: Dict[str, Union[str, np.ndarray]]
    ) -> Dict[str, np.ndarray]:
        """Replace ``data`` with a freshly sampled dynamic mixture (train-time only)."""
        # TODO(Chenda): need to test for multi-channel data.
        assert (
            len(data[self.mixture_source_name].shape) == 1
        ), "Multi-channel input has not been tested"

        if self.train:
            data = self._mix_speech_(uid, data)

        assert check_return_type(data)
        return data


class EnhPreprocessor(CommonPreprocessor):
    """Preprocessor for Speech Enhancement (Enh) task."""

    def __init__(
        self,
        train: bool,
        rir_scp: str = None,
        rir_apply_prob: float = 1.0,
        noise_scp: str = None,
        noise_apply_prob: float = 1.0,
        noise_db_range: str = "3_10",
        short_noise_thres: float = 0.5,
        speech_volume_normalize: float = None,
        speech_name: str = "speech_mix",
        speech_ref_name_prefix: str = "speech_ref",
        noise_ref_name_prefix: str = "noise_ref",
        dereverb_ref_name_prefix: str = "dereverb_ref",
        use_reverberant_ref: bool = False,
        num_spk: int = 1,
        num_noise_type: int = 1,
        sample_rate: int = 8000,
        force_single_channel: bool = False,
    ):
        super().__init__(
            train=train,
            token_type=None,
            token_list=None,
            bpemodel=None,
            text_cleaner=None,
            g2p_type=None,
            unk_symbol="<unk>",
            space_symbol="<space>",
            non_linguistic_symbols=None,
            delimiter=None,
            rir_scp=rir_scp,
            rir_apply_prob=rir_apply_prob,
            noise_scp=noise_scp,
            noise_apply_prob=noise_apply_prob,
            noise_db_range=noise_db_range,
            short_noise_thres=short_noise_thres,
            speech_volume_normalize=speech_volume_normalize,
            speech_name=speech_name,
        )
        self.speech_ref_name_prefix = speech_ref_name_prefix
        self.noise_ref_name_prefix = noise_ref_name_prefix
        self.dereverb_ref_name_prefix = dereverb_ref_name_prefix
        self.use_reverberant_ref = use_reverberant_ref
        self.num_spk = num_spk
        self.num_noise_type = num_noise_type
        self.sample_rate = sample_rate
        self.force_single_channel = force_single_channel

        if self.speech_volume_normalize is not None:
            sps = speech_volume_normalize.split("_")
            if len(sps) == 1:
                self.volume_low, self.volume_high = float(sps[0])
            elif len(sps) == 2:
                self.volume_low, self.volume_high = float(sps[0]), float(
                    sps[1]
                )
            else:
                raise ValueError(
                    "Format error for --speech_volume_normalize: "
                    f"'{speech_volume_normalize}'"
                )

    def _ensure_2d(self, signal):
        """Recursively transpose array(s) to (Nmic, Time), adding the mic axis if needed."""
        if isinstance(signal, tuple):
            return tuple(self._ensure_2d(sig) for sig in signal)
        elif isinstance(signal, list):
            return [self._ensure_2d(sig) for sig in signal]
        else:
            # (Nmic, Time)
            return signal[None, :] if signal.ndim == 1 else signal.T

    def _get_early_signal(
        self, speech: np.ndarray, rir: np.ndarray, power: float
    ) -> np.ndarray:
        """Convolve ``speech`` with only the early (pre-50ms) part of ``rir``.

        Used to derive the "clean speech with early reflections" target,
        rescaled back to the original ``power``.
        """
        predelay = 50  # milliseconds
        dt = np.argmax(rir, axis=1).min()
        et = dt + (predelay * self.sample_rate) // 1000
        rir_early = rir[:, :et]
        speech2 = scipy.signal.convolve(speech, rir_early, mode="full")[
            :, : speech.shape[1]
        ]
        # Reverse mean power to the original power
        power2 = (speech2[detect_non_silence(speech2)] ** 2).mean()
        speech2 = np.sqrt(power / max(power2, 1e-10)) * speech2
        return speech2

    def _apply_to_all_signals(self, data_dict, func):
        """Apply ``func`` in place to the mixture, noise refs, speech refs, and dereverb refs."""
        data_dict[self.speech_name] = func(data_dict[self.speech_name])

        for n in range(self.num_noise_type):
            noise_name = self.noise_ref_name_prefix + str(n + 1)
            if noise_name in data_dict:
                data_dict[noise_name] = func(data_dict[noise_name])

        for spk in range(self.num_spk):
            speech_ref_name = self.speech_ref_name_prefix + str(spk + 1)
            if self.train or speech_ref_name in data_dict:
                data_dict[speech_ref_name] = func(data_dict[speech_ref_name])

            dereverb_ref_name = self.dereverb_ref_name_prefix + str(spk + 1)
            if dereverb_ref_name in data_dict:
                data_dict[dereverb_ref_name] = func(
                    data_dict[dereverb_ref_name]
                )

    def _apply_rir_augmentation(
        self,
        data: Dict[str, Union[str, np.ndarray]],
        speech_ref: List[np.ndarray],
        power_ref: List[float],
        dereverb_speech_ref: Optional[List[np.ndarray]],
        speech_mix: np.ndarray,
    ) -> np.ndarray:
        """Convolve each speaker's reference with an independently sampled RIR.

        Updates ``data`` in place with the (reverberant or early-reflection)
        speech references and, when requested, the dereverberated
        references. Returns the re-derived speech mixture.
        """
        if self.noise_ref_name_prefix + "1" in data:
            noise = data[self.noise_ref_name_prefix + "1"]
            np.testing.assert_allclose(
                np.squeeze(sum(speech_ref) + noise),
                np.squeeze(speech_mix),
            )
        else:
            np.testing.assert_allclose(
                np.squeeze(sum(speech_ref)), np.squeeze(speech_mix)
            )

        speech_ref, rir_ref = zip(
            *[
                self._convolve_rir(sp, power)
                for sp, power in zip(speech_ref, power_ref)
            ]
        )
        if self.force_single_channel:
            speech_ref = list(
                map(
                    lambda x: x if x.shape[0] == 1 else x[:1],
                    speech_ref,
                )
            )
            rir_ref = list(
                map(lambda x: x if x.shape[0] == 1 else x[:1], rir_ref)
            )

        if self.use_reverberant_ref:
            for spk in range(self.num_spk):
                suffix = str(spk + 1)
                speech_ref_name = self.speech_ref_name_prefix + suffix
                # (Time, Nmic)
                data[speech_ref_name] = speech_ref[spk].T

                if dereverb_speech_ref is not None:
                    if spk == 0 or len(dereverb_speech_ref) > 1:
                        dereverb_name = self.dereverb_ref_name_prefix + suffix
                        data[dereverb_name] = self._get_early_signal(
                            speech_ref[spk],
                            rir_ref[spk],
                            power_ref[spk],
                        ).T
        else:
            for spk in range(self.num_spk):
                suffix = str(spk + 1)
                speech_ref_name = self.speech_ref_name_prefix + suffix
                # clean speech with early reflections (Time, Nmic)
                data[speech_ref_name] = self._get_early_signal(
                    speech_ref[spk], rir_ref[spk], power_ref[spk]
                ).T

                if dereverb_speech_ref is not None:
                    if spk == 0 or len(dereverb_speech_ref) > 1:
                        dereverb_name = self.dereverb_ref_name_prefix + suffix
                        data[dereverb_name] = data[speech_ref_name]

        if self.noise_ref_name_prefix + "1" in data:
            speech_mix = sum(speech_ref) + noise
        else:
            speech_mix = sum(speech_ref)
        return speech_mix

    def _apply_noise_augmentation(
        self, data: Dict[str, Union[str, np.ndarray]], speech_mix: np.ndarray
    ) -> np.ndarray:
        """Add noise to the mixture, replacing any pre-existing ``noise_ref1``.

        Updates ``data`` in place with the new ``noise_ref1`` (and drops any
        extra noise-type refs, since only one noise clip is added here).
        Returns the noisy speech mixture.
        """
        if self.noise_ref_name_prefix + "1" in data:
            speech_mix -= data[self.noise_ref_name_prefix + "1"]
        power_mix = (speech_mix[detect_non_silence(speech_mix)] ** 2).mean()
        speech_mix, noise = self._add_noise(speech_mix, power_mix)
        if self.force_single_channel:
            if speech_mix.shape[0] > 1:
                speech_mix = speech_mix[:1]
            if noise.shape[0] > 1:
                noise = noise[:1]

        for n in range(1, self.num_noise_type):
            name = self.noise_ref_name_prefix + str(n + 1)
            data.pop(name, None)
        data[self.noise_ref_name_prefix + "1"] = noise.T
        return speech_mix

    def _speech_process(
        self, data: Dict[str, Union[str, np.ndarray]]
    ) -> Dict[str, Union[str, np.ndarray]]:
        """Augment the mixture (train-time RIR/noise), then normalize volume.

        At train time: derive per-speaker references, optionally convolve
        them with RIRs and re-derive the mixture, optionally add noise, then
        peak-normalize the mixture and squeeze all signals. At all times:
        optionally force single-channel, and optionally apply the
        configured (random at train / fixed at eval) volume normalization.
        """
        assert check_argument_types()

        if self.speech_name not in data:
            assert check_return_type(data)
            return data

        if self.train:
            # clean speech signal (Nmic, Time)
            speech_ref = [
                self._ensure_2d(data[self.speech_ref_name_prefix + str(i + 1)])
                for i in range(self.num_spk)
            ]

            # dereverberated (noisy) signal (Nmic, Time)
            if "dereverb_ref1" in data:
                dereverb_speech_ref = [
                    self._ensure_2d(
                        data[self.dereverb_ref_name_prefix + str(i + 1)]
                    )
                    for i in range(self.num_spk)
                    if self.dereverb_ref_name_prefix + str(i + 1) in data
                ]
                assert len(dereverb_speech_ref) in (1, self.num_spk), len(
                    dereverb_speech_ref
                )
            else:
                dereverb_speech_ref = None

            # Calc power on non silence region
            power_ref = [
                (sref[detect_non_silence(sref)] ** 2).mean()
                for sref in speech_ref
            ]

            speech_mix = self._ensure_2d(data[self.speech_name])
            # 1. Convolve RIR
            if (
                self.rirs is not None
                and self.rir_apply_prob >= np.random.random()
            ):
                speech_mix = self._apply_rir_augmentation(
                    data, speech_ref, power_ref, dereverb_speech_ref, speech_mix
                )

            # 2. Add Noise
            if (
                self.noises is not None
                and self.noise_apply_prob >= np.random.random()
            ):
                speech_mix = self._apply_noise_augmentation(data, speech_mix)

            speech_mix = speech_mix.T
            data[self.speech_name] = speech_mix
            ma = np.max(np.abs(speech_mix))
            if ma > 1.0:
                self._apply_to_all_signals(data, lambda x: x / ma)

            self._apply_to_all_signals(data, lambda x: x.squeeze())

        if self.force_single_channel:
            self._apply_to_all_signals(
                data, lambda x: x if x.ndim == 1 else x[:, 0]
            )

        if self.speech_volume_normalize is not None:
            if self.train:
                volume_scale = np.random.uniform(
                    self.volume_low, self.volume_high
                )
            else:
                # use a fixed scale to make it deterministic
                volume_scale = self.volume_low
            speech_mix = data[self.speech_name]
            ma = np.max(np.abs(speech_mix))
            self._apply_to_all_signals(data, lambda x: x * volume_scale / ma)

        assert check_return_type(data)
        return data


class SVSPreprocessor(AbsPreprocessor):
    """Preprocessor for Sing Voice Sythesis (SVS) task."""

    def __init__(
        self,
        train: bool,
        token_type: str = None,
        token_list: Union[Path, str, Iterable[str]] = None,
        bpemodel: Union[Path, str, Iterable[str]] = None,
        text_cleaner: Collection[str] = None,
        g2p_type: str = None,
        unk_symbol: str = "<unk>",
        space_symbol: str = "<space>",
        non_linguistic_symbols: Union[Path, str, Iterable[str]] = None,
        delimiter: str = None,
        singing_volume_normalize: float = None,
        singing_name: str = "singing",
        text_name: str = "text",
        label_name: str = "label",
        midi_name: str = "score",
        fs: np.int32 = 0,
        hop_length: np.int32 = 256,
        phn_seg: dict = {
            1: [1],
            2: [0.25, 1],
            3: [0.1, 0.5, 1],
            4: [0.05, 0.1, 0.5, 1],
        },
    ):
        super().__init__(train)
        self.train = train
        self.singing_name = singing_name
        self.text_name = text_name
        self.label_name = label_name
        self.midi_name = midi_name
        self.fs = fs
        self.hop_length = hop_length
        self.singing_volume_normalize = singing_volume_normalize
        self.phn_seg = phn_seg
        self.time_shift = hop_length / fs
        if token_type is not None:
            if token_list is None:
                raise ValueError(
                    "token_list is required if token_type is not None"
                )
            self.text_cleaner = TextCleaner(text_cleaner)

            self.tokenizer = build_tokenizer(
                token_type=token_type,
                bpemodel=bpemodel,
                delimiter=delimiter,
                space_symbol=space_symbol,
                non_linguistic_symbols=non_linguistic_symbols,
                g2p_type=g2p_type,
            )
            self.token_id_converter = TokenIDConverter(
                token_list=token_list,
                unk_symbol=unk_symbol,
            )
        else:
            self.text_cleaner = None
            self.tokenizer = None
            self.token_id_converter = None

    def _normalize_singing_volume(
        self, data: Dict[str, Union[str, np.ndarray, tuple]]
    ) -> Dict[str, Union[str, np.ndarray, tuple]]:
        """Peak-normalize ``singing_name`` to ``singing_volume_normalize``, if set."""
        if self.singing_name in data:
            if self.singing_volume_normalize is not None:
                singing = data[self.singing_name]
                ma = np.max(np.abs(singing))
                data[self.singing_name] = (
                    singing * self.singing_volume_normalize / ma
                )
        return data

    def _align_label_and_score(
        self, data: Dict[str, Union[str, np.ndarray, tuple]]
    ) -> Dict[str, Union[str, np.ndarray, tuple]]:
        """Align phoneme labels with the musical score into per-phone arrays.

        Only runs when both ``midi_name`` and ``label_name`` are present;
        consumes (pops) both, and adds "label", "midi", "duration_phn",
        "duration_ruled_phn", "duration_syb", "phn_cnt", and "slur", one
        entry per phone.
        """
        if not (self.midi_name in data and self.label_name in data):
            return data

        # Load label info
        lab_timeseq, text = data[self.label_name]
        lab_len = len(text)
        text = " ".join(text)
        text = self.text_cleaner(text)
        text = text.split(" ")
        text_ints = self.token_id_converter.tokens2ids(text)
        data.pop(self.label_name)

        label = np.zeros((lab_len))
        midi = np.zeros((lab_len))
        duration_phn = np.zeros((lab_len))
        duration_ruled_phn = np.zeros((lab_len))
        duration_syb = np.zeros((lab_len))
        slur = np.zeros((lab_len))
        # Load score info
        tempo, syb_info = data[self.midi_name]
        phn_cnt = []

        # Calculate features
        index_lab = 0

        for st, et, syb, note, phns in syb_info:
            dur = et - st
            _duration_syb = int(dur / self.time_shift + 0.5)
            phone = phns.split("_")
            phn_num = len(phone)
            phn_cnt.append(phn_num)
            pre_seg = 0
            for k in range(phn_num):
                _duration_ruled_phn = int(
                    (self.phn_seg[phn_num][k] - pre_seg)
                    * dur
                    / self.time_shift
                    + 0.5
                )
                pre_seg = self.phn_seg[phn_num][k]
                # timeseq from lab
                assert text[index_lab] == phone[k]
                _duration_phn = int(
                    (lab_timeseq[index_lab][1] - lab_timeseq[index_lab][0])
                    / self.time_shift
                    + 0.5
                )
                # phone level feature
                label[index_lab] = text_ints[index_lab]
                midi[index_lab] = note
                duration_phn[index_lab] = _duration_phn
                duration_ruled_phn[index_lab] = _duration_ruled_phn
                duration_syb[index_lab] = _duration_syb
                if syb == "—":
                    slur[index_lab] = 1
                else:
                    slur[index_lab] = 0
                index_lab += 1

        assert index_lab == lab_len
        data.pop(self.midi_name)

        phn_cnt = np.array(phn_cnt)
        label.astype(np.int64)
        midi.astype(np.int64)
        duration_phn.astype(np.int64)
        duration_syb.astype(np.int64)
        duration_ruled_phn.astype(np.int64)
        phn_cnt.astype(np.int64)
        slur.astype(np.int64)

        data["label"] = label
        data["midi"] = midi
        data["duration_phn"] = duration_phn
        data["duration_ruled_phn"] = duration_ruled_phn
        data["duration_syb"] = duration_syb
        data["phn_cnt"] = phn_cnt
        data["slur"] = slur
        return data

    def _process_text(
        self, data: Dict[str, Union[str, np.ndarray, tuple]]
    ) -> Dict[str, Union[str, np.ndarray, tuple]]:
        """Clean and tokenize ``text_name`` into integer ids, if not already an array."""
        if self.text_name in data and self.tokenizer is not None:
            # FIX ME (Yuning): wrong transfer happen in pyopenjtalk
            text = data[self.text_name]
            if not isinstance(text, np.ndarray):
                if not isinstance(text, str):
                    text = " ".join(text)
                text = self.text_cleaner(text)
                tokens = self.tokenizer.text2tokens(text)
                _text_ints = self.token_id_converter.tokens2ids(tokens)
                data[self.text_name] = np.array(_text_ints, dtype=np.int64)
        return data

    def __call__(
        self,
        uid: str,
        data: Dict[str, Union[str, np.ndarray, tuple]],
    ) -> Dict[str, np.ndarray]:
        """Normalize volume, align label/score to per-phone arrays, and tokenize text."""
        assert check_argument_types()

        data = self._normalize_singing_volume(data)
        data = self._align_label_and_score(data)

        # TODO(Yuning): Add score from midi

        data = self._process_text(data)
        return data


class TSEPreprocessor(EnhPreprocessor):
    """Preprocessor for Target Speaker Extraction."""

    def __init__(
        self,
        train: bool,
        train_spk2enroll: str = None,
        enroll_segment: int = None,
        load_spk_embedding: bool = False,
        load_all_speakers: bool = False,
        # inherited from EnhPreprocessor
        rir_scp: str = None,
        rir_apply_prob: float = 1.0,
        noise_scp: str = None,
        noise_apply_prob: float = 1.0,
        noise_db_range: str = "3_10",
        short_noise_thres: float = 0.5,
        speech_volume_normalize: float = None,
        speech_name: str = "speech_mix",
        speech_ref_name_prefix: str = "speech_ref",
        noise_ref_name_prefix: str = "noise_ref",
        dereverb_ref_name_prefix: str = "dereverb_ref",
        use_reverberant_ref: bool = False,
        num_spk: int = 1,
        num_noise_type: int = 1,
        sample_rate: int = 8000,
        force_single_channel: bool = False,
    ):
        super().__init__(
            train,
            rir_scp=rir_scp,
            rir_apply_prob=rir_apply_prob,
            noise_scp=noise_scp,
            noise_apply_prob=noise_apply_prob,
            noise_db_range=noise_db_range,
            short_noise_thres=short_noise_thres,
            speech_volume_normalize=speech_volume_normalize,
            speech_name=speech_name,
            speech_ref_name_prefix=speech_ref_name_prefix,
            noise_ref_name_prefix=noise_ref_name_prefix,
            dereverb_ref_name_prefix=dereverb_ref_name_prefix,
            use_reverberant_ref=use_reverberant_ref,
            num_spk=num_spk,
            num_noise_type=num_noise_type,
            sample_rate=sample_rate,
            force_single_channel=force_single_channel,
        )
        # If specified, the enrollment will be chomped to the specified length
        self.enroll_segment = enroll_segment
        # If True, the speaker embedding will be loaded instead of enrollment audios
        self.load_spk_embedding = load_spk_embedding
        # If False, only one of the speakers in each mixture sample will be loaded
        self.load_all_speakers = load_all_speakers

        if train and rir_scp is not None and rir_apply_prob > 0:
            logging.warning(
                "Be cautious when applying RIRs on the fly in the TSE task! "
                "Please ensure `speech_ref` sums up to `speech_mix` for each sample."
            )
        print(train_spk2enroll, flush=True)
        if train:
            if train_spk2enroll is None:
                logging.info("Using fixed enrollment for each sample")
                self.train_spk2enroll = None
            else:
                logging.info(
                    "Using dynamically sampled enrollment for each sample"
                )
                with open(train_spk2enroll, "r", encoding="utf-8") as f:
                    # {spkID: [(uid1, path1), (uid2, path2), ...]}
                    self.train_spk2enroll = json.load(f)
        else:
            self.train_spk2enroll = None

    def _read_audio_segment(
        self, path: str, seg_len: Optional[int] = None
    ) -> np.ndarray:
        """Read (a random ``seg_len``-sample crop of) the mono audio at ``path``.

        Shorter files are wrap-padded to ``seg_len`` at a random offset;
        longer files are cropped starting at a random offset.
        """
        with soundfile.SoundFile(path) as f:
            if seg_len is None or f.frames == seg_len:
                audio = f.read(dtype=np.float32, always_2d=True)
            elif f.frames < seg_len:
                offset = np.random.randint(0, seg_len - f.frames)
                # audio: (Time, Nmic)
                audio = f.read(dtype=np.float32, always_2d=True)
                # Repeat audio
                audio = np.pad(
                    audio,
                    [(offset, seg_len - f.frames - offset), (0, 0)],
                    mode="wrap",
                )
            else:
                offset = np.random.randint(0, f.frames - seg_len)
                f.seek(offset)
                # audio: (Time, Nmic)
                audio = f.read(seg_len, dtype=np.float32, always_2d=True)
            if len(audio) != seg_len:
                raise RuntimeError(f"Something wrong: {path}")
        return audio[:, 0]

    def _pick_target_speaker_and_trim_refs(
        self,
        data: Dict[str, Union[str, np.ndarray]],
        ref_names: List[str],
        num_spk: int,
    ) -> int:
        """Randomly pick one target speaker and drop the other speech_ref entries.

        Avoids picking a speaker whose enrollment is the dummy-label marker.

        Returns:
            The picked speaker index. ``data[ref_names[0]]`` ends up holding
            that speaker's reference and the other ``ref_names`` are removed.
        """
        spk = np.random.randint(0, num_spk)
        while (
            hasattr(self, "dummy_label")
            and data[f"enroll_ref{spk+1}"]
            == f"*{self.dummy_label} {self.dummy_label}"
        ):
            spk = np.random.randint(0, num_spk)

        for i, name in enumerate(ref_names):
            if i == 0:
                data[name] = data[ref_names[spk]]
            else:
                data.pop(name)
                continue
        return spk

    def _load_enrollment_audio(
        self, data: Dict[str, Union[str, np.ndarray]], name: str, i: int
    ) -> None:
        """Resolve one train-time enrollment entry into an array, in place.

        Supports three formats for ``data[name]``: a plain audio/embedding
        path, the dummy-label marker (zeroed out), or a
        ``*MIXTURE_UID SPEAKER_ID`` marker that samples a random enrollment
        utterance for that speaker from ``self.train_spk2enroll`` (retrying
        until it differs from the mixture's own uid).
        """
        if self.train_spk2enroll is None:
            # normal format in `enroll_spk?.scp`:
            # MIXTURE_UID /path/to/enrollment_or_embedding
            aux_audio = data[name]
        elif (
            hasattr(self, "dummy_label")
            and data[name] == f"*{self.dummy_label} {self.dummy_label}"
        ):
            data[name] = np.zeros(1, dtype=data["speech_mix"].dtype)
            return
        else:
            # a special format in `enroll_spk?.scp`:
            # MIXTURE_UID *UID SPEAKER_ID
            assert data[name].startswith("*"), data[name]
            cur_uid, spkid = data[name][1:].strip().split(maxsplit=1)
            aux_uid, aux_audio = random.choice(self.train_spk2enroll[spkid])
            while aux_uid == cur_uid:
                aux_uid, aux_audio = random.choice(self.train_spk2enroll[spkid])
        if getattr(self, "load_spk_embedding", False):
            data[name] = np.load(aux_audio)[None, :]  # force 2D
        elif self.enroll_segment:
            try:
                data[name] = self._read_audio_segment(
                    aux_audio, self.enroll_segment
                )
            except:
                raise RuntimeError(f"Something wrong 1: {aux_audio}, i:{i}")
        else:
            data[name] = soundfile.read(aux_audio)[0]

    def _load_enrollment_audio_eval(
        self, data: Dict[str, Union[str, np.ndarray]], name: str
    ) -> None:
        """Resolve one enrollment entry at inference/eval time, in place.

        Unlike :meth:`_load_enrollment_audio`, there is no random sampling:
        the given path/embedding is loaded as-is (or zeroed for a
        dummy/stats-collection marker).
        """
        if data[name].startswith("*"):
            # in case of collecting stats for training data
            data[name] = np.zeros(1, dtype=data["speech_mix"].dtype)
        elif hasattr(self, "dummy_label") and self.dummy_label == data[name]:
            # in case of collecting stats for training data
            data[name] = np.zeros(1, dtype=data["speech_mix"].dtype)
        else:
            if getattr(self, "load_spk_embedding", False):
                data[name] = np.load(data[name])[None, :]  # force 2D
            elif self.enroll_segment:
                data[name] = self._read_audio_segment(
                    data[name], self.enroll_segment
                )
            else:
                data[name] = soundfile.read(data[name])[0]

    def _speech_process(
        self, uid: str, data: Dict[str, Union[str, np.ndarray]]
    ) -> Dict[str, Union[str, np.ndarray]]:
        """Load enrollment audio/embeddings for the target speaker(s).

        At train time, optionally subsamples to a single target speaker
        (see :meth:`_pick_target_speaker_and_trim_refs`) and resolves each
        remaining ``enroll_ref{n}`` entry via
        :meth:`_load_enrollment_audio`. At eval time, every ``enroll_ref{n}``
        is resolved via :meth:`_load_enrollment_audio_eval` (no random
        subsampling or enrollment sampling).
        """
        assert check_argument_types()

        ref_names = [k for k in data.keys() if re.match(r"speech_ref\d+", k)]
        num_spk = len(ref_names)

        aux_names = [k for k in data.keys() if re.match(r"enroll_ref\d+", k)]
        if self.train:
            assert len(ref_names) == len(aux_names), (
                len(ref_names),
                len(aux_names),
            )
            if not self.load_all_speakers:
                # only load one target-speaker data
                spk = self._pick_target_speaker_and_trim_refs(
                    data, ref_names, num_spk
                )

            for i, name in enumerate(aux_names):
                if not self.load_all_speakers:
                    if i == 0:
                        data[name] = data[aux_names[spk]]
                    else:
                        data.pop(name)
                        continue
                self._load_enrollment_audio(data, name, i)
        else:
            for name in aux_names:
                self._load_enrollment_audio_eval(data, name)
        assert check_return_type(data)
        return data

    def __call__(
        self, uid: str, data: Dict[str, Union[str, np.ndarray]]
    ) -> Dict[str, np.ndarray]:
        """Augment speech (via ``EnhPreprocessor``) then load enrollment audio."""
        assert check_argument_types()

        data = super()._speech_process(data)
        data = self._speech_process(uid, data)
        return data


class EnhTsePreprocessor(TSEPreprocessor):
    """Preprocessor for unified speech enhancement/separation/extraction."""

    def __init__(
        self,
        train: bool,
        task: str,
        dummy_label: str = "dummy",
        speech_segment: int = None,
        # inherited from TSEPreprocessor
        train_spk2enroll: str = None,
        enroll_segment: int = None,
        load_spk_embedding: bool = False,
        load_all_speakers: bool = False,
        # inherited from EnhPreprocessor
        rir_scp: str = None,
        rir_apply_prob: float = 1.0,
        noise_scp: str = None,
        noise_apply_prob: float = 1.0,
        noise_db_range: str = "3_10",
        short_noise_thres: float = 0.5,
        speech_volume_normalize: float = None,
        speech_name: str = "speech_mix",
        speech_ref_name_prefix: str = "speech_ref",
        noise_ref_name_prefix: str = "noise_ref",
        dereverb_ref_name_prefix: str = "dereverb_ref",
        use_reverberant_ref: bool = False,
        num_spk: int = 1,
        num_noise_type: int = 1,
        sample_rate: int = 8000,
        force_single_channel: bool = False,
    ):
        super().__init__(
            train,
            train_spk2enroll=train_spk2enroll,
            enroll_segment=enroll_segment,
            load_spk_embedding=load_spk_embedding,
            load_all_speakers=load_all_speakers,
            rir_scp=rir_scp,
            rir_apply_prob=rir_apply_prob,
            noise_scp=noise_scp,
            noise_apply_prob=noise_apply_prob,
            noise_db_range=noise_db_range,
            short_noise_thres=short_noise_thres,
            speech_volume_normalize=speech_volume_normalize,
            speech_name=speech_name,
            speech_ref_name_prefix=speech_ref_name_prefix,
            noise_ref_name_prefix=noise_ref_name_prefix,
            dereverb_ref_name_prefix=dereverb_ref_name_prefix,
            use_reverberant_ref=use_reverberant_ref,
            num_spk=num_spk,
            num_noise_type=num_noise_type,
            sample_rate=sample_rate,
            force_single_channel=force_single_channel,
        )
        # This defines the dummy label for handling the variable number of speakers
        self.dummy_label = dummy_label
        self.speech_segment = speech_segment
        self.load_enrollment = "tse" in task

    def _zero_out_dummy_speech_refs(
        self, data: Dict[str, Union[str, np.ndarray]]
    ) -> List[str]:
        """Zero out ``speech_ref{n}`` entries equal to the dummy label.

        Returns the names of the real (non-dummy) ``speech_ref`` entries.
        """
        to_remove, ref_names = [], []
        for k, v in data.items():
            if re.match(r"speech_ref\d+", k):
                if v == self.dummy_label:
                    # remove dummy references
                    to_remove.append(k)
                else:
                    ref_names.append(k)
        for k in to_remove:
            if k in data:
                data[k] = np.zeros(1)
        return ref_names

    def _read_speech_ref_segment(
        self,
        data: Dict[str, Union[str, np.ndarray]],
        name: str,
        spk: int,
        start: int,
        frames: int,
        mix_len: int,
    ) -> np.ndarray:
        """Read the ``[start, start + frames)`` segment of one speech_ref file.

        Falls back to reading the whole file and slicing in numpy (logging a
        warning) if the soundfile-level segment read fails.
        """
        try:
            audio = soundfile.read(
                data[name],
                start=start,
                frames=frames,
                dtype=np.float32,
                always_2d=False,
            )[0]
        except:
            audio = soundfile.read(data[name], dtype=np.float32, always_2d=False)[0]
            if frames != -1:
                assert self.speech_segment is not None, self.speech_segment
                audio = audio[..., start : start + frames]
            logging.warning(
                f"Something wrong {name}, spk{spk}, {mix_len} {start}, {frames}"
            )
        return audio

    def _speech_process(
        self, uid: str, data: Dict[str, Union[str, np.ndarray]]
    ) -> Dict[str, Union[str, np.ndarray]]:
        """Zero dummy references, then crop a shared segment across mix + refs.

        Retries with a new random offset (see ``self.speech_segment``) if a
        cropped reference turns out to be all-zero, so that the sampled
        window always contains signal for every real speaker.
        """
        assert check_argument_types()
        ref_names = self._zero_out_dummy_speech_refs(data)

        # speech segment for sequence-based iterator
        has_zero_ref = (
            True  # to avoid zero-reference when using partly-overlapped data
        )
        mix_len = data["speech_mix"].shape[-1]
        offset = mix_len - self.speech_segment
        while has_zero_ref:
            loaded_audios = {}
            if self.train and self.speech_segment is not None:
                mix_len = data["speech_mix"].shape[-1]
                if offset > 0:
                    start = np.random.randint(0, offset)
                    frames = self.speech_segment
                    loaded_audios["speech_mix"] = data["speech_mix"][
                        ..., start : start + frames
                    ]
                else:
                    start, frames = 0, -1
                    loaded_audios["speech_mix"] = data["speech_mix"]
            else:
                start, frames = 0, -1
                loaded_audios["speech_mix"] = data["speech_mix"]

            num_spk = len(ref_names)
            data["num_spk"] = np.array([num_spk])
            zero_ref = []
            for spk in range(1, num_spk + 1):
                name = f"speech_ref{spk}"
                # make sure the dummy references only exist after the real ones
                assert (
                    name in data
                ), "dummy reference must appear after the real ones"
                # Read the speech references
                loaded_audios[name] = self._read_speech_ref_segment(
                    data, name, spk, start, frames, mix_len
                )
                assert (
                    loaded_audios["speech_mix"].shape[-1]
                    == loaded_audios[name].shape[-1]
                ), (
                    mix_len,
                    data["speech_mix"].shape[-1],
                    loaded_audios[name].shape[-1],
                    start,
                    frames,
                )
                zero_ref.append((abs(loaded_audios[name]).sum() == 0.0))

            # if there is no zero-reference, break the loop
            if not np.any(zero_ref):
                has_zero_ref = False
                for spk in range(1, num_spk + 1):
                    name = f"speech_ref{spk}"
                    data[name] = loaded_audios[name]
                data["speech_mix"] = loaded_audios["speech_mix"]
            else:
                offset = start

        assert check_return_type(data)
        return data

    def __call__(
        self, uid: str, data: Dict[str, Union[str, np.ndarray]]
    ) -> Dict[str, np.ndarray]:
        """Crop mix/refs to a shared segment, then load enrollment if requested.

        When ``self.load_enrollment`` is False (task doesn't need target
        enrollment), any ``enroll_ref{n}`` fields are dropped instead.
        """
        assert check_argument_types()

        data = self._speech_process(uid, data)
        if self.load_enrollment:
            data = super()._speech_process(uid, data)
        else:
            to_remove = []
            for k, v in data.items():
                if re.match(r"enroll_ref\d+", k):
                    to_remove.append(k)
            for k in to_remove:
                if k in data:
                    data.pop(k)
        return data
