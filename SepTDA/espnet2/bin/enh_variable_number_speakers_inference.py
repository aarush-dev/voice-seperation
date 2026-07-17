#!/usr/bin/env python3
"""CLI for variable speaker-count target-speech extraction / separation inference.

Unlike a fixed-speaker-count enhancement model, the model run here (built by
``TargetSpeakerExtractionAndEnhancementTask``) can be asked to estimate its own
number of active speakers for each mixture, rather than always producing a
fixed number of output streams. This script therefore has to reconcile, for
every utterance, the number of sources the model *estimated* with the number
of reference sources actually present:

* If the model under-estimates the speaker count, the missing streams are
  filled with near-silent placeholders so the reference set is still fully
  covered.
* If the model over-estimates, the extra streams are logged and discarded
  after permutation matching.

The inference flow is:

1. Build ``SeparateSpeech``, which wraps the trained encoder/extractor/decoder
   and (optionally) segments very long mixtures to avoid GPU OOM.
2. Iterate over the dataset; for each mixture, collect the non-dummy speaker
   references, run separation (optionally informed of the true speaker count
   via ``--use_true_nspk``), and align the estimated sources to the
   references with SI-SDR-based permutation search (``select_sources``).
3. Write the aligned waveforms to per-speaker SCP/wav files and record the
   estimated speaker count (``Est_num_spk``) for later scoring.
"""
import argparse
import logging
import re
import sys
from itertools import chain
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import fast_bss_eval
import humanfriendly
import numpy as np
import torch
import tqdm
import yaml
from typeguard import check_argument_types

from espnet2.enh.loss.criterions.tf_domain import FrequencyDomainMSE
from espnet2.enh.loss.criterions.time_domain import SISNRLoss
from espnet2.enh.loss.wrappers.pit_solver import PITSolver
from espnet2.fileio.datadir_writer import DatadirWriter
from espnet2.fileio.sound_scp import SoundScpWriter
from espnet2.tasks.enh_tse_ss import (
    TargetSpeakerExtractionAndEnhancementTask as TSESSTask,
)
from espnet2.torch_utils.device_funcs import to_device
from espnet2.torch_utils.set_all_random_seed import set_all_random_seed
from espnet2.train.abs_espnet_model import AbsESPnetModel
from espnet2.utils import config_argparse
from espnet2.utils.types import str2bool, str2triple_str, str_or_none
from espnet.utils.cli_utils import get_commandline_args


def get_train_config(
    train_config: Optional[Union[Path, str]],
    model_file: Optional[Union[Path, str]] = None,
) -> Path:
    """Resolve the training config path, defaulting to ``config.yaml`` next to the model."""
    if train_config is None:
        assert model_file is not None, (
            "The argument 'model_file' must be provided "
            "if the argument 'train_config' is not specified."
        )
        train_config = Path(model_file).parent / "config.yaml"
    else:
        train_config = Path(train_config)
    return train_config


def recursive_dict_update(
    dict_org: Dict[str, Any],
    dict_patch: Dict[str, Any],
    verbose: bool = False,
    log_prefix: str = "",
) -> None:
    """Update `dict_org` with `dict_patch` in-place recursively."""
    for key, value in dict_patch.items():
        if key not in dict_org:
            if verbose:
                logging.info(
                    "Overwriting config: [{}{}]: None -> {}".format(
                        log_prefix, key, value
                    )
                )
            dict_org[key] = value
        elif isinstance(value, dict):
            recursive_dict_update(
                dict_org[key], value, verbose=verbose, log_prefix=f"{key}."
            )
        else:
            if verbose and dict_org[key] != value:
                logging.info(
                    "Overwriting config: [{}{}]: {} -> {}".format(
                        log_prefix, key, dict_org[key], value
                    )
                )
            dict_org[key] = value


def build_model_from_args_and_file(
    task, args: argparse.Namespace, model_file: Optional[Union[Path, str]], device: str
) -> AbsESPnetModel:
    """Build a task model from a parsed args namespace and load its weights."""
    model = task.build_model(args)
    if not isinstance(model, AbsESPnetModel):
        raise RuntimeError(
            f"model must inherit {AbsESPnetModel.__name__}, but got {type(model)}"
        )
    model.to(device)
    if model_file is not None:
        if device == "cuda":
            # NOTE(kamo): "cuda" for torch.load always indicates cuda:0
            #   in PyTorch<=1.4
            device = f"cuda:{torch.cuda.current_device()}"
        model.load_state_dict(torch.load(model_file, map_location=device))
    return model


def _zero_pad_missing_sources(
    estimates: torch.Tensor, num_missing: int
) -> torch.Tensor:
    """Pad the estimated-source tensor with near-silent placeholder sources.

    Used when the model estimates fewer sources than there are references
    (under-separation), so that every reference still has a candidate
    estimate to be scored against during permutation search.
    """
    if num_missing <= 0:
        return estimates
    filler = (
        torch.zeros((num_missing, estimates.shape[-1]), dtype=estimates.dtype) + 1e-8
    )
    return torch.cat((estimates, filler), dim=0)


def select_sources(
    ref: List[torch.Tensor],
    inf: List[torch.Tensor],
    mix: torch.Tensor,
    writer: DatadirWriter,
    max_num_spk: int,
    key: str,
) -> List[np.ndarray]:
    """Align estimated sources to references and record the estimated speaker count.

    The model may estimate a different number of active speakers than the
    ground-truth mixture actually contains:

    * Under-separation (fewer estimates than references): the missing
      estimates are filled with near-silent placeholders (see
      ``_zero_pad_missing_sources``) so every reference has a candidate to be
      matched against.
    * Over-separation (more estimates than references): logged, and the
      extra estimates are discarded once the best permutation is found.

    The permutation that maximizes SI-SDR between references and estimates is
    applied so that estimate ``i`` corresponds to reference ``i``.

    Returns:
        List of ``len(ref)`` numpy arrays, each shaped ``(1, n_samples)``.
    """
    true_num_spk = len(ref)
    ref_cat = torch.cat(ref, dim=0)
    est_num_spk = len(inf)
    inf_cat = torch.cat(inf, dim=0).to(ref_cat.device)

    if est_num_spk < true_num_spk:
        inf_cat = _zero_pad_missing_sources(inf_cat, true_num_spk - est_num_spk)
    elif est_num_spk > true_num_spk:
        logging.info(
            f"Over-separation (Est: {est_num_spk}, Ref: {true_num_spk}). "
            "Discard over-separated sources"
        )

    _, perm = fast_bss_eval.si_sdr(ref_cat, inf_cat, return_perm=True)
    inf_cat = inf_cat[perm]

    assert ref_cat.shape == inf_cat.shape, f"{ref_cat.shape}, {inf_cat.shape}"
    writer["Est_num_spk"][key] = str(est_num_spk)

    assert inf_cat.ndim == 2, (
        "shape should be (n_spk, n_samples) and there must not be the batch dimension"
    )
    return [w[None].cpu().numpy() for w in inf_cat]


class SeparateSpeech:
    """Wraps a trained TSE/SS model to separate a mixture into speaker streams.

    Handles building the model (optionally overriding some of its submodule
    configs at inference time), running direct or segment-wise separation
    (segment-wise separation is a fallback used when a mixture is too long to
    fit in GPU memory in one pass), and stitching segment-wise results back
    into full-length waveforms.

    Examples:
        >>> import soundfile
        >>> separate_speech = SeparateSpeech("enh_config.yml", "enh.pth")
        >>> audio, rate = soundfile.read("speech.wav")
        >>> separate_speech(audio)
        [separated_audio1, separated_audio2, ...]

    """

    def __init__(
        self,
        train_config: Union[Path, str] = None,
        model_file: Union[Path, str] = None,
        inference_config: Union[Path, str] = None,
        segment_size: Optional[float] = None,
        hop_size: Optional[float] = None,
        normalize_segment_scale: bool = False,
        show_progressbar: bool = False,
        ref_channel: Optional[int] = None,
        normalize_output_wav: bool = False,
        use_true_nspk: bool = False,
        device: str = "cpu",
        dtype: str = "float32",
    ):
        assert check_argument_types()

        enh_model, enh_train_args = self._resolve_enh_model(
            train_config, model_file, inference_config, device
        )
        enh_model.to(dtype=getattr(torch, dtype)).eval()
        if hasattr(enh_model.extractor, "multi_decode"):
            # Always take the single-decode inference path, regardless of how
            # the model was trained.
            enh_model.extractor.multi_decode = False

        self.device = device
        self.dtype = dtype
        self.enh_train_args = enh_train_args
        self.enh_model = enh_model

        # only used when processing long speech, i.e.
        # segment_size is not None and hop_size is not None
        self.segment_size = segment_size
        self.hop_size = hop_size
        self.normalize_segment_scale = normalize_segment_scale
        self.normalize_output_wav = normalize_output_wav
        self.show_progressbar = show_progressbar

        self.ref_channel = self._resolve_ref_channel(enh_model, ref_channel)

        self.segmenting = segment_size is not None and hop_size is not None
        if self.segmenting:
            logging.info("Perform segment-wise speech")
            logging.info(
                "Segment length = {} sec, hop length = {} sec".format(
                    segment_size, hop_size
                )
            )
        else:
            logging.info("Perform direct speech on the input")

        # if true, number of speakers are informed to model
        self.use_true_nspk = use_true_nspk

    @staticmethod
    def _resolve_enh_model(
        train_config: Optional[Union[Path, str]],
        model_file: Optional[Union[Path, str]],
        inference_config: Optional[Union[Path, str]],
        device: str,
    ) -> Tuple[AbsESPnetModel, argparse.Namespace]:
        """Build the enhancement/extraction model and its training args.

        If ``inference_config`` is given, the encoder/extractor/decoder
        sections of the training config are overwritten before the model is
        built, allowing inference-time architecture tweaks without
        retraining.
        """
        if inference_config is None:
            return TSESSTask.build_model_from_file(train_config, model_file, device)

        train_config = get_train_config(train_config, model_file=model_file)
        with train_config.open("r", encoding="utf-8") as f:
            train_args = yaml.safe_load(f)

        with Path(inference_config).open("r", encoding="utf-8") as f:
            infer_args = yaml.safe_load(f)

        supported_keys = list(
            chain(*[[k, k + "_conf"] for k in ("encoder", "extractor", "decoder")])
        )
        for k in infer_args.keys():
            if k not in supported_keys:
                raise ValueError(
                    "Only the following top-level keys are supported: %s"
                    % ", ".join(supported_keys)
                )

        recursive_dict_update(train_args, infer_args, verbose=True)
        enh_train_args = argparse.Namespace(**train_args)
        enh_model = build_model_from_args_and_file(
            TSESSTask, enh_train_args, model_file, device
        )
        return enh_model, enh_train_args

    @staticmethod
    def _resolve_ref_channel(
        enh_model: AbsESPnetModel, ref_channel: Optional[int]
    ) -> int:
        """Optionally overwrite the model's reference channel for multi-channel input."""
        if ref_channel is None:
            return enh_model.ref_channel
        logging.info(
            "Overwrite enh_model.separator.ref_channel with {}".format(ref_channel)
        )
        enh_model.separator.ref_channel = ref_channel
        if hasattr(enh_model.separator, "beamformer"):
            enh_model.separator.beamformer.ref_channel = ref_channel
        return ref_channel

    def _prepare_input(
        self, speech_mix: Union[torch.Tensor, np.ndarray]
    ) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """Cast, batch-length, and move the input mixture to the target device."""
        if isinstance(speech_mix, np.ndarray):
            speech_mix = torch.as_tensor(speech_mix)
        assert speech_mix.dim() > 1, speech_mix.size()

        batch_size = speech_mix.size(0)
        speech_mix = speech_mix.to(getattr(torch, self.dtype))
        lengths = speech_mix.new_full(
            [batch_size], dtype=torch.long, fill_value=speech_mix.size(1)
        )

        speech_mix = to_device(speech_mix, device=self.device)
        lengths = to_device(lengths, device=self.device)
        return speech_mix, lengths, batch_size

    def _separate(
        self,
        speech_mix: torch.Tensor,
        lengths: torch.Tensor,
        num_spk: Optional[int],
    ) -> List[torch.Tensor]:
        """Run a single direct (non-segmented) forward pass through the model."""
        feats, f_lens = self.enh_model.encoder(speech_mix, lengths)
        feats, _, _ = self.enh_model.extractor(feats, f_lens, num_spk=num_spk)
        return [self.enh_model.decoder(f, lengths)[0] for f in feats]

    def _match_segment_energy(
        self,
        speech_seg: torch.Tensor,
        processed_wav: List[torch.Tensor],
        valid_length: int,
    ) -> List[torch.Tensor]:
        """Rescale separated segment waveforms to match the input mixture energy."""
        speech_seg_ref = (
            speech_seg[:, self.ref_channel] if speech_seg.dim() > 2 else speech_seg
        )
        mix_energy = torch.sqrt(
            torch.mean(speech_seg_ref[:, :valid_length].pow(2), dim=1, keepdim=True)
        )
        enh_energy = torch.sqrt(
            torch.mean(sum(processed_wav)[:, :valid_length].pow(2), dim=1, keepdim=True)
        )
        return [w * (mix_energy / enh_energy) for w in processed_wav]

    def _run_windowed_separation(
        self,
        speech_mix: torch.Tensor,
        lengths: torch.Tensor,
        fs: int,
        batch_size: int,
        num_spk: Optional[int],
    ) -> Tuple[List[torch.Tensor], int, int]:
        """Separate a long mixture segment-by-segment (used as an OOM fallback).

        Returns:
            enh_waves: per-segment separated waveforms, each shaped
                ``(num_spk, batch, segment_samples)``.
            overlap_length: number of overlapping samples between adjacent
                segments.
            valid_length_last_segment: the number of non-padded samples in
                the final (possibly shorter) segment.
        """
        overlap_length = int(np.round(fs * (self.segment_size - self.hop_size)))
        num_segments = int(
            np.ceil((speech_mix.size(1) - overlap_length) / (self.hop_size * fs))
        )
        segment_samples = int(self.segment_size * fs)
        valid_length = segment_samples
        pad_shape = speech_mix[:, :segment_samples].shape

        enh_waves = []
        range_ = trange if self.show_progressbar else range
        for i in range_(num_segments):
            start = int(i * self.hop_size * fs)
            end = start + segment_samples
            if end >= lengths[0]:
                # last segment: shorter than segment_samples, zero-pad the tail
                end = lengths[0]
                speech_seg = speech_mix.new_zeros(pad_shape)
                valid_length = end - start
                speech_seg[:, :valid_length] = speech_mix[:, start:end]
            else:
                valid_length = segment_samples
                speech_seg = speech_mix[:, start:end]  # B x T [x C]

            lengths_seg = speech_mix.new_full(
                [batch_size], dtype=torch.long, fill_value=segment_samples
            )
            feats, f_lens = self.enh_model.encoder(speech_seg, lengths_seg)
            feats, _, _ = self.enh_model.extractor(feats, f_lens, num_spk=num_spk)
            processed_wav = [
                self.enh_model.decoder(f, lengths_seg)[0] for f in feats
            ]

            if self.normalize_segment_scale:
                processed_wav = self._match_segment_energy(
                    speech_seg, processed_wav, valid_length
                )

            # List[torch.Tensor(num_spk, B, T)]
            enh_waves.append(torch.stack(processed_wav, dim=0))

        return enh_waves, overlap_length, valid_length

    @staticmethod
    def _match_speaker_counts(
        front: torch.Tensor, back: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Zero-pad whichever of two adjacent segments estimated fewer speakers."""
        front_numspk = front.shape[0]
        back_numspk = back.shape[0]
        if front_numspk < back_numspk:
            filler = (
                torch.zeros(
                    (back_numspk - front_numspk, front.shape[1], front.shape[2]),
                    dtype=front.dtype,
                )
                + 1e-8
            )
            front = torch.cat((front, filler.to(front.device)), dim=0)
        elif front_numspk > back_numspk:
            filler = (
                torch.zeros(
                    (front_numspk - back_numspk, back.shape[1], back.shape[2]),
                    dtype=back.dtype,
                )
                + 1e-8
            )
            back = torch.cat((back, filler.to(front.device)), dim=0)
        return front, back

    def _stitch_windowed_separation(
        self,
        enh_waves: List[torch.Tensor],
        overlap_length: int,
        valid_length_last_segment: int,
        speech_mix: torch.Tensor,
        batch_size: int,
    ) -> Tuple[torch.Tensor, ...]:
        """Overlap-add per-segment estimates into full-length waveforms.

        Adjacent segments may disagree on the estimated speaker count and on
        the ordering of streams, so for each new segment this:

        1. zero-pads the segment with fewer estimated speakers so the counts
           match (``_match_speaker_counts``),
        2. finds the permutation that best aligns the new segment to the tail
           of the stitched-so-far waveform via SI-SDR on the overlapping
           region,
        3. averages the overlapping region and appends the remainder.
        """
        num_segments = len(enh_waves)
        waves = enh_waves[0]
        for i in range(1, num_segments):
            waves, enh_waves[i] = self._match_speaker_counts(waves, enh_waves[i])

            _, perm = fast_bss_eval.si_sdr(
                waves[:, :, -overlap_length:],
                enh_waves[i][:, :, :overlap_length],
                return_perm=True,
            )
            for batch in range(batch_size):
                enh_waves[i][:, batch] = enh_waves[i][perm[batch], batch]

            if i == num_segments - 1:
                enh_waves[i][:, :, valid_length_last_segment:] = 0
                residual = enh_waves[i][:, :, overlap_length:valid_length_last_segment]
            else:
                residual = enh_waves[i][:, :, overlap_length:]

            # overlap-and-add (average over the overlapped part)
            waves[:, :, -overlap_length:] = (
                waves[:, :, -overlap_length:] + enh_waves[i][:, :, :overlap_length]
            ) / 2
            waves = torch.cat([waves, residual], dim=2)

        assert waves.size(2) == speech_mix.size(1), (waves.shape, speech_mix.shape)
        return torch.unbind(waves, dim=0)

    def _separate_by_segments(
        self,
        speech_mix: torch.Tensor,
        lengths: torch.Tensor,
        fs: int,
        batch_size: int,
        num_spk: Optional[int],
    ) -> Tuple[torch.Tensor, ...]:
        """Segment-wise separation + stitching, used when a direct pass runs OOM."""
        print(f"segementing audio length: {lengths[0]} > {self.segment_size * fs}")
        enh_waves, overlap_length, valid_length_last_segment = (
            self._run_windowed_separation(
                speech_mix, lengths, fs, batch_size, num_spk
            )
        )
        return self._stitch_windowed_separation(
            enh_waves, overlap_length, valid_length_last_segment, speech_mix, batch_size
        )

    def _postprocess_waves(
        self, waves: Sequence[torch.Tensor]
    ) -> List[torch.Tensor]:
        """Optionally peak-normalize each separated waveform to [-0.9, 0.9]."""
        if self.normalize_output_wav:
            return [(w / abs(w).max(dim=1, keepdim=True)[0] * 0.9) for w in waves]
        return list(waves)

    @torch.no_grad()
    def __call__(
        self,
        speech_mix: Union[torch.Tensor, np.ndarray],
        fs: int = 8000,
        num_spk: int = None,
        key: str = None,
    ) -> List[torch.Tensor]:
        """Separate one mixture into its estimated speaker streams.

        Args:
            speech_mix: Input speech data (Batch, Nsamples [, Channels])
            fs: sample rate
            num_spk: true number of speakers, used only if the instance was
                constructed with ``use_true_nspk=True``
            key: utterance id, used only for logging/error messages
        Returns:
            [separated_audio1, separated_audio2, ...]

        """
        assert check_argument_types()

        speech_mix, lengths, batch_size = self._prepare_input(speech_mix)

        from espnet2.enh.espnet_model_tse_ss import normalization

        speech_mix, _mean, _std = normalization(speech_mix)

        if self.use_true_nspk:
            assert num_spk is not None, "num_spk must be specified"
        else:
            num_spk = None

        try:
            waves = self._separate(speech_mix, lengths, num_spk)
        except RuntimeError as e:
            if "out of memory" not in str(e):
                raise e
            print(f"key {key} out of memory")
            torch.cuda.empty_cache()
            assert (
                batch_size == 1
            ), "Segmenting mode currently only supports batch_size = 1"

            if self.segmenting and lengths[0] > self.segment_size * fs:
                waves = self._separate_by_segments(
                    speech_mix, lengths, fs, batch_size, num_spk
                )
            else:
                raise e

        return self._postprocess_waves(waves)

    @torch.no_grad()
    def cal_permumation(
        self,
        ref_wavs: List[torch.Tensor],
        enh_wavs: List[torch.Tensor],
        criterion: str = "si_snr",
    ) -> torch.Tensor:
        """Calculate the permutation between separated streams in two adjacent segments.

        Args:
            ref_wavs (List[torch.Tensor]): [(Batch, Nsamples)]
            enh_wavs (List[torch.Tensor]): [(Batch, Nsamples)]
            criterion (str): one of ("si_snr", "mse", "corr)
        Returns:
            perm (torch.Tensor): permutation for enh_wavs (Batch, num_spk)
        """

        criterion_class = {"si_snr": SISNRLoss, "mse": FrequencyDomainMSE}[criterion]

        pit_solver = PITSolver(criterion=criterion_class())

        _, _, others = pit_solver(ref_wavs, enh_wavs)
        perm = others["perm"]
        return perm

    @staticmethod
    def from_pretrained(
        model_tag: Optional[str] = None,
        **kwargs: Optional[Any],
    ) -> "SeparateSpeech":
        """Build SeparateSpeech instance from the pretrained model.

        Args:
            model_tag (Optional[str]): Model tag of the pretrained models.
                Currently, the tags of espnet_model_zoo are supported.

        Returns:
            SeparateSpeech: SeparateSpeech instance.

        """
        if model_tag is not None:
            try:
                from espnet_model_zoo.downloader import ModelDownloader

            except ImportError:
                logging.error(
                    "`espnet_model_zoo` is not installed. "
                    "Please install via `pip install -U espnet_model_zoo`."
                )
                raise
            d = ModelDownloader()
            kwargs.update(**d.download_and_unpack(model_tag))

        return SeparateSpeech(**kwargs)


def humanfriendly_or_none(value: str) -> Optional[float]:
    """Parse a human-friendly size string (e.g. '8k'), or None for 'none'."""
    if value in ("none", "None", "NONE"):
        return None
    return humanfriendly.parse_size(value)


def _build_separate_speech(
    train_config: Optional[str],
    model_file: Optional[str],
    model_tag: Optional[str],
    inference_config: Optional[str],
    segment_size: Optional[float],
    hop_size: Optional[float],
    normalize_segment_scale: bool,
    show_progressbar: bool,
    ref_channel: Optional[int],
    normalize_output_wav: bool,
    use_true_nspk: bool,
    device: str,
    dtype: str,
) -> SeparateSpeech:
    """Build (or download) the SeparateSpeech model used for inference."""
    return SeparateSpeech.from_pretrained(
        model_tag=model_tag,
        train_config=train_config,
        model_file=model_file,
        inference_config=inference_config,
        segment_size=segment_size,
        hop_size=hop_size,
        normalize_segment_scale=normalize_segment_scale,
        show_progressbar=show_progressbar,
        ref_channel=ref_channel,
        normalize_output_wav=normalize_output_wav,
        use_true_nspk=use_true_nspk,
        device=device,
        dtype=dtype,
    )


def _build_data_iterator(
    separate_speech: SeparateSpeech,
    data_path_and_name_and_type: Sequence[Tuple[str, str, str]],
    dtype: str,
    batch_size: int,
    key_file: Optional[str],
    num_workers: int,
    allow_variable_data_keys: bool,
):
    """Build the streaming iterator that yields (keys, batch) per utterance."""
    return TSESSTask.build_streaming_iterator(
        data_path_and_name_and_type,
        dtype=dtype,
        batch_size=batch_size,
        key_file=key_file,
        num_workers=num_workers,
        preprocess_fn=TSESSTask.build_preprocess_fn(
            separate_speech.enh_train_args, False
        ),
        collate_fn=TSESSTask.build_collate_fn(separate_speech.enh_train_args, False),
        allow_variable_data_keys=allow_variable_data_keys,
        inference=True,
    )


def _build_speaker_writers(
    output_dir: Path, max_num_spk: int
) -> List[SoundScpWriter]:
    """Create one SoundScpWriter per possible speaker slot (1..max_num_spk)."""
    return [
        SoundScpWriter(f"{output_dir}/wavs/{i + 1}", f"{output_dir}/spk{i + 1}.scp")
        for i in range(max_num_spk)
    ]


def _extract_references(
    batch: Dict[str, torch.Tensor], max_num_spk: int
) -> List[torch.Tensor]:
    """Collect the per-speaker reference waveforms actually present in this utterance.

    Mixtures with fewer than ``max_num_spk`` active speakers have their
    unused ``speech_refK`` entries filled with a dummy (length-1) zero tensor
    by the dataset loader; those placeholders are dropped here so
    ``len(ref)`` reflects the true number of speakers in the mixture.
    """
    ref_by_name = {k: v for k, v in batch.items() if re.match(r"speech_ref\d+", k)}
    ref = [
        ref_by_name.get(
            f"speech_ref{spk + 1}",
            torch.zeros_like(ref_by_name["speech_ref1"]),
        )
        for spk in range(max_num_spk)
        if f"speech_ref{spk + 1}" in ref_by_name
    ]
    return [s for s in ref if s.shape[-1] > 1]


def _write_speaker_outputs(
    writers: List[SoundScpWriter],
    waves: List[np.ndarray],
    keys: Sequence[str],
    fs: int,
    batch_size: int,
    max_num_spk: int,
) -> None:
    """Write separated waveforms, padding unused speaker slots with a 'dummy' line."""
    for spk in range(max_num_spk):
        if spk < len(waves):
            for b in range(batch_size):
                writers[spk][keys[b]] = fs, waves[spk][b]
        else:
            writers[spk].fscp.write(f"{keys[b]} dummy\n")


def _process_utterance(
    separate_speech: SeparateSpeech,
    keys: Sequence[str],
    batch: Dict[str, torch.Tensor],
    fs: int,
    max_num_spk: int,
    batch_size: int,
    writers: List[SoundScpWriter],
    score_writer: DatadirWriter,
) -> None:
    """Separate one utterance, align it to its references, and write the results."""
    assert isinstance(batch, dict), type(batch)
    assert all(isinstance(s, str) for s in keys), keys
    batch_len = len(next(iter(batch.values())))
    assert len(keys) == batch_len, f"{len(keys)} != {batch_len}"

    ref = _extract_references(batch, max_num_spk)
    mix_batch = {k: v for k, v in batch.items() if k == "speech_mix"}

    waves = separate_speech(**mix_batch, fs=fs, num_spk=len(ref), key=keys[0])
    waves = select_sources(
        ref,
        waves,
        mix_batch["speech_mix"],
        score_writer,
        max_num_spk,
        keys[0],
    )

    _write_speaker_outputs(writers, waves, keys, fs, batch_size, max_num_spk)


def inference(
    nj: int,
    output_dir: str,
    score_output_dir: str,
    batch_size: int,
    max_num_spk: int,
    dtype: str,
    fs: int,
    ngpu: int,
    seed: int,
    num_workers: int,
    log_level: Union[int, str],
    data_path_and_name_and_type: Sequence[Tuple[str, str, str]],
    key_file: Optional[str],
    train_config: Optional[str],
    model_file: Optional[str],
    model_tag: Optional[str],
    inference_config: Optional[str],
    allow_variable_data_keys: bool,
    segment_size: Optional[float],
    hop_size: Optional[float],
    normalize_segment_scale: bool,
    use_true_nspk: bool,
    show_progressbar: bool,
    ref_channel: Optional[int],
    normalize_output_wav: bool,
) -> None:
    """Run variable-speaker-count separation inference over an SCP-defined dataset.

    For every utterance in the input iterator this:

    1. builds the mixture and the variable-length list of speaker references,
    2. runs :class:`SeparateSpeech` (optionally informed of the true speaker
       count via ``use_true_nspk``),
    3. aligns/pads the estimated sources against the references and records
       the estimated speaker count for later scoring (``select_sources``),
    4. writes the resulting waveforms to per-speaker SCP files, using a
       'dummy' placeholder line for slots beyond the estimated speaker count.
    """
    assert check_argument_types()
    if batch_size > 1:
        raise NotImplementedError("batch decoding is not implemented")

    logging.basicConfig(
        level=log_level,
        format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
    )

    device = f"cuda:{(nj - 1) % ngpu}" if ngpu >= 1 else "cpu"

    set_all_random_seed(seed)

    separate_speech = _build_separate_speech(
        train_config=train_config,
        model_file=model_file,
        model_tag=model_tag,
        inference_config=inference_config,
        segment_size=segment_size,
        hop_size=hop_size,
        normalize_segment_scale=normalize_segment_scale,
        show_progressbar=show_progressbar,
        ref_channel=ref_channel,
        normalize_output_wav=normalize_output_wav,
        use_true_nspk=use_true_nspk,
        device=device,
        dtype=dtype,
    )

    loader = _build_data_iterator(
        separate_speech,
        data_path_and_name_and_type,
        dtype,
        batch_size,
        key_file,
        num_workers,
        allow_variable_data_keys,
    )

    output_dir = Path(output_dir).expanduser().resolve()

    # if n_mix was specified during training, we evaluate only N-mix data
    n_mix = separate_speech.enh_train_args.n_mix
    if n_mix is not None:
        assert max_num_spk == max(n_mix), (max_num_spk, n_mix)
        logging.info(f"Inference is done with only {n_mix}-mix data")

    writers = _build_speaker_writers(output_dir, max_num_spk)

    with DatadirWriter(score_output_dir) as score_writer:
        for i, (keys, batch) in tqdm.tqdm(enumerate(loader)):
            # skip samples when evaluating only N-mix
            if n_mix is not None and int(keys[0][0]) not in n_mix:
                continue
            logging.info(f"[{i}] Enhancing {keys}")
            _process_utterance(
                separate_speech,
                keys,
                batch,
                fs,
                max_num_spk,
                batch_size,
                writers,
                score_writer,
            )

    for writer in writers:
        writer.close()


def get_parser() -> config_argparse.ArgumentParser:
    parser = config_argparse.ArgumentParser(
        description="Frontend inference",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Note(kamo): Use '_' instead of '-' as separator.
    # '-' is confusing if written in yaml.

    # General / runtime options
    parser.add_argument(
        "--log_level",
        type=lambda x: x.upper(),
        default="INFO",
        choices=("CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG", "NOTSET"),
        help="The verbose level of logging",
    )
    parser.add_argument(
        "--nj",
        type=int,
        default=1,
        help="The number of parallel jobs",
    )
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--score_output_dir", type=str, required=True)
    parser.add_argument(
        "--ngpu",
        type=int,
        default=0,
        help="The number of gpus. 0 indicates CPU mode",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument(
        "--dtype",
        default="float32",
        choices=["float16", "float32", "float64"],
        help="Data type",
    )
    parser.add_argument(
        "--fs", type=humanfriendly_or_none, default=8000, help="Sampling rate"
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=1,
        help="The number of workers used for DataLoader",
    )

    group = parser.add_argument_group("Input data related")
    group.add_argument(
        "--data_path_and_name_and_type",
        type=str2triple_str,
        required=True,
        action="append",
    )
    group.add_argument("--key_file", type=str_or_none)
    group.add_argument("--allow_variable_data_keys", type=str2bool, default=False)

    group = parser.add_argument_group("Output data related")
    group.add_argument(
        "--normalize_output_wav",
        type=str2bool,
        default=True,
        help="Whether to normalize the predicted wav to [-1~1]",
    )

    group = parser.add_argument_group("The model configuration related")
    group.add_argument(
        "--train_config",
        type=str,
        help="Training configuration file",
    )
    group.add_argument(
        "--model_file",
        type=str,
        help="Model parameter file",
    )
    group.add_argument(
        "--model_tag",
        type=str,
        help="Pretrained model tag. If specify this option, train_config and "
        "model_file will be overwritten",
    )
    group.add_argument(
        "--inference_config",
        type=str_or_none,
        default=None,
        help="Optional configuration file for overwriting enh model attributes "
        "during inference",
    )
    group.add_argument(
        "--max_num_spk",
        type=int,
        help="maximum number of speakers",
    )
    group.add_argument(
        "--use_true_nspk",
        type=str2bool,
        default=False,
        help="Whether to inform true number of speakers to model",
    )

    group = parser.add_argument_group("Data loading related")
    group.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="The batch size for inference",
    )

    group = parser.add_argument_group("SeparateSpeech related")
    group.add_argument(
        "--segment_size",
        type=float,
        default=4,
        help="Segment length in seconds for segment-wise speech enhancement/separation",
    )
    group.add_argument(
        "--hop_size",
        type=float,
        default=2,
        help="Hop length in seconds for segment-wise speech enhancement/separation",
    )
    group.add_argument(
        "--normalize_segment_scale",
        type=str2bool,
        default=False,
        help="Whether to normalize the energy of the separated streams in each segment",
    )
    group.add_argument(
        "--show_progressbar",
        type=str2bool,
        default=False,
        help="Whether to show a progress bar when performing segment-wise speech "
        "enhancement/separation",
    )
    group.add_argument(
        "--ref_channel",
        type=int,
        default=None,
        help="If not None, this will overwrite the ref_channel defined in the "
        "separator module (for multi-channel speech processing)",
    )

    return parser


def main(cmd=None):
    print(get_commandline_args(), file=sys.stderr)
    parser = get_parser()
    args = parser.parse_args(cmd)
    kwargs = vars(args)
    kwargs.pop("config", None)
    inference(**kwargs)


if __name__ == "__main__":
    main()
