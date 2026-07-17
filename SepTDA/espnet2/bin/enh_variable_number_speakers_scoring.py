#!/usr/bin/env python3
"""CLI for scoring separated speech against references for variable speaker counts.

This complements ``enh_variable_number_speakers_inference``: the inference
stage already resolves the permutation between estimated and reference
sources (padding under-estimated slots and discarding over-estimated ones),
so this script assumes ``ref`` and ``inf`` are already aligned one-to-one and
just computes objective metrics per speaker:

* STOI / ESTOI (short-time objective intelligibility)
* (narrowband, and wideband when fs == 16 kHz) PESQ
* SNR / SI-SNR and BSS-eval SDR/SIR/SAR, SI-SDR/SI-SIR/SI-SAR

When ``--flexible_numspk`` is set, the number of reference and estimated
speakers for a given utterance may differ from ``len(ref_scp)`` /
``len(inf_scp)`` (some SCP files may lack an entry, or contain a 'dummy'
placeholder for that utterance); those slots are simply skipped rather than
scored.
"""
import argparse
import logging
import re
import sys
from pathlib import Path
from typing import List, Tuple, Union

import numpy as np
import torch
from fast_bss_eval import bss_eval_sources, si_bss_eval_sources
from pesq import PesqError, pesq
from pystoi import stoi
from typeguard import check_argument_types

from espnet2.enh.loss.criterions.time_domain import SNRLoss
from espnet2.fileio.datadir_writer import DatadirWriter
from espnet2.fileio.sound_scp import SoundScpReader
from espnet2.train.dataset import kaldi_loader
from espnet2.utils import config_argparse
from espnet2.utils.types import str2bool
from espnet.utils.cli_utils import get_commandline_args

snr_loss = SNRLoss()
DUMMY_SYMBOL = "dummy"


def get_readers(scps: List[str], dtype: str) -> Tuple[List, str]:
    """Open one reader per SCP file, auto-detecting sound vs. kaldi_ark format."""
    with open(scps[0], "r") as f:
        line = f.readline()
        filename = Path(line.strip().split(maxsplit=1)[1]).name
    if re.fullmatch(r".*\.ark(:\d+)?", filename):
        # xxx.ark or xxx.ark:123
        readers = [kaldi_loader(f, float_dtype=dtype) for f in scps]
        audio_format = "kaldi_ark"
    else:
        readers = [SoundScpReader(f, dtype=dtype) for f in scps]
        audio_format = "sound"
    return readers, audio_format


def read_audio(reader, key: str, audio_format: str = "sound") -> np.ndarray:
    """Read a single utterance's audio from a reader of the given format."""
    if audio_format == "sound":
        return reader[key][1]
    elif audio_format == "kaldi_ark":
        return reader[key]
    else:
        raise ValueError(f"Unknown audio format: {audio_format}")


def _read_keys(key_file: str) -> List[str]:
    with open(key_file, encoding="utf-8") as f:
        return [line.rstrip().split(maxsplit=1)[0] for line in f]


def _get_sample_rate(ref_readers: List, ref_audio_format: str, first_key: str) -> int:
    retval = ref_readers[0][first_key]
    if ref_audio_format == "kaldi_ark":
        sample_rate = ref_readers[0].rate
    elif ref_audio_format == "sound":
        sample_rate = retval[0]
    else:
        raise NotImplementedError(ref_audio_format)
    assert sample_rate is not None, (sample_rate, ref_audio_format)
    return sample_rate


def _collect_audios(
    key: str,
    ref_readers: List,
    ref_audio_format: str,
    inf_readers: List,
    inf_audio_format: str,
    flexible_numspk: bool,
) -> Tuple[np.ndarray, np.ndarray]:
    """Gather the non-dummy reference/estimate audios available for one utterance."""
    if not flexible_numspk:
        ref_audios = [
            read_audio(ref_reader, key, audio_format=ref_audio_format)
            for ref_reader in ref_readers
            if ref_reader.data[key] != DUMMY_SYMBOL
        ]
        inf_audios = [
            read_audio(inf_reader, key, audio_format=inf_audio_format)
            for (inf_reader, ref_reader) in zip(inf_readers, ref_readers)
            if ref_reader.data[key] != DUMMY_SYMBOL
        ]
    else:
        ref_audios = [
            read_audio(ref_reader, key, audio_format=ref_audio_format)
            for ref_reader in ref_readers
            if key in ref_reader.keys() and ref_reader.data[key] != DUMMY_SYMBOL
        ]
        inf_audios = [
            read_audio(inf_reader, key, audio_format=inf_audio_format)
            for inf_reader in inf_readers
            if key in inf_reader.keys() and inf_reader.data[key] != DUMMY_SYMBOL
        ]
    return np.array(ref_audios), np.array(inf_audios)


def _align_channel_dims(
    ref: np.ndarray, inf: np.ndarray, ref_channel: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Reduce whichever of ref/inf is multi-channel down to a single channel.

    Channel-count mismatches (and the choice of channel) are already handled
    upstream at the inference stage; this only normalizes ndim so that
    metrics can be computed on matching (n_spk, n_samples) arrays.
    """
    if ref.ndim > inf.ndim:
        # multi-channel reference and single-channel output
        ref = ref[..., ref_channel]
    elif ref.ndim < inf.ndim:
        # single-channel reference and multi-channel output
        inf = inf[..., ref_channel]
    elif ref.ndim == inf.ndim == 3:
        # multi-channel reference and output
        ref = ref[..., ref_channel]
        inf = inf[..., ref_channel]
    return ref, inf


def _compute_batch_bss_metrics(
    ref: np.ndarray, inf: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute (SDR, SIR, SAR, SI-SDR, SI-SIR, SI-SAR) for every speaker at once.

    The permutation between ``ref`` and ``inf`` is already resolved (either by
    the inference-stage SI-SDR alignment, or trivially for the observation
    baseline), so ``compute_permutation=False``.
    """
    tensor_ref = torch.from_numpy(ref).float()
    tensor_inf = torch.from_numpy(inf).float()
    if torch.cuda.is_available():
        tensor_ref = tensor_ref.to("cuda:0")
        tensor_inf = tensor_inf.to("cuda:0")
    sdr, sir, sar = bss_eval_sources(
        tensor_ref, tensor_inf, compute_permutation=False, clamp_db=80
    )
    si_snr, si_sir, si_sar = si_bss_eval_sources(
        tensor_ref, tensor_inf, compute_permutation=False, clamp_db=80
    )
    return (
        sdr.cpu().numpy(),
        sir.cpu().numpy(),
        sar.cpu().numpy(),
        si_snr.cpu().numpy(),
        si_sir.cpu().numpy(),
        si_sar.cpu().numpy(),
    )


def _write_active_speaker_scores(
    writer: DatadirWriter,
    key: str,
    spk_idx: int,
    ref_i: np.ndarray,
    inf_i: np.ndarray,
    sample_rate: int,
    bss_metrics: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    inf_readers: List,
    inf_audio_format: str,
    perm: np.ndarray,
    ref_scp_count: int,
) -> None:
    """Score and record one speaker slot that has both a reference and an estimate."""
    sdr, sir, sar, si_snr, si_sir, si_sar = bss_metrics

    stoi_score = stoi(ref_i, inf_i, fs_sig=sample_rate)
    estoi_score = stoi(ref_i, inf_i, fs_sig=sample_rate, extended=True)
    snr_score = -float(
        snr_loss(
            torch.from_numpy(ref_i[None, ...]),
            torch.from_numpy(inf_i[None, ...]),
        )
    )

    writer[f"STOI_spk{spk_idx + 1}"][key] = str(stoi_score * 100)  # in percentage
    writer[f"ESTOI_spk{spk_idx + 1}"][key] = str(estoi_score * 100)
    writer[f"SNR_spk{spk_idx + 1}"][key] = str(snr_score)
    writer[f"SI_SNR_spk{spk_idx + 1}"][key] = str(si_snr[spk_idx])
    writer[f"SI_SAR_spk{spk_idx + 1}"][key] = str(si_sar[spk_idx])
    writer[f"SI_SIR_spk{spk_idx + 1}"][key] = str(si_sir[spk_idx])
    writer[f"SDR_spk{spk_idx + 1}"][key] = str(sdr[spk_idx])
    writer[f"SAR_spk{spk_idx + 1}"][key] = str(sar[spk_idx])
    writer[f"SIR_spk{spk_idx + 1}"][key] = str(sir[spk_idx])

    # PESQ
    if sample_rate == 16000:
        wbpesq_score = pesq(
            sample_rate, ref_i, inf_i, mode="wb", on_error=PesqError.RETURN_VALUES
        )
    nbpesq_score = pesq(
        sample_rate, ref_i, inf_i, mode="nb", on_error=PesqError.RETURN_VALUES
    )
    if sample_rate == 16000:
        if wbpesq_score == PesqError.NO_UTTERANCES_DETECTED:
            print(key, flush=True)
        else:
            writer[f"WBPESQ_spk{spk_idx + 1}"][key] = str(wbpesq_score)
    if nbpesq_score == PesqError.NO_UTTERANCES_DETECTED:
        print(key, flush=True)
    else:
        writer[f"NBPESQ_spk{spk_idx + 1}"][key] = str(nbpesq_score)

    # save permutation-assigned script file
    if spk_idx < ref_scp_count:
        if inf_audio_format == "sound":
            writer[f"wav_spk{spk_idx + 1}"][key] = inf_readers[perm[spk_idx]].data[key]
        elif inf_audio_format == "kaldi_ark":
            # NOTE: SegmentsExtractor is not supported
            writer[f"wav_spk{spk_idx + 1}"][key] = inf_readers[
                perm[spk_idx]
            ].loader._dict[key]
        else:
            raise ValueError(f"Unknown audio format: {inf_audio_format}")


def _write_dummy_speaker_scores(
    writer: DatadirWriter, key: str, spk_idx: int, sample_rate: int
) -> None:
    """Record placeholder scores for a speaker slot with no reference/estimate."""
    writer[f"STOI_spk{spk_idx + 1}"][key] = DUMMY_SYMBOL
    writer[f"ESTOI_spk{spk_idx + 1}"][key] = DUMMY_SYMBOL
    if sample_rate == 16000:
        writer[f"WBPESQ_spk{spk_idx + 1}"][key] = DUMMY_SYMBOL
    writer[f"NBPESQ_spk{spk_idx + 1}"][key] = DUMMY_SYMBOL
    writer[f"SNR_spk{spk_idx + 1}"][key] = DUMMY_SYMBOL
    writer[f"SI_SNR_spk{spk_idx + 1}"][key] = DUMMY_SYMBOL
    writer[f"SI_SIR_spk{spk_idx + 1}"][key] = DUMMY_SYMBOL
    writer[f"SI_SAR_spk{spk_idx + 1}"][key] = DUMMY_SYMBOL
    writer[f"SDR_spk{spk_idx + 1}"][key] = DUMMY_SYMBOL
    writer[f"SAR_spk{spk_idx + 1}"][key] = DUMMY_SYMBOL
    writer[f"SIR_spk{spk_idx + 1}"][key] = DUMMY_SYMBOL
    writer[f"wav_spk{spk_idx + 1}"][key] = DUMMY_SYMBOL


def _score_utterance(
    writer: DatadirWriter,
    key: str,
    ref_readers: List,
    ref_audio_format: str,
    inf_readers: List,
    inf_audio_format: str,
    ref_scp: List[str],
    ref_channel: int,
    flexible_numspk: bool,
    num_spk: int,
    sample_rate: int,
) -> None:
    """Compute and write every metric for a single utterance."""
    ref, inf = _collect_audios(
        key, ref_readers, ref_audio_format, inf_readers, inf_audio_format, flexible_numspk
    )
    ref, inf = _align_channel_dims(ref, inf, ref_channel)
    # Speaker-count mismatches were already resolved (padded/discarded and
    # permuted) at the inference stage, so ref and inf must match here.
    assert ref.shape == inf.shape, (ref.shape, inf.shape)

    bss_metrics = _compute_batch_bss_metrics(ref, inf)
    perm = np.arange(num_spk)

    for spk_idx in range(num_spk):
        if spk_idx < ref.shape[0]:
            _write_active_speaker_scores(
                writer,
                key,
                spk_idx,
                ref[spk_idx],
                inf[spk_idx],
                sample_rate,
                bss_metrics,
                inf_readers,
                inf_audio_format,
                perm,
                len(ref_scp),
            )
        else:
            _write_dummy_speaker_scores(writer, key, spk_idx, sample_rate)


def scoring(
    output_dir: str,
    dtype: str,
    log_level: Union[int, str],
    key_file: str,
    ref_scp: List[str],
    inf_scp: List[str],
    ref_channel: int,
    flexible_numspk: bool,
) -> None:
    """Score every utterance listed in ``key_file`` against its references.

    Writes one score file per metric per speaker slot (e.g. ``SI_SNR_spk1``)
    under ``output_dir``, in the ``DatadirWriter`` key-value text format.
    """
    assert check_argument_types()

    logging.basicConfig(
        level=log_level,
        format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
    )

    if not flexible_numspk:
        assert len(ref_scp) == len(inf_scp), (ref_scp, inf_scp)
    num_spk = len(inf_scp)

    keys = _read_keys(key_file)

    ref_readers, ref_audio_format = get_readers(ref_scp, dtype)
    inf_readers, inf_audio_format = get_readers(inf_scp, dtype)
    sample_rate = _get_sample_rate(ref_readers, ref_audio_format, keys[0])

    with DatadirWriter(output_dir) as writer:
        for n, key in enumerate(keys):
            logging.info(f"[{n}] Scoring {keys}")
            _score_utterance(
                writer,
                key,
                ref_readers,
                ref_audio_format,
                inf_readers,
                inf_audio_format,
                ref_scp,
                ref_channel,
                flexible_numspk,
                num_spk,
                sample_rate,
            )


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
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument(
        "--dtype",
        default="float32",
        choices=["float16", "float32", "float64"],
        help="Data type",
    )

    group = parser.add_argument_group("Input data related")
    group.add_argument(
        "--ref_scp",
        type=str,
        required=True,
        action="append",
    )
    group.add_argument(
        "--inf_scp",
        type=str,
        required=True,
        action="append",
    )
    group.add_argument("--key_file", type=str)
    group.add_argument("--ref_channel", type=int, default=0)
    group.add_argument("--flexible_numspk", type=str2bool, default=False)

    return parser


def main(cmd=None):
    print(get_commandline_args(), file=sys.stderr)
    parser = get_parser()
    args = parser.parse_args(cmd)
    kwargs = vars(args)
    kwargs.pop("config", None)
    scoring(**kwargs)


if __name__ == "__main__":
    main()
