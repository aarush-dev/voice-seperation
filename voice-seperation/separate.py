"""Separate concurrent speakers from a mono recording using SR-CorrNet.

Usage:
    python separate.py input.wav -o output/
    python separate.py input.wav -o output/ --n-spks 4
    python separate.py input.wav -o output/ --no-route --no-mc   # plain baseline

Routes N=2 to a specialized checkpoint and applies mixture consistency by
default; both are measured wins on WSJ0 (see results_tier1.md). Disable either
with --no-route / --no-mc.
"""

import argparse
import json
import pathlib

import librosa
import soundfile as sf
import torch

from pipeline import Separator, rescale_for_write

MODEL_SR = 8000  # the checkpoints are 8 kHz only; anything else must be resampled


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("input", help="input .wav (any sample rate; resampled to 8 kHz)")
    ap.add_argument("-o", "--output-dir", default="output")
    ap.add_argument(
        "--n-spks",
        type=int,
        default=None,
        help="speaker count hint (2-5). Omit to let the model decide.",
    )
    ap.add_argument("--no-route", action="store_true", help="always use the general checkpoint")
    ap.add_argument("--no-mc", action="store_true", help="disable mixture consistency")
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    # librosa defaults to res_type='soxr_hq' -- verified to suppress above-Nyquist
    # content by ~54 dB. Do not swap this for naive decimation.
    audio, _ = librosa.load(args.input, sr=MODEL_SR, mono=True)
    wav = torch.from_numpy(audio).float()

    sep = Separator(device=args.device, route=not args.no_route, mc=not args.no_mc)
    est, n, repo = sep.separate(wav, args.n_spks)

    # ONE shared gain across streams -- never normalize streams independently.
    streams, gain = rescale_for_write(est.detach().cpu())

    out = pathlib.Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    stem = pathlib.Path(args.input).stem
    files = []
    for i, s in enumerate(streams, 1):
        path = out / f"{stem}_spk{i}.wav"
        sf.write(path, s.numpy(), MODEL_SR, subtype="PCM_16")
        files.append(path.name)
        print(f"wrote {path}")

    # Record the gain so the raw model output stays recoverable (raw = wav * gain).
    # Audio with no provenance is nearly worthless later.
    (out / "manifest.json").write_text(
        json.dumps(
            {
                "source": str(args.input),
                "checkpoint": repo,
                "n_spks": n,
                "n_spks_given": args.n_spks is not None,
                "routing": not args.no_route,
                "mixture_consistency": not args.no_mc,
                "gain": gain,
                "peak": float(streams.abs().max()),
                "sample_rate": MODEL_SR,
                "files": files,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"{n} speaker(s) separated from {args.input} using {repo}")


if __name__ == "__main__":
    main()
