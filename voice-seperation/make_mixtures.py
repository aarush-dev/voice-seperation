"""Build N-speaker monaural test mixtures following the WSJ0-mix protocol.

Corpus-agnostic: give it a {speaker_id: [utterance paths]} mapping and it emits
the layout SR-CorrNet's scp scripts expect:

    <out>/{N}speakers/wav8k/min/tt/{mix,s1..sN}/<key>.wav

Protocol (mirrors WSJ0-mix): pick N distinct speakers, one utterance each, hold
speaker 1 at unity and scale the rest to a random relative SNR, then sum. "min"
mode truncates every source to the shortest utterance in the mixture.
"""

import argparse
import pathlib
import random

import librosa
import numpy as np
import soundfile as sf

SR = 8000


def build_mixture(paths, rng, snr_range):
    """Returns (mixture, [sources]) all at SR, truncated to the shortest ('min' mode)."""
    sources = [librosa.load(p, sr=SR, mono=True)[0] for p in paths]

    n = min(len(s) for s in sources)
    sources = [s[:n] for s in sources]

    # Speaker 1 is the 0 dB reference; the others get a random relative gain.
    scaled = []
    for i, s in enumerate(sources):
        s = s - s.mean()
        rms = np.sqrt((s**2).mean()) + 1e-12
        s = s / rms  # unit RMS, so the SNR below is exact
        if i > 0:
            s = s * 10 ** (rng.uniform(*snr_range) / 20.0)
        scaled.append(s)

    mix = np.sum(scaled, axis=0)

    # Guard against clipping; apply the SAME gain to sources so SI-SNR refs stay aligned.
    peak = np.abs(mix).max()
    if peak > 0.99:
        g = 0.99 / peak
        mix = mix * g
        scaled = [s * g for s in scaled]

    return mix, scaled


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True, choices=["librispeech", "wsj0"])
    ap.add_argument("--root", required=True, help="corpus root dir")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-spks", type=int, nargs="+", default=[2, 3, 4, 5])
    ap.add_argument("--num", type=int, default=200, help="mixtures per speaker count")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--snr-range",
        type=float,
        nargs=2,
        default=[-5.0, 5.0],
        help="relative gain range in dB for speakers 2..N",
    )
    ap.add_argument(
        "--split",
        choices=["all", "a", "b"],
        default="all",
        help="draw speakers from a disjoint half of the corpus. Halves are assigned by "
        "sorted speaker id (even index -> a, odd -> b), so 'a' and 'b' never share a "
        "speaker. Use different halves for val and test to avoid tuning on test speakers.",
    )
    args = ap.parse_args()

    root = pathlib.Path(args.root)
    if args.corpus == "librispeech":
        exts = ("*.flac",)
    else:
        exts = ("*.wav",)

    # speaker id = the directory two levels up (LibriSpeech: <spk>/<chapter>/x.flac;
    # WSJ0 converted: <spk>/x.wav -> handled by taking the parent that isn't a chapter)
    speakers = {}
    for ext in exts:
        for f in root.rglob(ext):
            spk = f.parts[len(root.parts)] if len(f.parts) > len(root.parts) else "unk"
            speakers.setdefault(spk, []).append(f)

    speakers = {k: v for k, v in speakers.items() if v}
    print(f"found {len(speakers)} speakers, {sum(len(v) for v in speakers.values())} utterances")

    out_root = pathlib.Path(args.out)
    spk_ids = sorted(speakers)
    if args.split != "all":
        want = 0 if args.split == "a" else 1
        spk_ids = [s for i, s in enumerate(spk_ids) if i % 2 == want]
        print(f"split '{args.split}': {len(spk_ids)} speakers -> {' '.join(spk_ids)}")

    if len(spk_ids) < max(args.n_spks):
        raise SystemExit(f"not enough speakers ({len(spk_ids)}) for {max(args.n_spks)}-spk mixes")

    for n in args.n_spks:
        rng = random.Random(args.seed + n)
        base = out_root / f"{n}speakers" / "wav8k" / "min" / "tt"
        (base / "mix").mkdir(parents=True, exist_ok=True)
        for i in range(n):
            (base / f"s{i+1}").mkdir(parents=True, exist_ok=True)

        for j in range(args.num):
            chosen = rng.sample(spk_ids, n)
            paths = [rng.choice(speakers[s]) for s in chosen]
            mix, sources = build_mixture(paths, rng, args.snr_range)

            key = f"{j:04d}_" + "_".join(chosen) + ".wav"
            sf.write(base / "mix" / key, mix, SR)
            for i, s in enumerate(sources):
                sf.write(base / f"s{i+1}" / key, s, SR)

        print(f"{n}-spk: wrote {args.num} mixtures -> {base}")


if __name__ == "__main__":
    main()
