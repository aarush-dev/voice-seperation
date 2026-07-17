"""Benchmark SR-CorrNet: PIT-matched SI-SNRi per speaker count + counting accuracy.

Reads the layout make_mixtures.py emits (or the standard wsj0_kmix layout):
    <root>/{N}speakers/wav8k/min/tt/{mix,s1..sN}/<key>.wav

Reuses the authors' own PIT_SISNRi from the SR-CorrNet repo rather than a
reimplementation, so the numbers are computed the same way the paper's are.
"""

import argparse
import pathlib
import time

import librosa
import numpy as np
import soundfile as sf
import torch

from pipeline import GENERAL, Separator, rescale_for_write
from sr_corrnet.models.SR_CorrNet_SS.loss import PIT_SISNRi

REPO = GENERAL
SR = 8000


def load_pair(base, key, n):
    mix, _ = librosa.load(base / "mix" / key, sr=SR, mono=True)
    refs = [librosa.load(base / f"s{i+1}" / key, sr=SR, mono=True)[0] for i in range(n)]
    m = min([len(mix)] + [len(r) for r in refs])
    return mix[:m], [r[:m] for r in refs]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--n-spks", type=int, nargs="+", default=[2, 3, 4, 5])
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--dataset-label", default="unknown", help="for the report header")
    ap.add_argument("--skip-counting", action="store_true")
    ap.add_argument(
        "--save-dir",
        default=None,
        help="write separated wavs (+ the mixture) here, per speaker count",
    )
    ap.add_argument(
        "--save-n", type=int, default=20, help="how many mixtures to save audio for, per count"
    )
    # Both default OFF so the bare command still reproduces the verified headline
    # numbers (23.39/23.84/22.20/20.57). Turning them on silently would make the
    # documented reproduce command mean something different.
    ap.add_argument(
        "--route", action="store_true", help="route N=2 to fix-2spk-l-dm (+0.57 dB, results_tier1.md)"
    )
    ap.add_argument(
        "--mc", action="store_true", help="apply mixture consistency (+0.42 dB at N=2, free)"
    )
    args = ap.parse_args()

    device = torch.device(args.device)
    sep = Separator(device=args.device, route=args.route, mc=args.mc)
    sisnri = PIT_SISNRi(scale_inv=True, device=device)
    if args.route or args.mc:
        print(f"[cfg] routing={args.route} mixture_consistency={args.mc}")

    root = pathlib.Path(args.root)
    rows = []

    for n in args.n_spks:
        base = root / f"{n}speakers" / "wav8k" / "min" / "tt"
        if not base.exists():
            print(f"[skip] {base} missing")
            continue
        keys = sorted(p.name for p in (base / "mix").glob("*.wav"))[: args.limit]
        if not keys:
            print(f"[skip] no mixtures in {base}")
            continue

        scores, correct, audio_s, wall_s = [], 0, 0.0, 0.0
        t_start = time.time()

        for idx, key in enumerate(keys):
            if idx and idx % 25 == 0:
                done = time.time() - t_start
                eta = done / idx * (len(keys) - idx)
                print(
                    f"  N={n} {idx}/{len(keys)} "
                    f"SI-SNRi so far {np.mean(scores):.2f} dB  ETA {eta/60:.1f} min",
                    flush=True,
                )
            mix, refs = load_pair(base, key, n)
            mix_t = torch.from_numpy(mix).float()

            t0 = time.time()
            est, _, _ = sep.separate(mix_t, n_spks=n)
            wall_s += time.time() - t0
            audio_s += len(mix) / SR

            L = min([len(mix)] + [est.shape[-1]] + [len(r) for r in refs])

            # PIT_SISNRi wants lists of (B, L) tensors and the (B, L) mixture.
            est_l = [e[:L].reshape(1, -1).float().to(device) for e in est]
            ref_l = [torch.from_numpy(r[:L]).reshape(1, -1).float().to(device) for r in refs]
            mix_b = mix_t[:L].reshape(1, -1).to(device)

            with torch.no_grad():
                scores.append(sisnri(est_l, ref_l, mix_b).item())

            if args.save_dir and idx < args.save_n:
                # ONE shared gain across streams -- preserves relative speaker
                # levels and stops every stream clipping to +/-1. Never normalize
                # streams independently.
                d = pathlib.Path(args.save_dir) / f"{n}spk" / key.replace(".wav", "")
                d.mkdir(parents=True, exist_ok=True)
                streams, _ = rescale_for_write(est.detach().cpu())
                sf.write(d / "mixture.wav", mix, SR)
                for i, s in enumerate(streams, 1):
                    sf.write(d / f"est_spk{i}.wav", s.numpy(), SR)
                for i, r in enumerate(refs, 1):
                    sf.write(d / f"ref_spk{i}.wav", r, SR)

            if not args.skip_counting:
                # Counting always uses the general checkpoint -- it is the only one
                # that can infer a count (the routed 2-spk model always emits 2).
                pred = len(
                    sep.model(REPO).process_waveform(mix_t.unsqueeze(0))["waveforms"]
                )
                correct += int(pred == n)

        arr = np.array(scores)
        rtf = wall_s / audio_s
        acc = None if args.skip_counting else 100.0 * correct / len(keys)
        rows.append((n, len(keys), arr.mean(), arr.std(), acc, rtf))
        acc_s = "n/a" if acc is None else f"{acc:5.1f}%"
        print(
            f"N={n}  n={len(keys):4d}  SI-SNRi={arr.mean():6.2f} dB "
            f"(sd {arr.std():4.2f})  count-acc={acc_s}  RTF={rtf:.4f}"
        )

    print(f"\n=== SR-CorrNet var-2-5spk on {args.dataset_label} ===")
    print(f"{'N':>2} {'mixes':>6} {'SI-SNRi (dB)':>13} {'count acc':>10} {'RTF':>8}")
    for n, cnt, mean, sd, acc, rtf in rows:
        acc_s = "n/a" if acc is None else f"{acc:.1f}%"
        print(f"{n:>2} {cnt:>6} {mean:>13.2f} {acc_s:>10} {rtf:>8.4f}")


if __name__ == "__main__":
    main()
