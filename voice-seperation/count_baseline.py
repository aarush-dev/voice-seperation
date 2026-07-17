"""Step 0 baseline: speaker-count confusion matrix + end-to-end SI-SNRi gap.

benchmark.py answers "how good is separation when we already know N". This answers
"what does it cost us that we don't", which is the metric the counting work is
actually trying to move.

Per mixture it runs two passes:
    known   = process_waveform(mix, n_spks=N)      -> oracle SI-SNRi (upper bound)
    unknown = process_waveform(mix, n_spks=None)   -> predicted count AND its audio

The unknown-count pass does double duty: the number of streams it returns is the
prediction, and those streams are what we score. So the gap costs one extra
forward pass, not two.

## Scoring a count mismatch

The authors' PIT_SISNRi asserts len(estims) == len(targets), so it cannot score a
mismatch directly. Rather than reimplement SI-SNRi (which the project forbids, and
which is how the scale-invariance bug hid), note that PIT_SISNRi's objective is the
MEAN of per-pair SI-SNRi over a permutation. Maximising a mean over permutations is
a linear assignment problem. So we call the authors' PIT_SISNRi on 1-element lists
to fill a pairwise matrix M[ref, est], then solve the assignment ourselves. On a
square matrix this is provably identical to calling PIT_SISNRi directly -- asserted
by --self-test.

Do NOT pad the short side with silence to force the counts equal. PIT_SISNRi's
scale_inv branch projects the reference onto the estimate, so a silent estimate
gives 20*log10(eps) = -400 dB and swamps every real number.

Two mismatch policies are reported, because which one is right depends on the
judge's rules (unknown at time of writing) and they disagree about what a good
counter even is:

  lenient  -- extra streams are ignored; a missed speaker is not.
              over-predict : each ref claims a distinct est, spare ests unused (free)
              under-predict: refs share ests (models the real failure -- two speakers
                             merged into one stream, so that stream partly serves both)
              NOTE: over-prediction is FREE here, so "always predict 5" is optimal.
              Under this policy a pruning counter can only lose.

  strict   -- the judge wants exactly N streams; a wrong count fails the item (0 dB).
              NOTE: this makes the gap a monotone function of counting accuracy, so
              gap-ranking and accuracy-ranking coincide.

Neither is neutral. Report both and let the judge's rules pick.
"""

import argparse
import itertools
import pathlib

import librosa
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

from sr_corrnet import SSInference
from sr_corrnet.models.SR_CorrNet_SS.loss import PIT_SISNRi

REPO = "shinuh/sr-corrnet-ss-1ch-wsj-var-2-5spk"
SR = 8000
MAX_SPKS = 5  # the checkpoint's ceiling; predictions clamp here


def load_pair(base, key, n):
    mix, _ = librosa.load(base / "mix" / key, sr=SR, mono=True)
    refs = [librosa.load(base / f"s{i+1}" / key, sr=SR, mono=True)[0] for i in range(n)]
    m = min([len(mix)] + [len(r) for r in refs])
    return mix[:m], [r[:m] for r in refs]


class Scorer:
    """Wraps the authors' PIT_SISNRi into a rectangular-capable scorer."""

    def __init__(self, device):
        self.device = device
        self.pit = PIT_SISNRi(scale_inv=True, device=device)

    def pair_matrix(self, ests, refs, mix):
        """M[i, j] = SI-SNRi of est j against ref i, via the authors' own loss."""
        M = np.zeros((len(refs), len(ests)), dtype=np.float64)
        with torch.no_grad():
            for i, r in enumerate(refs):
                for j, e in enumerate(ests):
                    M[i, j] = self.pit([e], [r], mix).item()
        return M

    def score(self, M, policy):
        """Reduce a (n_refs, n_ests) pairwise matrix to one SI-SNRi number."""
        n_refs, n_ests = M.shape
        if policy == "strict" and n_refs != n_ests:
            return 0.0
        if n_ests >= n_refs:
            # Each ref takes a distinct est; spare ests go unused (and unpenalised).
            rows, cols = linear_sum_assignment(-M)
            return float(M[rows, cols].mean())
        # Fewer streams than speakers: refs share streams (merged-speaker reality).
        return float(M.max(axis=1).mean())


def self_test(scorer, device):
    """Assignment on a square matrix must equal calling PIT_SISNRi directly.

    Also checks the null baseline (estimate == mixture) scores ~0 dB, the same
    harness check benchmark.py's numbers rest on.
    """
    g = torch.Generator().manual_seed(0)
    for n in (2, 3, 4):
        refs = [torch.randn(1, 8000, generator=g) for _ in range(n)]
        mix = sum(refs)
        # Estimates: the refs, permuted and perturbed, so PIT has real work to do.
        ests = [refs[(i + 1) % n] + 0.15 * torch.randn(1, 8000, generator=g) for i in range(n)]
        refs_d = [r.to(device) for r in refs]
        ests_d = [e.to(device) for e in ests]
        mix_d = mix.to(device)

        with torch.no_grad():
            direct = scorer.pit(ests_d, refs_d, mix_d).item()
        viaM = scorer.score(scorer.pair_matrix(ests_d, refs_d, mix_d), "lenient")
        assert abs(direct - viaM) < 1e-4, f"n={n}: assignment {viaM} != PIT {direct}"
        print(f"  [self-test] n={n}: assignment == PIT_SISNRi ({direct:.4f} dB) OK")

        # Null baseline: every estimate IS the mixture -> no improvement.
        null = scorer.score(scorer.pair_matrix([mix_d] * n, refs_d, mix_d), "lenient")
        assert abs(null) < 1e-3, f"n={n}: null baseline {null} != 0"
        print(f"  [self-test] n={n}: null baseline = {null:.4f} dB OK")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--n-spks", type=int, nargs="+", default=[2, 3, 4, 5])
    ap.add_argument("--limit", type=int, default=500)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--label", default="unknown", help="for the report header")
    ap.add_argument("--self-test", action="store_true", help="validate the harness and exit")
    ap.add_argument("--out", default=None, help="write per-mixture rows to this .npz")
    args = ap.parse_args()

    device = torch.device(args.device)
    scorer = Scorer(device)

    if args.self_test:
        print("Validating scorer against the authors' PIT_SISNRi:")
        self_test(scorer, device)
        print("harness OK")
        return

    model = SSInference.from_pretrained(checkpoint_path=REPO, device=args.device)
    root = pathlib.Path(args.root)

    counts = sorted(set(args.n_spks))
    conf = {n: {} for n in counts}  # conf[true][pred] = tally
    rows = []

    for n in counts:
        base = root / f"{n}speakers" / "wav8k" / "min" / "tt"
        if not base.exists():
            print(f"[skip] {base} missing")
            continue
        keys = sorted(p.name for p in (base / "mix").glob("*.wav"))[: args.limit]

        oracle_s, len_s, strict_s, preds = [], [], [], []

        for idx, key in enumerate(keys):
            mix, refs = load_pair(base, key, n)
            mix_t = torch.from_numpy(mix).float().unsqueeze(0)
            mix_b = torch.from_numpy(mix).float().reshape(1, -1).to(device)
            refs_d = [torch.from_numpy(r).reshape(1, -1).float().to(device) for r in refs]

            # Oracle: separation given the true count.
            est_o = model.process_waveform(mix_t, n_spks=torch.tensor(n))["waveforms"]
            # Unknown-count: the prediction AND the audio we'd actually ship.
            est_p = model.process_waveform(mix_t)["waveforms"]
            pred = min(len(est_p), MAX_SPKS)
            preds.append(pred)
            conf[n][pred] = conf[n].get(pred, 0) + 1

            def prep(est):
                L = min([mix_b.shape[-1]] + [e.shape[-1] for e in est])
                return (
                    [e[:L].reshape(1, -1).float().to(device) for e in est],
                    [r[..., :L] for r in refs_d],
                    mix_b[..., :L],
                )

            eo, ro, mo = prep(est_o)
            oracle_s.append(scorer.score(scorer.pair_matrix(eo, ro, mo), "lenient"))

            ep, rp, mp = prep(est_p)
            Mp = scorer.pair_matrix(ep, rp, mp)
            len_s.append(scorer.score(Mp, "lenient"))
            strict_s.append(scorer.score(Mp, "strict"))

            if idx and idx % 50 == 0:
                acc = 100.0 * np.mean([p == n for p in preds])
                print(
                    f"  N={n} {idx}/{len(keys)}  oracle {np.mean(oracle_s):.2f}  "
                    f"lenient {np.mean(len_s):.2f}  strict {np.mean(strict_s):.2f}  "
                    f"acc {acc:.1f}%",
                    flush=True,
                )

        rows.append(
            dict(
                n=n,
                cnt=len(keys),
                oracle=float(np.mean(oracle_s)),
                lenient=float(np.mean(len_s)),
                strict=float(np.mean(strict_s)),
                acc=100.0 * float(np.mean([p == n for p in preds])),
                preds=preds,
            )
        )
        r = rows[-1]
        print(
            f"N={n}  n={r['cnt']:4d}  oracle={r['oracle']:6.2f}  "
            f"lenient={r['lenient']:6.2f} (gap {r['oracle']-r['lenient']:5.2f})  "
            f"strict={r['strict']:6.2f} (gap {r['oracle']-r['strict']:5.2f})  "
            f"acc={r['acc']:5.1f}%",
            flush=True,
        )

    # ---- report ----
    print(f"\n=== Step 0 baseline: built-in counter on {args.label} ===\n")

    seen = sorted({p for n in conf for p in conf[n]})
    print("Confusion matrix (rows = true N, cols = predicted N, % of row):")
    print("        " + "".join(f"{p:>8}" for p in seen) + "     n")
    for r in rows:
        n = r["n"]
        tot = r["cnt"]
        cells = "".join(f"{100.0*conf[n].get(p,0)/tot:>7.1f}%" for p in seen)
        print(f"  N={n} |{cells}  {tot:>5}")

    print("\nMean predicted count vs true (direction of error):")
    for r in rows:
        mp = float(np.mean(r["preds"]))
        bias = mp - r["n"]
        print(f"  N={r['n']}: mean pred {mp:5.2f}  bias {bias:+5.2f}")

    print("\nEnd-to-end SI-SNRi (dB) and gap vs oracle:")
    print(f"{'N':>2} {'mixes':>6} {'acc':>7} {'oracle':>8} {'lenient':>8} {'gap':>6} {'strict':>8} {'gap':>6}")
    for r in rows:
        print(
            f"{r['n']:>2} {r['cnt']:>6} {r['acc']:>6.1f}% {r['oracle']:>8.2f} "
            f"{r['lenient']:>8.2f} {r['oracle']-r['lenient']:>6.2f} "
            f"{r['strict']:>8.2f} {r['oracle']-r['strict']:>6.2f}"
        )

    if rows:
        print(
            f"\nMEAN gap  lenient {np.mean([r['oracle']-r['lenient'] for r in rows]):.2f} dB"
            f"   strict {np.mean([r['oracle']-r['strict'] for r in rows]):.2f} dB"
        )

    if args.out:
        np.savez(
            args.out,
            **{f"preds_{r['n']}": np.array(r["preds"]) for r in rows},
            summary=np.array(
                [[r["n"], r["cnt"], r["acc"], r["oracle"], r["lenient"], r["strict"]] for r in rows]
            ),
        )
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
