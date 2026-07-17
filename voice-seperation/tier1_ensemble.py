"""Tier 1.3: test-time ensembling over input offsets.

The model is single-pass (no CSS chunking -- verified by reading
engine_infer._single_pass_session), so "chunk offset" here means shifting the
input relative to the STFT frame grid. Sub-hop shifts change which samples land
in which analysis frame, so the artifacts differ between runs while the signal
does not. Averaging aligned runs should cancel artifacts -- which is the right
target, since results_bleed.md found the error is SAR-type.

TWO TRAPS, both silent:

1. Each forward pass has its own ARBITRARY global gain, including SIGN (the
   output is polarity-inverted, gain ~= -0.011 -- see scratch_probe_scale.py).
   Averaging runs without sign/scale matching lets opposite polarities CANCEL,
   destroying the signal. Every run is least-squares projected onto run 0 first.

2. Stream order is not stable across runs -- speaker 1 in run 0 can emerge as
   speaker 3 in run 1. Streams are Hungarian-matched on |cosine| before
   averaging. (This is the permutation-drift problem that does NOT exist across
   time for this model, but DOES exist across independent runs.)

    python tier1_ensemble.py --root data/wsj0_kmix --n-spks 2 3 4 5 --limit 25
"""

import argparse
import pathlib
import sys
import time

import librosa
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm

from sr_corrnet import SSInference
from sr_corrnet.models.SR_CorrNet_SS.loss import PIT_SISNRi
from tier1_eval import mixture_consistency

SR = 8000
REPO = "shinuh/sr-corrnet-ss-1ch-wsj-var-2-5spk"


def emit(msg):
    tqdm.write(msg)
    sys.stdout.flush()


def load_pair(base, key, n):
    mix, _ = librosa.load(base / "mix" / key, sr=SR, mono=True)
    refs = [librosa.load(base / f"s{i + 1}" / key, sr=SR, mono=True)[0] for i in range(n)]
    m = min([len(mix)] + [len(r) for r in refs])
    return mix[:m], [r[:m] for r in refs]


def run_at_offset(model, mix_t, n, d, L):
    """Front-pad by d samples, run, then undo the shift. Returns (N, L)."""
    x = torch.nn.functional.pad(mix_t, (d, 0)) if d else mix_t
    out = model.process_waveform(x.unsqueeze(0), n_spks=torch.tensor(n))["waveforms"]
    est = torch.stack([e.float().cpu() for e in out])
    est = est[:, d:] if d else est
    return est[:, :L]


def align_to(ref, cand):
    """Permute + sign/scale-match cand (N,L) onto ref (N,L) by least squares."""
    n = ref.shape[0]
    C = np.zeros((n, n))
    rn = ref.norm(dim=1).numpy() + 1e-12
    cn = cand.norm(dim=1).numpy() + 1e-12
    for i in range(n):
        for j in range(n):
            C[i, j] = abs(torch.dot(cand[i], ref[j]).item()) / (cn[i] * rn[j])
    ci, ri = linear_sum_assignment(-C)
    out = torch.zeros_like(ref)
    for i, j in zip(ci, ri):
        # signed least-squares gain -- this is what fixes the polarity flip
        a = torch.dot(cand[i], ref[j]) / (torch.dot(cand[i], cand[i]) + 1e-12)
        out[j] = a * cand[i]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--n-spks", type=int, nargs="+", default=[2, 3, 4, 5])
    ap.add_argument("--limit", type=int, default=25)
    ap.add_argument("--offsets", type=int, nargs="+", default=[0, 16, 32, 48])
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default="cache_tier1/ensemble.npz")
    args = ap.parse_args()

    out = pathlib.Path(args.out)
    cache = {}
    if out.exists():
        with np.load(out, allow_pickle=True) as z:
            cache = {k: z[k] for k in z.files}

    root = pathlib.Path(args.root)
    # Resume keys off ensmc_, the last-added column: a cache from before ensmc
    # existed must recompute, not silently report a missing column as done.
    todo = [
        n for n in args.n_spks if f"ensmc_{n}" not in cache or len(cache[f"ensmc_{n}"]) < args.limit
    ]
    have = sorted(int(k.split("_")[1]) for k in cache if k.startswith("ensmc_"))
    emit(f"resume: already has {have or '[]'}; still need {todo or '[]'}")
    if not todo:
        emit("nothing to do")
        return

    model = SSInference.from_pretrained(checkpoint_path=REPO, device=args.device)
    device = torch.device(args.device)
    sisnri = PIT_SISNRi(scale_inv=True, device=device)

    for n in todo:
        base = root / f"{n}speakers" / "wav8k" / "min" / "tt"
        keys = sorted(p.name for p in (base / "mix").glob("*.wav"))[: args.limit]
        single, ens, ensmc = [], [], []
        t0 = time.time()

        for key in tqdm(keys, desc=f"ens N={n}", disable=None):
            mix, refs = load_pair(base, key, n)
            mix_t = torch.from_numpy(mix).float()
            L = len(mix)

            runs = [run_at_offset(model, mix_t, n, d, L) for d in args.offsets]
            L = min(r.shape[1] for r in runs)
            runs = [r[:, :L] for r in runs]

            ref0 = runs[0]
            stack = torch.stack([ref0] + [align_to(ref0, r) for r in runs[1:]])
            avg = stack.mean(dim=0)

            # do the two artifact fixes stack? MC applied on top of the ensemble
            avg_mc = mixture_consistency(avg.to(device), mix_t[:L].to(device))

            ref_l = [torch.from_numpy(r[:L]).reshape(1, -1).float().to(device) for r in refs]
            mix_b = mix_t[:L].reshape(1, -1).to(device)
            with torch.no_grad():
                single.append(
                    sisnri([e.reshape(1, -1).to(device) for e in ref0], ref_l, mix_b).item()
                )
                ens.append(sisnri([e.reshape(1, -1).to(device) for e in avg], ref_l, mix_b).item())
                ensmc.append(
                    sisnri([e.reshape(1, -1) for e in avg_mc], ref_l, mix_b).item()
                )

        cache[f"single_{n}"] = np.array(single)
        cache[f"ens_{n}"] = np.array(ens)
        cache[f"ensmc_{n}"] = np.array(ensmc)
        cache[f"keys_{n}"] = np.array(keys)
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = out.with_suffix(".tmp.npz")
        np.savez(tmp, **cache)
        tmp.replace(out)

        d = np.array(ens) - np.array(single)
        emit(
            f"N={n} n={len(keys)} single={np.mean(single):6.2f} ens={np.mean(ens):6.2f} "
            f"paired delta={d.mean():+.3f} +/- {1.96 * d.std(ddof=1) / np.sqrt(len(d)):.3f}  "
            f"[{time.time() - t0:.0f}s]"
        )


if __name__ == "__main__":
    main()
