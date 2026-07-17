"""Tier 1 evaluation: per-mixture SI-SNRi for a given checkpoint, with and
without mixture consistency, at oracle speaker count.

Emits PER-MIXTURE scores keyed by filename so two checkpoints can be compared
*paired* on identical mixtures. At the sample sizes used here (25/config) an
unpaired comparison of two means cannot resolve anything under ~1.5 dB; the
paired delta cancels per-mixture difficulty and resolves ~0.2 dB.

Oracle count is used throughout (n_spks=N is passed): this measures separation
quality, not counting. Counting is a separate, already-answered question.

Resumable per (repo, count): rerun the identical command and it computes only
the missing counts. No --resume flag.

    python tier1_eval.py --repo shinuh/sr-corrnet-ss-1ch-wsj-fix-2spk \
        --root data/wsj0_kmix --n-spks 2 --limit 25 --out cache_tier1/wsj0_fix2.npz
"""

import argparse
import pathlib
import sys
import time

import librosa
import numpy as np
import torch
from tqdm import tqdm

from sr_corrnet import SSInference
from sr_corrnet.models.SR_CorrNet_SS.loss import PIT_SISNRi

SR = 8000


def emit(msg):
    """tqdm.write() does NOT flush; a redirected stdout is block-buffered."""
    tqdm.write(msg)
    sys.stdout.flush()


def load_pair(base, key, n):
    mix, _ = librosa.load(base / "mix" / key, sr=SR, mono=True)
    refs = [librosa.load(base / f"s{i + 1}" / key, sr=SR, mono=True)[0] for i in range(n)]
    m = min([len(mix)] + [len(r) for r in refs])
    return mix[:m], [r[:m] for r in refs]


# Imported from the shipping path on purpose: the thing measured must be the
# thing that ships. See pipeline.py for why the textbook formula needs the
# global-gain correction, and probe_output_scale.py for the evidence.
from pipeline import mixture_consistency  # noqa: E402


def load_cache(path):
    if not path.exists():
        return {}
    try:
        with np.load(path, allow_pickle=True) as z:
            return {k: z[k] for k in z.files}
    except Exception as e:
        emit(f"[warn] unreadable cache {path} ({e}) -- treating as empty")
        return {}


def save_cache(path, data):
    """Atomic: np.savez silently appends .npz, so the temp name MUST end in .npz
    or the replace() targets a file that was never written (see HANDOFF gotcha 10)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.npz")
    np.savez(tmp, **data)
    tmp.replace(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--root", required=True)
    ap.add_argument("--n-spks", type=int, nargs="+", default=[2, 3, 4, 5])
    ap.add_argument("--limit", type=int, default=25)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", required=True)
    ap.add_argument(
        "--seed",
        type=int,
        default=None,
        help="randomly sample --limit mixtures with this seed instead of taking the "
        "alphabetically-first. The default (sorted prefix) is NOT a random sample: WSJ0 keys "
        "lead with a speaker id, so the first N mixtures all share a leading speaker and the "
        "first 25 are nearly one session. Use a seed to check a result generalizes.",
    )
    args = ap.parse_args()

    out = pathlib.Path(args.out)
    cache = load_cache(out)
    root = pathlib.Path(args.root)

    # Short-circuit BEFORE loading the model, so a cached rerun costs a second.
    todo = []
    for n in args.n_spks:
        k = f"base_{n}"
        if k in cache and len(cache[k]) >= args.limit:
            continue
        if not (root / f"{n}speakers" / "wav8k" / "min" / "tt" / "mix").exists():
            emit(f"[skip] N={n}: no mixtures under {root}")
            continue
        todo.append(n)

    have = sorted(int(k.split("_")[1]) for k in cache if k.startswith("base_"))
    emit(f"resume: {out.name} already has {have or '[]'}; still need {todo or '[]'}")
    if not todo:
        emit("nothing to do")
        return

    model = SSInference.from_pretrained(checkpoint_path=args.repo, device=args.device)
    device = torch.device(args.device)
    sisnri = PIT_SISNRi(scale_inv=True, device=device)

    for n in todo:
        base = root / f"{n}speakers" / "wav8k" / "min" / "tt"
        keys = sorted(p.name for p in (base / "mix").glob("*.wav"))
        if args.seed is not None:
            # Seeded so the same --seed gives the same mixtures across checkpoints,
            # which is what keeps the comparison paired.
            rng = np.random.default_rng(args.seed)
            keys = sorted(rng.choice(keys, size=min(args.limit, len(keys)), replace=False).tolist())
        else:
            keys = keys[: args.limit]
        b_scores, m_scores, wall_s, audio_s = [], [], 0.0, 0.0
        t0 = time.time()

        bar = tqdm(keys, desc=f"{args.repo.split('/')[-1]} N={n}", disable=None)
        for key in bar:
            mix, refs = load_pair(base, key, n)
            mix_t = torch.from_numpy(mix).float()

            t = time.time()
            res = model.process_waveform(mix_t.unsqueeze(0), n_spks=torch.tensor(n))
            wall_s += time.time() - t
            audio_s += len(mix) / SR

            est = res["waveforms"]
            L = min([len(mix)] + [e.shape[-1] for e in est] + [len(r) for r in refs])
            est_stack = torch.stack([e[:L].float() for e in est]).to(device)

            mix_dev = mix_t[:L].to(device)
            est_mc = mixture_consistency(est_stack, mix_dev)

            ref_l = [torch.from_numpy(r[:L]).reshape(1, -1).float().to(device) for r in refs]
            mix_b = mix_dev.reshape(1, -1)

            with torch.no_grad():
                b = sisnri([e.reshape(1, -1) for e in est_stack], ref_l, mix_b).item()
                m = sisnri([e.reshape(1, -1) for e in est_mc], ref_l, mix_b).item()
            b_scores.append(b)
            m_scores.append(m)
            bar.set_postfix(base=f"{np.mean(b_scores):.2f}", mc=f"{np.mean(m_scores):.2f}")

        cache[f"base_{n}"] = np.array(b_scores)
        cache[f"mc_{n}"] = np.array(m_scores)
        cache[f"keys_{n}"] = np.array(keys)
        cache[f"rtf_{n}"] = np.array([wall_s / audio_s])
        save_cache(out, cache)
        emit(
            f"N={n} n={len(keys)} base={np.mean(b_scores):6.2f} dB  "
            f"mc={np.mean(m_scores):6.2f} dB  delta={np.mean(m_scores) - np.mean(b_scores):+.3f}  "
            f"RTF={wall_s / audio_s:.3f}  [{time.time() - t0:.0f}s]"
        )


if __name__ == "__main__":
    main()
