"""Track A: count speakers by pruning the separator's own n_spks=5 output.

Idea: always separate at the ceiling (5), then decide which streams are real.
The attractors already encode who is present -- read that rather than asking a
second model to re-derive it.

Two distinct phantom flavours, and an energy threshold only catches the first:
  1. low-energy junk      -> caught by the relative-dB prune
  2. duplicated / split   -> NOT low-energy; caught by the correlation prune
Missing (2) is the most likely way this track underdelivers, so both are here and
--ablate reports each prune alone.

Cost structure: separation is expensive, thresholding is free. So `cache` runs the
model once per mixture and stores everything a threshold could possibly need --
the pairwise SI-SNRi matrix M[ref, est] against the references, the per-stream
relative dB, and the 5x5 stream correlation matrix. `sweep` then grid-searches in
pure numpy against that cache, never touching the GPU. Retuning is seconds.
"""

import argparse
import itertools
import pathlib

import numpy as np
import torch

from sr_corrnet import SSInference

from count_baseline import MAX_SPKS, Scorer, load_pair

REPO = "shinuh/sr-corrnet-ss-1ch-wsj-var-2-5spk"


def stream_stats(streams):
    """Per-stream relative energy (dB below the loudest) and |cosine| similarity.

    Relative, not absolute, dB -- invariant to input gain, so a threshold tuned on
    one corpus is not silently a threshold on that corpus's loudness.
    """
    S = torch.stack([s.flatten() for s in streams]).float()
    p = (S**2).mean(dim=1)
    p_db = 10 * torch.log10(p + 1e-10)
    p_rel = (p_db - p_db.max()).cpu().numpy()

    Sn = S - S.mean(dim=1, keepdim=True)
    Sn = Sn / (Sn.norm(dim=1, keepdim=True) + 1e-10)
    C = (Sn @ Sn.T).abs().cpu().numpy()
    np.fill_diagonal(C, 0.0)
    return p_rel, C


def count_from_stats(p_rel, C, energy_db_thresh, corr_thresh, use_energy=True, use_corr=True):
    """Return a boolean keep-mask over streams. p_rel/C come from stream_stats()."""
    k = len(p_rel)
    keep = np.ones(k, dtype=bool)

    if use_energy:
        keep &= p_rel > -energy_db_thresh

    if use_corr:
        # Drop the quieter member of any highly-correlated surviving pair.
        order = np.argsort(-p_rel)  # loudest first, so we drop the quieter one
        for i, j in itertools.combinations(order, 2):
            if keep[i] and keep[j] and C[i, j] > corr_thresh:
                keep[j if p_rel[j] < p_rel[i] else i] = False

    if not keep.any():  # never return zero speakers
        keep[int(np.argmax(p_rel))] = True
    return keep


def build_cache(root, counts, limit, device, out_path):
    """One n_spks=5 pass per mixture; store everything a threshold could need.

    Resumable at speaker-count granularity: the .npz is rewritten after each count
    completes, and a rerun skips counts already present. Kill it any time and rerun
    the identical command; you lose at most the count in flight, not the whole run.
    """
    out_path = pathlib.Path(out_path)
    blob = {}
    if out_path.exists():
        with np.load(out_path) as z:
            blob = {k: z[k] for k in z.files}
        have = sorted({int(k.split("_")[1]) for k in blob if k.startswith("M_")})
        print(f"resume: {out_path} already has counts {have}", flush=True)

    # Resume at PARTIAL-count granularity: a count already cached with k < limit
    # mixtures only needs its tail computed. Both this and any earlier run take
    # sorted(keys)[:limit], so the cached k rows are exactly a prefix of the wanted
    # limit rows -- the tail concatenates on without recomputing the head.
    todo = []
    for n in counts:
        k = blob[f"M_{n}"].shape[0] if f"M_{n}" in blob else 0
        if k >= limit:
            continue  # already has at least what was asked for; never truncate
        todo.append((n, k))

    if not todo:
        print(f"nothing to do -- {out_path} already covers {counts} at >={limit}", flush=True)
        return
    for n, k in todo:
        if k:
            print(f"resume: N={n} has {k}, extending to {limit} ({limit-k} to compute)", flush=True)
        else:
            print(f"resume: N={n} from scratch ({limit} to compute)", flush=True)

    # Only pay the model load if there is real work left.
    model = SSInference.from_pretrained(checkpoint_path=REPO, device=device)
    scorer = Scorer(torch.device(device))
    root = pathlib.Path(root)

    for n, start in todo:
        base = root / f"{n}speakers" / "wav8k" / "min" / "tt"
        if not base.exists():
            print(f"[skip] {base} missing")
            continue
        keys = sorted(p.name for p in (base / "mix").glob("*.wav"))[:limit]
        todo_keys = keys[start:]  # only the tail; keys[:start] are already cached
        Ms, Ps, Cs, oracles = [], [], [], []

        for idx, key in enumerate(todo_keys):
            mix, refs = load_pair(base, key, n)
            mix_t = torch.from_numpy(mix).float().unsqueeze(0)
            mix_b = torch.from_numpy(mix).float().reshape(1, -1).to(device)
            refs_d = [torch.from_numpy(r).reshape(1, -1).float().to(device) for r in refs]

            est5 = model.process_waveform(mix_t, n_spks=torch.tensor(MAX_SPKS))["waveforms"]
            est_o = model.process_waveform(mix_t, n_spks=torch.tensor(n))["waveforms"]

            L = min([mix_b.shape[-1]] + [e.shape[-1] for e in est5] + [e.shape[-1] for e in est_o])
            e5 = [e[:L].reshape(1, -1).float().to(device) for e in est5]
            eo = [e[:L].reshape(1, -1).float().to(device) for e in est_o]
            rr = [r[..., :L] for r in refs_d]
            mm = mix_b[..., :L]

            p_rel, C = stream_stats(e5)
            Ms.append(scorer.pair_matrix(e5, rr, mm))  # (n, 5)
            Ps.append(p_rel)
            Cs.append(C)
            oracles.append(scorer.score(scorer.pair_matrix(eo, rr, mm), "lenient"))

            if idx and idx % 50 == 0:
                print(f"  cache N={n} {start+idx}/{len(keys)}", flush=True)

        new = {
            f"M_{n}": np.stack(Ms),
            f"P_{n}": np.stack(Ps),
            f"C_{n}": np.stack(Cs),
            f"O_{n}": np.array(oracles),
        }
        if start:  # extending: keep the cached head, append the freshly computed tail
            for k, v in new.items():
                new[k] = np.concatenate([blob[k], v], axis=0)
            assert new[f"M_{n}"].shape[0] == len(keys), "extend produced wrong row count"
        blob.update(new)

        # Checkpoint after every count, via a temp file + atomic replace so a kill
        # mid-write cannot leave a truncated .npz that poisons the next resume.
        # The temp name MUST end in .npz: np.savez silently appends ".npz" when the
        # path lacks it, so ".npz.tmp" makes numpy write "<name>.npz.tmp.npz" and the
        # replace() below then fails on a file that was never created.
        tmp = out_path.with_suffix(".tmp.npz")
        np.savez(tmp, **blob)
        tmp.replace(out_path)
        print(f"cached N={n}: {len(keys)} mixtures -> checkpointed {out_path}", flush=True)

    print(f"wrote {out_path}")


def evaluate(cache, counts, e_th, c_th, use_energy=True, use_corr=True):
    """Score a threshold pair against a cache. Pure numpy -- no GPU, no model."""
    scorer = Scorer(torch.device("cpu"))  # only uses .score(), which is numpy
    per_n, conf = {}, {}

    for n in counts:
        M, P, C, O = (cache[f"{k}_{n}"] for k in "MPCO")
        len_s, str_s, preds = [], [], []
        for i in range(len(M)):
            keep = count_from_stats(P[i], C[i], e_th, c_th, use_energy, use_corr)
            pred = int(keep.sum())
            preds.append(pred)
            Mk = M[i][:, keep]
            len_s.append(scorer.score(Mk, "lenient"))
            str_s.append(scorer.score(Mk, "strict"))
        per_n[n] = dict(
            oracle=float(O.mean()),
            lenient=float(np.mean(len_s)),
            strict=float(np.mean(str_s)),
            acc=100.0 * float(np.mean([p == n for p in preds])),
            mean_pred=float(np.mean(preds)),
        )
        conf[n] = preds
    return per_n, conf


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("cache", help="run the model once per mixture and cache stats")
    c.add_argument("--root", required=True)
    c.add_argument("--out", required=True)
    c.add_argument("--n-spks", type=int, nargs="+", default=[2, 3, 4, 5])
    c.add_argument("--limit", type=int, default=500)
    c.add_argument("--device", default="cuda:0")

    s = sub.add_parser("sweep", help="grid-search thresholds against a cache")
    s.add_argument("--cache", required=True)
    s.add_argument("--n-spks", type=int, nargs="+", default=[2, 3, 4, 5])
    s.add_argument("--policy", choices=["lenient", "strict"], default="strict")
    s.add_argument("--ablate", action="store_true", help="report each prune alone")

    e = sub.add_parser("eval", help="evaluate one threshold pair against a cache")
    e.add_argument("--cache", required=True)
    e.add_argument("--n-spks", type=int, nargs="+", default=[2, 3, 4, 5])
    e.add_argument("--energy-db", type=float, required=True)
    e.add_argument("--corr", type=float, required=True)
    e.add_argument("--label", default="")

    args = ap.parse_args()

    if args.cmd == "cache":
        build_cache(args.root, args.n_spks, args.limit, args.device, args.out)
        return

    cache = np.load(args.cache)
    counts = args.n_spks

    if args.cmd == "eval":
        per_n, conf = evaluate(cache, counts, args.energy_db, args.corr)
        report(per_n, conf, counts, f"{args.label} (energy {args.energy_db} dB, corr {args.corr})")
        return

    # ---- sweep ----
    # Grids must bracket the optimum on BOTH sides. The first sweep put the winner at
    # the low edge of both (10 dB / 0.5), which means the range was wrong, not that the
    # edge was optimal -- so both extend well below the plan's suggested [10,40]/[0.5,0.95].
    e_grid = [2, 4, 6, 8, 10, 12, 15, 20, 25, 30, 40, 60]  # 60 ~= energy prune off
    c_grid = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.7, 0.9, 1.01]  # 1.01 ~= off

    def objective(per_n):
        return float(np.mean([per_n[n]["oracle"] - per_n[n][args.policy] for n in counts]))

    print(f"Sweeping on {args.cache}; objective = mean end-to-end gap under '{args.policy}'\n")
    results = []
    for e_th in e_grid:
        for c_th in c_grid:
            per_n, _ = evaluate(cache, counts, e_th, c_th)
            gap = objective(per_n)
            acc = float(np.mean([per_n[n]["acc"] for n in counts]))
            results.append((gap, acc, e_th, c_th))
    results.sort()

    print(f"{'gap':>7} {'meanacc':>8} {'energy_db':>10} {'corr':>6}")
    for gap, acc, e_th, c_th in results[:10]:
        print(f"{gap:>7.3f} {acc:>7.1f}% {e_th:>10} {c_th:>6}")
    best = results[0]
    print(f"\nBEST: energy_db={best[2]} corr={best[3]}  gap={best[0]:.3f} dB  meanacc={best[1]:.1f}%")

    if args.ablate:
        print("\nAblation (best thresholds, each prune alone):")
        for name, ue, uc in [("energy only", True, False), ("corr only", False, True), ("both", True, True)]:
            per_n, _ = evaluate(cache, counts, best[2], best[3], ue, uc)
            print(
                f"  {name:12s} gap {objective(per_n):6.3f}  "
                f"meanacc {np.mean([per_n[n]['acc'] for n in counts]):5.1f}%"
            )

    per_n, conf = evaluate(cache, counts, best[2], best[3])
    report(per_n, conf, counts, f"Track A @ energy {best[2]} dB, corr {best[3]}")


def report(per_n, conf, counts, title):
    print(f"\n=== {title} ===\n")
    seen = sorted({p for n in counts for p in conf[n]})
    print("Confusion (rows true N, cols predicted N, % of row):")
    print("        " + "".join(f"{p:>8}" for p in seen))
    for n in counts:
        tot = len(conf[n])
        cells = "".join(f"{100.0*sum(1 for x in conf[n] if x==p)/tot:>7.1f}%" for p in seen)
        print(f"  N={n} |{cells}")
    print(f"\n{'N':>2} {'acc':>7} {'meanpred':>9} {'oracle':>8} {'lenient':>8} {'gap':>6} {'strict':>8} {'gap':>6}")
    for n in counts:
        r = per_n[n]
        print(
            f"{n:>2} {r['acc']:>6.1f}% {r['mean_pred']:>9.2f} {r['oracle']:>8.2f} "
            f"{r['lenient']:>8.2f} {r['oracle']-r['lenient']:>6.2f} "
            f"{r['strict']:>8.2f} {r['oracle']-r['strict']:>6.2f}"
        )
    print(
        f"\nMEAN gap  lenient {np.mean([per_n[n]['oracle']-per_n[n]['lenient'] for n in counts]):.2f} dB"
        f"   strict {np.mean([per_n[n]['oracle']-per_n[n]['strict'] for n in counts]):.2f} dB"
    )


if __name__ == "__main__":
    main()
