"""Diagnose separation bleed: BSS_Eval (SDR/SIR/SAR) + inter-stream correlation.

Runs with the count KNOWN (n_spks=tensor(N)) so this measures separation quality
alone, with counting taken out of the picture.

Two questions:
  1. SIR vs SAR -- is the residual cross-talker leakage or processing artifact?
  2. off-diagonal correlation between ESTIMATED streams -- did two attractors latch
     onto the same speaker (attractor collapse)?

Resumable: checkpoints every CKPT mixtures per (dataset, N) and resumes by
inspecting the output npz. Rerun the identical command to continue.
"""

import argparse
import pathlib
import sys

import librosa
import numpy as np
import torch
from mir_eval.separation import bss_eval_sources
from tqdm import tqdm

REPO = "shinuh/sr-corrnet-ss-1ch-wsj-var-2-5spk"
SR = 8000
CKPT = 20


def emit(msg):
    """Print a progress line that ACTUALLY reaches a redirected log.

    tqdm.write() does not flush, and a redirected python stdout is block-buffered, so
    without the explicit flush a long run's log sits empty for its whole duration and
    is indistinguishable from a hang. This bit us: an entire 400-mixture pass logged
    nothing but its stderr warning while quietly working fine.
    """
    tqdm.write(msg)
    sys.stdout.flush()


def load_pair(base, key, n):
    mix, _ = librosa.load(base / "mix" / key, sr=SR, mono=True)
    refs = [librosa.load(base / f"s{i+1}" / key, sr=SR, mono=True)[0] for i in range(n)]
    m = min([len(mix)] + [len(r) for r in refs])
    return mix[:m], np.stack([r[:m] for r in refs])


def save_atomic(path, **arrays):
    # Must end in .npz -- np.savez silently appends .npz to anything that doesn't.
    tmp = path.with_suffix(".tmp.npz")
    np.savez(tmp, **arrays)
    tmp.replace(path)


def max_offdiag_corr(est):
    """est: [N, L] float64. Returns max |correlation| between distinct streams."""
    S = torch.from_numpy(est)
    Sn = S - S.mean(1, keepdim=True)
    Sn = Sn / (Sn.norm(dim=1, keepdim=True) + 1e-10)
    C = (Sn @ Sn.T).abs()
    C.fill_diagonal_(0.0)
    return C.max().item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--label", required=True, help="dataset label, names the checkpoint")
    ap.add_argument("--n-spks", type=int, nargs="+", default=[2, 3, 4, 5])
    ap.add_argument("--limit", type=int, default=100)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out-dir", default="results_bleed")
    args = ap.parse_args()

    root = pathlib.Path(args.root)
    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- work out what's outstanding BEFORE loading the model (short-circuit) ---
    todo = {}
    for n in args.n_spks:
        base = root / f"{n}speakers" / "wav8k" / "min" / "tt"
        if not base.exists():
            print(f"[skip] {base} missing")
            continue
        keys = sorted(p.name for p in (base / "mix").glob("*.wav"))[: args.limit]
        ck = out_dir / f"{args.label}_{n}spk.npz"
        done = set()
        if ck.exists():
            done = set(np.load(ck)["keys"].tolist())
        remaining = [k for k in keys if k not in done]
        if remaining:
            todo[n] = (base, remaining, ck)
        print(
            f"resume: {args.label} N={n}: have {len(done)}/{len(keys)}, "
            f"need {len(remaining)}",
            flush=True,
        )

    if not todo:
        print(f"\n{args.label}: nothing outstanding, all counts complete.", flush=True)
        return

    model = SSInference.from_pretrained(checkpoint_path=REPO, device=args.device)

    for n, (base, remaining, ck) in todo.items():
        rows = {k: [] for k in ("keys", "sdr", "sir", "sar", "corr")}
        if ck.exists():
            old = np.load(ck)
            for k in rows:
                rows[k] = old[k].tolist()

        # disable=None -> tqdm auto-disables when stdout is not a TTY, so redirected
        # runs keep clean line-oriented logs instead of megabytes of \r spam.
        pbar = tqdm(
            remaining,
            desc=f"{args.label} N={n}",
            unit="mix",
            disable=None,
            dynamic_ncols=True,
            initial=len(rows["keys"]),
            total=len(rows["keys"]) + len(remaining),
        )
        for i, key in enumerate(pbar, 1):
            mix, refs = load_pair(base, key, n)
            out = model.process_waveform(
                torch.from_numpy(mix).float().unsqueeze(0), n_spks=torch.tensor(n)
            )
            est = out["waveforms"]
            L = min([refs.shape[1]] + [e.shape[-1] for e in est])
            est = np.stack([e[:L].detach().cpu().float().numpy() for e in est]).astype(np.float64)
            ref = refs[:, :L].astype(np.float64)

            # mir_eval rejects all-silent sources; skip rather than crash the run.
            if (np.abs(ref).max(1) < 1e-8).any() or (np.abs(est).max(1) < 1e-8).any():
                emit(f"  [skip] {key}: silent source")
                continue

            sdr, sir, sar, _ = bss_eval_sources(ref, est)

            rows["keys"].append(key)
            rows["sdr"].append(float(np.mean(sdr)))
            rows["sir"].append(float(np.mean(sir)))
            rows["sar"].append(float(np.mean(sar)))
            rows["corr"].append(max_offdiag_corr(est))

            pbar.set_postfix(
                SDR=f"{np.mean(rows['sdr']):.2f}",
                SIR=f"{np.mean(rows['sir']):.2f}",
                SAR=f"{np.mean(rows['sar']):.2f}",
                corr=f"{np.mean(rows['corr']):.3f}",
            )

            if i % CKPT == 0 or i == len(remaining):
                save_atomic(ck, **{k: np.array(v) for k, v in rows.items()})
                # Also emit a plain line: the bar is invisible in a redirected log,
                # and a long run with no log progress is indistinguishable from a hang.
                emit(
                    f"  {args.label} N={n} {i}/{len(remaining)}  "
                    f"SDR {np.mean(rows['sdr']):.2f}  SIR {np.mean(rows['sir']):.2f}  "
                    f"SAR {np.mean(rows['sar']):.2f}  corr {np.mean(rows['corr']):.3f}"
                )
        pbar.close()

    print(f"\n{args.label}: done.")


if __name__ == "__main__":
    from sr_corrnet import SSInference  # noqa: E402  (kept off the short-circuit path)

    main()
