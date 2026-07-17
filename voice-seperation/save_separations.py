"""Save SR-CorrNet's separated streams to data/separations/ as durable artifacts.

Separate from diagnose_bleed.py on purpose: inference is cheap (RTF ~0.05-0.1) but
bss_eval_sources is not, so regenerating audio costs ~1 min/count while re-running the
metrics costs hours. Keeping them apart means audio can be (re)generated at any time
without touching an in-flight metrics run.

Saves ONLY the estimated streams. The mixture and references already exist under
data/{wsj0_kmix,libri_kmix_test} -- re-saving them would duplicate ~4.7 GB of corpus
to no benefit. manifest.json records the source root + key so every stream is traceable
back to its inputs.

CRITICAL: raw streams peak at ~40-70 and clip to garbage if written
straight to a wav. They are rescaled by ONE shared gain across the streams of a mixture,
which preserves relative speaker levels. The gain is recorded in the manifest, so the raw
model output is recoverable exactly: raw = wav * gain.

    python save_separations.py --root data/wsj0_kmix --label wsj0 --limit 100
"""

import argparse
import json
import pathlib
import sys

import librosa
import numpy as np
import soundfile as sf
import torch
from tqdm import tqdm

REPO = "shinuh/sr-corrnet-ss-1ch-wsj-var-2-5spk"
SR = 8000
CKPT = 20


def emit(msg):
    """Progress line that actually reaches a redirected log -- tqdm.write() does not
    flush, and a redirected python stdout is block-buffered."""
    tqdm.write(msg)
    sys.stdout.flush()


def load_mix(base, key):
    mix, _ = librosa.load(base / "mix" / key, sr=SR, mono=True)
    return mix


def save_manifest(path, data):
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data, indent=1, sort_keys=True))
    tmp.replace(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--n-spks", type=int, nargs="+", default=[2, 3, 4, 5])
    ap.add_argument("--limit", type=int, default=100)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out-dir", default="data/separations")
    args = ap.parse_args()

    root = pathlib.Path(args.root)
    out_root = pathlib.Path(args.out_dir) / args.label

    # Work out what's outstanding BEFORE loading the model.
    todo = {}
    for n in args.n_spks:
        base = root / f"{n}speakers" / "wav8k" / "min" / "tt"
        if not base.exists():
            print(f"[skip] {base} missing")
            continue
        keys = sorted(p.name for p in (base / "mix").glob("*.wav"))[: args.limit]
        mpath = out_root / f"{n}spk" / "manifest.json"
        man = {}
        if mpath.exists():
            try:
                man = json.loads(mpath.read_text())
            except ValueError:
                man = {}
        # Trust the manifest only if the wavs it claims are actually on disk.
        done = {
            k
            for k, v in man.items()
            if all((out_root / f"{n}spk" / f).exists() for f in v["files"])
        }
        remaining = [k for k in keys if k not in done]
        if remaining:
            todo[n] = (base, remaining, mpath, man)
        print(f"resume: {args.label} N={n}: have {len(done)}/{len(keys)}, need {len(remaining)}", flush=True)

    if not todo:
        print(f"\n{args.label}: nothing outstanding, all separations saved.")
        return

    from sr_corrnet import SSInference

    model = SSInference.from_pretrained(checkpoint_path=REPO, device=args.device)

    for n, (base, remaining, mpath, man) in todo.items():
        d_out = mpath.parent
        d_out.mkdir(parents=True, exist_ok=True)
        pbar = tqdm(
            remaining,
            desc=f"save {args.label} N={n}",
            unit="mix",
            disable=None,
            dynamic_ncols=True,
            initial=len(man),
            total=len(man) + len(remaining),
        )
        for i, key in enumerate(pbar, 1):
            mix = load_mix(base, key)
            out = model.process_waveform(
                torch.from_numpy(mix).float().unsqueeze(0), n_spks=torch.tensor(n)
            )
            raw = [e.detach().cpu().float() for e in out["waveforms"]]

            # ONE shared gain across streams -- never normalise streams independently.
            peak = max(s.abs().max().item() for s in raw)
            gain = (peak / 0.9) if peak > 0 else 1.0

            stem = key.replace(".wav", "")
            files = []
            for j, s in enumerate(raw, 1):
                rel = f"{stem}/est_spk{j}.wav"
                (d_out / stem).mkdir(parents=True, exist_ok=True)
                sf.write(d_out / rel, (s / gain).numpy(), SR, subtype="PCM_16")
                files.append(rel)

            man[key] = {
                "n_spks": n,
                "gain": gain,  # raw = wav * gain
                "source_root": str(base).replace("\\", "/"),
                "source_key": key,
                "files": files,
                "peak_raw": peak,
            }

            if i % CKPT == 0 or i == len(remaining):
                save_manifest(mpath, man)
                emit(f"  {args.label} N={n} {i}/{len(remaining)} saved -> {d_out}")
        pbar.close()
        save_manifest(mpath, man)

    print(f"\n{args.label}: done.")


if __name__ == "__main__":
    main()
