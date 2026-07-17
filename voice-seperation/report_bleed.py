"""Turn diagnose_bleed.py checkpoints into results_bleed.md -- a self-contained report.

Writes the markdown file directly with encoding="utf-8" rather than printing to a shell
redirect: on Windows a redirected stdout encodes as cp1252 and silently turns non-ASCII
(±, arrows) into replacement chars in the saved file.

    python report_bleed.py                 # regenerate results_bleed.md
"""

import pathlib
import sys

import numpy as np

DATASETS = [
    ("wsj0", "WSJ0-mix", "IN DOMAIN", "data/wsj0_kmix", "official WSJ0-{2,3,4,5}mix `tt` — the corpus the model was trained on"),
    ("libri", "LibriSpeech dev-clean", "OUT OF DOMAIN", "data/libri_kmix_test", "built by `make_mixtures.py` from LibriSpeech dev-clean — the model never saw this corpus"),
]
COUNTS = (2, 3, 4, 5)


def load(out_dir, label, n):
    f = out_dir / f"{label}_{n}spk.npz"
    return np.load(f) if f.exists() else None


def table(out_dir, label):
    rows = ["| N | mixes | SDR (dB) | SIR (dB) | SAR (dB) | SIR−SAR | mean max-corr | % mixes >0.5 |",
            "|---|-------|----------|----------|----------|---------|---------------|--------------|"]
    stats = {}
    for n in COUNTS:
        d = load(out_dir, label, n)
        if d is None:
            continue
        sdr, sir, sar, corr = d["sdr"], d["sir"], d["sar"], d["corr"]
        pct = 100.0 * float((corr > 0.5).mean())
        gap = sir.mean() - sar.mean()
        stats[n] = dict(sdr=sdr.mean(), sir=sir.mean(), sar=sar.mean(), gap=gap,
                        corr=corr.mean(), pct=pct, n=len(sdr))
        rows.append(
            f"| {n} | {len(sdr)} | {sdr.mean():.2f} ± {sdr.std():.2f} | "
            f"{sir.mean():.2f} ± {sir.std():.2f} | {sar.mean():.2f} ± {sar.std():.2f} | "
            f"+{gap:.1f} | {corr.mean():.3f} | {pct:.1f}% |"
        )
    return "\n".join(rows), stats


def main():
    out_dir = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "results_bleed")
    dest = pathlib.Path("results_bleed.md")

    parts = [
        "# Bleed diagnosis — SR-CorrNet separated streams",
        "",
        "**Verdict: processing artifacts, NOT cross-talker bleed.** Generated 2026-07-17.",
        "",
        "Question asked: *why do the separated streams have audible bleed?* Answer: the measured",
        "residual is dominated by artifacts, not by other talkers. Details below.",
        "",
        "> Regenerate this file: `SR_CorrNet_SS/.venv/Scripts/python.exe report_bleed.py`",
        "",
        "---",
        "",
        "## Results",
        "",
    ]

    all_stats = {}
    for label, title, domain, root, note in DATASETS:
        tbl, stats = table(out_dir, label)
        all_stats[label] = stats
        parts += [f"### {title} — {domain}", "", f"`{root}` — {note}", "", tbl, ""]

    w, l = all_stats.get("wsj0", {}), all_stats.get("libri", {})

    def gaps(s):
        return " → ".join(f"{s[n]['gap']:.1f}" for n in COUNTS if n in s)

    parts += [
        "---",
        "",
        "## Diagnosis",
        "",
        "### 1. Cross-talker leakage — MEASURED FALSE (as the dominant error)",
        "",
        "SIR runs **4–11 dB above** SAR at every speaker count in both domains, and SAR tracks",
        "SDR to within ~0.4–2 dB. SDR is therefore capped by the **artifact** term, not the",
        "interference term. This is the *\"SIR high, SAR low → processing artifacts\"* case.",
        "",
        "Real cross-talker leakage would show SIR around 16–18 dB. In-domain SIR is **23–35 dB**.",
        "A discriminative second stage conditioned on all streams jointly would be attacking a",
        "problem that is not the bottleneck.",
        "",
        f"The SIR−SAR margin *does* shrink monotonically with N (WSJ0: {gaps(w)} dB; ",
        f"LibriSpeech: {gaps(l)} dB), so interference grows in relative importance with speaker",
        "count — but never overtakes artifacts anywhere in 2–5 speakers.",
        "",
        "### 2. Attractor collapse — not dominant, but not zero",
        "",
        "Mean max off-diagonal correlation is **0.009–0.108**, far below the 0.5 threshold. But it",
        "is not identically zero, and that tail should not be rounded away:",
        "",
        "| case | mixtures with any off-diag pair >0.5 |",
        "|------|--------------------------------------|",
    ]
    for label, title, *_ in DATASETS:
        for n in COUNTS:
            s = all_stats.get(label, {}).get(n)
            if s and s["pct"] > 0:
                parts.append(f"| {title} N={n} | {s['pct']:.0f}% ({int(round(s['pct']*s['n']/100))}/{s['n']}) |")
    parts += [
        "",
        "So collapse is a **real minority tail at high speaker counts**, not the main failure.",
        "Do not claim it never happens.",
        "",
        "### 3. Eval set / domain — this confounds the premise",
        "",
        "The SI-SNRi figures quoted as \"current results\" (**19.14 / 18.61 / 18.04 / 16.48** at",
        "N=2/3/4/5) are the **LibriSpeech dev-clean (out-of-domain)** row, not",
        "WSJ0-mix. The in-domain WSJ0 figures are **23.39 / 23.84 / 22.20 / 20.57** SI-SNRi, which",
        "match the paper.",
        "",
        "The suspected 18.61-vs-24.2 dB gap at 3 speakers is therefore **corpus, not a separation",
        "defect**. Any conclusion about model quality drawn from those numbers is confounded.",
        "",
        "---",
        "",
        "## Method",
        "",
        "| | |",
        "|---|---|",
        "| Model | `shinuh/sr-corrnet-ss-1ch-wsj-var-2-5spk` |",
        "| Speaker count | **KNOWN** (`n_spks=torch.tensor(N)`) — isolates separation from counting |",
        "| Eval size | 100 mixtures per count per dataset (**800 total**) |",
        "| Audio | 8 kHz mono (the model is 8 kHz only) |",
        "",
        "**SDR / SIR / SAR** — `mir_eval.separation.bss_eval_sources`, PIT-matched (permutation",
        "from mir_eval). These are **absolute** values, **not improvements**. Do *not* compare them",
        "to the SI-SNRi numbers in `results_wsj0.txt`, which are improvements over",
        "the mixture. Reported as mean ± sd across mixtures.",
        "",
        "**mean max-corr** — per mixture, the estimated streams are mean-removed and L2-normalised,",
        "`C = (Sn @ Sn.T).abs()`, diagonal zeroed, take the max; then averaged over mixtures.",
        "",
        "**% mixes >0.5** — share of mixtures where *any* off-diagonal pair exceeds 0.5. This is the",
        "attractor-collapse test (two attractors latched onto one speaker while another got split).",
        "",
        "### Reproduce",
        "",
        "Runs are resumable and these are also the resume commands — rerunning after an",
        "interruption continues where it stopped; rerunning when complete is a ~1 s no-op.",
        "",
        "```bash",
        "P=SR_CorrNet_SS/.venv/Scripts/python.exe",
        "$P diagnose_bleed.py --root data/wsj0_kmix       --label wsj0  --limit 100",
        "$P diagnose_bleed.py --root data/libri_kmix_test --label libri --limit 100",
        "$P report_bleed.py            # regenerates this file",
        "$P progress.py --watch        # live bars + ETA, safe to run against a live job",
        "```",
        "",
        "### Artifacts on disk",
        "",
        "| path | what |",
        "|---|---|",
        "| `results_bleed/{wsj0,libri}_{N}spk.npz` | per-mixture `keys/sdr/sir/sar/corr` — the expensive part (~1.5 h of `bss_eval`) |",
        "| `data/separations/{wsj0,libri}/{N}spk/<key>/est_spk*.wav` | separated audio, shared-gain rescaled |",
        "| `data/separations/.../manifest.json` | source root/key, gain, peak — `raw = wav * gain` |",
        "",
        "Audio was written by `save_separations.py`. Mixtures and references are **not** duplicated;",
        "they already exist under `data/wsj0_kmix` and `data/libri_kmix_test`, and the manifest",
        "points back at them.",
        "",
        "### Precision",
        "",
        "100 mixtures/count with sd 1.4–5.1 dB gives a standard error of roughly **±0.15–0.5 dB**.",
        "Ample to resolve a 4–11 dB SIR/SAR gap; **do not read the third digit**. Larger sets are on",
        "disk if tighter bounds are ever needed (3000/count for WSJ0, 500 for LibriSpeech) — pass a",
        "bigger `--limit`.",
        "",
    ]

    dest.write_text("\n".join(parts), encoding="utf-8")
    print(f"wrote {dest} ({len('\n'.join(parts))} bytes)")


if __name__ == "__main__":
    main()
