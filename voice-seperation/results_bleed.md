# Bleed diagnosis — SR-CorrNet separated streams

**Verdict: processing artifacts, NOT cross-talker bleed.** Generated 2026-07-17.

Question asked: *why do the separated streams have audible bleed?* Answer: the measured
residual is dominated by artifacts, not by other talkers. Details below.

> Regenerate this file: `SR_CorrNet_SS/.venv/Scripts/python.exe report_bleed.py`

---

## Results

### WSJ0-mix — IN DOMAIN

`data/wsj0_kmix` — official WSJ0-{2,3,4,5}mix `tt` — the corpus the model was trained on

| N | mixes | SDR (dB) | SIR (dB) | SAR (dB) | SIR−SAR | mean max-corr | % mixes >0.5 |
|---|-------|----------|----------|----------|---------|---------------|--------------|
| 2 | 100 | 23.94 ± 1.49 | 35.10 ± 2.63 | 24.35 ± 1.45 | +10.7 | 0.009 | 0.0% |
| 3 | 100 | 21.16 ± 1.49 | 31.90 ± 2.18 | 21.58 ± 1.44 | +10.3 | 0.016 | 0.0% |
| 4 | 100 | 17.72 ± 3.52 | 27.00 ± 4.76 | 18.54 ± 2.93 | +8.5 | 0.053 | 1.0% |
| 5 | 100 | 14.84 ± 3.86 | 23.14 ± 5.13 | 15.95 ± 3.12 | +7.2 | 0.108 | 4.0% |

### LibriSpeech dev-clean — OUT OF DOMAIN

`data/libri_kmix_test` — built by `make_mixtures.py` from LibriSpeech dev-clean — the model never saw this corpus

| N | mixes | SDR (dB) | SIR (dB) | SAR (dB) | SIR−SAR | mean max-corr | % mixes >0.5 |
|---|-------|----------|----------|----------|---------|---------------|--------------|
| 2 | 100 | 19.52 ± 4.53 | 28.33 ± 5.30 | 20.61 ± 3.91 | +7.7 | 0.024 | 0.0% |
| 3 | 100 | 16.77 ± 3.59 | 24.28 ± 4.42 | 18.08 ± 3.10 | +6.2 | 0.033 | 0.0% |
| 4 | 100 | 14.20 ± 3.32 | 21.23 ± 4.05 | 15.73 ± 2.79 | +5.5 | 0.060 | 1.0% |
| 5 | 100 | 11.33 ± 3.46 | 17.32 ± 4.20 | 13.27 ± 2.73 | +4.0 | 0.106 | 0.0% |

---

## Diagnosis

### 1. Cross-talker leakage — MEASURED FALSE (as the dominant error)

SIR runs **4–11 dB above** SAR at every speaker count in both domains, and SAR tracks
SDR to within ~0.4–2 dB. SDR is therefore capped by the **artifact** term, not the
interference term. This is the *"SIR high, SAR low → processing artifacts"* case.

Real cross-talker leakage would show SIR around 16–18 dB. In-domain SIR is **23–35 dB**.
A discriminative second stage conditioned on all streams jointly would be attacking a
problem that is not the bottleneck.

The SIR−SAR margin *does* shrink monotonically with N (WSJ0: 10.7 → 10.3 → 8.5 → 7.2 dB; 
LibriSpeech: 7.7 → 6.2 → 5.5 → 4.0 dB), so interference grows in relative importance with speaker
count — but never overtakes artifacts anywhere in 2–5 speakers.

### 2. Attractor collapse — not dominant, but not zero

Mean max off-diagonal correlation is **0.009–0.108**, far below the 0.5 threshold. But it
is not identically zero, and that tail should not be rounded away:

| case | mixtures with any off-diag pair >0.5 |
|------|--------------------------------------|
| WSJ0-mix N=4 | 1% (1/100) |
| WSJ0-mix N=5 | 4% (4/100) |
| LibriSpeech dev-clean N=4 | 1% (1/100) |

So collapse is a **real minority tail at high speaker counts**, not the main failure.
Do not claim it never happens.

### 3. Eval set / domain — this confounds the premise

The SI-SNRi figures quoted as "current results" (**19.14 / 18.61 / 18.04 / 16.48** at
N=2/3/4/5) are the **LibriSpeech dev-clean (out-of-domain)** row, not
WSJ0-mix. The in-domain WSJ0 figures are **23.39 / 23.84 / 22.20 / 20.57** SI-SNRi, which
match the paper.

The suspected 18.61-vs-24.2 dB gap at 3 speakers is therefore **corpus, not a separation
defect**. Any conclusion about model quality drawn from those numbers is confounded.

---

## Method

| | |
|---|---|
| Model | `shinuh/sr-corrnet-ss-1ch-wsj-var-2-5spk` |
| Speaker count | **KNOWN** (`n_spks=torch.tensor(N)`) — isolates separation from counting |
| Eval size | 100 mixtures per count per dataset (**800 total**) |
| Audio | 8 kHz mono (the model is 8 kHz only) |

**SDR / SIR / SAR** — `mir_eval.separation.bss_eval_sources`, PIT-matched (permutation
from mir_eval). These are **absolute** values, **not improvements**. Do *not* compare them
to the SI-SNRi numbers in `results_wsj0.txt`, which are improvements over
the mixture. Reported as mean ± sd across mixtures.

**mean max-corr** — per mixture, the estimated streams are mean-removed and L2-normalised,
`C = (Sn @ Sn.T).abs()`, diagonal zeroed, take the max; then averaged over mixtures.

**% mixes >0.5** — share of mixtures where *any* off-diagonal pair exceeds 0.5. This is the
attractor-collapse test (two attractors latched onto one speaker while another got split).

### Reproduce

Runs are resumable and these are also the resume commands — rerunning after an
interruption continues where it stopped; rerunning when complete is a ~1 s no-op.

```bash
P=SR_CorrNet_SS/.venv/Scripts/python.exe
$P diagnose_bleed.py --root data/wsj0_kmix       --label wsj0  --limit 100
$P diagnose_bleed.py --root data/libri_kmix_test --label libri --limit 100
$P report_bleed.py            # regenerates this file
$P progress.py --watch        # live bars + ETA, safe to run against a live job
```

### Artifacts on disk

| path | what |
|---|---|
| `results_bleed/{wsj0,libri}_{N}spk.npz` | per-mixture `keys/sdr/sir/sar/corr` — the expensive part (~1.5 h of `bss_eval`) |
| `data/separations/{wsj0,libri}/{N}spk/<key>/est_spk*.wav` | separated audio, shared-gain rescaled |
| `data/separations/.../manifest.json` | source root/key, gain, peak — `raw = wav * gain` |

Audio was written by `save_separations.py`. Mixtures and references are **not** duplicated;
they already exist under `data/wsj0_kmix` and `data/libri_kmix_test`, and the manifest
points back at them.

### Precision

100 mixtures/count with sd 1.4–5.1 dB gives a standard error of roughly **±0.15–0.5 dB**.
Ample to resolve a 4–11 dB SIR/SAR gap; **do not read the third digit**. Larger sets are on
disk if tighter bounds are ever needed (3000/count for WSJ0, 500 for LibriSpeech) — pass a
bigger `--limit`.
