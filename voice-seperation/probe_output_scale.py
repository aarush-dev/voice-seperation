"""Evidence for the output-gain finding in results_tier1.md section 1.2.

Answers: what scale/polarity does this model's output have, is that gain GLOBAL
across streams, and does sum(gain * est) reconstruct the mixture?

Why it matters (two live consequences, both silent under SI-SNR):
  1. Mixture consistency is NOT scale invariant. The textbook formula assumes
     sum(est) ~= mix. Here it is off by ~87x AND sign-flipped, so applied naively
     it annihilates the estimates -- and SI-SNR, being scale invariant, reads that
     as an ordinary regression rather than a bug.
  2. separate.py rescales all streams by ONE shared gain, on the assumption that
     doing so preserves relative speaker levels. That assumption is only true if
     the model's arbitrary gain is global rather than per-stream. This measures it.

Costs ~30 s of GPU. Run it before trusting either.

    SR_CorrNet_SS/.venv/Scripts/python.exe probe_output_scale.py
"""

import pathlib

import librosa
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

from sr_corrnet import SSInference

SR = 8000
REPO = "shinuh/sr-corrnet-ss-1ch-wsj-var-2-5spk"

root = pathlib.Path("data/wsj0_kmix")
model = SSInference.from_pretrained(checkpoint_path=REPO, device="cuda:0")

print(f"model: {REPO}\n")
for n in (2, 3, 4):
    base = root / f"{n}speakers" / "wav8k" / "min" / "tt"
    keys = sorted(p.name for p in (base / "mix").glob("*.wav"))[:3]
    print(f"=== N={n} ===")
    for key in keys:
        mix, _ = librosa.load(base / "mix" / key, sr=SR, mono=True)
        refs = [librosa.load(base / f"s{i + 1}" / key, sr=SR, mono=True)[0] for i in range(n)]
        mix_t = torch.from_numpy(mix).float()
        out = model.process_waveform(mix_t.unsqueeze(0), n_spks=torch.tensor(n))["waveforms"]
        L = min([len(mix)] + [e.shape[-1] for e in out] + [len(r) for r in refs])
        E = torch.stack([e[:L].float().cpu() for e in out]).numpy()
        R = np.stack([r[:L] for r in refs])

        # PIT-align estimates to references by |cosine|
        C = np.array(
            [
                [
                    abs(np.dot(E[i], R[j])) / (np.linalg.norm(E[i]) * np.linalg.norm(R[j]))
                    for j in range(n)
                ]
                for i in range(n)
            ]
        )
        ei, rj = linear_sum_assignment(-C)

        # per-stream optimal gain mapping est_i -> its matched reference
        a = np.array([np.dot(E[i], R[j]) / np.dot(E[i], E[i]) for i, j in zip(ei, rj)])
        # blind gain: least squares against the MIXTURE only (available at test time)
        s = E.sum(0)
        a_blind = np.dot(mix[:L], s) / np.dot(s, s)
        recon = sum(a[k] * E[i] for k, i in enumerate(ei))
        rel = np.linalg.norm(recon - mix[:L]) / np.linalg.norm(mix[:L])

        print(f"  {key[:30]}")
        print(f"    |cos| to refs      = {np.round(C[ei, rj], 3)}   (separation is good)")
        print(f"    per-stream gain a_i = {np.round(a, 5)}")
        print(f"    gain spread max/min = {abs(a).max() / abs(a).min():.3f}   (~1.0 => GLOBAL, not per-stream)")
        print(f"    blind gain (mix only) = {a_blind:.6f}  vs reference-derived mean {a.mean():.6f}")
        print(f"    ||mix - sum(a_i*est_i)||/||mix|| = {rel:.4f}   (small => gains restore consistency)")
    print()

print(
    "Expected: gains ~-0.011 (NEGATIVE => output is polarity-inverted), spread ~1.00-1.01\n"
    "(=> the gain is global, so one shared rescale is correct), blind gain matching the\n"
    "reference-derived gain to ~4 decimals (=> mixture consistency needs no references),\n"
    "and reconstruction error ~3% (=> a real but small residual for MC to redistribute)."
)
