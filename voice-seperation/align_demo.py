"""Align saved estimates to their matching references and write a listening manifest.

Separation output has no inherent speaker order, so est_spk1 may correspond to
ref_spk3. This finds the best (PIT) assignment per example, rewrites the estimates
in reference order, and reports per-speaker SI-SNRi so each file has a number.
"""

import glob
import os
import pathlib
from itertools import permutations

import librosa
import numpy as np
import soundfile as sf

SR = 8000


def si_snr(est, ref, eps=1e-20):
    est = est - est.mean()
    ref = ref - ref.mean()
    proj = np.dot(est, ref) / (np.dot(ref, ref) + eps) * ref
    return 10 * np.log10((np.sum(proj**2) + eps) / (np.sum((est - proj) ** 2) + eps))


rows = []
for n in [2, 3, 4, 5]:
    for d in sorted(glob.glob(f"separated_wsj0/{n}spk/*/")):
        d = pathlib.Path(d)
        mix, _ = librosa.load(d / "mixture.wav", sr=SR)
        ests = [librosa.load(d / f"est_spk{i+1}.wav", sr=SR)[0] for i in range(n)]
        refs = [librosa.load(d / f"ref_spk{i+1}.wav", sr=SR)[0] for i in range(n)]
        L = min([len(mix)] + [len(x) for x in ests + refs])
        mix, ests, refs = mix[:L], [e[:L] for e in ests], [r[:L] for r in refs]

        # best assignment of estimates -> references
        best, best_perm = -1e9, None
        for p in permutations(range(n)):
            score = np.mean([si_snr(ests[p[j]], refs[j]) for j in range(n)])
            if score > best:
                best, best_perm = score, p

        per_spk = []
        for j in range(n):
            imp = si_snr(ests[best_perm[j]], refs[j]) - si_snr(mix, refs[j])
            per_spk.append(imp)
            sf.write(d / f"aligned_spk{j+1}.wav", ests[best_perm[j]], SR)

        rows.append((n, d.name, np.mean(per_spk), per_spk))

print(f"{'N':>2}  {'mean SI-SNRi':>12}  per-speaker SI-SNRi (dB)")
for n, name, mean, per in rows:
    print(f"{n:>2}  {mean:>9.2f} dB  " + "  ".join(f"{x:5.1f}" for x in per))

print("\nBest example per speaker count (loudest improvement):")
for n in [2, 3, 4, 5]:
    sub = [r for r in rows if r[0] == n]
    b = max(sub, key=lambda r: r[2])
    print(f"  {n}spk  {b[2]:.2f} dB  separated_wsj0/{n}spk/{b[1][:52]}")
