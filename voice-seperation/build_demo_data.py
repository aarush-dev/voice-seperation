"""Build demo_data.js -- everything index.html displays, generated from the caches.

The demo site must never contain a hand-typed number: hand-maintained tables go stale
silently, and this project has already burned time re-deriving what a stale number meant.
So every figure on the page comes from this script, which reads the same artifacts the
results_*.md reports read:

    cache_tier1/*.npz      -> routing / mixture-consistency / ensembling paired deltas
    results_bleed/*.npz    -> BSS_Eval SDR/SIR/SAR + inter-stream correlation
    results_count_test.npz -> built-in counter confusion matrix (LibriSpeech, out of domain)
    results_track_a_test.txt -> Track A vs built-in, like-for-like
    results_wsj0.txt       -> the headline table's count accuracy + RTF
    separated_wsj0/, data/separations/ -> the listening samples

Per-sample SI-SNRi uses the authors' own PIT_SISNRi via count_baseline.Scorer -- NOT a
reimplementation (reimplementing it is forbidden, and a reimplementation is how the scale-invariance
bug hid). CPU only, ~1 min, no GPU and no model load.

    SR_CorrNet_SS/.venv/Scripts/python.exe build_demo_data.py

Rerunning is safe and cheap; it is the resume command (it recomputes everything from
scratch in about a minute, so there is nothing to checkpoint).
"""

import json
import pathlib
import re

import librosa
import numpy as np
import torch
from scipy import stats
from scipy.optimize import linear_sum_assignment

from count_baseline import Scorer, load_pair

SR = 8000
COUNTS = (2, 3, 4, 5)
ENV_BINS = 420  # waveform envelope resolution drawn on canvas
DEVICE = torch.device("cpu")
OUT = pathlib.Path("demo_data.js")

# The paper's published comparison point. This is the ONE set of numbers not derived from
# our own artifacts -- it is quoted from SepTDA's paper (known-count row), which is why it
# is isolated here and labelled as such on the page.
SEPTDA_KNOWN_COUNT = {2: 23.6, 3: 23.5, 4: 22.0, 5: 21.0}
SEPTDA_UNKNOWN_COUNT = {2: 23.6, 3: 22.1, 4: 19.5, 5: 16.9}


# --------------------------------------------------------------------------- helpers
def envelope(x, bins=ENV_BINS):
    """Peak envelope, absolute (NOT per-stream normalised).

    Returned raw so the page can scale a whole example by one shared maximum. Normalising
    each stream to its own peak would misrepresent relative speaker levels -- the same
    mistake the shared-gain rescale in separate.py exists to avoid.
    """
    n = len(x)
    if n == 0:
        return []
    idx = np.linspace(0, n, bins + 1).astype(int)
    # 4 dp is well below one screen pixel of envelope height; full float64 precision here
    # tripled the file size for nothing.
    return [round(float(np.abs(x[a:b]).max()), 4) if b > a else 0.0
            for a, b in zip(idx[:-1], idx[1:])]


def crest_db(x):
    """Crest factor: ~18 dB is natural speech, ~2 dB means clipped."""
    rms = float(np.sqrt(np.mean(x**2)))
    pk = float(np.abs(x).max())
    if rms <= 0 or pk <= 0:
        return None
    return 20 * np.log10(pk / rms)


def paired(a, b, n, key_a="base", key_b="base"):
    """Paired delta on the intersection of mixture keys. Mirrors report_tier1.paired()."""
    ka, kb = list(a[f"keys_{n}"]), list(b[f"keys_{n}"])
    sb = set(kb)
    common = [k for k in ka if k in sb]
    xa = a[f"{key_a}_{n}"][[ka.index(k) for k in common]]
    xb = b[f"{key_b}_{n}"][[kb.index(k) for k in common]]
    d = xb - xa
    se = d.std(ddof=1) / np.sqrt(len(d))
    return dict(
        a=float(xa.mean()),
        b=float(xb.mean()),
        delta=float(d.mean()),
        ci=float(stats.t.ppf(0.975, len(d) - 1) * se),
        p=float(stats.ttest_rel(xb, xa).pvalue),
        # NOT "n": callers add the speaker count as `n`, which would silently clobber the
        # sample size and report "n=2" for a 100-mixture comparison.
        n_mix=int(len(d)),
    )


def load_npz(path):
    with np.load(path, allow_pickle=True) as z:
        return {k: z[k] for k in z.files}


# --------------------------------------------------------------------------- samples
def build_example(scorer, mix, refs, est_paths, audio):
    """Score one example and emit everything the player needs.

    Estimates carry no inherent speaker order, so the estimate->reference assignment is
    solved here (linear assignment on the authors' pairwise SI-SNRi) rather than trusting
    file order. `audio` maps logical names to server-relative URLs.
    """
    ests = [librosa.load(p, sr=SR, mono=True)[0] for p in est_paths]
    L = min([len(mix)] + [len(x) for x in ests + refs])
    mix, ests, refs = mix[:L], [e[:L] for e in ests], [r[:L] for r in refs]

    mix_t = torch.from_numpy(mix).reshape(1, -1).float().to(DEVICE)
    est_t = [torch.from_numpy(e).reshape(1, -1).float().to(DEVICE) for e in ests]
    ref_t = [torch.from_numpy(r).reshape(1, -1).float().to(DEVICE) for r in refs]

    M = scorer.pair_matrix(est_t, ref_t, mix_t)  # M[ref, est]
    rows, cols = linear_sum_assignment(-M)
    order = {int(r): int(c) for r, c in zip(rows, cols)}  # ref slot -> est index

    scale = max([max(np.abs(mix))] + [float(np.abs(r).max()) for r in refs]
                + [float(np.abs(e).max()) for e in ests]) or 1.0

    speakers = []
    for i in range(len(refs)):
        j = order[i]
        speakers.append(dict(
            sisnri=float(M[i, j]),
            est_url=audio["est"][j],
            ref_url=audio["ref"][i],
            est_env=envelope(ests[j] / scale),
            ref_env=envelope(refs[i] / scale),
            est_crest=crest_db(ests[j]),
        ))

    return dict(
        n=len(refs),
        seconds=round(L / SR, 2),
        mean_sisnri=float(np.mean([s["sisnri"] for s in speakers])),
        mix_url=audio["mix"],
        mix_env=envelope(mix / scale),
        speakers=speakers,
    )


def wsj0_samples(scorer):
    """separated_wsj0/ -- 8 examples per count, with mixture + refs + estimates alongside."""
    out = []
    for n in COUNTS:
        for d in sorted(pathlib.Path(f"separated_wsj0/{n}spk").glob("*/")):
            mix, _ = librosa.load(d / "mixture.wav", sr=SR, mono=True)
            refs = [librosa.load(d / f"ref_spk{i+1}.wav", sr=SR, mono=True)[0] for i in range(n)]
            ex = build_example(
                scorer, mix, refs,
                [d / f"est_spk{i+1}.wav" for i in range(n)],
                dict(mix=f"{d.as_posix()}/mixture.wav",
                     est=[f"{d.as_posix()}/est_spk{i+1}.wav" for i in range(n)],
                     ref=[f"{d.as_posix()}/ref_spk{i+1}.wav" for i in range(n)]),
            )
            ex["key"] = d.name
            ex["dataset"] = "wsj0"
            out.append(ex)
            print(f"  wsj0 N={n} {ex['mean_sisnri']:6.2f} dB  {d.name[:44]}", flush=True)
    return out


def libri_samples(scorer, per_count=4):
    """data/separations/libri -- out of domain. Mix/refs are NOT duplicated on disk; they
    live in data/libri_kmix_test and are referenced there, per the manifest's contract."""
    out = []
    for n in COUNTS:
        base = pathlib.Path(f"data/libri_kmix_test/{n}speakers/wav8k/min/tt")
        sep = pathlib.Path(f"data/separations/libri/{n}spk")
        keys = sorted(json.loads((sep / "manifest.json").read_text()).keys())[:per_count]
        for key in keys:
            stem = key[:-4]
            mix, refs = load_pair(base, key, n)
            ex = build_example(
                scorer, mix, refs,
                [sep / stem / f"est_spk{i+1}.wav" for i in range(n)],
                dict(mix=f"{base.as_posix()}/mix/{key}",
                     est=[f"{sep.as_posix()}/{stem}/est_spk{i+1}.wav" for i in range(n)],
                     ref=[f"{base.as_posix()}/s{i+1}/{key}" for i in range(n)]),
            )
            ex["key"] = stem
            ex["dataset"] = "libri"
            out.append(ex)
            print(f"  libri N={n} {ex['mean_sisnri']:6.2f} dB  {stem}", flush=True)
    return out


# --------------------------------------------------------------------------- results
def headline():
    """WSJ0 headline: SI-SNRi from the tier1 cache, count-acc + RTF from results_wsj0.txt.

    benchmark.py prints count accuracy but does not cache it per mixture, so the summary
    table it wrote is the source for those two columns.
    """
    var25 = load_npz("cache_tier1/wsj0_var25.npz")
    txt = pathlib.Path("results_wsj0.txt").read_text()
    acc, rtf = {}, {}
    for m in re.finditer(r"N=(\d)\s+n=\s*(\d+)\s+SI-SNRi=\s*([\d.]+) dB \(sd ([\d.]+)\)\s+"
                         r"count-acc=\s*([\d.]+)%\s+RTF=([\d.]+)", txt):
        n = int(m.group(1))
        acc[n], rtf[n] = float(m.group(5)), float(m.group(6))

    rows = []
    for n in COUNTS:
        x = var25[f"base_{n}"]
        rows.append(dict(
            n=n, mixes=int(len(x)), sisnri=float(x.mean()), sd=float(x.std()),
            count_acc=acc[n], rtf=rtf[n],
            septda=SEPTDA_KNOWN_COUNT[n], septda_unknown=SEPTDA_UNKNOWN_COUNT[n],
        ))
    return rows


def tier1():
    var25 = load_npz("cache_tier1/wsj0_var25.npz")
    libri25 = load_npz("cache_tier1/libri_var25.npz")
    ens = load_npz("cache_tier1/ensemble.npz")

    routing = []
    for name, cache, ns in [
        ("fix-2spk", "wsj0_fix2.npz", [2]),
        ("fix-2spk-l-dm", "wsj0_fix2ldm.npz", [2]),
        ("var-2-3spk", "wsj0_var23.npz", [2, 3]),
    ]:
        c = load_npz(f"cache_tier1/{cache}")
        for n in ns:
            r = paired(var25, c, n)
            r.update(n=n, challenger=name, shipped=(name == "fix-2spk-l-dm" and n == 2))
            routing.append(r)

    mc = []
    for label, domain, cache in [("WSJ0-mix tt", "in", var25), ("LibriSpeech dev-clean", "out", libri25)]:
        for n in COUNTS:
            r = paired(cache, cache, n, key_a="base", key_b="mc")
            r.update(n=n, dataset=label, domain=domain)
            mc.append(r)

    ensemble = []
    for n in COUNTS:
        e = paired(ens, ens, n, key_a="single", key_b="ens")
        em = paired(ens, ens, n, key_a="single", key_b="ensmc")
        ensemble.append(dict(n=n, single=e["a"], ens=e["b"], delta=e["delta"], ci=e["ci"],
                             p=e["p"], ensmc=em["b"], delta_ensmc=em["delta"], n_mix=e["n_mix"]))

    # Sampling robustness: the default subset is sorted(keys)[:limit], which is NOT random
    # (WSJ0-mix keys lead with a speaker id). The shipped wins were re-measured on a random
    # sample (--seed 1234) to check they were not an artifact of that.
    rand25 = load_npz("cache_tier1/rand_var25.npz")
    rand_ldm = load_npz("cache_tier1/rand_fix2ldm.npz")
    fix2ldm = load_npz("cache_tier1/wsj0_fix2ldm.npz")
    sampling = [dict(
        what="routing: fix-2spk-l-dm vs general @ N=2",
        sorted=paired(var25, fix2ldm, 2)["delta"],
        sorted_ci=paired(var25, fix2ldm, 2)["ci"],
        random=paired(rand25, rand_ldm, 2)["delta"],
        random_ci=paired(rand25, rand_ldm, 2)["ci"],
    )]
    for n in COUNTS:
        s = paired(var25, var25, n, key_a="base", key_b="mc")
        r = paired(rand25, rand25, n, key_a="base", key_b="mc")
        sampling.append(dict(what=f"mixture consistency @ N={n}", sorted=s["delta"],
                             sorted_ci=s["ci"], random=r["delta"], random_ci=r["ci"]))
    # The absolute baseline differs between subsets even though the paired deltas transfer.
    baselines = [dict(n=n, sorted=float(var25[f"base_{n}"].mean()),
                      random=float(rand25[f"base_{n}"].mean())) for n in COUNTS]

    return dict(routing=routing, mc=mc, ensemble=ensemble, sampling=sampling,
                baselines=baselines)


def out_of_domain():
    """LibriSpeech dev-clean, 500/count, oracle count -- parsed from the report
    count_baseline.py wrote. This is the OUT OF DOMAIN headline; it is not comparable to
    published WSJ0 numbers and must never be quoted as if it were."""
    txt = pathlib.Path("results_count_test.txt").read_text(encoding="utf-8", errors="replace")
    rows = []
    for m in re.finditer(r"^\s*(\d)\s+(\d+)\s+([\d.]+)%\s+([\d.]+)\s+([\d.]+)\s+([-\d.]+)\s+"
                         r"([\d.]+)\s+([-\d.]+)\s*$", txt, re.M):
        rows.append(dict(n=int(m.group(1)), mixes=int(m.group(2)), acc=float(m.group(3)),
                         oracle=float(m.group(4)), lenient=float(m.group(5)),
                         lenient_gap=float(m.group(6)), strict=float(m.group(7)),
                         strict_gap=float(m.group(8))))
    return rows


def bleed():
    out = {}
    for label in ("wsj0", "libri"):
        rows = []
        for n in COUNTS:
            f = pathlib.Path(f"results_bleed/{label}_{n}spk.npz")
            if not f.exists():
                continue
            d = load_npz(f)
            sdr, sir, sar, corr = d["sdr"], d["sir"], d["sar"], d["corr"]
            rows.append(dict(
                n=n, mixes=int(len(sdr)),
                sdr=float(sdr.mean()), sdr_sd=float(sdr.std()),
                sir=float(sir.mean()), sir_sd=float(sir.std()),
                sar=float(sar.mean()), sar_sd=float(sar.std()),
                gap=float(sir.mean() - sar.mean()),
                corr=float(corr.mean()), pct_collapse=100.0 * float((corr > 0.5).mean()),
            ))
        out[label] = rows
    return out


def counting():
    d = load_npz("results_count_test.npz")
    confusion = []
    for n in COUNTS:
        preds = d[f"preds_{n}"]
        row = [100.0 * float((preds == p).mean()) for p in COUNTS]
        confusion.append(dict(true_n=n, row=row, n_mix=int(len(preds)),
                              acc=100.0 * float((preds == n).mean()),
                              mean_pred=float(preds.mean())))

    # Track A vs built-in: parsed from the report compare_counters.py wrote.
    txt = pathlib.Path("results_track_a_test.txt").read_text()
    track_a = []
    for m in re.finditer(r"^\s*(\d)\s+(\d+)\s+([\d.]+)%\s+([\d.]+)%\s+([+-][\d.]+)\s*$", txt, re.M):
        track_a.append(dict(n=int(m.group(1)), n_mix=int(m.group(2)),
                            builtin=float(m.group(3)), track_a=float(m.group(4)),
                            delta=float(m.group(5))))
    gaps = []
    for m in re.finditer(r"^\s*(\d)\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)\s*$", txt, re.M):
        gaps.append(dict(n=int(m.group(1)), lenient_builtin=float(m.group(2)),
                         lenient_tracka=float(m.group(3)), strict_builtin=float(m.group(4)),
                         strict_tracka=float(m.group(5))))
    mean_gap = re.search(r"^mean\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s*$", txt, re.M)
    return dict(
        confusion=confusion,
        track_a=track_a,
        gaps=gaps,
        mean_gap=dict(lenient_builtin=float(mean_gap.group(1)), lenient_tracka=float(mean_gap.group(2)),
                      strict_builtin=float(mean_gap.group(3)), strict_tracka=float(mean_gap.group(4)))
        if mean_gap else None,
    )


# --------------------------------------------------------------------------- main
def main():
    scorer = Scorer(DEVICE)
    print("scoring samples with the authors' PIT_SISNRi (CPU)...", flush=True)
    samples = wsj0_samples(scorer) + libri_samples(scorer)

    data = dict(
        headline=headline(),
        out_of_domain=out_of_domain(),
        tier1=tier1(),
        bleed=bleed(),
        counting=counting(),
        samples=samples,
        meta=dict(
            model="shinuh/sr-corrnet-ss-1ch-wsj-var-2-5spk",
            sr=SR,
            n_samples=len(samples),
            paper_url="https://arxiv.org/abs/2603.29097",
            hf_url="https://huggingface.co/shinuh/sr-corrnet-ss-1ch-wsj-var-2-5spk",
        ),
    )

    js = "window.DEMO = " + json.dumps(data, separators=(",", ":")) + ";\n"
    OUT.write_text(js, encoding="utf-8")
    print(f"\nwrote {OUT} ({len(js)/1024:.0f} KB, {len(samples)} samples)", flush=True)


if __name__ == "__main__":
    main()
