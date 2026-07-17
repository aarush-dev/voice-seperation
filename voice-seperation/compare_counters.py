"""Side-by-side: built-in counter vs Track A, on the SAME test mixtures.

The two were measured at different sample sizes (the built-in at 500/count from
count_baseline.py; Track A's N=4/5 at 150 to save GPU time). Comparing a 500-mixture
mean against a 150-mixture mean would confound the method with the sample, so the
built-in's accuracy is recomputed here on exactly the subset Track A used.

Both scripts take `sorted(keys)[:limit]`, so Track A's first K mixtures of a count are
a prefix of the built-in's 500 -- truncating the saved per-mixture predictions to K
lines up mixture-for-mixture.

Caveat, stated rather than hidden: count_baseline.py saved per-mixture *predictions*
but only aggregate SI-SNRi, so the built-in's gap CANNOT be recomputed on the subset.
Accuracy is compared like-for-like; the gap columns compare Track A's subset mean
against the built-in's full-500 mean. Both estimate the same population, but they are
not the identical sample -- read the gap difference as approximate.
"""

import numpy as np

from track_a import evaluate

E_TH, C_TH = 10.0, 0.2  # tuned on cache_val.npz ONLY -- never on test
COUNTS = [2, 3, 4, 5]


def main():
    base = np.load("results_count_test.npz")
    cache = np.load("cache_test.npz")

    have = [n for n in COUNTS if f"M_{n}" in cache.files]
    per_n, conf = evaluate(cache, have, E_TH, C_TH)

    # Built-in summary rows: [n, cnt, acc, oracle, lenient, strict]
    summary = {int(r[0]): r for r in base["summary"]}

    print(f"Track A thresholds: energy_db={E_TH}, corr={C_TH} (tuned on val only)\n")

    print("COUNTING ACCURACY -- identical mixtures, like-for-like")
    print(f"{'N':>2} {'n':>5} {'built-in':>9} {'Track A':>9} {'delta':>7}")
    deltas = []
    for n in have:
        k = len(conf[n])
        b_pred = base[f"preds_{n}"][:k]  # same prefix of the same sorted keys
        b_acc = 100.0 * float((b_pred == n).mean())
        a_acc = per_n[n]["acc"]
        deltas.append(a_acc - b_acc)
        print(f"{n:>2} {k:>5} {b_acc:>8.1f}% {a_acc:>8.1f}% {a_acc-b_acc:>+6.1f}")
    print(f"{'mean':>2} {'':>5} {np.mean([100.0*(base[f'preds_{n}'][:len(conf[n])]==n).mean() for n in have]):>8.1f}%"
          f" {np.mean([per_n[n]['acc'] for n in have]):>8.1f}% {np.mean(deltas):>+6.1f}")

    print("\nEND-TO-END SI-SNRi GAP vs oracle (lower is better)")
    sizes = ", ".join(f"N={n}:{len(conf[n])}" for n in have)
    print(f"  built-in gaps are its full-500 means; Track A's are [{sizes}] -- where these")
    print("  differ the comparison is approximate (same population, different sample)")
    print(f"{'N':>2} {'lenient':>18} {'strict':>18}")
    print(f"{'':>2} {'built-in':>8} {'TrackA':>9} {'built-in':>8} {'TrackA':>9}")
    for n in have:
        b = summary[n]
        b_len, b_str = b[3] - b[4], b[3] - b[5]
        a_len = per_n[n]["oracle"] - per_n[n]["lenient"]
        a_str = per_n[n]["oracle"] - per_n[n]["strict"]
        print(f"{n:>2} {b_len:>8.2f} {a_len:>9.2f} {b_str:>8.2f} {a_str:>9.2f}")

    b_len_m = float(np.mean([summary[n][3] - summary[n][4] for n in have]))
    b_str_m = float(np.mean([summary[n][3] - summary[n][5] for n in have]))
    a_len_m = float(np.mean([per_n[n]["oracle"] - per_n[n]["lenient"] for n in have]))
    a_str_m = float(np.mean([per_n[n]["oracle"] - per_n[n]["strict"] for n in have]))
    print(f"{'mean':>2} {b_len_m:>8.2f} {a_len_m:>9.2f} {b_str_m:>8.2f} {a_str_m:>9.2f}")

    print("\nVERDICT")
    print(f"  judge IGNORES extra streams (lenient): "
          f"{'Track A' if a_len_m < b_len_m else 'BUILT-IN'} wins "
          f"({min(a_len_m, b_len_m):.2f} vs {max(a_len_m, b_len_m):.2f} dB gap)")
    print(f"  judge WANTS EXACTLY N  (strict):       "
          f"{'Track A' if a_str_m < b_str_m else 'BUILT-IN'} wins "
          f"({min(a_str_m, b_str_m):.2f} vs {max(a_str_m, b_str_m):.2f} dB gap)")

    print("\nTrack A confusion (rows true N, cols predicted N, % of row):")
    seen = sorted({p for n in have for p in conf[n]})
    print("        " + "".join(f"{p:>8}" for p in seen))
    for n in have:
        tot = len(conf[n])
        print(f"  N={n} |" + "".join(f"{100.0*sum(1 for x in conf[n] if x==p)/tot:>7.1f}%" for p in seen))


if __name__ == "__main__":
    main()
