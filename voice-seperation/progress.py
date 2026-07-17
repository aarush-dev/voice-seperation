"""Live progress bars + ETA for a diagnose_bleed.py run, read from its checkpoints.

Works against a run in flight, including one whose stdout is redirected to a log
(where a \\r-style live bar would be unreadable). Reads the same npz checkpoints the
run resumes from, so it needs no cooperation from the running process.

    python progress.py                 # one snapshot
    python progress.py --watch         # auto-refreshing, redraws in place, with ETA

ETA note: bss_eval_sources cost grows steeply with speaker count, so a single global
rate would badly misestimate the remaining N=4/5 work. Rate is tracked PER speaker
count from observed throughput; counts not yet started are extrapolated from the
nearest observed count via COST (relative cost per mixture).
"""

import argparse
import json
import pathlib
import sys
import time

import numpy as np

WIDTH = 32
COUNTS = (2, 3, 4, 5)
LABELS = [("wsj0", "WSJ0-mix (in domain)"), ("libri", "LibriSpeech dev-clean (out of domain)")]

# Relative per-mixture cost by speaker count. Only the RATIOS matter -- they seed the
# ETA for counts that have not started yet, and are replaced by measured rates as soon
# as a count produces two observations.
COST = {2: 1.0, 3: 1.9, 4: 3.4, 5: 5.6}


def fmt_eta(sec):
    if sec is None or not np.isfinite(sec):
        return "--:--"
    sec = int(max(0, sec))
    h, m, s = sec // 3600, (sec % 3600) // 60, sec % 60
    return f"{h:d}:{m:02d}:{s:02d}" if h else f"{m:d}:{s:02d}"


def bar(done, total):
    frac = 0.0 if not total else min(1.0, done / total)
    fill = int(round(frac * WIDTH))
    return f"[{'#' * fill}{'-' * (WIDTH - fill)}] {frac*100:5.1f}% {done:>4}/{total}"


def read_done(out_dir, label, n):
    """Returns (done, mtime). mtime is when that checkpoint was actually WRITTEN.

    Timing off mtime rather than off poll time matters: checkpoints land every CKPT
    mixtures, so sampling at poll time measures the gap between our polls, not the
    work -- which makes the rate look several times faster than it is.
    """
    f = out_dir / f"{label}_{n}spk.npz"
    if not f.exists():
        return 0, None
    try:
        return len(np.load(f)["keys"]), f.stat().st_mtime
    except (OSError, ValueError, KeyError):
        return 0, None  # mid-write or truncated: unknown, not progress


class Tracker:
    """Turns repeated (time, done) samples into a per-speaker-count rate and an ETA."""

    def __init__(self, expect, hist_path=None):
        self.expect = expect
        self.hist = {}  # (label, n) -> list[(t, done)]
        # Persisted so ETA is available on a cold snapshot instead of waiting ~4 min
        # for two checkpoints to land.
        self.hist_path = hist_path
        if hist_path and hist_path.exists():
            try:
                raw = json.loads(hist_path.read_text())
                self.hist = {tuple(k.split("|")): [tuple(p) for p in v] for k, v in raw.items()}
                self.hist = {(l, int(n)): v for (l, n), v in self.hist.items()}
            except (OSError, ValueError):
                self.hist = {}  # unreadable history is not worth failing over

    def observe(self, key, t, done):
        h = self.hist.setdefault(key, [])
        if h and done < h[-1][1]:
            h.clear()  # count went backwards: a different run, old samples are lies
        if not h or h[-1][1] != done:
            h.append((t, done))
        del h[:-8]  # keep a short recent window

    def save(self):
        if not self.hist_path:
            return
        raw = {f"{l}|{n}": v for (l, n), v in self.hist.items()}
        tmp = self.hist_path.with_suffix(".json.tmp")
        try:
            tmp.write_text(json.dumps(raw))
            tmp.replace(self.hist_path)
        except OSError:
            pass  # progress reporting must never break the thing it reports on

    def sec_per_mix(self, n, label=None):
        """Measured seconds/mixture for count n, else None.

        Pooled across datasets by default (cost depends on speaker count, not corpus);
        pass label to restrict to one config's own measurements.
        """
        samples = []
        for (ll, nn), h in self.hist.items():
            if nn != n or (label is not None and ll != label) or len(h) < 2:
                continue
            dt, dn = h[-1][0] - h[0][0], h[-1][1] - h[0][1]
            if dn > 0 and dt > 0:
                samples.append(dt / dn)
        return float(np.mean(samples)) if samples else None

    def estimate(self, n):
        """Seconds/mixture for count n: measured, else scaled from any measured count."""
        direct = self.sec_per_mix(n)
        if direct:
            return direct
        for m in sorted(COUNTS, key=lambda m: abs(m - n)):
            s = self.sec_per_mix(m)
            if s:
                return s * COST[n] / COST[m]
        return None

    def eta(self, out_dir):
        total = 0.0
        for label, _ in LABELS:
            for n in COUNTS:
                rem = self.expect - read_done(out_dir, label, n)[0]
                if rem <= 0:
                    continue
                s = self.estimate(n)
                if s is None:
                    return None
                total += rem * s
        return total


def render(out_dir, expect, tracker, now):
    lines, tot_done, tot_exp = [], 0, 0
    for label, title in LABELS:
        lines.append(title)
        for n in COUNTS:
            done, mtime = read_done(out_dir, label, n)
            if mtime is not None:
                tracker.observe((label, n), mtime, done)
            tot_done += done
            tot_exp += expect
            # Only this config's own measured rate -- showing a rate pooled from the
            # other dataset on a 0/100 row reads as progress that hasn't happened.
            rate = tracker.sec_per_mix(n, label=label) if done else None
            note = "  done" if done >= expect else (f"  {rate:.1f}s/mix" if rate else "")
            lines.append(f"  N={n}  {bar(done, expect)}{note}")
        lines.append("")
    eta = tracker.eta(out_dir)
    lines.append(f"  TOTAL {bar(tot_done, tot_exp)}   ETA {fmt_eta(eta)}")
    return lines, tot_done, tot_exp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="results_bleed")
    ap.add_argument("--expect", type=int, default=100, help="mixtures per speaker count")
    ap.add_argument("--watch", action="store_true")
    ap.add_argument("--interval", type=float, default=5.0)
    args = ap.parse_args()

    out_dir = pathlib.Path(args.out_dir)
    tracker = Tracker(args.expect, out_dir / ".progress_history.json")
    # Redraw in place only on a TTY; piped/redirected output gets plain snapshots
    # instead of a screenful of ANSI escapes.
    tty = sys.stdout.isatty()
    prev = 0

    while True:
        lines, done, exp = render(out_dir, args.expect, tracker, time.time())
        if tty and prev:
            sys.stdout.write(f"\033[{prev}A\033[J")
        print("\n".join(lines), flush=True)
        tracker.save()
        prev = len(lines)
        if not args.watch or done >= exp:
            break
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
