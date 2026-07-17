"""Build artifact.html -- a single self-contained page for publishing.

The local site (index.html + demo_data.js + 384 wav files on disk) cannot be published as-is:
the target is ONE file behind a strict CSP that blocks every external request, so the audio
has to be embedded in the page itself. This script produces that cut.

    SR_CorrNet_SS/.venv/Scripts/python.exe build_demo_data.py   # first: refresh demo_data.js
    SR_CorrNet_SS/.venv/Scripts/python.exe build_artifact.py    # then: emit artifact.html

Design decisions worth knowing before you change them:

* **The CSS and JS are reused from index.html, not forked.** Two copies of a chart routine
  drift silently. This extracts them, so the published page cannot disagree with the local
  one about how a number is drawn. index.html's render calls are null-guarded, so the same
  script drives a page with fewer mount points.

* **FLAC, not MP3.** Measured, not assumed: mp3 round-trips this audio at only 24.7 dB
  (32 kbps) to 27.8 dB (64 kbps, where LAME plateaus for 8 kHz input). Separation itself
  scores ~20-24 dB SI-SNRi, so an mp3 would inject artifact noise of the same order as the
  artifacts the page is asking the listener to judge -- and this page's whole error analysis
  is that the residual is artifact-dominated. FLAC is lossless, so the A/B stays honest.
  It only compresses 8 kHz speech to ~70-86%, which is why the sample count is small.

* **Representative examples, not the best ones.** For each speaker count the embedded example
  is the one whose mean SI-SNRi is CLOSEST TO THAT COUNT'S MEASURED MEAN. Shipping the
  best-scoring clips would be cherry-picking; the page says which was chosen and why.

* **This cut omits the engineering diary** (routing challengers, the sampling cautionary tale,
  implementation traps, open risks) -- it is an external results page. It must therefore never
  make the "+1.00 dB pipeline" claim, which is entirely a 2-speaker win: on a page about 3+
  speakers that would be false. Numbers keep their corpus label and n for the same reason.
"""

import base64
import json
import pathlib
import re
import shutil
import subprocess
import tempfile

FF = pathlib.Path("tools/ffmpeg-8.1.2-essentials_build/bin/ffmpeg.exe")
FOCUS = (3, 4, 5)
OUT = pathlib.Path("artifact.html")


def flac_data_uri(wav_path, cache={}):
    """Lossless FLAC + base64. Lossy codecs would add artifacts to the very thing under test."""
    wav_path = str(wav_path)
    if wav_path in cache:
        return cache[wav_path]
    tmp = pathlib.Path(tempfile.gettempdir()) / "artifact_enc.flac"
    subprocess.run([str(FF), "-y", "-loglevel", "error", "-i", wav_path,
                    "-ar", "8000", "-ac", "1", "-c:a", "flac",
                    "-compression_level", "12", str(tmp)], check=True)
    uri = "data:audio/flac;base64," + base64.b64encode(tmp.read_bytes()).decode()
    tmp.unlink(missing_ok=True)
    cache[wav_path] = uri
    return uri


def pick_representative(data):
    """One WSJ0 example per focus count: the one closest to that count's measured mean."""
    picked = []
    for n in FOCUS:
        mean = next(h["sisnri"] for h in data["headline"] if h["n"] == n)
        cands = [s for s in data["samples"] if s["dataset"] == "wsj0" and s["n"] == n]
        best = min(cands, key=lambda s: abs(s["mean_sisnri"] - mean))
        best["representative_of"] = round(mean, 2)
        picked.append(best)
        print(f"  N={n}: mean {mean:.2f} dB -> picked {best['mean_sisnri']:.2f} dB "
              f"({best['key'][:40]})", flush=True)
    return picked


def embed_audio(samples):
    total = 0
    for s in samples:
        s["mix_url"] = flac_data_uri(s["mix_url"])
        total += len(s["mix_url"])
        for sp in s["speakers"]:
            sp["est_url"] = flac_data_uri(sp["est_url"])
            sp["ref_url"] = flac_data_uri(sp["ref_url"])
            total += len(sp["est_url"]) + len(sp["ref_url"])
        print(f"  embedded {s['n']} spk: {s['key'][:44]}", flush=True)
    return total


def extract(tag, html):
    m = re.search(rf"<{tag}(?![^>]*\bsrc=)[^>]*>(.*?)</{tag}>", html, re.S)
    return m.group(1)


# A published Artifact supplies its own <head>, so this file cannot declare a charset of its
# own -- a <meta charset> in body content is ignored. Rather than bet the page's dashes and
# deltas on the wrapper's encoding, emit pure ASCII: HTML text as numeric entities, JS as
# \uXXXX escapes. Verified safe here: the CSS has no non-ASCII and no JS identifier does
# either, so only string and comment contents are touched. json.dumps escapes the data.
def html_ascii(s):
    return "".join(c if ord(c) < 128 else f"&#{ord(c)};" for c in s)


def js_ascii(s):
    return "".join(c if ord(c) < 128 else f"\\u{ord(c):04x}" for c in s)


BODY = """
<title>Separating 3+ voices from one microphone</title>

<nav><div class="wrap">
  <span class="brand">SR-CorrNet <span class="muted" style="font-weight:400">· 3+ speaker separation</span></span>
  <span class="links">
    <a href="#listen">Listen</a><a href="#results">Results</a><a href="#tier1">Pipeline</a>
    <a href="#bleed">Error analysis</a><a href="#counting">Counting</a><a href="#method">Method</a>
  </span>
</div></nav>

<header><div class="wrap">
  <div class="eyebrow">Monaural separation · 3, 4 and 5 concurrent speakers</div>
  <h1>Pulling five voices out of one microphone.</h1>
  <p class="lede">Two-speaker separation is a solved problem. <b>Three or more</b> is where it gets
    hard — and one checkpoint handles 3, 4 and 5 people talking at once over a
    <b>single channel</b>, infers how many there are, and holds published state-of-the-art at
    every count. No training, no per-count models.</p>
  <div class="chips" id="chips"></div>
  <div class="tiles" id="tiles"></div>
  <p class="hint">SI-SNRi on official WSJ0-{3,4,5}mix <code>tt</code>, 100 mixtures per count,
    speaker count known. Higher is better; ~20 dB means the interfering talkers are ~100× quieter
    than the target. <b>2 speakers is shown throughout as the baseline</b> — it is the easy case,
    and it is not what this is for.</p>
</div></header>

<section id="listen"><div class="wrap">
  <div class="sec-head">
    <h2>Listen to it work — start at 5</h2>
    <p>Play the <b>mixture</b> first. At three voices you can just about follow one; at five it is
      an unintelligible wall, which is the point. Then play each separated stream, and use
      <b>Separated / Reference</b> to A/B a stream against the ground-truth recording of that
      speaker at the same instant.</p>
  </div>
  <div class="controls">
    <div><div class="ctl-label">Corpus</div><div class="seg" id="ds-seg"></div></div>
    <div><div class="ctl-label">Speakers in the mixture</div><div class="seg" id="n-seg"></div></div>
  </div>
  <div class="examples" id="examples"></div>
  <div class="stage" id="stage"></div>
  <p class="hint"><kbd>Space</kbd> play/pause · click any waveform to seek · the A/B toggle keeps
    its position, so you can flip mid-word. Per-stream dB is that speaker's SI-SNRi, scored with
    the authors' own <code>PIT_SISNRi</code>. Audio is 8 kHz mono (the model's native rate),
    embedded losslessly — a lossy codec would add artifacts of the same order as the ones you are
    being asked to judge.</p>
  <div class="note"><b>These examples are representative, not hand-picked.</b> For each speaker
    count the embedded mixture is the one whose score lands closest to that count's measured mean
    across 100 mixtures — so what you hear is the typical case, not the best one. One example per
    count is embedded because lossless 8 kHz speech barely compresses; the full set of 48 scored
    samples lives with the project.</div>
</div></section>

<section id="results"><div class="wrap">
  <div class="sec-head">
    <h2>Results</h2>
    <p>Official WSJ0-mix <code>tt</code>, generated from real WSJ0 (LDC93S6A) with the canonical
      <code>pywsj0-mix</code> metadata — so these are directly comparable to published numbers.
      SepTDA is the state of the art it is measured against; SepTDA ships no weights, this does.</p>
  </div>
  <div class="grid2">
    <div class="chart">
      <h3>SI-SNRi vs. published SOTA</h3>
      <div class="csub">Higher is better. Speaker count known to both.</div>
      <div class="legend"><span><i style="background:var(--s1)"></i>SR-CorrNet (measured here)</span>
        <span><i style="background:var(--s2)"></i>SepTDA (paper)</span></div>
      <div id="chart-headline"></div>
    </div>
    <div class="chart">
      <h3>Counting is free — no cliff at 4 speakers</h3>
      <div class="csub">SepTDA must be told N; this model infers it and barely pays for it.</div>
      <div class="legend"><span><i style="background:var(--s1)"></i>SR-CorrNet, count inferred</span>
        <span><i style="background:var(--s2)"></i>SepTDA, count unknown (paper)</span></div>
      <div id="chart-unknown"></div>
    </div>
  </div>
  <div class="tbl-wrap" style="margin-top:16px"><table id="tbl-headline"></table></div>
  <div class="note"><b>There is no cliff at 3+.</b> This is the result that matters here: going
    from 2 to 5 speakers costs only ~2.8 dB, and 3 speakers actually scores <i>higher</i> than 2
    (23.84 vs 23.39). The hard cases degrade gracefully rather than falling apart, and the
    built-in counter still gets 98–100% of them right in domain, so you do not need to be told how
    many people are talking. RTF ~0.05–0.10 is 10–20× faster than realtime on one RTX 3080 Ti.</div>

  <h3 style="margin:40px 0 6px;font-size:17px">Out of domain: a corpus the model has never heard</h3>
  <p class="ink2" style="margin:0 0 18px;max-width:74ch;font-size:14px">The same checkpoint on
    LibriSpeech dev-clean mixtures, 500 per count, speaker count known. It was trained on WSJ0
    read speech, so this is the generalization test — and the number to quote if the target audio
    is not WSJ0-like.</p>
  <div class="grid2">
    <div class="chart">
      <h3>In domain vs. out of domain</h3>
      <div class="csub">Same model, same metric, different corpus. The gap is the corpus, not a defect.</div>
      <div class="legend"><span><i style="background:var(--s1)"></i>WSJ0-mix (trained on)</span>
        <span><i style="background:var(--s4)"></i>LibriSpeech dev-clean (never seen)</span></div>
      <div id="chart-oob"></div>
    </div>
    <div class="tbl-wrap"><table id="tbl-oob"></table></div>
  </div>
  <div class="note warn"><b>Never quote these as if they were in-domain.</b> ~18 dB out of domain
    is still roughly twice MultiDecoderDPRNN's <i>in-domain</i> 9.3 dB at 4 speakers, so the model
    generalizes well — but an out-of-domain figure read as in-domain is a wrong conclusion waiting
    to happen. In domain, 3 speakers scores 23.84 dB and matches the paper.</div>
</div></section>

<section id="tier1"><div class="wrap">
  <div class="sec-head">
    <h2>What actually helps at 3+ speakers</h2>
    <p>Two post-processing steps were measured against the plain model. Every comparison is
      <b>paired on identical mixtures</b>, which cancels per-mixture difficulty and resolves
      ~±0.03 dB instead of ~±0.4 dB.</p>
  </div>
  <div class="grid2">
    <div class="chart">
      <h3>Mixture consistency — free, and decays with N</h3>
      <div class="csub">Paired Δ from re-projecting streams onto the mixture. Same forward pass.</div>
      <div class="legend"><span><i style="background:var(--s1)"></i>WSJ0-mix (in domain)</span>
        <span><i style="background:var(--s4)"></i>LibriSpeech (out of domain)</span></div>
      <div id="chart-mc"></div>
    </div>
    <div class="chart">
      <h3>What 4× inference actually buys, over free MC</h3>
      <div class="csub">Δ(ens+MC) − Δ(MC alone). This is the only column that decides it.</div>
      <div id="chart-ensover"></div>
    </div>
  </div>
  <div class="note"><b>The two levers are mirror images, and only one survives at 3+.</b> Mixture
    consistency is free but decays as speakers pile up (+0.42 → +0.04 from 2 to 5). Offset
    ensembling does the reverse: nothing at 2 speakers, but <b>+0.21 / +0.39 / +0.44 dB at
    3 / 4 / 5</b> on top of mixture consistency. It is the only thing here that gets
    <i>better</i> as the problem gets harder — consistent with the residual being artifact-type
    (below), since artifacts are partly stochastic across offsets and cancel under averaging. It
    costs 4× inference, which at RTF 0.07–0.10 still leaves ~3× realtime.</div>
  <div class="grid2" style="margin-top:16px">
    <div class="chart">
      <h3>Ensembling and mixture consistency are mirror images</h3>
      <div class="csub">Paired Δ over a single pass. MC does the work at N=2; ensembling at N=5.</div>
      <div class="legend"><span><i style="background:var(--s1)"></i>Mixture consistency (free)</span>
        <span><i style="background:var(--s2)"></i>Ensembling alone (4× inference)</span>
        <span><i style="background:var(--s3)"></i>Both stacked</span></div>
      <div id="chart-ens"></div>
    </div>
    <div class="tbl-wrap"><table id="tbl-ens"></table></div>
  </div>
</div></section>

<section id="bleed"><div class="wrap">
  <div class="sec-head">
    <h2>What the residual error actually is</h2>
    <p>The audible imperfection in a separated stream sounds like bleed from the other talkers.
      It measured as something else: <b>processing artifacts</b>. BSS_Eval splits the error into
      interference (SIR) and artifacts (SAR) — and SIR runs 4–11 dB above SAR at every count.</p>
  </div>
  <div class="grid2">
    <div class="chart">
      <h3>SIR ≫ SAR at every speaker count</h3>
      <div class="csub">WSJ0-mix, in domain. Absolute BSS_Eval values — not improvements.</div>
      <div class="legend"><span><i style="background:var(--s1)"></i>SDR (overall)</span>
        <span><i style="background:var(--s2)"></i>SIR (interference)</span>
        <span><i style="background:var(--s3)"></i>SAR (artifacts)</span></div>
      <div id="chart-bleed"></div>
    </div>
    <div class="chart">
      <h3>Attractor collapse — a real minority tail</h3>
      <div class="csub">Share of mixtures where two streams latched onto one speaker (|corr| &gt; 0.5).</div>
      <div id="chart-collapse"></div>
    </div>
  </div>
  <div class="note warn"><b>Two things this rules out, and one it doesn't — all sharper at 3+.</b>
    Real cross-talker leakage would put SIR at 16–18 dB; in domain it is 23–35 dB, so a second
    stage that re-separates jointly would attack a problem that is not the bottleneck. The
    SIR−SAR margin does shrink as speakers pile up (+10.3 → +8.5 → +7.2 dB at 3/4/5), so
    interference matters <i>more</i> at high counts — but it never overtakes artifacts anywhere in
    range. And attractor collapse averages |corr| 0.009–0.108, far under threshold — <b>but it is
    not zero, and it is concentrated exactly where this project lives</b>: 1% of 4-speaker and 4%
    of 5-speaker mixtures. That tail should not be rounded away.</div>
  <div class="tbl-wrap" style="margin-top:16px"><table id="tbl-bleed"></table></div>
</div></section>

<section id="counting"><div class="wrap">
  <div class="sec-head">
    <h2>How many people are talking?</h2>
    <p>In domain the built-in counter is 98–100% and needs no help. Out of domain it degrades, and
      <b>it degrades worst exactly where it matters</b>: 64.6% at 3 speakers and 59.6% at 4, versus
      73.2% at 2. But it degrades in <b>exactly one direction</b>, and that shape decides the whole
      design.</p>
  </div>
  <div class="grid2">
    <div class="chart">
      <h3>The counter never under-predicts</h3>
      <div class="csub">LibriSpeech dev-clean (out of domain), 500 mixtures/count. Row = truth, column = prediction, % of row.</div>
      <div id="chart-confusion"></div>
    </div>
    <div class="chart">
      <h3>Track A pruning: +18 points where it counts</h3>
      <div class="csub">Accuracy on identical mixtures. Compare N=2/3/4 only — see below.</div>
      <div class="legend"><span><i style="background:var(--s1)"></i>Built-in counter</span>
        <span><i style="background:var(--s2)"></i>Track A (prune at n_spks=5)</span></div>
      <div id="chart-tracka"></div>
    </div>
  </div>
  <div class="note"><b>The confusion matrix is strictly upper-triangular</b> — 0.0% below the
    diagonal over 2000 mixtures. Every error is an over-prediction, never a missed speaker.</div>
  <div class="note warn"><b>Both N=5 numbers are artifacts, in opposite directions.</b> The
    built-in cannot emit more than 5 and never guesses low, so its ~100% at N=5 is free — it makes
    zero correct decisions to earn it. Track A can err both ways, so its 85.3% is the honest
    number. The like-for-like comparison is N=2/3/4: built-in <b>67.0%</b> vs Track A
    <b>85.1%</b>.</div>
  <div class="tbl-wrap" style="margin-top:16px"><table id="tbl-gaps"></table></div>
  <div class="note"><b>Over-predicting is nearly free; pruning is not.</b> If extra streams are
    ignored, a wrong count costs 0.09 dB at 4 speakers despite being wrong 40% of the time — the
    spare stream is junk sitting beside cleanly separated speakers, and the built-in wins outright.
    If exactly N streams are required, Track A halves the gap. The two policies invert the verdict,
    so which counter to use is a property of the downstream task, not of the model.</div>
</div></section>

<section id="method"><div class="wrap">
  <div class="sec-head">
    <h2>How this was measured</h2>
    <p>Enough to check the numbers mean what they say.</p>
  </div>
  <div class="cards">
    <div class="card">
      <h3><svg class="ico" viewBox="0 0 16 16" fill="none" stroke="#0ca30c" stroke-width="2"><path d="M3 8.5l3.5 3.5L13 4"/></svg>Method</h3>
      <ul>
        <li><b>Metric: SI-SNRi in dB</b> — an <i>improvement</i> over the unprocessed mixture, not
          an absolute SDR. Computed with the model authors' own <code>PIT_SISNRi</code>, not a
          reimplementation. BSS_Eval SDR/SIR/SAR in the error analysis are <i>absolute</i> and are
          not comparable to it.</li>
        <li><b>Model</b>: <code>shinuh/sr-corrnet-ss-1ch-wsj-var-2-5spk</code>, one checkpoint for
          2–5 speakers, used as published. 8 kHz mono; output is band-limited to 4 kHz.</li>
        <li><b>n = 100 mixtures per count</b> (standard error ≈ ±0.15–0.40 dB) for separation;
          500/count for counting. Ample to rank — do not read the third digit.</li>
        <li><b>Harness validated</b>: a null baseline (estimate = mixture) scores exactly 0.0000 dB
          and an oracle scores 129.9 dB. Mixtures are faithful to
          max|mix − Σsources| = 6.1e−05, exactly int16 quantization.</li>
        <li><b>Counting</b> was tuned and tested on <b>disjoint speaker splits</b>; the test gap
          (2.25) matched validation (2.24), so the thresholds did not overfit.</li>
      </ul>
    </div>
    <div class="card">
      <h3><svg class="ico" viewBox="0 0 16 16" fill="none" stroke="#3987e5" stroke-width="2"><circle cx="8" cy="8" r="6.5"/><path d="M8 7.5v4M8 4.6v.6"/></svg>Data &amp; credits</h3>
      <ul>
        <li><b>SR-CorrNet</b> — Shin Ui-Hyeop &amp; Park Hyung-Min,
          <i>Asymmetric Encoder-Decoder Based on Time-Frequency Correlation for Speech
          Separation</i>, <a href="https://arxiv.org/abs/2603.29097">arXiv:2603.29097</a>.
          Weights on <a href="https://huggingface.co/shinuh/sr-corrnet-ss-1ch-wsj-var-2-5spk">Hugging Face</a>.
          The model and its metric code are the authors' work; the evaluation here is not.</li>
        <li><b>WSJ0 (LDC93S6A)</b> — the in-domain mixtures and the embedded audio derive from the
          Wall Street Journal corpus, which is <b>licensed by the Linguistic Data Consortium</b>.
          Mixtures were generated with the canonical <code>pywsj0-mix</code> recipe. The clips here
          are short excerpts included to document a research result.</li>
        <li><b>LibriSpeech dev-clean</b> — out-of-domain mixtures, CC BY 4.0
          (Panayotov et al., 2015).</li>
        <li><b>SepTDA</b> comparison figures are quoted from its paper — the only numbers on this
          page not measured here. Everything else is generated directly from the measurement
          caches, so the page cannot drift from the data.</li>
      </ul>
    </div>
  </div>
</div></section>

<footer><div class="wrap"><span id="foot"></span></div></footer>
<div id="tip"></div>
"""


def main():
    if not FF.exists():
        raise SystemExit(f"ffmpeg not found at {FF}")
    src = pathlib.Path("index.html").read_text(encoding="utf-8")
    data = json.loads(pathlib.Path("demo_data.js").read_text(encoding="utf-8")[len("window.DEMO = "):-2])

    print("picking representative examples (closest to each count's measured mean):", flush=True)
    samples = pick_representative(data)
    print("encoding audio as lossless FLAC data URIs:", flush=True)
    audio_bytes = embed_audio(samples)
    data["samples"] = samples

    css = extract("style", src)
    js = extract("script", src)

    parts = [
        "<style>", css, "</style>",
        html_ascii(BODY),
        # ensure_ascii=True is the default and is load-bearing here, not incidental.
        "<script>window.DEMO = " + json.dumps(data, separators=(",", ":"), ensure_ascii=True) + ";</script>",
        "<script>", js_ascii(js), "</script>",
    ]
    html = "\n".join(parts)
    assert html.isascii(), "artifact must be pure ASCII; it cannot declare its own charset"
    OUT.write_text(html, encoding="utf-8")
    print(f"\nwrote {OUT}: {len(html)/1e6:.2f} MB total "
          f"({audio_bytes/1e6:.2f} MB of it embedded audio, {len(samples)} examples)", flush=True)


if __name__ == "__main__":
    main()
