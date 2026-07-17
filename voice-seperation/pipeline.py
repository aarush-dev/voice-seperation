"""The shipping separation path: count-routed checkpoints + mixture consistency.

This is the product. `tier1_eval.py` imports `mixture_consistency` from here on
purpose, so the thing measured is the thing that ships.

Both behaviours below are MEASURED, not assumed -- see results_tier1.md.

ROUTING (+0.57 dB at N=2, p<0.001, n=100, in domain)
    Only ONE routing rule survived benchmarking. `fix-2spk` (-1.15 dB) and
    `var-2-3spk` (-0.90 at N=2, -1.91 at N=3) are both WORSE than the general
    checkpoint -- do not add them back without re-measuring. The winner is the
    dynamic-mixing model, so the gain looks like a DM effect rather than a
    specialization effect.

    Routing needs N *before* separating, but N is normally inferred by the
    general model. That is safe here because of a property verified over 2000
    mixtures: the built-in counter NEVER under-predicts (0.0% below the
    diagonal over 2000 mixtures). So predicted==2 => true<=2 => true==2, since 2
    is the minimum. A predicted 2 is trustworthy even out of domain, where raw
    N=2 counting accuracy is only 73%.

MIXTURE CONSISTENCY (+0.42/+0.19/+0.08/+0.04 dB at N=2/3/4/5, all p<0.001)
    Free -- same forward pass, no extra inference. Decays with N, and is worth
    ~nothing out of domain above N=3 (measured -0.003 dB at N=4 on LibriSpeech).

CAVEAT: both were measured on WSJ0 (IN DOMAIN) only. If the judge's audio is not
WSJ0-like read speech, neither gain is verified -- see Tier 0 in the roadmap.
"""

import torch

GENERAL = "shinuh/sr-corrnet-ss-1ch-wsj-var-2-5spk"

# count -> checkpoint. Absent counts use GENERAL. Deliberately sparse: every
# other specialized checkpoint tested LOST. See results_tier1.md 1.1.
ROUTING = {2: "shinuh/sr-corrnet-ss-1ch-wsj-fix-2spk-l-dm"}


def mixture_consistency(est, mix):
    """Wisdom et al. 1811.08521, closed form -- with the global gain fixed first.

    est: (N, L) raw model output.  mix: (L,) the raw mixture.  Returns (N, L).

    The textbook formula assumes sum(est) ~= mix. That is FALSE for this model:
    it is trained with a scale-invariant loss, so its output gain is arbitrary --
    measured ~87x too large and POLARITY-INVERTED (best-fit gain ~= -0.011). Applied
    naively the residual is ~87x too big and annihilates the estimates.

    SI-SNR is scale invariant and will NOT catch that -- it reads as an ordinary
    regression, not a bug (same silent-failure class as the clipping bug). Verify
    with probe_output_scale.py, not by reading the metric.

    It works because the arbitrary gain is GLOBAL, identical across streams to
    within 1% (measured spread 1.00-1.01), so a single scalar restores the
    premise. `a` is least-squares against the MIXTURE only -- available at test
    time; using references would be cheating. Blind vs reference-derived gain
    agree to 4 decimals.
    """
    s = est.sum(dim=0)
    a = torch.dot(mix, s) / (torch.dot(s, s) + 1e-12)
    est_s = a * est
    residual = mix - est_s.sum(dim=0)
    return est_s + residual / est.shape[0]


class Separator:
    """Loads checkpoints lazily and caches them (each is ~hundreds of MB)."""

    def __init__(self, device="cuda:0", route=True, mc=True):
        self.device = device
        self.route = route
        self.mc = mc
        self._models = {}

    def model(self, repo):
        from sr_corrnet import SSInference

        if repo not in self._models:
            # the repo id must go in checkpoint_path -- the README snippet is wrong
            self._models[repo] = SSInference.from_pretrained(
                checkpoint_path=repo, device=self.device
            )
        return self._models[repo]

    def separate(self, wav, n_spks=None):
        """wav: 1-D float tensor @ 8 kHz. Returns (streams (N,L) tensor, n, repo).

        Streams are RAW model output (peaks ~40-70) unless mixture consistency is
        on, in which case they are in the mixture's own scale. Either way, callers
        must rescale before writing a wav -- see rescale_for_write().
        """
        if n_spks is None:
            # Infer with the general model; it is the only one that can count.
            hint = None
            repo = GENERAL
        else:
            repo = ROUTING.get(int(n_spks), GENERAL) if self.route else GENERAL
            hint = torch.tensor(int(n_spks))

        out = self.model(repo).process_waveform(wav.unsqueeze(0), n_spks=hint)["waveforms"]
        n = len(out)

        # Count was inferred: if it came back 2 we can trust it (the counter never
        # under-predicts), so re-run on the specialist. Costs a second pass, but
        # RTF ~0.05 and it buys +0.57 dB.
        if n_spks is None and self.route and n in ROUTING:
            repo = ROUTING[n]
            out = self.model(repo).process_waveform(
                wav.unsqueeze(0), n_spks=torch.tensor(n)
            )["waveforms"]
            n = len(out)

        L = min(e.shape[-1] for e in out)
        est = torch.stack([e[:L].float() for e in out])

        if self.mc:
            est = mixture_consistency(est, wav[:L].to(est.device))
        return est, n, repo


def rescale_for_write(est):
    """Return (streams, gain) safe to write as wav, using ONE shared gain.

    Raw streams peak at ~40-70 and clip to garbage if written directly -- audibly
    destroyed but INVISIBLE to SI-SNR, which is scale invariant. Crest factor is
    the tell: ~2 dB means clipped, ~18 dB is natural speech.

    The shared gain preserves relative speaker levels. That is not folklore: the
    model's arbitrary output gain is global across streams to within 1%
    (probe_output_scale.py), so one gain is the correct choice and per-stream
    normalization would actively destroy the relative levels.

    Record the returned gain in a manifest so the raw output stays recoverable.
    """
    peak = est.abs().max().item()
    gain = (peak / 0.9) if peak > 0 else 1.0
    return est / gain, gain
