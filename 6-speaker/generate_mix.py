"""Generate {2..6}Mix: N-speaker mixtures from LibriSpeech or WSJ0, for SR-CorrNet fine-tuning.

Layout (matches the repo's SCP convention, scp_dir/<n>mix/{tr,cv,tt}_{mix,sK}.scp):
    out_dir/<n>mix/{tr,cv,tt}/{mix,s1..sN}/<key>.wav      8 kHz, 16-bit PCM, 4 s
    scp_dir/<n>mix/{tr,cv,tt}_mix.scp                      lines: "<key> <abs path>"
    scp_dir/<n>mix/{tr,cv,tt}_s<K>.scp

Construction per mixture: N distinct speakers, one random utterance each, resampled
to 8k, random 4s crop (zero-padded if shorter), per-source gain U[-5,+5] dB,
mixture = sum(sources); if peak > 0.9 the mixture AND all sources are rescaled by the
same factor (never independently -- that would corrupt SI-SNR targets).

Two corpora, selected with --corpus:

  libri  LibriSpeech .flac, speaker id from the path (<root>/<spk>/<chapter>/x.flac).
         Default split: train speakers from --train_roots; --eval_roots speakers are
         halved into cv and tt, so all three partitions are speaker-disjoint.

  wsj0   WSJ0 .wv1 (NIST SPHERE, shorten-compressed -- decoded via sph2pipe, which is
         the only thing that reads them). Speaker id from the path (<root>/<spk>/x.wv1).
         Canonical WSJ0-mix split (--valid_source train_utterances): tr and cv share
         si_tr_s speakers but draw from disjoint utterance pools, and tt comes from
         si_dt_05 + si_et_05 pooled. This is what WSJ0-2mix/3mix do, and what the
         pretrained var-2-5spk checkpoint was trained under.

.wv2 files are the second (secondary-mic) channel of the same recording and are
ignored -- WSJ0-mix uses .wv1 only.
"""
import argparse
import io
import random
import re
import shutil
import subprocess
from pathlib import Path

import numpy as np
import librosa
import soundfile as sf

SR = 8000
MAX_LEN = 32000  # 4 s @ 8 kHz
MIN_DUR = MAX_LEN / SR  # 4 s: shorter utterances are dropped (see index_speakers)

# Fraction of each train speaker's utterances held out for cv when the corpus draws
# validation from the training speakers (canonical WSJ0-mix).
CV_UTT_FRACTION = 0.1


# ---------------------------------------------------------------- corpus readers

def sphere_header(path):
    """{field: value} from a NIST SPHERE header, without decoding the audio.

    Layout: b"NIST_1A\n", then the header size in bytes as ASCII, then
    "<name> -i <int>" / "-s<len> <str>" lines until "end_head". Parsing this is what
    lets the >=4 s filter run over the whole corpus without paying for a shorten
    decode of every file.
    """
    with open(path, "rb") as f:
        magic = f.readline().strip()
        if magic != b"NIST_1A":
            raise ValueError(f"{path}: not a SPHERE file (magic={magic!r})")
        size = int(f.readline().strip())
        head = f.read(size - f.tell()).decode("ascii", "replace")
    fields = {}
    for m in re.finditer(r"^(\S+)\s+-(?:i|r)\s+(\S+)$", head, re.M):
        fields[m.group(1)] = float(m.group(2))
    return fields


def sphere_duration(path):
    h = sphere_header(path)
    return h["sample_count"] / h["sample_rate"]


def flac_duration(path):
    info = sf.info(str(path))
    return info.frames / info.samplerate


def load_sphere(path, sph2pipe):
    """Decode a shorten-compressed .wv1 to a float array at SR via sph2pipe -> stdout."""
    out = subprocess.run([sph2pipe, "-f", "wav", str(path)],
                         stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
    wav, sr = sf.read(io.BytesIO(out.stdout), dtype="float32")
    if wav.ndim > 1:
        wav = wav[:, 0]
    return librosa.resample(wav, orig_sr=sr, target_sr=SR) if sr != SR else wav


def load_flac(path, _sph2pipe):
    wav, _ = librosa.load(str(path), sr=SR)
    return wav


CORPORA = {
    "libri": {
        "ext": "*.flac",
        "spk_of": lambda p: p.parts[-3],
        "duration": flac_duration,
        "load": load_flac,
        "needs_sph2pipe": False,
    },
    "wsj0": {
        "ext": "*.wv1",
        "spk_of": lambda p: p.parts[-2],
        "duration": sphere_duration,
        "load": load_sphere,
        "needs_sph2pipe": True,
    },
}


# ---------------------------------------------------------------- indexing

def index_speakers(roots, corpus):
    """{speaker_id: [utterance paths]} across one or more corpus subset roots.

    Only utterances >= 4 s are kept: a silent (all-zero) source segment gives the
    SI-SNR loss a NaN gradient (torch.norm at 0), which destroys training.
    """
    c = CORPORA[corpus]
    spk = {}
    for root in roots:
        for f in Path(root).rglob(c["ext"]):
            try:
                if c["duration"](f) >= MIN_DUR:
                    spk.setdefault(c["spk_of"](f), []).append(f)
            except (ValueError, KeyError, RuntimeError) as e:
                print(f"  skipping unreadable {f}: {e}")
    return {k: sorted(v) for k, v in spk.items() if len(v) >= 2}


def split_utterances(speakers, fraction, seed):
    """Partition each speaker's utterance list into (majority, holdout).

    Speakers are shared across the two halves; utterances are not. Every speaker
    contributes at least one utterance to each half, so speakers with a single
    usable utterance are dropped (index_speakers already requires >= 2).
    """
    keep, held = {}, {}
    for sid, utts in speakers.items():
        u = list(utts)
        random.Random(f"{seed}:{sid}").shuffle(u)
        n_held = max(1, int(round(len(u) * fraction)))
        n_held = min(n_held, len(u) - 1)  # always leave >=1 for the majority half
        held[sid], keep[sid] = u[:n_held], u[n_held:]
    return keep, held


# ---------------------------------------------------------------- generation

def load_clip(path, rng, corpus, sph2pipe):
    wav = CORPORA[corpus]["load"](path, sph2pipe)
    if len(wav) > MAX_LEN:
        off = rng.integers(0, len(wav) - MAX_LEN)
        wav = wav[off:off + MAX_LEN]
    else:
        wav = np.pad(wav, (0, MAX_LEN - len(wav)))
    return wav


def make_partition(speakers, n_spks, n_mix, out_wav, out_scp, part, rng, corpus, sph2pipe):
    spk_ids = sorted(speakers)
    if len(spk_ids) < n_spks:
        raise SystemExit(
            f"{part}/{n_spks}mix: need {n_spks} distinct speakers, have {len(spk_ids)}")
    dirs = ["mix"] + [f"s{k}" for k in range(1, n_spks + 1)]
    for d in dirs:
        (out_wav / d).mkdir(parents=True, exist_ok=True)
    scp = {d: [] for d in dirs}
    for i in range(n_mix):
        chosen = rng.choice(spk_ids, size=n_spks, replace=False)
        srcs = []
        for sid in chosen:
            utt = speakers[sid][rng.integers(len(speakers[sid]))]
            gain = 10 ** (rng.uniform(-5, 5) / 20)
            srcs.append(load_clip(utt, rng, corpus, sph2pipe) * gain)
        mixture = np.sum(srcs, axis=0)
        peak = np.abs(mixture).max()
        if peak > 0.9:
            srcs = [s * (0.9 / peak) for s in srcs]
            mixture = mixture * (0.9 / peak)
        key = f"{part}_{n_spks}mix_{i:06d}"
        for d, wav in zip(dirs, [mixture] + srcs):
            p = out_wav / d / f"{key}.wav"
            sf.write(p, wav.astype(np.float32), SR, subtype="PCM_16")
            scp[d].append(f"{key} {p.resolve()}\n")
    out_scp.mkdir(parents=True, exist_ok=True)
    for d in dirs:
        name = f"{part}_{'mix' if d == 'mix' else d}.scp"
        with open(out_scp / name, "w") as f:
            f.writelines(scp[d])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", choices=sorted(CORPORA), default="libri")
    ap.add_argument("--train_roots", nargs="+", required=True,
                    help="subset dirs holding training speakers "
                         "(libri: train-clean-100; wsj0: si_tr_s)")
    ap.add_argument("--eval_roots", nargs="+", required=True,
                    help="subset dirs holding held-out speakers "
                         "(libri: dev-clean; wsj0: si_dt_05 si_et_05)")
    ap.add_argument("--valid_source", choices=["eval_speakers", "train_utterances"],
                    default=None,
                    help="where cv comes from. eval_speakers: halve --eval_roots "
                         "speakers into cv/tt (fully speaker-disjoint; libri default). "
                         "train_utterances: cv reuses train speakers via a disjoint "
                         "utterance holdout and tt takes all of --eval_roots "
                         "(canonical WSJ0-mix; wsj0 default).")
    ap.add_argument("--sph2pipe", default="sph2pipe",
                    help="sph2pipe binary, required for --corpus wsj0 (.wv1 is "
                         "shorten-compressed SPHERE and nothing else decodes it)")
    ap.add_argument("--out_wav", required=True)
    ap.add_argument("--out_scp", required=True)
    ap.add_argument("--counts", default="2,3,4,5,6")
    ap.add_argument("--n_train", default="2000,2000,3000,5000,8000",
                    help="mixtures per count (same order as --counts); weights training toward high N")
    ap.add_argument("--n_valid", default="500,500,750,1250,2000")
    ap.add_argument("--n_test", default="500,500,500,500,500")
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()

    if args.valid_source is None:
        args.valid_source = "train_utterances" if args.corpus == "wsj0" else "eval_speakers"

    sph2pipe = args.sph2pipe
    if CORPORA[args.corpus]["needs_sph2pipe"]:
        resolved = shutil.which(sph2pipe) or (sph2pipe if Path(sph2pipe).is_file() else None)
        if resolved is None:
            raise SystemExit(
                f"--corpus {args.corpus} needs sph2pipe to decode .wv1, but {sph2pipe!r} "
                f"was not found. Build it from https://github.com/robd003/sph2pipe and "
                f"pass --sph2pipe /path/to/sph2pipe.")
        sph2pipe = resolved

    counts = [int(c) for c in args.counts.split(",")]
    n_tr = [int(c) for c in args.n_train.split(",")]
    n_cv = [int(c) for c in args.n_valid.split(",")]
    n_tt = [int(c) for c in args.n_test.split(",")]

    print(f"indexing {args.corpus} (utterances >= {MIN_DUR:g}s only)...")
    train_spk = index_speakers(args.train_roots, args.corpus)
    eval_spk = index_speakers(args.eval_roots, args.corpus)
    if not train_spk or not eval_spk:
        raise SystemExit("no usable speakers found -- check --train_roots/--eval_roots")

    if args.valid_source == "train_utterances":
        # Canonical WSJ0-mix: cv shares tr's speakers but not its utterances; tt is
        # the full held-out speaker set (si_dt_05 + si_et_05).
        train_spk, cv_spk = split_utterances(train_spk, CV_UTT_FRACTION, args.seed)
        tt_spk = eval_spk
    else:
        # valid and test must not share speakers: split the eval speaker set in half
        eval_ids = sorted(eval_spk)
        random.Random(args.seed).shuffle(eval_ids)
        cv_spk = {k: eval_spk[k] for k in eval_ids[: len(eval_ids) // 2]}
        tt_spk = {k: eval_spk[k] for k in eval_ids[len(eval_ids) // 2:]}

    print(f"speakers: train={len(train_spk)} valid={len(cv_spk)} test={len(tt_spk)} "
          f"(valid_source={args.valid_source})")

    for n, a, b, c in zip(counts, n_tr, n_cv, n_tt):
        sub = f"{n}mix"
        for part, spk, cnt in [("tr", train_spk, a), ("cv", cv_spk, b), ("tt", tt_spk, c)]:
            # fixed seed per (count, partition): valid/test are identical across runs
            rng = np.random.default_rng(args.seed + 1000 * n + {"tr": 0, "cv": 1, "tt": 2}[part])
            make_partition(spk, n, cnt, Path(args.out_wav) / sub / part,
                           Path(args.out_scp) / sub, part, rng, args.corpus, sph2pipe)
            print(f"{sub}/{part}: {cnt} mixtures done")


if __name__ == "__main__":
    main()
