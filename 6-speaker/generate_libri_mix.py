"""Generate Libri{2..6}Mix: N-speaker mixtures from LibriSpeech for SR-CorrNet fine-tuning.

Layout (matches the repo's SCP convention, scp_dir/<n>mix/{tr,cv,tt}_{mix,sK}.scp):
    out_dir/<n>mix/{tr,cv,tt}/{mix,s1..sN}/<key>.wav      8 kHz, 16-bit PCM, 4 s
    scp_dir/<n>mix/{tr,cv,tt}_mix.scp                      lines: "<key> <abs path>"
    scp_dir/<n>mix/{tr,cv,tt}_s<K>.scp

Construction per mixture: N distinct speakers, one random utterance each, resampled
16k->8k, random 4s crop (zero-padded if shorter), per-source gain U[-5,+5] dB,
mixture = sum(sources); if peak > 0.9 the mixture AND all sources are rescaled by the
same factor (never independently — that would corrupt SI-SNR targets).
"""
import argparse
import random
from pathlib import Path

import numpy as np
import librosa
import soundfile as sf

SR = 8000
MAX_LEN = 32000  # 4 s @ 8 kHz


def index_speakers(roots):
    """{speaker_id: [flac paths]} across one or more LibriSpeech subset roots.

    Only utterances >= 4 s are kept: a silent (all-zero) source segment gives the
    SI-SNR loss a NaN gradient (torch.norm at 0), which destroys training.
    """
    spk = {}
    for root in roots:
        for f in Path(root).rglob("*.flac"):
            info = sf.info(str(f))
            if info.frames / info.samplerate >= MAX_LEN / SR:
                spk.setdefault(f.parts[-3], []).append(f)
    return {k: sorted(v) for k, v in spk.items() if len(v) >= 2}


def load_clip(path, rng):
    wav, _ = librosa.load(path, sr=SR)
    if len(wav) > MAX_LEN:
        off = rng.integers(0, len(wav) - MAX_LEN)
        wav = wav[off:off + MAX_LEN]
    else:
        wav = np.pad(wav, (0, MAX_LEN - len(wav)))
    return wav


def make_partition(speakers, n_spks, n_mix, out_wav, out_scp, part, rng):
    spk_ids = sorted(speakers)
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
            srcs.append(load_clip(utt, rng) * gain)
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
    prefix = {"tr": "tr", "cv": "cv", "tt": "tt"}[part]
    out_scp.mkdir(parents=True, exist_ok=True)
    for d in dirs:
        name = f"{prefix}_{'mix' if d == 'mix' else d}.scp"
        with open(out_scp / name, "w") as f:
            f.writelines(scp[d])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_roots", nargs="+", required=True,
                    help="LibriSpeech subset dirs for training speakers (e.g. train-clean-100)")
    ap.add_argument("--eval_roots", nargs="+", required=True,
                    help="LibriSpeech subset dirs for valid/test speakers (e.g. dev-clean)")
    ap.add_argument("--out_wav", required=True)
    ap.add_argument("--out_scp", required=True)
    ap.add_argument("--counts", default="2,3,4,5,6")
    ap.add_argument("--n_train", default="2000,2000,3000,5000,8000",
                    help="mixtures per count (same order as --counts); weights training toward high N")
    ap.add_argument("--n_valid", default="500,500,750,1250,2000")
    ap.add_argument("--n_test", default="500,500,500,500,500")
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()

    counts = [int(c) for c in args.counts.split(",")]
    n_tr = [int(c) for c in args.n_train.split(",")]
    n_cv = [int(c) for c in args.n_valid.split(",")]
    n_tt = [int(c) for c in args.n_test.split(",")]

    train_spk = index_speakers(args.train_roots)
    eval_spk = index_speakers(args.eval_roots)
    # valid and test must not share speakers: split the eval speaker set in half
    eval_ids = sorted(eval_spk)
    random.Random(args.seed).shuffle(eval_ids)
    cv_spk = {k: eval_spk[k] for k in eval_ids[: len(eval_ids) // 2]}
    tt_spk = {k: eval_spk[k] for k in eval_ids[len(eval_ids) // 2:]}
    print(f"speakers: train={len(train_spk)} valid={len(cv_spk)} test={len(tt_spk)}")

    for n, a, b, c in zip(counts, n_tr, n_cv, n_tt):
        sub = f"{n}mix"
        for part, spk, cnt in [("tr", train_spk, a), ("cv", cv_spk, b), ("tt", tt_spk, c)]:
            # fixed seed per (count, partition): valid/test are identical across runs
            rng = np.random.default_rng(args.seed + 1000 * n + {"tr": 0, "cv": 1, "tt": 2}[part])
            make_partition(spk, n, cnt, Path(args.out_wav) / sub / part,
                           Path(args.out_scp) / sub, part, rng)
            print(f"{sub}/{part}: {cnt} mixtures done")


if __name__ == "__main__":
    main()
