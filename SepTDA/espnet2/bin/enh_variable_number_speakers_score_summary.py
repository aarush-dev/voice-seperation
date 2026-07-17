#!/usr/bin/env python3
"""CLI that aggregates per-utterance scores into per-mixture-size summaries.

``enh_variable_number_speakers_scoring`` writes one score per utterance per
speaker slot. This script reads those files back in, groups them by the
number of speakers in the mixture (encoded as the first character of each
utterance key, e.g. a key starting with "3" belongs to a 3-speaker mixture),
and reports:

* for each requested metric protocol (e.g. SI_SNR, STOI, ...): the average
  score per mixture size,
* for the special "Est_num_spk" protocol: the speaker-counting accuracy,
  i.e. what fraction of mixtures of each true size were estimated to have
  each possible speaker count.

wsj0-mix 4- and 5-speaker mixtures sometimes reuse the same speaker's
utterances within one mixture, so this also emits duplication-filtered
summaries (``with_duplication`` / ``without_duplication``) in addition to
the unfiltered one.
"""
import json
from pathlib import Path
from typing import Dict, List, Optional

from espnet2.fileio.read_text import read_2columns_text
from espnet2.utils import config_argparse

ScoreDict = Dict[str, str]


def has_speaker_duplicates(speaker_ids: List[str]) -> bool:
    """Return True if any speaker id appears more than once in the mixture."""
    if len(speaker_ids) <= 1:
        return False
    return len(speaker_ids) != len(set(speaker_ids))


def filter_result(
    result: ScoreDict, filter_type: Optional[str] = None
) -> ScoreDict:
    """Keep only utterances with/without speaker duplication, or all of them.

    Each key is expected to look like ``"<nmix>mix_<spk1>_<spk2>_..."``; the
    leading digit gives the mixture size, and the following ``nmix`` fields
    are the speaker ids used to build the mixture.
    """
    assert filter_type in (None, "with_duplication", "without_duplication")
    if filter_type is None:
        return result

    filtered_result = {}
    for key, score in result.items():
        nmix = int(key[0])
        spk_ids = key.split("_")[1 : (nmix + 1)]
        has_duplicate = has_speaker_duplicates(spk_ids)
        if (filter_type == "with_duplication" and has_duplicate) or (
            filter_type == "without_duplication" and not has_duplicate
        ):
            filtered_result[key] = score
    return filtered_result


def score_summary(args, filter_type: Optional[str] = None) -> Dict:
    """Read per-utterance scores for every protocol and summarize per mixture size."""
    results = {}
    for protocol in args.protocols:
        if protocol == "Est_num_spk":
            scores = read_2columns_text(args.score_dir / f"{protocol}")
            scores = filter_result(scores, filter_type=filter_type)
            results[protocol] = speaker_count(scores, args.max_num_spk)
        else:
            scores = []
            for spk in range(1, args.max_num_spk + 1):
                score = read_2columns_text(args.score_dir / f"{protocol}_spk{spk}")
                score = filter_result(score, filter_type=filter_type)
                scores.append(score)
            results[protocol] = get_score(scores, args.max_num_spk)
    return results


def get_score(scores: List[ScoreDict], max_num_spk: int) -> Dict[int, str]:
    """Average a speech-quality metric across speaker slots, grouped by mixture size."""
    score_total, count = {}, {}
    for score in scores:
        for key, value in score.items():
            # first word of key must be like 2mix, 3mix, ...
            nmix = int(key[0])
            assert nmix <= max_num_spk, (nmix, max_num_spk)
            if value == "dummy":
                continue
            if nmix not in score_total:
                score_total[nmix] = float(value)
                count[nmix] = 1
            else:
                score_total[nmix] += float(value)
                count[nmix] += 1
    for nmix in score_total:
        score_total[nmix] = str(round(score_total[nmix] / count[nmix], 4))
    return score_total


def speaker_count(score: ScoreDict, max_num_spk: int) -> Dict[int, Dict[str, str]]:
    """Compute the distribution of estimated speaker counts per true mixture size."""
    counts_by_nmix = {}
    for key, est_num_spk in score.items():
        # first word of key must be like 2mix, 3mix, ...
        nmix = int(key[0])
        counts_by_nmix.setdefault(nmix, {})
        counts_by_nmix[nmix][est_num_spk] = (
            counts_by_nmix[nmix].get(est_num_spk, 0) + 1
        )

    # convert raw counts to "count  percentage[%]" strings
    for nmix, counts in counts_by_nmix.items():
        total = sum(counts.values())
        for est_num_spk, count in counts.items():
            counts_by_nmix[nmix][
                est_num_spk
            ] = f"{count}  {round(100 * count / total, 2)}[%]"
    return counts_by_nmix


def write_results(
    output_dir: Path,
    results: Dict,
    max_num_spk: int,
    filename: str = "score_summary.json",
) -> None:
    """Write per-mixture-size result files plus one combined summary JSON."""
    for protocol, scores in results.items():
        for nmix, score in scores.items():
            output_dir_nmix = output_dir / f"{nmix}mix_results"
            output_dir_nmix.mkdir(exist_ok=True)
            if protocol != "Est_num_spk":
                with open(output_dir_nmix / f"{protocol}.txt", "w") as f:
                    f.write(score)
            else:
                with open(output_dir_nmix / f"{protocol}.json", "w") as f:
                    json.dump(score, f)
    with open(output_dir / filename, "w") as f:
        json.dump(results, f, indent=4)


def get_parser() -> config_argparse.ArgumentParser:
    parser = config_argparse.ArgumentParser()
    parser.add_argument("--score_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--protocols", type=str, required=True)
    parser.add_argument("--max_num_spk", type=int, required=True)

    return parser


def main() -> None:
    parser = get_parser()
    args = parser.parse_args()
    args.protocols = args.protocols.split(" ")
    results = score_summary(args)
    write_results(
        args.output_dir,
        results,
        args.max_num_spk,
        filename="score_summary.json",
    )

    # 4- and 5-speaker wsj0-mix mixtures sometimes reuse utterances from the
    # same speaker, so also report results split by speaker duplication.
    if args.max_num_spk >= 4:
        for filter_type in ["with_duplication", "without_duplication"]:
            results = score_summary(args, filter_type=filter_type)
            write_results(
                args.output_dir,
                results,
                args.max_num_spk,
                filename=f"score_summary_{filter_type}.json",
            )


if __name__ == "__main__":
    main()
