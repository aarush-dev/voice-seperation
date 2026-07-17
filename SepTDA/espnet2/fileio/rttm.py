"""Reader for the (espnet-extended) RTTM speaker-diarization label format.

Standard RTTM (Rich Transcription Time Marked, see
https://catalog.ldc.upenn.edu/docs/LDC2004T12/RTTM-format-v13.pdf) has one
``SPEAKER`` line per speech turn, space-separated with 9 fields::

    SPEAKER file1 1 0 1023 <NA> <NA> spk1 <NA>
    SPEAKER file1 2 4000 3023 <NA> <NA> spk2 <NA>

Espnet extends this with a few conventions:

1. Times are given in sample numbers rather than absolute (wall-clock) time.
2. The 5th field is the segment *end* time rather than its duration.
3. An additional ``END`` line records the total duration of the recording::

    END     file1 <NA> 4023 <NA> <NA> <NA> <NA>

Only speaker labels are supported; other RTTM record types are rejected.
"""

import collections.abc
import re
from pathlib import Path
from typing import Dict, List, Tuple, Union

import numpy as np
from typeguard import check_argument_types

#: Per-utterance accumulator: (speaker ids seen, (spk_id, start, end) events,
#: recording duration in samples -- filled in once the "END" line is seen).
_UttState = Tuple[List[str], List[Tuple[str, int, int]], int]


def load_rttm_text(
    path: Union[Path, str]
) -> Dict[str, Tuple[List[str], List[Tuple[str, int, int]], int]]:
    """Read an RTTM file into per-utterance speaker events.

    Note: only supports speaker information (``SPEAKER``/``END`` lines) now.

    Returns:
        Mapping from utterance id to a tuple of
        ``(speaker_ids, [(speaker_id, start, end), ...], num_samples)``.
    """

    assert check_argument_types()
    data: Dict[str, _UttState] = {}
    with Path(path).open("r", encoding="utf-8") as f:
        for linenum, line in enumerate(f, 1):
            fields = re.split(" +", line.rstrip())

            # RTTM format must have exactly 9 fields
            assert len(fields) == 9, "{} does not have exactly 9 fields".format(path)
            label_type, utt_id, _channel, start, end, _, _, spk_id, _ = fields

            # Only support speaker label now
            assert label_type in ["SPEAKER", "END"]

            spk_list, spk_event, max_duration = data.get(utt_id, ([], [], 0))
            if label_type == "END":
                data[utt_id] = (spk_list, spk_event, int(end))
                continue
            if spk_id not in spk_list:
                spk_list.append(spk_id)

            data[utt_id] = (
                spk_list,
                spk_event + [(spk_id, int(float(start)), int(float(end)))],
                max_duration,
            )

    return data


class RttmReader(collections.abc.Mapping):
    """Reader class for 'rttm.scp'.

    Examples:
        SPEAKER file1 1 0 1023 <NA> <NA> spk1 <NA>
        SPEAKER file1 2 4000 3023 <NA> <NA> spk2 <NA>
        SPEAKER file1 3 500 4023 <NA> <NA> spk1 <NA>
        END     file1 <NA> 4023 <NA> <NA> <NA> <NA>

        This is an extend version of standard RTTM format for espnet.
        The difference including:
        1. Use sample number instead of absolute time
        2. has a END label to represent the duration of a recording
        3. replace duration (5th field) with end time
        (For standard RTTM,
            see https://catalog.ldc.upenn.edu/docs/LDC2004T12/RTTM-format-v13.pdf)
        ...

        >>> reader = RttmReader('rttm')
        >>> spk_label = reader["file1"]

    """

    def __init__(
        self,
        fname: str,
    ):
        assert check_argument_types()
        super().__init__()

        self.fname = fname
        self.data = load_rttm_text(path=fname)

    def __getitem__(self, key: str) -> np.ndarray:
        """Build a dense (num_samples, num_speakers) binary activity matrix."""
        spk_list, spk_event, max_duration = self.data[key]
        spk_label = np.zeros((max_duration, len(spk_list)))
        for spk_id, start, end in spk_event:
            spk_label[start : end + 1, spk_list.index(spk_id)] = 1
        return spk_label

    def __contains__(self, item):
        return item

    def __len__(self):
        return len(self.data)

    def __iter__(self):
        return iter(self.data)

    def keys(self):
        return self.data.keys()
