"""Readers/writer for singing-voice score data: MusicXML and JSON scores.

``xml.scp`` / ``score.scp`` are ordinary Kaldi-style 2-column scp files
mapping an utterance id to a MusicXML or JSON score file path::

    key1 /some/path/a.musicxml
    key2 /some/path/b.musicxml
    key3 /some/path/c.musicxml

``XMLReader`` parses the referenced MusicXML file (via ``music21``) into a
``(tempo, notes)`` pair; ``SingingScoreReader`` just loads the referenced
JSON file, which has the shape described in
:meth:`SingingScoreWriter.__setitem__`.
"""

import collections.abc
import json
from pathlib import Path
from typing import Dict, List, Tuple, Union

import numpy as np
from typeguard import check_argument_types

from espnet2.fileio.read_text import read_2columns_text

try:
    import music21 as m21  # for CI import
except ImportError or ModuleNotFoundError:
    m21 = None


class NOTE(object):
    """A single parsed musical note/rest: lyric, MIDI pitch, start/end time."""

    def __init__(self, lyric: str, midi: int, st: float, et: float):
        self.lyric = lyric
        self.midi = midi
        self.st = st
        self.et = et


class XMLReader(collections.abc.Mapping):
    """Reader class for 'xml.scp'.

    Examples:
        key1 /some/path/a.xml
        key2 /some/path/b.xml
        key3 /some/path/c.xml
        key4 /some/path/d.xml
        ...

        >>> reader = XMLScpReader('xml.scp')
        >>> lyrics_array, notes_array, segs_array = reader['key1']
    """

    def __init__(
        self,
        fname,
        dtype=np.int16,
    ):
        assert check_argument_types()
        assert m21 is not None, (
            "Cannot load music21 package. ",
            "Please install Muskit modules via ",
            "(cd tools && make muskit.done)",
        )
        self.fname = fname
        self.dtype = dtype
        self.data = read_2columns_text(fname)  # get key-value dict

    def __getitem__(self, key) -> Tuple[int, List["NOTE"]]:
        """Parse the MusicXML file for ``key`` into ``(tempo, notes)``.

        Walks the flattened notes/rests of the first part, merging tied or
        un-lyriced notes into the preceding syllable and expanding ``<br>``
        markers and breath marks into their own pseudo-notes.
        """
        score = m21.converter.parse(self.data[key])
        tempo_marks = score.metronomeMarkBoundaries()
        tempo = int(tempo_marks[0][2].number)
        part = score.parts[0].flat
        notes_list: List[NOTE] = []
        prev_pitch = -1
        start_time = 0
        for note in part.notesAndRests:
            duration = note.seconds
            if not note.isRest:  # Note or Chord
                lyric = note.lyric
                if note.isChord:
                    for candidate in note:
                        if candidate.pitch.midi != prev_pitch:  # Ignore repeat note
                            note = candidate
                            break
                if lyric is None or lyric == "":  # multi note in one syllable
                    if note.pitch.midi == prev_pitch:  # same pitch
                        notes_list[-1].et += duration
                    else:  # different pitch
                        notes_list.append(
                            NOTE("—", note.pitch.midi, start_time, start_time + duration)
                        )
                elif lyric == "br":  # <br> is tagged as a note
                    if prev_pitch == 0:
                        notes_list[-1].et += duration
                    else:
                        notes_list.append(NOTE("P", 0, start_time, start_time + duration))
                    prev_pitch = 0
                else:  # normal note for one syllable
                    notes_list.append(
                        NOTE(note.lyric, note.pitch.midi, start_time, start_time + duration)
                    )
                prev_pitch = note.pitch.midi
                for articulation in note.articulations:  # <br> is tagged as a notation
                    if articulation.name in ["breath mark"]:  # up-bow?
                        notes_list.append(NOTE("B", 0, start_time, start_time))
            else:  # rest note
                if prev_pitch == 0:
                    notes_list[-1].et += duration
                else:
                    notes_list.append(NOTE("P", 0, start_time, start_time + duration))
                prev_pitch = 0
            start_time += duration
        # NOTE(Yuning): implicit rest at the end of xml file should be removed.
        if notes_list[-1].midi == 0 and notes_list[-1].lyric == "P":
            notes_list.pop()
        return tempo, notes_list

    def get_path(self, key):
        return self.data[key]

    def __contains__(self, item):
        return item

    def __len__(self):
        return len(self.data)

    def __iter__(self):
        return iter(self.data)

    def keys(self):
        return self.data.keys()


class XMLWriter:
    """Writer class for 'midi.scp'

    Examples:
        key1 /some/path/a.musicxml
        key2 /some/path/b.musicxml
        key3 /some/path/c.musicxml
        key4 /some/path/d.musicxml
        ...

        >>> writer = XMLScpWriter('./data/', './data/xml.scp')
        >>> writer['aa'] = xml_obj
        >>> writer['bb'] = xml_obj

    """

    def __init__(
        self,
        outdir: Union[Path, str],
        scpfile: Union[Path, str],
    ):
        assert check_argument_types()
        self.dir = Path(outdir)
        self.dir.mkdir(parents=True, exist_ok=True)
        scpfile = Path(scpfile)
        scpfile.parent.mkdir(parents=True, exist_ok=True)
        self.fscp = scpfile.open("w", encoding="utf-8")
        self.data: Dict[str, str] = {}

    def __setitem__(self, key: str, value: tuple):
        """Render ``(lyrics_seq, notes_seq, segs_seq, tempo)`` to a MusicXML file.

        Each note's duration is derived from its segment boundaries
        (``segs_seq``) and the tempo, quantized to eighth notes (with a
        minimum of a sixteenth note).
        """
        assert (
            len(value) == 4
        ), "The xml values should include lyrics, note, segmentations and tempo"
        lyrics_seq, notes_seq, segs_seq, tempo = value
        xml_path = self.dir / f"{key}.musicxml"
        xml_path.parent.mkdir(parents=True, exist_ok=True)

        stream = m21.stream.Stream()
        stream.insert(m21.tempo.MetronomeMark(number=tempo))
        beats_per_second = 1.0 * tempo / 60
        offset = 0
        for idx in range(len(lyrics_seq)):
            duration = int(
                8 * (segs_seq[idx][1] - segs_seq[idx][0]) * beats_per_second + 0.5
            )
            duration = 1.0 * duration / 8
            if duration == 0:
                duration = 1 / 16
            if notes_seq[idx] != -1:  # isNote
                note_obj = m21.note.Note(notes_seq[idx])
                if lyrics_seq[idx] != "—":
                    note_obj.lyric = lyrics_seq[idx]
            else:  # isRest
                note_obj = m21.note.Rest()
            note_obj.offset = offset
            note_obj.duration = m21.duration.Duration(duration)
            stream.insert(note_obj)
            offset += duration
        stream.write("xml", fp=xml_path)
        self.fscp.write(f"{key} {xml_path}\n")
        self.data[key] = str(xml_path)

    def get_path(self, key):
        return self.data[key]

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def close(self):
        self.fscp.close()


class SingingScoreReader(collections.abc.Mapping):
    """Reader class for 'score.scp'.

    Examples:
        key1 /some/path/score.json
        key2 /some/path/score.json
        key3 /some/path/score.json
        key4 /some/path/score.json
        ...

        >>> reader = SoundScpReader('score.scp')
        >>> score = reader['key1']

    """

    def __init__(
        self,
        fname,
        dtype=np.int16,
    ):
        assert check_argument_types()
        self.fname = fname
        self.dtype = dtype
        self.data = read_2columns_text(fname)

    def __getitem__(self, key):
        with open(self.data[key], "r") as f:
            score = json.load(f)
        return score

    def get_path(self, key):
        return self.data[key]

    def __contains__(self, item):
        return item

    def __len__(self):
        return len(self.data)

    def __iter__(self):
        return iter(self.data)

    def keys(self):
        return self.data.keys()


class SingingScoreWriter:
    """Writer class for 'score.scp'

    Examples:
        key1 /some/path/score.json
        key2 /some/path/score.json
        key3 /some/path/score.json
        key4 /some/path/score.json
        ...

        >>> writer = SingingScoreWriter('./data/', './data/score.scp')
        >>> writer['aa'] = score_obj
        >>> writer['bb'] = score_obj

    """

    def __init__(
        self,
        outdir: Union[Path, str],
        scpfile: Union[Path, str],
    ):
        assert check_argument_types()
        self.dir = Path(outdir)
        self.dir.mkdir(parents=True, exist_ok=True)
        scpfile = Path(scpfile)
        scpfile.parent.mkdir(parents=True, exist_ok=True)
        self.fscp = scpfile.open("w", encoding="utf-8")
        self.data: Dict[str, str] = {}

    def __setitem__(self, key: str, value: dict):
        """Score should be a dict

        Example:
        {
            "tempo": bpm,
            "item_list": a subset of ["st", "et", "lyric", "midi", "phn"],
            "note": [
                [start_time1, end_time1, lyric1, midi1, phn1],
                [start_time2, end_time2, lyric2, midi2, phn2],
                ...
            ]
        }

        The itmes in each note correspond to the "item_list".

        """

        score_path = self.dir / f"{key}.json"
        score_path.parent.mkdir(parents=True, exist_ok=True)
        with open(score_path, "w") as f:
            json.dump(value, f, ensure_ascii=False, indent=2)
        self.fscp.write(f"{key} {score_path}\n")
        self.data[key] = str(score_path)

    def get_path(self, key):
        return self.data[key]

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def close(self):
        self.fscp.close()
