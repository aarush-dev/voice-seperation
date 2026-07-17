# Referenced from https://github.com/hccho2/Tacotron-Wavenet-Vocoder-Korean
"""Korean text normalizer: spells out digits and Latin letters in Hangul."""

import re


class KoreanCleaner:
    """Normalizes Korean text for TTS/ASR by spelling out digits and letters.

    Example:
        >>> KoreanCleaner.normalize_text("A1")
        '에이일'
    """

    _NUMBER_TO_KOREAN = {
        "0": "영",
        "1": "일",
        "2": "이",
        "3": "삼",
        "4": "사",
        "5": "오",
        "6": "육",
        "7": "칠",
        "8": "팔",
        "9": "구",
    }

    _UPPER_ALPHABET_TO_KOREAN = {
        "A": "에이",
        "B": "비",
        "C": "씨",
        "D": "디",
        "E": "이",
        "F": "에프",
        "G": "지",
        "H": "에이치",
        "I": "아이",
        "J": "제이",
        "K": "케이",
        "L": "엘",
        "M": "엠",
        "N": "엔",
        "O": "오",
        "P": "피",
        "Q": "큐",
        "R": "알",
        "S": "에스",
        "T": "티",
        "U": "유",
        "V": "브이",
        "W": "더블유",
        "X": "엑스",
        "Y": "와이",
        "Z": "지",
    }

    @classmethod
    def _normalize_numbers(cls, text: str) -> str:
        """Replace each ASCII digit with its Korean number word."""
        return "".join(
            cls._NUMBER_TO_KOREAN.get(char, char) for char in text
        )

    @classmethod
    def _normalize_english_text(cls, text: str) -> str:
        """Upper-case lowercase Latin runs, then spell each letter in Hangul."""
        uppercased = re.sub("[a-z]+", lambda m: str.upper(m.group()), text)
        return "".join(
            cls._UPPER_ALPHABET_TO_KOREAN.get(char, char) for char in uppercased
        )

    @classmethod
    def normalize_text(cls, text: str) -> str:
        """Strip, then spell out digits and Latin letters in Hangul."""
        text = text.strip()
        text = cls._normalize_numbers(text)
        text = cls._normalize_english_text(text)
        return text
