"""Grapheme-to-phoneme (G2P) tokenizer with many pluggable backends.

:class:`PhonemeTokenizer` converts raw text into phoneme-level tokens by
delegating to one of many G2P backends (english, Japanese, Chinese, Korean,
Icelandic, or generic espeak-based backends for many other languages). The
backend is selected by the ``g2p_type`` string, matching the module-level
``g2p_choices`` list.
"""

import logging
import re
import warnings
from pathlib import Path
from typing import Iterable, List, Optional, Set, Union

import g2p_en
import jamo
from packaging.version import parse as V
from typeguard import check_argument_types

from espnet2.text.abs_tokenizer import AbsTokenizer

g2p_choices = [
    None,
    "g2p_en",
    "g2p_en_no_space",
    "pyopenjtalk",
    "pyopenjtalk_kana",
    "pyopenjtalk_accent",
    "pyopenjtalk_accent_with_pause",
    "pyopenjtalk_prosody",
    "pypinyin_g2p",
    "pypinyin_g2p_phone",
    "pypinyin_g2p_phone_without_prosody",
    "espeak_ng_arabic",
    "espeak_ng_german",
    "espeak_ng_french",
    "espeak_ng_spanish",
    "espeak_ng_russian",
    "espeak_ng_greek",
    "espeak_ng_finnish",
    "espeak_ng_hungarian",
    "espeak_ng_dutch",
    "espeak_ng_english_us_vits",
    "espeak_ng_hindi",
    "espeak_ng_italian",
    "espeak_ng_ukrainian",
    "espeak_ng_polish",
    "g2pk",
    "g2pk_no_space",
    "g2pk_explicit_space",
    "korean_jaso",
    "korean_jaso_no_space",
    "g2p_is",
]


def split_by_space(text) -> List[str]:
    """Split ``text`` on single spaces, treating runs of 3 spaces as a literal space.

    Example:
        >>> split_by_space("a b   c")
        ['a', 'b', ' c']
    """
    if "   " in text:
        text = text.replace("   ", " <space> ")
        return [c.replace("<space>", " ") for c in text.split(" ")]
    else:
        return text.split(" ")


def pyopenjtalk_g2p(text) -> List[str]:
    """Convert Japanese ``text`` to a flat phoneme sequence via pyopenjtalk."""
    import pyopenjtalk

    # phones is a str object separated by space
    phones = pyopenjtalk.g2p(text, kana=False)
    phones = phones.split(" ")
    return phones


def _extract_fullcontext_label(text):
    """Return pyopenjtalk's full-context HTS labels for ``text``."""
    import pyopenjtalk

    if V(pyopenjtalk.__version__) >= V("0.3.0"):
        return pyopenjtalk.make_label(pyopenjtalk.run_frontend(text))
    else:
        return pyopenjtalk.run_frontend(text)[1]


def pyopenjtalk_g2p_accent(text) -> List[str]:
    """Convert Japanese ``text`` to phonemes interleaved with accent info."""
    phones = []
    for labels in _extract_fullcontext_label(text):
        p = re.findall(r"\-(.*?)\+.*?\/A:([0-9\-]+).*?\/F:.*?_([0-9]+)", labels)
        if len(p) == 1:
            phones += [p[0][0], p[0][2], p[0][1]]
    return phones


def pyopenjtalk_g2p_accent_with_pause(text) -> List[str]:
    """Like :func:`pyopenjtalk_g2p_accent`, but also emits ``"pau"`` for pauses."""
    phones = []
    for labels in _extract_fullcontext_label(text):
        if labels.split("-")[1].split("+")[0] == "pau":
            phones += ["pau"]
            continue
        p = re.findall(r"\-(.*?)\+.*?\/A:([0-9\-]+).*?\/F:.*?_([0-9]+)", labels)
        if len(p) == 1:
            phones += [p[0][0], p[0][2], p[0][1]]
    return phones


def pyopenjtalk_g2p_kana(text) -> List[str]:
    """Convert Japanese ``text`` to a sequence of kana characters."""
    import pyopenjtalk

    kanas = pyopenjtalk.g2p(text, kana=True)
    return list(kanas)


def pyopenjtalk_g2p_prosody(text: str, drop_unvoiced_vowels: bool = True) -> List[str]:
    """Extract phoneme + prosoody symbol sequence from input full-context labels.

    The algorithm is based on `Prosodic features control by symbols as input of
    sequence-to-sequence acoustic modeling for neural TTS`_ with some r9y9's tweaks.

    Args:
        text (str): Input text.
        drop_unvoiced_vowels (bool): whether to drop unvoiced vowels.

    Returns:
        List[str]: List of phoneme + prosody symbols.

    Examples:
        >>> from espnet2.text.phoneme_tokenizer import pyopenjtalk_g2p_prosody
        >>> pyopenjtalk_g2p_prosody("こんにちは。")
        ['^', 'k', 'o', '[', 'N', 'n', 'i', 'ch', 'i', 'w', 'a', '$']

    .. _`Prosodic features control by symbols as input of sequence-to-sequence acoustic
        modeling for neural TTS`: https://doi.org/10.1587/transinf.2020EDP7104

    """
    labels = _extract_fullcontext_label(text)
    N = len(labels)

    phones = []
    for n in range(N):
        lab_curr = labels[n]

        # current phoneme
        p3 = re.search(r"\-(.*?)\+", lab_curr).group(1)

        # deal unvoiced vowels as normal vowels
        if drop_unvoiced_vowels and p3 in "AEIOU":
            p3 = p3.lower()

        # deal with sil at the beginning and the end of text
        if p3 == "sil":
            assert n == 0 or n == N - 1
            if n == 0:
                phones.append("^")
            elif n == N - 1:
                # check question form or not
                e3 = _numeric_feature_by_regex(r"!(\d+)_", lab_curr)
                if e3 == 0:
                    phones.append("$")
                elif e3 == 1:
                    phones.append("?")
            continue
        elif p3 == "pau":
            phones.append("_")
            continue
        else:
            phones.append(p3)

        # accent type and position info (forward or backward)
        a1 = _numeric_feature_by_regex(r"/A:([0-9\-]+)\+", lab_curr)
        a2 = _numeric_feature_by_regex(r"\+(\d+)\+", lab_curr)
        a3 = _numeric_feature_by_regex(r"\+(\d+)/", lab_curr)

        # number of mora in accent phrase
        f1 = _numeric_feature_by_regex(r"/F:(\d+)_", lab_curr)

        a2_next = _numeric_feature_by_regex(r"\+(\d+)\+", labels[n + 1])
        # accent phrase border
        if a3 == 1 and a2_next == 1 and p3 in "aeiouAEIOUNcl":
            phones.append("#")
        # pitch falling
        elif a1 == 0 and a2_next == a2 + 1 and a2 != f1:
            phones.append("]")
        # pitch rising
        elif a2 == 1 and a2_next == 2:
            phones.append("[")

    return phones


def _numeric_feature_by_regex(regex, s):
    """Extract the first numeric capture group of ``regex`` from ``s``, or -50."""
    match = re.search(regex, s)
    if match is None:
        return -50
    return int(match.group(1))


def pypinyin_g2p(text) -> List[str]:
    """Convert Chinese ``text`` to pinyin syllables with tone digits."""
    from pypinyin import Style, pinyin

    phones = [phone[0] for phone in pinyin(text, style=Style.TONE3)]
    return phones


def pypinyin_g2p_phone(text) -> List[str]:
    """Convert Chinese ``text`` to initial/final phoneme pairs with tone digits."""
    from pypinyin import Style, pinyin
    from pypinyin.style._utils import get_finals, get_initials

    phones = [
        p
        for phone in pinyin(text, style=Style.TONE3)
        for p in [
            get_initials(phone[0], strict=True),
            get_finals(phone[0][:-1], strict=True) + phone[0][-1]
            if phone[0][-1].isdigit()
            else get_finals(phone[0], strict=True)
            if phone[0][-1].isalnum()
            else phone[0],
        ]
        # Remove the case of individual tones as a phoneme
        if len(p) != 0 and not p.isdigit()
    ]
    return phones


def pypinyin_g2p_phone_without_prosody(text) -> List[str]:
    """Convert Chinese ``text`` to initial/final phoneme pairs, without tones."""
    from pypinyin import Style, pinyin
    from pypinyin.style._utils import get_finals, get_initials

    phones = []
    for phone in pinyin(text, style=Style.NORMAL, strict=False):
        initial = get_initials(phone[0], strict=False)
        final = get_finals(phone[0], strict=False)
        if len(initial) != 0:
            if initial in ["x", "y", "j", "q"]:
                if final == "un":
                    final = "vn"
                elif final == "uan":
                    final = "van"
                elif final == "u":
                    final = "v"
            if final == "ue":
                final = "ve"
            phones.append(initial + "_" + final)
        else:
            phones.append(final)
    return phones


class G2p_en:
    """On behalf of g2p_en.G2p.

    g2p_en.G2p isn't pickalable and it can't be copied to the other processes
    via multiprocessing module.
    As a workaround, g2p_en.G2p is instantiated upon calling this class.

    """

    def __init__(self, no_space: bool = False):
        self.no_space = no_space
        self.g2p = None

    def __call__(self, text) -> List[str]:
        if self.g2p is None:
            self.g2p = g2p_en.G2p()

        phones = self.g2p(text)
        if self.no_space:
            # remove space which represents word serapater
            phones = list(filter(lambda s: s != " ", phones))
        return phones


class G2pk:
    """On behalf of g2pk.G2p.

    g2pk.G2p isn't pickalable and it can't be copied to the other processes
    via multiprocessing module.
    As a workaround, g2pk.G2p is instantiated upon calling this class.

    """

    def __init__(
        self,
        descritive=False,
        group_vowels=False,
        to_syl=False,
        no_space=False,
        explicit_space=False,
        space_symbol="<space>",
    ):
        self.descritive = descritive
        self.group_vowels = group_vowels
        self.to_syl = to_syl
        self.no_space = no_space
        self.explicit_space = explicit_space
        self.space_symbol = space_symbol
        self.g2p = None

    def __call__(self, text) -> List[str]:
        if self.g2p is None:
            import g2pk

            self.g2p = g2pk.G2p()

        phones = list(
            self.g2p(
                text,
                descriptive=self.descritive,
                group_vowels=self.group_vowels,
                to_syl=self.to_syl,
            )
        )
        if self.no_space:
            # remove space which represents word serapater
            phones = list(filter(lambda s: s != " ", phones))

        if self.explicit_space:
            # replace space as explicit space symbol
            phones = list(map(lambda s: s if s != " " else self.space_symbol, phones))
        return phones


class Jaso:
    PUNC = "!'(),-.:;?"
    SPACE = " "

    JAMO_LEADS = "".join([chr(_) for _ in range(0x1100, 0x1113)])
    JAMO_VOWELS = "".join([chr(_) for _ in range(0x1161, 0x1176)])
    JAMO_TAILS = "".join([chr(_) for _ in range(0x11A8, 0x11C3)])

    VALID_CHARS = JAMO_LEADS + JAMO_VOWELS + JAMO_TAILS + PUNC + SPACE

    def __init__(self, space_symbol=" ", no_space=False):
        self.space_symbol = space_symbol
        self.no_space = no_space

    def _text_to_jaso(self, line: str) -> List[str]:
        jasos = list(jamo.hangul_to_jamo(line))
        return jasos

    def _remove_non_korean_characters(self, tokens):
        new_tokens = [token for token in tokens if token in self.VALID_CHARS]
        return new_tokens

    def __call__(self, text) -> List[str]:
        graphemes = [x for x in self._text_to_jaso(text)]
        graphemes = self._remove_non_korean_characters(graphemes)

        if self.no_space:
            graphemes = list(filter(lambda s: s != " ", graphemes))
        else:
            graphemes = [x if x != " " else self.space_symbol for x in graphemes]
        return graphemes


class Phonemizer:
    """Phonemizer module for various languages.

    This is wrapper module of https://github.com/bootphon/phonemizer.
    You can define various g2p modules by specifying options for phonemizer.

    See available options:
        https://github.com/bootphon/phonemizer/blob/master/phonemizer/phonemize.py#L32

    """

    def __init__(
        self,
        backend,
        word_separator: Optional[str] = None,
        syllable_separator: Optional[str] = None,
        phone_separator: Optional[str] = " ",
        strip=False,
        split_by_single_token: bool = False,
        **phonemizer_kwargs,
    ):
        # delayed import
        from phonemizer.backend import BACKENDS
        from phonemizer.separator import Separator

        self.separator = Separator(
            word=word_separator,
            syllable=syllable_separator,
            phone=phone_separator,
        )

        # define logger to suppress the warning in phonemizer
        logger = logging.getLogger("phonemizer")
        logger.setLevel(logging.ERROR)
        self.phonemizer = BACKENDS[backend](
            **phonemizer_kwargs,
            logger=logger,
        )
        self.strip = strip
        self.split_by_single_token = split_by_single_token

    def __call__(self, text) -> List[str]:
        tokens = self.phonemizer.phonemize(
            [text],
            separator=self.separator,
            strip=self.strip,
            njobs=1,
        )[0]
        if not self.split_by_single_token:
            return tokens.split()
        else:
            # "a: ab" -> ["a", ":", "<space>",  "a", "b"]
            # TODO(kan-bayashi): space replacement should be dealt in PhonemeTokenizer
            return [c.replace(" ", "<space>") for c in tokens]


class IsG2p:  # pylint: disable=too-few-public-methods
    """Minimal wrapper for https://github.com/grammatek/ice-g2p

    The g2p module uses a Bi-LSTM model along with
    a pronunciation dictionary to generate phonemization
    Unfortunately does not support multi-thread phonemization as of yet
    """

    def __init__(
        self,
        dialect: str = "standard",
        syllabify: bool = True,
        word_sep: str = ",",
        use_dict: bool = True,
    ):
        self.dialect = dialect
        self.syllabify = syllabify
        self.use_dict = use_dict
        from ice_g2p.transcriber import Transcriber

        self.transcriber = Transcriber(
            use_dict=self.use_dict,
            syllab_symbol=".",
            stress_label=True,
            word_sep=word_sep,
            lang_detect=True,
        )

    def __call__(self, text) -> List[str]:
        return self.transcriber.transcribe(text).split()


def _build_espeak_phonemizer(language: str, **extra_kwargs) -> Phonemizer:
    """Build an espeak-ng-backed :class:`Phonemizer` for ``language``.

    All of the ``espeak_ng_*`` backends (other than the VITS variant) share
    the same options, differing only by language code.
    """
    return Phonemizer(
        language=language,
        backend="espeak",
        with_stress=True,
        preserve_punctuation=True,
        **extra_kwargs,
    )


def _build_g2p_function(g2p_type: Union[None, str], space_symbol: str):
    """Instantiate the callable G2P backend named by ``g2p_type``.

    Args:
        g2p_type: One of the strings in ``g2p_choices``.
        space_symbol: Token used for whitespace by backends that support it.

    Returns:
        A callable ``text -> List[str]`` implementing that backend.

    Raises:
        NotImplementedError: If ``g2p_type`` is not a recognized backend.
    """
    if g2p_type is None:
        return split_by_space
    elif g2p_type == "g2p_en":
        return G2p_en(no_space=False)
    elif g2p_type == "g2p_en_no_space":
        return G2p_en(no_space=True)
    elif g2p_type == "pyopenjtalk":
        return pyopenjtalk_g2p
    elif g2p_type == "pyopenjtalk_kana":
        return pyopenjtalk_g2p_kana
    elif g2p_type == "pyopenjtalk_accent":
        return pyopenjtalk_g2p_accent
    elif g2p_type == "pyopenjtalk_accent_with_pause":
        return pyopenjtalk_g2p_accent_with_pause
    elif g2p_type == "pyopenjtalk_prosody":
        return pyopenjtalk_g2p_prosody
    elif g2p_type == "pypinyin_g2p":
        return pypinyin_g2p
    elif g2p_type == "pypinyin_g2p_phone":
        return pypinyin_g2p_phone
    elif g2p_type == "pypinyin_g2p_phone_without_prosody":
        return pypinyin_g2p_phone_without_prosody
    elif g2p_type == "espeak_ng_arabic":
        return _build_espeak_phonemizer("ar")
    elif g2p_type == "espeak_ng_german":
        return _build_espeak_phonemizer("de")
    elif g2p_type == "espeak_ng_french":
        return _build_espeak_phonemizer("fr-fr")
    elif g2p_type == "espeak_ng_spanish":
        return _build_espeak_phonemizer("es")
    elif g2p_type == "espeak_ng_russian":
        return _build_espeak_phonemizer("ru")
    elif g2p_type == "espeak_ng_greek":
        return _build_espeak_phonemizer("el")
    elif g2p_type == "espeak_ng_finnish":
        return _build_espeak_phonemizer("fi")
    elif g2p_type == "espeak_ng_hungarian":
        return _build_espeak_phonemizer("hu")
    elif g2p_type == "espeak_ng_dutch":
        return _build_espeak_phonemizer("nl")
    elif g2p_type == "espeak_ng_hindi":
        return _build_espeak_phonemizer("hi")
    elif g2p_type == "espeak_ng_italian":
        return _build_espeak_phonemizer("it")
    elif g2p_type == "espeak_ng_polish":
        return _build_espeak_phonemizer("pl")
    elif g2p_type == "g2pk":
        return G2pk(no_space=False)
    elif g2p_type == "g2pk_no_space":
        return G2pk(no_space=True)
    elif g2p_type == "g2pk_explicit_space":
        return G2pk(explicit_space=True, space_symbol=space_symbol)
    elif g2p_type == "espeak_ng_english_us_vits":
        # VITS official implementation-like processing
        # Reference: https://github.com/jaywalnut310/vits
        return Phonemizer(
            language="en-us",
            backend="espeak",
            with_stress=True,
            preserve_punctuation=True,
            strip=True,
            word_separator=" ",
            phone_separator="",
            split_by_single_token=True,
        )
    elif g2p_type == "korean_jaso":
        return Jaso(space_symbol=space_symbol, no_space=False)
    elif g2p_type == "korean_jaso_no_space":
        return Jaso(no_space=True)
    elif g2p_type == "g2p_is":
        return IsG2p()
    elif g2p_type == "g2p_is_north":
        return IsG2p(dialect="north")
    else:
        raise NotImplementedError(f"Not supported: g2p_type={g2p_type}")


class PhonemeTokenizer(AbsTokenizer):
    """Tokenizes text into phonemes by delegating to a pluggable G2P backend.

    Example:
        >>> tok = PhonemeTokenizer(g2p_type="g2p_en")
        >>> tok.text2tokens("hello")  # doctest: +SKIP
        ['HH', 'AH0', 'L', 'OW1']
    """

    def __init__(
        self,
        g2p_type: Union[None, str],
        non_linguistic_symbols: Union[Path, str, Iterable[str]] = None,
        space_symbol: str = "<space>",
        remove_non_linguistic_symbols: bool = False,
    ):
        """Initialize the tokenizer.

        Args:
            g2p_type: Name of the G2P backend to use; see ``g2p_choices``.
            non_linguistic_symbols: Symbols (e.g. ``"<noise>"``), or a path
                to a file listing one per line, that are stripped out (or
                kept verbatim) before the text is handed to the G2P backend.
            space_symbol: Token string used for whitespace by backends that
                support it.
            remove_non_linguistic_symbols: If ``True``, matched
                ``non_linguistic_symbols`` are dropped instead of being kept
                verbatim in the pre-G2P text.
        """
        assert check_argument_types()
        self.g2p = _build_g2p_function(g2p_type, space_symbol)

        self.g2p_type = g2p_type
        self.space_symbol = space_symbol
        self.non_linguistic_symbols = self._load_symbol_set(non_linguistic_symbols)
        self.remove_non_linguistic_symbols = remove_non_linguistic_symbols

    @staticmethod
    def _load_symbol_set(
        symbols: Union[Path, str, Iterable[str], None]
    ) -> Set[str]:
        """Load a set of symbols from an iterable, or one-per-line from a file."""
        if symbols is None:
            return set()
        if isinstance(symbols, (Path, str)):
            symbols_path = Path(symbols)
            try:
                with symbols_path.open("r", encoding="utf-8") as f:
                    return set(line.rstrip() for line in f)
            except FileNotFoundError:
                warnings.warn(f"{symbols_path} doesn't exist.")
                return set()
        return set(symbols)

    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f'g2p_type="{self.g2p_type}", '
            f'space_symbol="{self.space_symbol}", '
            f'non_linguistic_symbols="{self.non_linguistic_symbols}"'
            ")"
        )

    def _strip_non_linguistic_symbols(self, line: str) -> str:
        """Consume ``non_linguistic_symbols`` prefixes from ``line``.

        Matched symbols are kept verbatim unless
        ``self.remove_non_linguistic_symbols`` is set, in which case they are
        dropped. Everything else passes through one character at a time.
        """
        tokens = []
        while len(line) != 0:
            for symbol in self.non_linguistic_symbols:
                if line.startswith(symbol):
                    if not self.remove_non_linguistic_symbols:
                        tokens.append(line[: len(symbol)])
                    line = line[len(symbol) :]
                    break
            else:
                tokens.append(line[0])
                line = line[1:]
        return "".join(tokens)

    def text2tokens(self, line: str) -> List[str]:
        """Strip non-linguistic symbols, then run the configured G2P backend."""
        line = self._strip_non_linguistic_symbols(line)
        return self.g2p(line)

    def tokens2text(self, tokens: Iterable[str]) -> str:
        """Concatenate phoneme tokens; G2P is not invertible in general."""
        # phoneme type is not invertible
        return "".join(tokens)

    def text2tokens_svs(self, syllable: str) -> List[str]:
        """G2P a single singing-voice-synthesis syllable, with dictionary overrides.

        Some Japanese syllables have a syllable-to-phoneme mapping that the
        general G2P backend gets wrong for singing voice synthesis; those are
        overridden by ``customed_dic``.
        """
        # Note(Yuning): fix syllabel2phoneme mismatch
        # If needed, customed_dic can be changed into extra input
        customed_dic = {
            "へ": ["h", "e"],
            "は": ["h", "a"],
            "シ": ["sh", "I"],
            "ヴぁ": ["v", "a"],
            "ヴぃ": ["v", "i"],
            "ヴぇ": ["v", "e"],
            "ヴぉ": ["v", "o"],
            "でぇ": ["dy", "e"],
            "くぁ": ["k", "w", "a"],
            "くぃ": ["k", "w", "i"],
            "くぅ": ["k", "w", "u"],
            "くぇ": ["k", "w", "e"],
            "くぉ": ["k", "w", "o"],
            "ぐぁ": ["g", "w", "a"],
            "ぐぃ": ["g", "w", "i"],
            "ぐぅ": ["g", "w", "u"],
            "ぐぇ": ["g", "w", "e"],
            "ぐぉ": ["g", "w", "o"],
            "くぉっ": ["k", "w", "o", "cl"],
        }
        tokens = self.g2p(syllable)
        if syllable in customed_dic:
            tokens = customed_dic[syllable]
        return tokens
