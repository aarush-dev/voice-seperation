"""Factory function that builds a concrete tokenizer from a string type name.

This is the single entry point used by config-driven code (e.g. training
YAML configs) to construct a tokenizer without importing the concrete class
directly: the ``token_type`` string (``"bpe"``, ``"char"``, ``"word"``, ...)
selects the implementation.
"""

from pathlib import Path
from typing import Iterable, Union

from typeguard import check_argument_types

from espnet2.text.abs_tokenizer import AbsTokenizer
from espnet2.text.char_tokenizer import CharTokenizer
from espnet2.text.hugging_face_tokenizer import HuggingFaceTokenizer
from espnet2.text.phoneme_tokenizer import PhonemeTokenizer
from espnet2.text.sentencepiece_tokenizer import SentencepiecesTokenizer
from espnet2.text.whisper_tokenizer import OpenAIWhisperTokenizer
from espnet2.text.word_tokenizer import WordTokenizer


def build_tokenizer(
    token_type: str,
    bpemodel: Union[Path, str, Iterable[str]] = None,
    non_linguistic_symbols: Union[Path, str, Iterable[str]] = None,
    remove_non_linguistic_symbols: bool = False,
    space_symbol: str = "<space>",
    delimiter: str = None,
    g2p_type: str = None,
    nonsplit_symbol: Iterable[str] = None,
) -> AbsTokenizer:
    """Instantiate a :class:`AbsTokenizer` subclass based on ``token_type``.

    Args:
        token_type: One of ``"bpe"``, ``"hugging_face"``, ``"word"``,
            ``"char"``, ``"phn"``, or a string containing ``"whisper"``.
        bpemodel: Path to a sentencepiece/hugging-face/whisper model,
            required for the corresponding token types.
        non_linguistic_symbols: Symbols (e.g. ``"<noise>"``) treated as
            atomic tokens by the char/word/phn tokenizers.
        remove_non_linguistic_symbols: If ``True``, strip
            ``non_linguistic_symbols`` from the output instead of keeping
            them as tokens.
        space_symbol: Token string used to represent whitespace.
        delimiter: Word-splitting delimiter for the ``"word"`` tokenizer.
        g2p_type: Grapheme-to-phoneme backend name for the ``"phn"``
            tokenizer.
        nonsplit_symbol: Symbols that the char tokenizer must never split,
            even when ``remove_non_linguistic_symbols`` is set.

    Returns:
        A ready-to-use tokenizer instance.

    Example:
        >>> tokenizer = build_tokenizer("char")
        >>> tokenizer.text2tokens("hi there")
        ['h', 'i', '<space>', 't', 'h', 'e', 'r', 'e']

    Raises:
        ValueError: If ``token_type`` is unrecognized, or a required model
            path is missing.
        RuntimeError: If ``remove_non_linguistic_symbols`` is requested for
            a token type that does not support it.
    """
    assert check_argument_types()
    if token_type == "bpe":
        if bpemodel is None:
            raise ValueError('bpemodel is required if token_type = "bpe"')

        if remove_non_linguistic_symbols:
            raise RuntimeError(
                "remove_non_linguistic_symbols is not implemented for token_type=bpe"
            )
        return SentencepiecesTokenizer(bpemodel)

    if token_type == "hugging_face":
        if bpemodel is None:
            raise ValueError('bpemodel is required if token_type = "hugging_face"')

        if remove_non_linguistic_symbols:
            raise RuntimeError(
                "remove_non_linguistic_symbols is not "
                + "implemented for token_type=hugging_face"
            )
        return HuggingFaceTokenizer(bpemodel)

    elif token_type == "word":
        if remove_non_linguistic_symbols and non_linguistic_symbols is not None:
            return WordTokenizer(
                delimiter=delimiter,
                non_linguistic_symbols=non_linguistic_symbols,
                remove_non_linguistic_symbols=True,
            )
        else:
            return WordTokenizer(delimiter=delimiter)

    elif token_type == "char":
        return CharTokenizer(
            non_linguistic_symbols=non_linguistic_symbols,
            space_symbol=space_symbol,
            remove_non_linguistic_symbols=remove_non_linguistic_symbols,
            nonsplit_symbols=nonsplit_symbol,
        )

    elif token_type == "phn":
        return PhonemeTokenizer(
            g2p_type=g2p_type,
            non_linguistic_symbols=non_linguistic_symbols,
            space_symbol=space_symbol,
            remove_non_linguistic_symbols=remove_non_linguistic_symbols,
        )

    elif "whisper" in token_type:
        return OpenAIWhisperTokenizer(bpemodel)

    else:
        raise ValueError(
            f"token_mode must be one of bpe, word, char or phn: " f"{token_type}"
        )
