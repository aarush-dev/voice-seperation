"""Character-level tokenizer with optional multi-character symbols."""

import warnings
from pathlib import Path
from typing import Iterable, List, Set, Union

from typeguard import check_argument_types

from espnet2.text.abs_tokenizer import AbsTokenizer


class CharTokenizer(AbsTokenizer):
    """Splits text into individual characters.

    Whitespace is mapped to ``space_symbol`` so it survives as an explicit
    token. ``non_linguistic_symbols`` and ``nonsplit_symbols`` are treated as
    single, indivisible tokens whenever they occur as a prefix of the
    remaining text, instead of being split character-by-character.

    Example:
        >>> tok = CharTokenizer()
        >>> tok.text2tokens("hi there")
        ['h', 'i', '<space>', 't', 'h', 'e', 'r', 'e']
        >>> tok.tokens2text(['h', 'i', '<space>', 'a'])
        'hi a'
    """

    def __init__(
        self,
        non_linguistic_symbols: Union[Path, str, Iterable[str]] = None,
        space_symbol: str = "<space>",
        remove_non_linguistic_symbols: bool = False,
        nonsplit_symbols: Iterable[str] = None,
    ):
        """Initialize the tokenizer.

        Args:
            non_linguistic_symbols: A collection of symbols, or a path to a
                file listing one symbol per line, that should be matched as
                whole tokens rather than split into characters.
            space_symbol: Token string that whitespace is mapped to.
            remove_non_linguistic_symbols: If ``True``, symbols matched from
                ``non_linguistic_symbols`` are dropped from the output
                instead of emitted as tokens. Symbols in ``nonsplit_symbols``
                are always kept regardless of this flag.
            nonsplit_symbols: A collection of symbols, in the ``symbol`` or
                ``symbol:group`` format, that must never be split into
                characters and are always emitted as tokens.
        """
        assert check_argument_types()
        self.space_symbol = space_symbol
        self.non_linguistic_symbols = _load_symbol_set(non_linguistic_symbols)
        self.remove_non_linguistic_symbols = remove_non_linguistic_symbols
        self.nonsplit_symbols = (
            set()
            if nonsplit_symbols is None
            else set(symbol.split(":")[0] for symbol in nonsplit_symbols)
        )

    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f'space_symbol="{self.space_symbol}"'
            f'non_linguistic_symbols="{self.non_linguistic_symbols}"'
            f'nonsplit_symbols="{self.nonsplit_symbols}"'
            f")"
        )

    def text2tokens(self, line: str) -> List[str]:
        """Split ``line`` into a list of single-character tokens.

        Prefix matches against ``non_linguistic_symbols`` and
        ``nonsplit_symbols`` are consumed as one token each; everything else
        is emitted one character at a time, with spaces mapped to
        ``self.space_symbol``.
        """
        atomic_symbols = self.non_linguistic_symbols.union(self.nonsplit_symbols)
        tokens = []
        while len(line) != 0:
            matched_symbol = _match_prefix_symbol(line, atomic_symbols)
            if matched_symbol is not None:
                keep_symbol = (
                    matched_symbol in self.nonsplit_symbols
                    or not self.remove_non_linguistic_symbols
                )
                if keep_symbol:
                    tokens.append(matched_symbol)
                line = line[len(matched_symbol) :]
            else:
                char = line[0]
                if char == " ":
                    char = self.space_symbol
                tokens.append(char)
                line = line[1:]
        return tokens

    def tokens2text(self, tokens: Iterable[str]) -> str:
        """Join character tokens back into a string, restoring spaces."""
        chars = [t if t != self.space_symbol else " " for t in tokens]
        return "".join(chars)


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


def _match_prefix_symbol(line: str, symbols: Iterable[str]) -> Union[str, None]:
    """Return the symbol in ``symbols`` that prefixes ``line``, if any."""
    for symbol in symbols:
        if line.startswith(symbol):
            return symbol
    return None
