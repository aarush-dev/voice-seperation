"""Whitespace/delimiter-based word tokenizer."""

import warnings
from pathlib import Path
from typing import Iterable, List, Set, Union

from typeguard import check_argument_types

from espnet2.text.abs_tokenizer import AbsTokenizer


class WordTokenizer(AbsTokenizer):
    """Splits text into words on a fixed delimiter (default: whitespace).

    Example:
        >>> tok = WordTokenizer()
        >>> tok.text2tokens("hello world")
        ['hello', 'world']
        >>> tok.tokens2text(['hello', 'world'])
        'hello world'
    """

    def __init__(
        self,
        delimiter: str = None,
        non_linguistic_symbols: Union[Path, str, Iterable[str]] = None,
        remove_non_linguistic_symbols: bool = False,
    ):
        """Initialize the tokenizer.

        Args:
            delimiter: Separator passed to ``str.split``. ``None`` splits on
                any run of whitespace.
            non_linguistic_symbols: Words to drop from the output, or a path
                to a file listing one such word per line. Only used when
                ``remove_non_linguistic_symbols`` is ``True``.
            remove_non_linguistic_symbols: If ``True``, words found in
                ``non_linguistic_symbols`` are dropped from the output.
        """
        assert check_argument_types()
        self.delimiter = delimiter

        if not remove_non_linguistic_symbols and non_linguistic_symbols is not None:
            warnings.warn(
                "non_linguistic_symbols is only used "
                "when remove_non_linguistic_symbols = True"
            )

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
        return f'{self.__class__.__name__}(delimiter="{self.delimiter}")'

    def text2tokens(self, line: str) -> List[str]:
        """Split ``line`` on ``self.delimiter``, optionally dropping symbols."""
        tokens = []
        for word in line.split(self.delimiter):
            if (
                self.remove_non_linguistic_symbols
                and word in self.non_linguistic_symbols
            ):
                continue
            tokens.append(word)
        return tokens

    def tokens2text(self, tokens: Iterable[str]) -> str:
        """Join word tokens with ``self.delimiter`` (default: a space)."""
        delimiter = " " if self.delimiter is None else self.delimiter
        return delimiter.join(tokens)
