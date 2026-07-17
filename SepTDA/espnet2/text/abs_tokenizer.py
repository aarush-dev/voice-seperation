"""Abstract interface shared by all ESPnet text tokenizers.

A tokenizer converts between a raw text string and a sequence of token
strings (e.g. words, characters, BPE pieces, or phonemes). It is a purely
string-level operation: mapping tokens to integer ids is handled separately
by :class:`espnet2.text.token_id_converter.TokenIDConverter`.
"""

from abc import ABC, abstractmethod
from typing import Iterable, List


class AbsTokenizer(ABC):
    """Base class for text <-> token-string converters.

    Example:
        >>> tokenizer = SomeConcreteTokenizer(...)
        >>> tokens = tokenizer.text2tokens("hello world")
        >>> text = tokenizer.tokens2text(tokens)
    """

    @abstractmethod
    def text2tokens(self, line: str) -> List[str]:
        """Split a text string into a list of token strings."""
        raise NotImplementedError

    @abstractmethod
    def tokens2text(self, tokens: Iterable[str]) -> str:
        """Join a sequence of token strings back into a text string."""
        raise NotImplementedError
