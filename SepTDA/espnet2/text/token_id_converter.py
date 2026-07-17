"""Converts between token strings and integer ids using a fixed vocabulary."""

from pathlib import Path
from typing import Dict, Iterable, List, Union

import numpy as np
from typeguard import check_argument_types


class TokenIDConverter:
    """Maps tokens to ids (and back) using a vocabulary list.

    The vocabulary defines the id for each token as its line/list index.
    Tokens not found in the vocabulary are mapped to ``unk_symbol``'s id.

    Example:
        >>> conv = TokenIDConverter(["<unk>", "a", "b"])
        >>> conv.tokens2ids(["a", "b", "z"])
        [1, 2, 0]
        >>> conv.ids2tokens([1, 2, 0])
        ['a', 'b', '<unk>']
    """

    def __init__(
        self,
        token_list: Union[Path, str, Iterable[str]],
        unk_symbol: str = "<unk>",
    ):
        """Build the vocabulary and the token<->id lookup tables.

        Args:
            token_list: Either a path to a file with one token per line
                (line index is the token's id), or an in-memory list/iterable
                of token strings (list index is the token's id).
            unk_symbol: The token used for out-of-vocabulary lookups. It
                must already be present in ``token_list``.

        Raises:
            RuntimeError: If ``token_list`` contains a duplicate token, or
                if ``unk_symbol`` is not present in ``token_list``.
        """
        assert check_argument_types()

        if isinstance(token_list, (Path, str)):
            self.token_list, self.token_list_repr = self._load_token_list_from_file(
                token_list
            )
        else:
            self.token_list, self.token_list_repr = self._load_token_list_from_iterable(
                token_list
            )

        self.token2id: Dict[str, int] = self._build_token2id(self.token_list)

        self.unk_symbol = unk_symbol
        if self.unk_symbol not in self.token2id:
            raise RuntimeError(
                f"Unknown symbol '{unk_symbol}' doesn't exist in the token_list"
            )
        self.unk_id = self.token2id[self.unk_symbol]

    @staticmethod
    def _load_token_list_from_file(token_list_path: Union[Path, str]):
        """Read one token per line from a file, preserving a leading space."""
        token_list_path = Path(token_list_path)
        token_list: List[str] = []
        with token_list_path.open("r", encoding="utf-8") as f:
            for line in f:
                # Only the trailing newline is stripped; a leading space
                # (e.g. the sentencepiece "▁" convention) is preserved by
                # keeping the first character untouched.
                line = line[0] + line[1:].rstrip()
                token_list.append(line)
        return token_list, str(token_list_path)

    @staticmethod
    def _load_token_list_from_iterable(tokens: Iterable[str]):
        """Materialize an in-memory token list and a short repr for logging."""
        token_list: List[str] = list(tokens)
        preview = "".join(f"{t}, " for t in token_list[:3])
        token_list_repr = f"{preview}... (NVocab={len(token_list)})"
        return token_list, token_list_repr

    @staticmethod
    def _build_token2id(token_list: List[str]) -> Dict[str, int]:
        """Invert the token list into a token -> id lookup table."""
        token2id: Dict[str, int] = {}
        for token_id, token in enumerate(token_list):
            if token in token2id:
                raise RuntimeError(f'Symbol "{token}" is duplicated')
            token2id[token] = token_id
        return token2id

    def get_num_vocabulary_size(self) -> int:
        """Return the number of tokens in the vocabulary."""
        return len(self.token_list)

    def ids2tokens(self, integers: Union[np.ndarray, Iterable[int]]) -> List[str]:
        """Convert a 1-D sequence of ids to their token strings."""
        if isinstance(integers, np.ndarray) and integers.ndim != 1:
            raise ValueError(f"Must be 1 dim ndarray, but got {integers.ndim}")
        return [self.token_list[i] for i in integers]

    def tokens2ids(self, tokens: Iterable[str]) -> List[int]:
        """Convert token strings to ids, mapping unknown tokens to ``unk_id``."""
        return [self.token2id.get(token, self.unk_id) for token in tokens]
