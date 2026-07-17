"""Tokenizer backed by a Hugging Face `transformers` `AutoTokenizer`."""

from pathlib import Path
from typing import Iterable, List, Optional, Union

from typeguard import check_argument_types

from espnet2.text.abs_tokenizer import AbsTokenizer

try:
    from transformers import AutoTokenizer

    is_transformers_available = True
except ImportError:
    is_transformers_available = False


class HuggingFaceTokenizer(AbsTokenizer):
    """Wraps a pretrained Hugging Face tokenizer as an :class:`AbsTokenizer`.

    Example:
        >>> tok = HuggingFaceTokenizer("bert-base-uncased")
        >>> tok.text2tokens("hello world")
        ['hello', 'world']
    """

    def __init__(self, model: Union[Path, str]):
        """Store the model name/path; the tokenizer itself loads lazily.

        Args:
            model: Hugging Face model id or local path, as accepted by
                ``transformers.AutoTokenizer.from_pretrained``.
        """
        assert check_argument_types()

        if not is_transformers_available:
            raise ImportError(
                "`transformers` is not available. Please install it via `pip install"
                " transformers` or `cd /path/to/espnet/tools && . ./activate_python.sh"
                " && ./installers/install_transformers.sh`."
            )

        self.model = str(model)
        # NOTE(kamo):
        # Don't build tokenizer in __init__()
        # because it's not picklable and it may cause following error,
        # "TypeError: can't pickle SwigPyObject objects",
        # when giving it as argument of "multiprocessing.Process()".
        self.tokenizer: Optional["AutoTokenizer"] = None

    def __repr__(self):
        return f'{self.__class__.__name__}(model="{self.model}")'

    def _build_tokenizer(self) -> None:
        """Lazily load the underlying Hugging Face tokenizer."""
        if self.tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model)

    def text2tokens(self, line: str) -> List[str]:
        """Tokenize ``line`` using the underlying Hugging Face tokenizer."""
        self._build_tokenizer()
        return self.tokenizer.tokenize(line)

    def tokens2text(self, tokens: Iterable[str]) -> str:
        """Decode a token sequence back into text, dropping special tokens."""
        self._build_tokenizer()
        return self.tokenizer.batch_decode(
            [self.tokenizer.convert_tokens_to_ids(tokens)], skip_special_tokens=True
        )[0]
