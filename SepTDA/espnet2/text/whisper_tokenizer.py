"""Tokenizer backed by OpenAI Whisper's own BPE tokenizer."""

from typing import Iterable, List

from typeguard import check_argument_types

from espnet2.text.abs_tokenizer import AbsTokenizer


class OpenAIWhisperTokenizer(AbsTokenizer):
    """Splits text using Whisper's built-in tokenizer.

    Example:
        >>> tok = OpenAIWhisperTokenizer("whisper_en")
        >>> tokens = tok.text2tokens("hello world")
        >>> tok.tokens2text(tokens)
        'hello world'
    """

    def __init__(self, model_type: str):
        """Load the Whisper tokenizer for the given model type.

        Args:
            model_type: ``"whisper_en"`` for the English-only vocabulary, or
                ``"whisper_multilingual"`` for the multilingual vocabulary.
        """
        assert check_argument_types()

        try:
            import whisper.tokenizer
        except Exception as e:
            print("Error: whisper is not properly installed.")
            print(
                "Please install whisper with: cd ${MAIN_ROOT}/tools && "
                "./installers/install_whisper.sh"
            )
            raise e

        self.model = model_type
        if model_type == "whisper_en":
            self.tokenizer = whisper.tokenizer.get_tokenizer(multilingual=False)
        # TODO(Shih-Lun): should support feeding in
        #                  different languages (default is en)
        elif model_type == "whisper_multilingual":
            self.tokenizer = whisper.tokenizer.get_tokenizer(
                multilingual=True, language=None
            )
        else:
            raise ValueError("tokenizer unsupported:", model_type)

    def __repr__(self):
        return f'{self.__class__.__name__}(model="{self.model}")'

    def text2tokens(self, line: str) -> List[str]:
        """Tokenize ``line`` with Whisper's BPE tokenizer (no special tokens)."""
        return self.tokenizer.tokenizer.tokenize(line, add_special_tokens=False)

    def tokens2text(self, tokens: Iterable[str]) -> str:
        """Decode Whisper BPE tokens back into a text string."""
        return self.tokenizer.tokenizer.convert_tokens_to_string(tokens)
