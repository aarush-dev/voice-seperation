"""Token <-> id conversion for OpenAI Whisper's tokenizer/vocabulary.

Unlike :class:`espnet2.text.token_id_converter.TokenIDConverter`, this reads
the vocabulary directly from the Whisper tokenizer instead of a token-list
file, and prepends Whisper's start-of-transcript sequence when encoding.

Special-token ids for reference:

    <sos>/<eos> for Whisper multilingual:
        '<|startoftranscript|>': 50258
        '<|endoftext|>':         50257

    <sos>/<eos> for Whisper english:
        '<|startoftranscript|>': 50257
        '<|endoftext|>':         50256
"""

from typing import Iterable, List, Union

import numpy as np
from typeguard import check_argument_types


class OpenAIWhisperTokenIDConverter:
    """Converts between token strings/ids using Whisper's own vocabulary.

    Example:
        >>> conv = OpenAIWhisperTokenIDConverter("whisper_en")
        >>> ids = conv.tokens2ids(["hello"])
        >>> conv.ids2tokens(ids)
        ['hello']
    """

    def __init__(
        self,
        model_type: str = "whisper_multilingual",
    ):
        """Load the Whisper tokenizer that defines the vocabulary.

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

    def get_num_vocabulary_size(self) -> int:
        """Return the vocabulary size, including added special tokens."""
        return self.tokenizer.tokenizer.vocab_size + len(
            self.tokenizer.tokenizer.get_added_vocab()
        )

    def ids2tokens(self, integers: Union[np.ndarray, Iterable[int]]) -> List[str]:
        """Convert integer ids to token strings, dropping special tokens."""
        return self.tokenizer.tokenizer.convert_ids_to_tokens(
            integers, skip_special_tokens=True
        )

    def tokens2ids(self, tokens: Iterable[str]) -> List[int]:
        """Convert token strings to ids, prefixed with the start-of-transcript ids."""
        return list(
            self.tokenizer.sot_sequence_including_notimestamps[1:]
        ) + self.tokenizer.tokenizer.convert_tokens_to_ids(tokens)
