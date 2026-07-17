"""Text normalization pipeline used before tokenization."""

from typing import Collection, List, Optional

import tacotron_cleaner.cleaners
from jaconv import jaconv
from typeguard import check_argument_types

try:
    from vietnamese_cleaner import vietnamese_cleaners
except ImportError:
    vietnamese_cleaners = None

from espnet2.text.korean_cleaner import KoreanCleaner

try:
    from whisper.normalizers import BasicTextNormalizer, EnglishTextNormalizer
except (ImportError, SyntaxError):
    BasicTextNormalizer = None


class TextCleaner:
    """Applies a configurable chain of text-normalization steps.

    Each entry in ``cleaner_types`` names one normalizer, and they are
    applied in order. Supported names: ``"tacotron"``, ``"jaconv"``,
    ``"vietnamese"``, ``"korean_cleaner"``, ``"whisper_en"``,
    ``"whisper_basic"``.

    Examples:
        >>> cleaner = TextCleaner("tacotron")
        >>> cleaner("(Hello-World);   &  jr. & dr.")
        'HELLO WORLD, AND JUNIOR AND DOCTOR'

    """

    def __init__(self, cleaner_types: Collection[str] = None):
        """Initialize the cleaner chain.

        Args:
            cleaner_types: A single cleaner name, a collection of cleaner
                names applied in order, or ``None`` for a no-op cleaner.
        """
        assert check_argument_types()

        self.cleaner_types: List[str] = self._normalize_cleaner_types(cleaner_types)
        self.whisper_cleaner = self._build_whisper_cleaner(self.cleaner_types)

    @staticmethod
    def _normalize_cleaner_types(
        cleaner_types: Optional[Collection[str]],
    ) -> List[str]:
        if cleaner_types is None:
            return []
        if isinstance(cleaner_types, str):
            return [cleaner_types]
        return list(cleaner_types)

    @staticmethod
    def _build_whisper_cleaner(cleaner_types: List[str]):
        """Instantiate the Whisper text normalizer, if one was requested."""
        if BasicTextNormalizer is None:
            return None
        for cleaner_type in cleaner_types:
            if cleaner_type == "whisper_en":
                return EnglishTextNormalizer()
            elif cleaner_type == "whisper_basic":
                return BasicTextNormalizer()
        return None

    def __call__(self, text: str) -> str:
        """Run ``text`` through each configured cleaner, in order."""
        for cleaner_type in self.cleaner_types:
            if cleaner_type == "tacotron":
                text = tacotron_cleaner.cleaners.custom_english_cleaners(text)
            elif cleaner_type == "jaconv":
                text = jaconv.normalize(text)
            elif cleaner_type == "vietnamese":
                if vietnamese_cleaners is None:
                    raise RuntimeError("Please install underthesea")
                text = vietnamese_cleaners.vietnamese_cleaner(text)
            elif cleaner_type == "korean_cleaner":
                text = KoreanCleaner.normalize_text(text)
            elif "whisper" in cleaner_type and self.whisper_cleaner is not None:
                text = self.whisper_cleaner(text)
            else:
                raise RuntimeError(f"Not supported: type={cleaner_type}")

        return text
