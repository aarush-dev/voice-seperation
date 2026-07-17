"""Abstract interface for target-speaker-extraction (TSE) frontends.

A TSE extractor takes the encoded mixture plus an encoded enrollment
(auxiliary) utterance identifying the target speaker, and returns an
estimate of that speaker's encoded signal.
"""
from abc import ABC, abstractmethod
from collections import OrderedDict
from typing import Tuple

import torch


class AbsExtractor(torch.nn.Module, ABC):
    """Base class for modules that extract one target speaker's signal."""

    @abstractmethod
    def forward(
        self,
        input: torch.Tensor,
        ilens: torch.Tensor,
        input_aux: torch.Tensor,
        ilens_aux: torch.Tensor,
        suffix_tag: str = "",
    ) -> Tuple[Tuple[torch.Tensor], torch.Tensor, OrderedDict]:
        """Extract the target speaker's signal from an encoded mixture.

        Args:
            input: (Batch, Frames, Freq) encoded mixture feature.
            ilens: (Batch,) valid frame lengths of ``input``.
            input_aux: (Batch, Frames_aux, Freq) or (Batch, Emb) encoded
                enrollment feature identifying the target speaker.
            ilens_aux: (Batch,) valid frame lengths of ``input_aux``.
            suffix_tag: suffix appended to keys placed in the returned
                side-info dict, useful when the same extractor is invoked
                once per speaker.

        Returns:
            masked: extracted feature(s) for the target speaker.
            ilens: (Batch,) output frame lengths.
            others: OrderedDict of auxiliary outputs (e.g. predicted masks).
        """
        raise NotImplementedError
