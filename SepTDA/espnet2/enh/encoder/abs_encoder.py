"""Abstract base class for enhancement/separation frontend encoders."""
from abc import ABC, abstractmethod
from typing import List, Tuple

import torch


class AbsEncoder(torch.nn.Module, ABC):
    """Base class for encoders that turn a waveform into a feature representation.

    An encoder is the first stage of the enhancement/separation pipeline: it
    maps the raw (possibly multi-channel) mixture waveform to a sequence of
    features (e.g. a learned filterbank or an STFT spectrum) that the
    separator operates on. The matching :class:`AbsDecoder` maps the
    separator's output features back to a waveform.
    """

    @abstractmethod
    def forward(
        self,
        input: torch.Tensor,
        ilens: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode a batch of waveforms into features.

        Args:
            input: (Batch, samples) mixture waveform.
            ilens: (Batch,) valid length of each waveform in the batch.

        Returns:
            feature: (Batch, Frames, Channel) encoded feature sequence.
            flens: (Batch,) valid length of each feature sequence.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def output_dim(self) -> int:
        """Feature dimension produced by :meth:`forward`."""
        raise NotImplementedError

    def forward_streaming(self, input: torch.Tensor):
        """Encode a single streaming frame (see :meth:`streaming_frame`)."""
        raise NotImplementedError

    def streaming_frame(self, audio: torch.Tensor) -> List[torch.Tensor]:
        """Split a full-length audio signal into frame-level chunks.

        This simulates streaming inference: it splits the *entire* audio
        into the chunks that would be fed to :meth:`forward_streaming` one
        at a time in a real streaming application. It is noted that this
        function takes the entire long audio as input for a streaming
        simulation. You may refer to this function to manage your streaming
        input buffer in a real streaming application.

        Args:
            audio: (B, T)
        Returns:
            chunked: List [(B, frame_size),]
        """
        NotImplementedError
