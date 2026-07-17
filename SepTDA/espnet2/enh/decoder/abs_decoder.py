"""Abstract base class for enhancement/separation frontend decoders."""
from abc import ABC, abstractmethod
from typing import Tuple

import torch


class AbsDecoder(torch.nn.Module, ABC):
    """Base class for decoders that turn separator features back into a waveform.

    A decoder is the inverse of the matching :class:`AbsEncoder`: it maps
    the per-speaker features produced by the separator back to time-domain
    waveforms.
    """

    @abstractmethod
    def forward(
        self,
        input: torch.Tensor,
        ilens: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Decode a batch of features into waveforms.

        Args:
            input: (Batch, Frames, Channel) feature sequence for one speaker.
            ilens: (Batch,) valid length of each feature sequence.

        Returns:
            wav: (Batch, samples) reconstructed waveform.
            wav_lens: (Batch,) valid length of each waveform.
        """
        raise NotImplementedError

    def forward_streaming(self, input_frame: torch.Tensor):
        """Decode a single streaming feature frame into a waveform frame."""
        raise NotImplementedError

    def streaming_merge(self, chunks: torch.Tensor, ilens: torch.tensor = None):
        """Merge frame-level decoded chunks into a continuous waveform.

        It merges the frame-level processed audio chunks in the streaming
        *simulation*. It is noted that, in real applications, the processed
        audio should be sent to the output channel frame by frame. You may
        refer to this function to manage your streaming output buffer.

        Args:
            chunks: List [(B, frame_size),]
            ilens: [B]
        Returns:
            merge_audio: [B, T]
        """
        raise NotImplementedError
