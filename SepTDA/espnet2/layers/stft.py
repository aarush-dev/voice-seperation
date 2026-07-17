"""STFT / inverse-STFT front-end layer.

Wraps ``torch.stft`` (falling back to ``librosa.stft`` on devices without
MKL/native FFT support) to turn raw waveforms into complex time-frequency
representations, and provides the corresponding inverse transform. This is
typically the first layer in the separation pipeline, turning
(Batch, Nsamples) waveforms into (Batch, Frames, Freq, 2) real/imag
spectrograms consumed by the mask-estimation network.
"""
from typing import Optional, Tuple, Union

import librosa
import numpy as np
import torch
from packaging.version import parse as V
from torch_complex.tensor import ComplexTensor
from typeguard import typechecked

from espnet2.enh.layers.complex_utils import to_complex
from espnet2.layers.inversible_interface import InversibleInterface
from espnet.nets.pytorch_backend.nets_utils import make_pad_mask

is_torch_1_10_plus = V(torch.__version__) >= V("1.10.0")


class Stft(torch.nn.Module, InversibleInterface):
    """Short-time Fourier transform layer with an inverse STFT counterpart.

    Attributes:
        n_fft: FFT size.
        win_length: Analysis window length (defaults to ``n_fft``).
        hop_length: Number of samples between successive frames.
        center: Whether frames are centered (zero-padding at both ends),
            matching ``torch.stft(center=True)``.
        normalized: Whether the STFT is normalized by the window norm.
        onesided: Whether only the non-negative frequency half is returned.
        window: Name of the window function (e.g. "hann"), or None.
    """

    @typechecked
    def __init__(
        self,
        n_fft: int = 512,
        win_length: Optional[int] = None,
        hop_length: int = 128,
        window: Optional[str] = "hann",
        center: bool = True,
        normalized: bool = False,
        onesided: bool = True,
    ):
        super().__init__()
        self.n_fft = n_fft
        if win_length is None:
            self.win_length = n_fft
        else:
            self.win_length = win_length
        self.hop_length = hop_length
        self.center = center
        self.normalized = normalized
        self.onesided = onesided
        if window is not None and not hasattr(torch, f"{window}_window"):
            raise ValueError(f"{window} window is not implemented")
        self.window = window

    def extra_repr(self) -> str:
        return (
            f"n_fft={self.n_fft}, "
            f"win_length={self.win_length}, "
            f"hop_length={self.hop_length}, "
            f"center={self.center}, "
            f"normalized={self.normalized}, "
            f"onesided={self.onesided}"
        )

    def _build_window(
        self, dtype: torch.dtype, device: torch.device
    ) -> Optional[torch.Tensor]:
        """Instantiate the analysis window for the given dtype/device."""
        if self.window is not None:
            window_func = getattr(torch, f"{self.window}_window")
            return window_func(self.win_length, dtype=dtype, device=device)
        return None

    def _stft_torch(
        self, input: torch.Tensor, window: Optional[torch.Tensor]
    ) -> torch.Tensor:
        """Compute STFT via ``torch.stft`` (native CUDA/MKL/PyTorch>=1.10 path).

        Args:
            input: (Batch, Nsamples)
            window: (win_length,) or None
        Returns:
            output: (Batch, Freq, Frames, 2=real_imag)
        """
        stft_kwargs = dict(
            n_fft=self.n_fft,
            win_length=self.win_length,
            hop_length=self.hop_length,
            center=self.center,
            window=window,
            normalized=self.normalized,
            onesided=self.onesided,
        )
        stft_kwargs["return_complex"] = True
        # NOTE(Jinchuan) CuFFT is not compatible with bfloat16
        output = torch.stft(input.float(), **stft_kwargs)
        output = torch.view_as_real(output).type(input.dtype)
        return output

    def _stft_librosa(
        self, input: torch.Tensor, window: Optional[torch.Tensor]
    ) -> torch.Tensor:
        """Compute STFT via librosa (fallback for devices lacking MKL/native STFT).

        Args:
            input: (Batch, Nsamples)
            window: (win_length,) or None
        Returns:
            output: (Batch, Freq, Frames, 2=real_imag)
        """
        if self.training:
            raise NotImplementedError(
                "stft is implemented with librosa on this device, which does not "
                "support the training mode."
            )

        # use stft_kwargs to flexibly control different PyTorch versions' kwargs
        # note: librosa does not support a win_length that is < n_ftt
        # but the window can be manually padded (see below).
        stft_kwargs = dict(
            n_fft=self.n_fft,
            win_length=self.n_fft,
            hop_length=self.hop_length,
            center=self.center,
            window=window,
            pad_mode="reflect",
        )

        if window is not None:
            # pad the given window to n_fft
            n_pad_left = (self.n_fft - window.shape[0]) // 2
            n_pad_right = self.n_fft - window.shape[0] - n_pad_left
            stft_kwargs["window"] = torch.cat(
                [torch.zeros(n_pad_left), window, torch.zeros(n_pad_right)], 0
            ).numpy()
        else:
            win_length = (
                self.win_length if self.win_length is not None else self.n_fft
            )
            stft_kwargs["window"] = torch.ones(win_length)

        output = []
        # iterate over istances in a batch
        for i, instance in enumerate(input):
            stft = librosa.stft(input[i].numpy(), **stft_kwargs)
            output.append(torch.tensor(np.stack([stft.real, stft.imag], -1)))
        output = torch.stack(output, 0)
        if not self.onesided:
            len_conj = self.n_fft - output.shape[1]
            conj = output[:, 1 : 1 + len_conj].flip(1)
            conj[:, :, :, -1].data *= -1
            output = torch.cat([output, conj], 1)
        if self.normalized:
            output = output * (stft_kwargs["window"].shape[0] ** (-0.5))
        return output

    def _compute_output_lengths(
        self, output: torch.Tensor, ilens: Optional[torch.Tensor]
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Compute per-frame output lengths and zero out padded frames.

        Args:
            output: (Batch, Frames, ..., Freq, 2=real_imag)
            ilens: (Batch,) input sample lengths, or None
        Returns:
            output: Same shape as input, with padded frames zeroed.
            olens: (Batch,) output frame lengths, or None
        """
        if ilens is not None:
            if self.center:
                pad = self.n_fft // 2
                ilens = ilens + 2 * pad

            olens = (
                torch.div(ilens - self.n_fft, self.hop_length, rounding_mode="trunc")
                + 1
            )
            output.masked_fill_(make_pad_mask(olens, output, 1), 0.0)
        else:
            olens = None
        return output, olens

    def forward(
        self, input: torch.Tensor, ilens: torch.Tensor = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """STFT forward function.

        Args:
            input: (Batch, Nsamples) or (Batch, Nsample, Channels)
            ilens: (Batch)
        Returns:
            output: (Batch, Frames, Freq, 2) or (Batch, Frames, Channels, Freq, 2)

        """
        bs = input.size(0)
        if input.dim() == 3:
            multi_channel = True
            # input: (Batch, Nsample, Channels) -> (Batch * Channels, Nsample)
            input = input.transpose(1, 2).reshape(-1, input.size(1))
        else:
            multi_channel = False

        # NOTE(kamo):
        #   The default behaviour of torch.stft is compatible with librosa.stft
        #   about padding and scaling.
        #   Note that it's different from scipy.signal.stft

        # output: (Batch, Freq, Frames, 2=real_imag)
        # or (Batch, Channel, Freq, Frames, 2=real_imag)
        window = self._build_window(input.dtype, input.device)

        # For the compatibility of ARM devices, which do not support
        # torch.stft() due to the lack of MKL (on older pytorch versions),
        # there is an alternative replacement implementation with librosa.
        # Note: pytorch >= 1.10.0 now has native support for FFT and STFT
        # on all cpu targets including ARM.
        if input.is_cuda or torch.backends.mkl.is_available() or is_torch_1_10_plus:
            output = self._stft_torch(input, window)
        else:
            output = self._stft_librosa(input, window)

        # output: (Batch, Freq, Frames, 2=real_imag)
        # -> (Batch, Frames, Freq, 2=real_imag)
        output = output.transpose(1, 2)
        if multi_channel:
            # output: (Batch * Channel, Frames, Freq, 2=real_imag)
            # -> (Batch, Frame, Channel, Freq, 2=real_imag)
            output = output.view(bs, -1, output.size(1), output.size(2), 2).transpose(
                1, 2
            )

        output, olens = self._compute_output_lengths(output, ilens)
        return output, olens

    def inverse(
        self, input: Union[torch.Tensor, ComplexTensor], ilens: torch.Tensor = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Inverse STFT.

        Args:
            input: Tensor(batch, T, F, 2) or ComplexTensor(batch, T, F)
            ilens: (batch,)
        Returns:
            wavs: (batch, samples)
            ilens: (batch,)
        """
        input = to_complex(input)

        if self.window is not None:
            window_func = getattr(torch, f"{self.window}_window")
            datatype = input.real.dtype
            window = window_func(self.win_length, dtype=datatype, device=input.device)
        else:
            window = None

        input = input.transpose(1, 2)

        wavs = torch.functional.istft(
            input,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=window,
            center=self.center,
            normalized=self.normalized,
            onesided=self.onesided,
            length=ilens.max() if ilens is not None else ilens,
            return_complex=False,
        )

        return wavs, ilens
