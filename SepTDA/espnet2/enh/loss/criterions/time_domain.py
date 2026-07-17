"""Time-domain loss criterions for speech enhancement and separation."""
import logging
import math
from abc import ABC
from typing import List, Optional

import ci_sdr
import fast_bss_eval
import torch
from packaging.version import parse as V
from torch_complex.tensor import ComplexTensor

from espnet2.enh.layers.complex_utils import is_complex
from espnet2.enh.loss.criterions.abs_loss import AbsEnhLoss
from espnet2.layers.stft import Stft

is_torch_1_9_plus = V(torch.__version__) >= V("1.9.0")

EPS = torch.finfo(torch.get_default_dtype()).eps


class TimeDomainLoss(AbsEnhLoss, ABC):
    """Base class for all time-domain Enhancement loss modules.

    Subclasses operate directly on waveforms (as opposed to
    :class:`~espnet2.enh.loss.criterions.tf_domain.FrequencyDomainLoss`,
    which operates on STFT spectra/masks). This base class only manages the
    ``name`` reported to the training logger/stats, appending ``_noise`` or
    ``_dereverb`` when the loss is flagged as targeting the noise or
    dereverberation output rather than the main speech estimate.
    """

    @property
    def name(self) -> str:
        return self._name

    @property
    def only_for_test(self) -> bool:
        return self._only_for_test

    @property
    def is_noise_loss(self) -> bool:
        return self._is_noise_loss

    @property
    def is_dereverb_loss(self) -> bool:
        return self._is_dereverb_loss

    def __init__(
        self,
        name: str,
        only_for_test: bool = False,
        is_noise_loss: bool = False,
        is_dereverb_loss: bool = False,
    ):
        super().__init__()
        # only used during validation
        self._only_for_test = only_for_test
        # only used to calculate the noise-related loss
        self._is_noise_loss = is_noise_loss
        # only used to calculate the dereverberation-related loss
        self._is_dereverb_loss = is_dereverb_loss
        if is_noise_loss and is_dereverb_loss:
            raise ValueError(
                "`is_noise_loss` and `is_dereverb_loss` cannot be True at the same time"
            )
        if is_noise_loss and "noise" not in name:
            name = name + "_noise"
        if is_dereverb_loss and "dereverb" not in name:
            name = name + "_dereverb"
        self._name = name


class CISDRLoss(TimeDomainLoss):
    """CI-SDR loss

    Reference:
        Convolutive Transfer Function Invariant SDR Training
        Criteria for Multi-Channel Reverberant Speech Separation;
        C. Boeddeker et al., 2021;
        https://arxiv.org/abs/2011.15003
    Args:
        ref: (Batch, samples)
        inf: (Batch, samples)
        filter_length (int): a time-invariant filter that allows
                                slight distortion via filtering
    Returns:
        loss: (Batch,)
    """

    def __init__(
        self,
        filter_length: int = 512,
        name: Optional[str] = None,
        only_for_test: bool = False,
        is_noise_loss: bool = False,
        is_dereverb_loss: bool = False,
    ):
        _name = "ci_sdr_loss" if name is None else name
        super().__init__(
            _name,
            only_for_test=only_for_test,
            is_noise_loss=is_noise_loss,
            is_dereverb_loss=is_dereverb_loss,
        )

        self.filter_length = filter_length

    def forward(
        self,
        ref: torch.Tensor,
        inf: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the (negative, i.e. loss-form) CI-SDR between ref and inf.

        Args:
            ref: (Batch, samples) reference waveform.
            inf: (Batch, samples) estimated waveform.
        Returns:
            loss: (Batch,) CI-SDR loss (already permutation-fixed; the
                caller is expected to have already matched ref/inf pairs).
        """
        assert ref.shape == inf.shape, (ref.shape, inf.shape)

        return ci_sdr.pt.ci_sdr_loss(
            inf, ref, compute_permutation=False, filter_length=self.filter_length
        )


class SNRLoss(TimeDomainLoss):
    """Plain (non scale-invariant) signal-to-noise ratio loss."""

    def __init__(
        self,
        eps: float = EPS,
        name: Optional[str] = None,
        only_for_test: bool = False,
        is_noise_loss: bool = False,
        is_dereverb_loss: bool = False,
    ):
        _name = "snr_loss" if name is None else name
        super().__init__(
            _name,
            only_for_test=only_for_test,
            is_noise_loss=is_noise_loss,
            is_dereverb_loss=is_dereverb_loss,
        )

        self.eps = float(eps)

    def forward(self, ref: torch.Tensor, inf: torch.Tensor) -> torch.Tensor:
        """Negative SNR (in dB) between ref and inf.

        Args:
            ref: (Batch, samples) reference waveform.
            inf: (Batch, samples) estimated waveform.
        Returns:
            loss: (Batch,) negative SNR, i.e. lower is better.
        """
        noise = inf - ref

        snr = 20 * (
            torch.log10(torch.norm(ref, p=2, dim=1).clamp(min=self.eps))
            - torch.log10(torch.norm(noise, p=2, dim=1).clamp(min=self.eps))
        )
        return -snr


class SDRLoss(TimeDomainLoss):
    """SDR loss.

    filter_length: int
        The length of the distortion filter allowed (default: ``512``)
    use_cg_iter:
        If provided, an iterative method is used to solve for the distortion
        filter coefficients instead of direct Gaussian elimination.
        This can speed up the computation of the metrics in case the filters
        are long. Using a value of 10 here has been shown to provide
        good accuracy in most cases and is sufficient when using this
        loss to train neural separation networks.
    clamp_db: float
        clamp the output value in  [-clamp_db, clamp_db]
    zero_mean: bool
        When set to True, the mean of all signals is subtracted prior.
    load_diag:
        If provided, this small value is added to the diagonal coefficients of
        the system metrices when solving for the filter coefficients.
        This can help stabilize the metric in the case where some of the reference
        signals may sometimes be zero
    """

    def __init__(
        self,
        filter_length: int = 512,
        use_cg_iter: Optional[int] = None,
        clamp_db: Optional[float] = None,
        zero_mean: bool = True,
        load_diag: Optional[float] = None,
        name: Optional[str] = None,
        only_for_test: bool = False,
        is_noise_loss: bool = False,
        is_dereverb_loss: bool = False,
    ):
        _name = "sdr_loss" if name is None else name
        super().__init__(
            _name,
            only_for_test=only_for_test,
            is_noise_loss=is_noise_loss,
            is_dereverb_loss=is_dereverb_loss,
        )

        self.filter_length = filter_length
        self.use_cg_iter = use_cg_iter
        self.clamp_db = clamp_db
        self.zero_mean = zero_mean
        self.load_diag = load_diag

    def forward(self, ref: torch.Tensor, est: torch.Tensor) -> torch.Tensor:
        """SDR forward.

        Args:
            ref: Tensor, (..., n_samples)
                reference signal
            est: Tensor (..., n_samples)
                estimated signal

        Returns:
            loss: (...,)
                the SDR loss (negative sdr)
        """

        sdr_loss = fast_bss_eval.sdr_loss(
            est=est,
            ref=ref,
            filter_length=self.filter_length,
            use_cg_iter=self.use_cg_iter,
            zero_mean=self.zero_mean,
            clamp_db=self.clamp_db,
            load_diag=self.load_diag,
            pairwise=False,
        )

        return sdr_loss


class SISNRLoss(TimeDomainLoss):
    """SI-SNR (or named SI-SDR) loss

    A more stable SI-SNR loss with clamp from `fast_bss_eval`.

    Attributes:
        clamp_db: float
            clamp the output value in  [-clamp_db, clamp_db]
        zero_mean: bool
            When set to True, the mean of all signals is subtracted prior.
        eps: float
            Deprecated. Kept for compatibility.
    """

    def __init__(
        self,
        clamp_db: Optional[float] = None,
        zero_mean: bool = True,
        eps: Optional[float] = None,
        name: Optional[str] = None,
        only_for_test: bool = False,
        is_noise_loss: bool = False,
        is_dereverb_loss: bool = False,
    ):
        _name = "si_snr_loss" if name is None else name
        super().__init__(
            _name,
            only_for_test=only_for_test,
            is_noise_loss=is_noise_loss,
            is_dereverb_loss=is_dereverb_loss,
        )

        self.clamp_db = clamp_db
        self.zero_mean = zero_mean
        if eps is not None:
            logging.warning("Eps is deprecated in si_snr loss, set clamp_db instead.")
            if self.clamp_db is None:
                self.clamp_db = -math.log10(eps / (1 - eps)) * 10

    def forward(self, ref: torch.Tensor, est: torch.Tensor) -> torch.Tensor:
        """SI-SNR forward.

        Args:

            ref: Tensor, (..., n_samples)
                reference signal
            est: Tensor (..., n_samples)
                estimated signal

        Returns:
            loss: (...,)
                the SI-SDR loss (negative si-sdr)
        """
        assert torch.is_tensor(est) and torch.is_tensor(ref), est

        si_snr = fast_bss_eval.si_sdr_loss(
            est=est.float(),
            ref=ref.float(),
            zero_mean=self.zero_mean,
            clamp_db=self.clamp_db,
            pairwise=False,
        )

        return si_snr


class TimeDomainMSE(TimeDomainLoss):
    """Plain mean squared error between two waveforms."""

    def __init__(
        self,
        name: Optional[str] = None,
        only_for_test: bool = False,
        is_noise_loss: bool = False,
        is_dereverb_loss: bool = False,
    ):
        _name = "TD_MSE_loss" if name is None else name
        super().__init__(
            _name,
            only_for_test=only_for_test,
            is_noise_loss=is_noise_loss,
            is_dereverb_loss=is_dereverb_loss,
        )

    def forward(self, ref: torch.Tensor, inf: torch.Tensor) -> torch.Tensor:
        """Time-domain MSE loss forward.

        Args:
            ref: (Batch, T) or (Batch, T, C)
            inf: (Batch, T) or (Batch, T, C)
        Returns:
            loss: (Batch,)
        """
        assert ref.shape == inf.shape, (ref.shape, inf.shape)

        mseloss = (ref - inf).pow(2)
        if ref.dim() == 3:
            mseloss = mseloss.mean(dim=[1, 2])
        elif ref.dim() == 2:
            mseloss = mseloss.mean(dim=1)
        else:
            raise ValueError(
                "Invalid input shape: ref={}, inf={}".format(ref.shape, inf.shape)
            )
        return mseloss


class TimeDomainL1(TimeDomainLoss):
    """Plain mean absolute error (L1) between two waveforms."""

    def __init__(
        self,
        name: Optional[str] = None,
        only_for_test: bool = False,
        is_noise_loss: bool = False,
        is_dereverb_loss: bool = False,
    ):
        _name = "TD_L1_loss" if name is None else name
        super().__init__(
            _name,
            only_for_test=only_for_test,
            is_noise_loss=is_noise_loss,
            is_dereverb_loss=is_dereverb_loss,
        )

    def forward(self, ref: torch.Tensor, inf: torch.Tensor) -> torch.Tensor:
        """Time-domain L1 loss forward.

        Args:
            ref: (Batch, T) or (Batch, T, C)
            inf: (Batch, T) or (Batch, T, C)
        Returns:
            loss: (Batch,)
        """
        assert ref.shape == inf.shape, (ref.shape, inf.shape)

        l1loss = abs(ref - inf)
        if ref.dim() == 3:
            l1loss = l1loss.mean(dim=[1, 2])
        elif ref.dim() == 2:
            l1loss = l1loss.mean(dim=1)
        else:
            raise ValueError(
                "Invalid input shape: ref={}, inf={}".format(ref.shape, inf.shape)
            )
        return l1loss


def _build_multires_stft_encoders(
    window_sz: List[int], hop_sz: Optional[List[int]]
) -> torch.nn.ModuleList:
    """Build one :class:`Stft` module per analysis window size for multi-resolution
    spectral losses.

    Args:
        window_sz: list of STFT window sizes (each must be even).
        hop_sz: matching list of hop sizes, or ``None`` to default each hop
            size to half of its window size.
    Returns:
        A ``ModuleList`` of ``Stft`` modules, one per ``(window, hop)`` pair.
    """
    assert all(x % 2 == 0 for x in window_sz)
    if hop_sz is None:
        hop_sz = [w // 2 for w in window_sz]

    stft_encoders = torch.nn.ModuleList([])
    for w, h in zip(window_sz, hop_sz):
        stft_enc = Stft(
            n_fft=w,
            win_length=w,
            hop_length=h,
            window=None,
            center=True,
            normalized=False,
            onesided=True,
        )
        stft_encoders.append(stft_enc)
    return stft_encoders


class MultiResL1SpecLoss(TimeDomainLoss):
    """Multi-Resolution L1 time-domain + STFT mag loss

    Reference:
    Lu, Y. J., Cornell, S., Chang, X., Zhang, W., Li, C., Ni, Z., ... & Watanabe, S.
    Towards Low-Distortion Multi-Channel Speech Enhancement:
    The ESPNET-Se Submission to the L3DAS22 Challenge. ICASSP 2022 p. 9201-9205.

    Attributes:
        window_sz: (list)
            list of STFT window sizes.
        hop_sz: (list, optional)
            list of hop_sizes, default is each window_sz // 2.
        eps: (float)
            stability epsilon
        time_domain_weight: (float)
            weight for time domain loss.
        normalize_variance (bool)
            whether or not to normalize the variance when calculating the loss.
        reduction (str)
            select from "sum" and "mean"
    """

    def __init__(
        self,
        window_sz: List[int] = [512],
        hop_sz: Optional[List[int]] = None,
        eps: float = 1e-8,
        time_domain_weight: float = 0.5,
        normalize_variance: bool = False,
        reduction: str = "sum",
        name: Optional[str] = None,
        only_for_test: bool = False,
        is_noise_loss: bool = False,
        is_dereverb_loss: bool = False,
    ):
        _name = "TD_L1_loss" if name is None else name
        super().__init__(
            _name,
            only_for_test=only_for_test,
            is_noise_loss=is_noise_loss,
            is_dereverb_loss=is_dereverb_loss,
        )

        assert all([x % 2 == 0 for x in window_sz])
        self.window_sz = window_sz

        if hop_sz is None:
            self.hop_sz = [x // 2 for x in window_sz]
        else:
            self.hop_sz = hop_sz

        self.time_domain_weight = time_domain_weight
        self.normalize_variance = normalize_variance
        self.eps = eps
        self.stft_encoders = _build_multires_stft_encoders(
            self.window_sz, self.hop_sz
        )

        assert reduction in ("sum", "mean")
        self.reduction = reduction

    @property
    def name(self) -> str:
        return "l1_timedomain+magspec_loss"

    def get_magnitude(self, stft: torch.Tensor, eps: float = 1e-06) -> torch.Tensor:
        """Convert a real/imag-packed STFT output ``(..., 2)`` to magnitude.

        Args:
            stft: (..., 2) real STFT output as returned by ``Stft.forward``,
                with the last dim holding [real, imag].
            eps: stability epsilon used only in the ``ComplexTensor`` fallback
                path (pre torch-1.9 complex support).
        Returns:
            magnitude: (...,) magnitude spectrum.
        """
        if is_torch_1_9_plus:
            stft = torch.complex(stft[..., 0], stft[..., 1])
            return stft.abs()
        else:
            stft = ComplexTensor(stft[..., 0], stft[..., 1])
            return (stft.real.pow(2) + stft.imag.pow(2) + eps).sqrt()

    def _reduce(self, x: torch.Tensor, dim) -> torch.Tensor:
        """Sum- or mean-reduce ``x`` over ``dim`` according to ``self.reduction``."""
        if self.reduction == "sum":
            return torch.sum(x, dim=dim)
        else:
            return torch.mean(x, dim=dim)

    @torch.cuda.amp.autocast(enabled=False)
    def forward(
        self,
        target: torch.Tensor,
        estimate: torch.Tensor,
    ) -> torch.Tensor:
        """forward.

        Computes a scale-compensated L1 loss in the time domain, plus (if any
        STFT resolutions are configured) an L1 loss between the magnitude
        spectrograms of ``target`` and the scaled ``estimate``, averaged over
        resolutions and combined via ``time_domain_weight``.

        Args:
            target: (Batch, T)
            estimate: (Batch, T)
        Returns:
            loss: (Batch,)
        """
        assert target.shape == estimate.shape, (target.shape, estimate.shape)
        half_precision = (torch.float16, torch.bfloat16)
        if target.dtype in half_precision or estimate.dtype in half_precision:
            target = target.float()
            estimate = estimate.float()
        if self.normalize_variance:
            target = target / torch.std(target, dim=1, keepdim=True)
            estimate = estimate / torch.std(estimate, dim=1, keepdim=True)
        # shape bsz, samples
        # Least-squares scale factor that best aligns `estimate` to `target`
        # (removes any gain mismatch before computing the L1 distance).
        scaling_factor = torch.sum(estimate * target, -1, keepdim=True) / (
            torch.sum(estimate**2, -1, keepdim=True) + self.eps
        )
        time_domain_loss = self._reduce(
            (estimate * scaling_factor - target).abs(), dim=-1
        )

        if len(self.stft_encoders) == 0:
            return time_domain_loss
        else:
            spectral_loss = torch.zeros_like(time_domain_loss)
            for stft_enc in self.stft_encoders:
                target_mag = self.get_magnitude(stft_enc(target)[0])
                estimate_mag = self.get_magnitude(
                    stft_enc(estimate * scaling_factor)[0]
                )
                spectral_loss += self._reduce(
                    (estimate_mag - target_mag).abs(), dim=(1, 2)
                )

            return time_domain_loss * self.time_domain_weight + (
                1 - self.time_domain_weight
            ) * spectral_loss / len(self.stft_encoders)


class MultiResL1STFTLoss(TimeDomainLoss):
    """Multi-Resolution L1 STFT loss

    Attributes:
        window_sz: (list)
            list of STFT window sizes.
        hop_sz: (list, optional)
            list of hop_sizes, default is each window_sz // 2.
        eps: (float)
            stability epsilon
        normalize_variance (bool)
            whether or not to normalize the variance when calculating the loss.
        reduction (str)
            select from "sum" and "mean"
    """

    def __init__(
        self,
        window_sz: List[int] = [512],
        hop_sz: Optional[List[int]] = None,
        normalize_variance: bool = False,
        reduction: str = "mean",
        name: Optional[str] = None,
        only_for_test: bool = False,
        is_noise_loss: bool = False,
        is_dereverb_loss: bool = False,
    ):
        _name = "l1_mrstft_loss" if name is None else name
        super().__init__(
            _name,
            only_for_test=only_for_test,
            is_noise_loss=is_noise_loss,
            is_dereverb_loss=is_dereverb_loss,
        )

        assert all([x % 2 == 0 for x in window_sz])
        self.window_sz = window_sz

        if hop_sz is None:
            self.hop_sz = [x // 2 for x in window_sz]
        else:
            self.hop_sz = hop_sz

        self.normalize_variance = normalize_variance
        self.stft_encoders = _build_multires_stft_encoders(
            self.window_sz, self.hop_sz
        )

        assert reduction in ("sum", "mean")
        self.reduction = reduction

    @property
    def name(self) -> str:
        return "l1_mrstft_loss"

    def compute_l1_loss(self, ref: torch.Tensor, inf: torch.Tensor) -> torch.Tensor:
        """L1 distance between two STFT spectra, reduced over T (and F, C).

        Args:
            ref: (Batch, T, F) or (Batch, T, C, F) complex spectrum.
            inf: same shape as ``ref``.
        Returns:
            loss: (Batch,)
        """
        assert ref.shape == inf.shape, (ref.shape, inf.shape)

        if is_complex(inf):
            l1loss = (
                abs(ref.real - inf.real)
                + abs(ref.imag - inf.imag)
                + abs(ref.abs() - inf.abs())
            )
        else:
            l1loss = abs(ref - inf)
        if ref.dim() == 3:
            reduce_dims = [1, 2]
        elif ref.dim() == 4:
            reduce_dims = [1, 2, 3]
        else:
            raise ValueError(
                "Invalid input shape: ref={}, inf={}".format(ref.shape, inf.shape)
            )
        if self.reduction == "sum":
            l1loss = l1loss.sum(dim=reduce_dims)
        elif self.reduction == "mean":
            l1loss = l1loss.mean(dim=reduce_dims)

        return l1loss

    @torch.cuda.amp.autocast(enabled=False)
    def forward(
        self,
        target: torch.Tensor,
        estimate: torch.Tensor,
    ) -> torch.Tensor:
        """forward.

        Averages the L1 spectral distance (see :meth:`compute_l1_loss`) over
        all configured STFT resolutions.

        Args:
            target: (Batch, T)
            estimate: (Batch, T)
        Returns:
            loss: (Batch,)
        """
        assert target.shape == estimate.shape, (target.shape, estimate.shape)
        half_precision = (torch.float16, torch.bfloat16)
        if target.dtype in half_precision or estimate.dtype in half_precision:
            target = target.float()
            estimate = estimate.float()
        if self.normalize_variance:
            target = target / torch.std(target, dim=1, keepdim=True)
            estimate = estimate / torch.std(estimate, dim=1, keepdim=True)

        spectral_loss = torch.zeros(
            target.shape[0], dtype=target.dtype, device=target.device
        )
        for stft_enc in self.stft_encoders:
            target_spec = stft_enc(target)[0]
            estimate_spec = stft_enc(estimate)[0]
            l1_loss = self.compute_l1_loss(target_spec, estimate_spec)
            spectral_loss += l1_loss

        return spectral_loss / len(self.stft_encoders)
