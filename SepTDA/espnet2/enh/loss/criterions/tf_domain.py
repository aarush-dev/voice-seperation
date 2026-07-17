"""Time-frequency-domain loss criterions for speech enhancement and separation.

These criterions operate either directly on complex STFT spectra, or on
real-valued time-frequency *masks* derived from those spectra (selected via
each criterion's ``compute_on_mask`` / ``mask_type``). :func:`_create_mask_label`
implements the supported family of oracle mask definitions (IBM, IRM, IAM,
PSM, NPSM, PSM^2, CIRM) used as training targets for mask-based losses.
"""
import math
from abc import ABC, abstractmethod
from functools import reduce
from typing import List, Optional

import torch
import torch.nn.functional as F

from espnet2.enh.layers.complex_utils import complex_norm, is_complex, new_complex_like
from espnet2.enh.loss.criterions.abs_loss import AbsEnhLoss

EPS = torch.finfo(torch.get_default_dtype()).eps

_VALID_MASK_TYPES = ("IBM", "IRM", "IAM", "PSM", "NPSM", "PSM^2", "CIRM")


def _expand_to_mix_shape(spec, mix_spec):
    """Broadcast a (B, T, F) spectrum to (B, T, C, F) to match ``mix_spec``.

    Single-channel reference/noise spectra are expanded (not just reshaped)
    across the channel dimension so that per-channel masks can be computed
    against a multi-channel mixture.
    """
    if spec.ndim < mix_spec.ndim:
        return spec.unsqueeze(2).expand_as(mix_spec.real)
    return spec


def _ibm_mask(ref_spec: List, noise_spec, target_idx: int):
    """Ideal Binary Mask: 1 where the target is the loudest source, else 0."""
    target = ref_spec[target_idx]
    competitors = ref_spec if noise_spec is None else ref_spec + [noise_spec]
    dominance_flags = [abs(target) >= abs(other) for other in competitors]
    mask = reduce(lambda x, y: x * y, dominance_flags)
    return mask.int()


def _irm_mask(ref_spec: List, noise_spec, target_idx: int):
    """Ideal Ratio Mask: target power over (target + interference) power."""
    beta = 0.5
    target = ref_spec[target_idx]
    interference = sum(
        spec for i, spec in enumerate(ref_spec) if i != target_idx
    )
    if noise_spec is not None:
        interference = interference + noise_spec
    return (abs(target).pow(2) / (abs(interference).pow(2) + EPS)).pow(beta)


def _iam_mask(target, mix_spec):
    """Ideal Amplitude Mask: |target| / |mixture|, clamped to [0, 1]."""
    mask = abs(target) / (abs(mix_spec) + EPS)
    return mask.clamp(min=0, max=1)


def _cos_phase_similarity(target, mix_spec):
    """cos(angle(target) - angle(mixture)) via the dot product of unit phasors."""
    phase_target = target / (abs(target) + EPS)
    phase_mix = mix_spec / (abs(mix_spec) + EPS)
    # cos(a - b) = cos(a)*cos(b) + sin(a)*sin(b)
    return phase_target.real * phase_mix.real + phase_target.imag * phase_mix.imag


def _psm_mask(target, mix_spec, non_negative: bool):
    """(Non-negative) Phase-Sensitive Mask: amplitude ratio scaled by phase agreement."""
    cos_theta = _cos_phase_similarity(target, mix_spec)
    mask = (abs(target) / (abs(mix_spec) + EPS)) * cos_theta
    return mask.clamp(min=0, max=1) if non_negative else mask.clamp(min=-1, max=1)


def _psm_power_mask(target, mix_spec):
    """Squared-amplitude variant of the phase-sensitive mask, used for beamforming."""
    cos_theta = _cos_phase_similarity(target, mix_spec)
    mask = (abs(target).pow(2) / (abs(mix_spec).pow(2) + EPS)) * cos_theta
    return mask.clamp(min=-1, max=1)


def _cirm_mask(target, mix_spec):
    """Complex Ideal Ratio Mask.

    Reference: Complex Ratio Masking for Monaural Speech Separation.
    """
    denominator = mix_spec.real.pow(2) + mix_spec.imag.pow(2) + EPS
    mask_real = (mix_spec.real * target.real + mix_spec.imag * target.imag) / (
        denominator
    )
    mask_imag = (mix_spec.real * target.imag - mix_spec.imag * target.real) / (
        denominator
    )
    return new_complex_like(mix_spec, [mask_real, mask_imag])


def _create_mask_label(mix_spec, ref_spec, noise_spec=None, mask_type="IAM"):
    """Create oracle mask training targets for each reference source.

    Args:
        mix_spec: ComplexTensor(B, T, [C,] F)
        ref_spec: List[ComplexTensor(B, T, [C,] F), ...]
        noise_spec: ComplexTensor(B, T, [C,] F)
            only used for IBM and IRM
        mask_type: str
    Returns:
        labels: List[Tensor(B, T, [C,] F), ...] or List[ComplexTensor(B, T, F), ...]
    """

    mask_type = mask_type.upper()
    assert mask_type in _VALID_MASK_TYPES, f"mask type {mask_type} not supported"

    ref_spec = [_expand_to_mix_shape(r, mix_spec) for r in ref_spec]
    if noise_spec is not None:
        noise_spec = _expand_to_mix_shape(noise_spec, mix_spec)

    mask_label = []
    for idx, target in enumerate(ref_spec):
        if mask_type == "IBM":
            mask = _ibm_mask(ref_spec, noise_spec, idx)
        elif mask_type == "IRM":
            mask = _irm_mask(ref_spec, noise_spec, idx)
        elif mask_type == "IAM":
            mask = _iam_mask(target, mix_spec)
        elif mask_type in ("PSM", "NPSM"):
            mask = _psm_mask(target, mix_spec, non_negative=mask_type == "NPSM")
        elif mask_type == "PSM^2":
            # This is for training beamforming masks
            mask = _psm_power_mask(target, mix_spec)
        elif mask_type == "CIRM":
            mask = _cirm_mask(target, mix_spec)
        else:  # pragma: no cover - guarded by the assertion above
            raise ValueError(f"mask type {mask_type} not supported")
        mask_label.append(mask)
    return mask_label


class FrequencyDomainLoss(AbsEnhLoss, ABC):
    """Base class for all frequence-domain Enhancement loss modules."""

    # The loss will be computed on mask or on spectrum
    @property
    @abstractmethod
    def compute_on_mask() -> bool:
        pass

    # the mask type
    @property
    @abstractmethod
    def mask_type() -> str:
        pass

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
        self._name = name
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

    def create_mask_label(self, mix_spec, ref_spec, noise_spec=None):
        """Build the oracle mask targets for this criterion's ``mask_type``.

        Thin wrapper around the module-level :func:`_create_mask_label` that
        fills in ``self.mask_type``. See that function for the definitions
        of the supported mask types.
        """
        return _create_mask_label(
            mix_spec=mix_spec,
            ref_spec=ref_spec,
            noise_spec=noise_spec,
            mask_type=self.mask_type,
        )


class FrequencyDomainMSE(FrequencyDomainLoss):
    """Mean squared error, computed either on masks or on raw spectra."""

    def __init__(
        self,
        compute_on_mask: bool = False,
        mask_type: str = "IBM",
        name: Optional[str] = None,
        only_for_test: bool = False,
        is_noise_loss: bool = False,
        is_dereverb_loss: bool = False,
    ):
        if name is not None:
            _name = name
        elif compute_on_mask:
            _name = f"MSE_on_{mask_type}"
        else:
            _name = "MSE_on_Spec"
        super().__init__(
            _name,
            only_for_test=only_for_test,
            is_noise_loss=is_noise_loss,
            is_dereverb_loss=is_dereverb_loss,
        )

        self._compute_on_mask = compute_on_mask
        self._mask_type = mask_type

    @property
    def compute_on_mask(self) -> bool:
        return self._compute_on_mask

    @property
    def mask_type(self) -> str:
        return self._mask_type

    def forward(self, ref, inf) -> torch.Tensor:
        """time-frequency MSE loss.

        Args:
            ref: (Batch, T, F) or (Batch, T, C, F)
            inf: (Batch, T, F) or (Batch, T, C, F)
        Returns:
            loss: (Batch,)
        """
        assert ref.shape == inf.shape, (ref.shape, inf.shape)

        diff = ref - inf
        if is_complex(diff):
            mseloss = diff.real**2 + diff.imag**2
        else:
            mseloss = diff**2
        if ref.dim() == 3:
            mseloss = mseloss.mean(dim=[1, 2])
        elif ref.dim() == 4:
            mseloss = mseloss.mean(dim=[1, 2, 3])
        else:
            raise ValueError(
                "Invalid input shape: ref={}, inf={}".format(ref.shape, inf.shape)
            )
        return mseloss


class FrequencyDomainL1(FrequencyDomainLoss):
    """Mean absolute error (L1), computed either on masks or on raw spectra."""

    def __init__(
        self,
        compute_on_mask: bool = False,
        mask_type: str = "IBM",
        name: Optional[str] = None,
        only_for_test: bool = False,
        is_noise_loss: bool = False,
        is_dereverb_loss: bool = False,
    ):
        if name is not None:
            _name = name
        elif compute_on_mask:
            _name = f"L1_on_{mask_type}"
        else:
            _name = "L1_on_Spec"
        super().__init__(
            _name,
            only_for_test=only_for_test,
            is_noise_loss=is_noise_loss,
            is_dereverb_loss=is_dereverb_loss,
        )

        self._compute_on_mask = compute_on_mask
        self._mask_type = mask_type

    @property
    def compute_on_mask(self) -> bool:
        return self._compute_on_mask

    @property
    def mask_type(self) -> str:
        return self._mask_type

    def forward(self, ref, inf) -> torch.Tensor:
        """time-frequency L1 loss.

        Args:
            ref: (Batch, T, F) or (Batch, T, C, F)
            inf: (Batch, T, F) or (Batch, T, C, F)
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
            l1loss = l1loss.mean(dim=[1, 2])
        elif ref.dim() == 4:
            l1loss = l1loss.mean(dim=[1, 2, 3])
        else:
            raise ValueError(
                "Invalid input shape: ref={}, inf={}".format(ref.shape, inf.shape)
            )
        return l1loss


def _dpcl_onehot_targets(abs_ref: List[torch.Tensor], num_spk: int) -> torch.Tensor:
    """Build the one-hot Deep Clustering affinity target.

    For every T-F bin, the "dominant" speaker (largest magnitude) is
    assigned label ``i``; the result is one-hot encoded so that the ideal
    embedding for a bin lies entirely along the dominant speaker's axis.

    Returns:
        (B, T*F, num_spk) one-hot target embedding.
    """
    batch_size = abs_ref[0].shape[0]
    dominant_spk = torch.zeros_like(abs_ref[0])
    for i in range(num_spk):
        is_dominant = [abs_ref[i] >= other for other in abs_ref]
        dominant_flag = reduce(lambda x, y: x * y, is_dominant)
        dominant_spk += dominant_flag.int() * i
    dominant_spk = dominant_spk.contiguous().flatten().long()
    target = F.one_hot(dominant_spk, num_classes=num_spk)
    return target.contiguous().view(batch_size, -1, num_spk)


def _mdc_manifold_targets(
    abs_ref: List[torch.Tensor], num_spk: int, dtype: torch.dtype, device
) -> torch.Tensor:
    """Build the Manifold-aware Deep Clustering (MDC) regular-simplex target.

    Instead of a one-hot vector, each speaker is assigned a vertex of a
    regular simplex in ``num_spk``-dimensional space, chosen so that the
    angles between any two speaker vectors are maximized. Every T-F bin is
    then labeled with the simplex vertex of its dominant speaker.

    Returns:
        (B, T*F, num_spk) simplex-vertex target embedding.
    """
    off_axis_value = (-1 / num_spk) * math.sqrt(num_spk / (num_spk - 1))
    on_axis_value = ((num_spk - 1) / num_spk) * math.sqrt(num_spk / (num_spk - 1))
    manifold_vector = torch.full(
        (num_spk, num_spk), off_axis_value, dtype=dtype, device=device
    )
    for i in range(num_spk):
        manifold_vector[i][i] = on_axis_value

    batch_size, num_frames, num_freqs = abs_ref[0].shape
    target = torch.zeros(batch_size, num_frames, num_freqs, num_spk, device=device)
    for i in range(num_spk):
        is_dominant = [abs_ref[i] >= other for other in abs_ref]
        dominant_flag = reduce(lambda x, y: x * y, is_dominant).int()
        target[dominant_flag == 1] = manifold_vector[i]
    return target.contiguous().view(batch_size, -1, num_spk)


class FrequencyDomainDPCL(FrequencyDomainLoss):
    """Deep Clustering affinity loss for T-F-bin speaker embeddings."""

    def __init__(
        self,
        compute_on_mask: bool = False,
        mask_type: str = "IBM",
        loss_type: str = "dpcl",
        name: Optional[str] = None,
        only_for_test: bool = False,
        is_noise_loss: bool = False,
        is_dereverb_loss: bool = False,
    ):
        _name = "dpcl" if name is None else name
        super().__init__(
            _name,
            only_for_test=only_for_test,
            is_noise_loss=is_noise_loss,
            is_dereverb_loss=is_dereverb_loss,
        )
        self._compute_on_mask = compute_on_mask
        self._mask_type = mask_type
        self._loss_type = loss_type

    @property
    def compute_on_mask(self) -> bool:
        return self._compute_on_mask

    @property
    def mask_type(self) -> str:
        return self._mask_type

    def forward(self, ref, inf) -> torch.Tensor:
        """time-frequency Deep Clustering loss.

        Compares the Gram matrix of the network's learned T-F embeddings
        ``inf`` against the Gram matrix of an oracle affinity target built
        from ``ref`` (either a one-hot "dominant speaker" target for
        ``loss_type="dpcl"``, or a regular-simplex target for
        ``loss_type="mdc"``). Minimizing ``||V V^T - Y Y^T||_F^2`` (expanded
        below into ``V2 + Y2 - 2*VY``) encourages embeddings of T-F bins
        dominated by the same speaker to be close together, and embeddings
        of bins dominated by different speakers to be far apart -- without
        ever needing to know which output index corresponds to which
        speaker (hence being permutation-free by construction).

        References:
            [1] Deep clustering: Discriminative embeddings for segmentation and
                separation; John R. Hershey. et al., 2016;
                https://ieeexplore.ieee.org/document/7471631
            [2] Manifold-Aware Deep Clustering: Maximizing Angles Between Embedding
                Vectors Based on Regular Simplex; Tanaka, K. et al., 2021;
                https://www.isca-speech.org/archive/interspeech_2021/tanaka21_interspeech.html

        Args:
            ref: List[(Batch, T, F) * spks]
            inf: (Batch, T*F, D)
        Returns:
            loss: (Batch,)
        """  # noqa: E501
        assert len(ref) > 0
        num_spk = len(ref)

        abs_ref = [abs(n) for n in ref]
        if self._loss_type == "dpcl":
            target = _dpcl_onehot_targets(abs_ref, num_spk)
        elif self._loss_type == "mdc":
            target = _mdc_manifold_targets(
                abs_ref, num_spk, dtype=inf.dtype, device=inf.device
            )
        else:
            raise ValueError(
                f"Invalid loss type error: {self._loss_type}, "
                'the loss type must be "dpcl" or "mdc"'
            )

        # ||V V^T - Y Y^T||_F^2 = V2 + Y2 - 2*VY, computed without materializing
        # the full (T*F, T*F) Gram matrices.
        v2 = torch.matmul(torch.transpose(inf, 2, 1), inf).pow(2).sum(dim=(1, 2))
        y2 = (
            torch.matmul(torch.transpose(target, 2, 1).float(), target.float())
            .pow(2)
            .sum(dim=(1, 2))
        )
        vy = (
            torch.matmul(torch.transpose(inf, 2, 1), target.float())
            .pow(2)
            .sum(dim=(1, 2))
        )

        return v2 + y2 - 2 * vy


class FrequencyDomainAbsCoherence(FrequencyDomainLoss):
    """Magnitude (absolute) coherence loss between two complex spectra."""

    def __init__(
        self,
        compute_on_mask: bool = False,
        mask_type: Optional[str] = None,
        name: Optional[str] = None,
        only_for_test: bool = False,
        is_noise_loss: bool = False,
        is_dereverb_loss: bool = False,
    ):
        _name = "Coherence_on_Spec" if name is None else name
        super().__init__(
            _name,
            only_for_test=only_for_test,
            is_noise_loss=is_noise_loss,
            is_dereverb_loss=is_dereverb_loss,
        )

        self._compute_on_mask = False
        self._mask_type = None

    @property
    def compute_on_mask(self) -> bool:
        return self._compute_on_mask

    @property
    def mask_type(self) -> str:
        return self._mask_type

    def forward(self, ref, inf) -> torch.Tensor:
        """time-frequency absolute coherence loss.

        Reference:
            Independent Vector Analysis with Deep Neural Network Source Priors;
            Li et al 2020; https://arxiv.org/abs/2008.11273

        Args:
            ref: (Batch, T, F) or (Batch, T, C, F)
            inf: (Batch, T, F) or (Batch, T, C, F)
        Returns:
            loss: (Batch,)
        """
        assert ref.shape == inf.shape, (ref.shape, inf.shape)

        if is_complex(ref) and is_complex(inf):
            # sqrt( E[|inf|^2] * E[|ref|^2] )
            denom = (
                complex_norm(ref, dim=1) * complex_norm(inf, dim=1) / ref.size(1) + EPS
            )
            coh = (inf * ref.conj()).mean(dim=1).abs() / denom
            if ref.dim() == 3:
                coh_loss = 1.0 - coh.mean(dim=1)
            elif ref.dim() == 4:
                coh_loss = 1.0 - coh.mean(dim=[1, 2])
            else:
                raise ValueError(
                    "Invalid input shape: ref={}, inf={}".format(ref.shape, inf.shape)
                )
        else:
            raise ValueError("`ref` and `inf` must be complex tensors.")
        return coh_loss


class FrequencyDomainCrossEntropy(FrequencyDomainLoss):
    """Cross-entropy loss for classification-style (e.g. discretized) targets."""

    def __init__(
        self,
        compute_on_mask: bool = False,
        mask_type: Optional[str] = None,
        ignore_id: int = -100,
        name: Optional[str] = None,
        only_for_test: bool = False,
        is_noise_loss: bool = False,
        is_dereverb_loss: bool = False,
    ):
        if name is not None:
            _name = name
        elif compute_on_mask:
            _name = f"CE_on_{mask_type}"
        else:
            _name = "CE_on_Spec"
        super().__init__(
            _name,
            only_for_test=only_for_test,
            is_noise_loss=is_noise_loss,
            is_dereverb_loss=is_dereverb_loss,
        )

        self._compute_on_mask = compute_on_mask
        self._mask_type = mask_type
        self.cross_entropy = torch.nn.CrossEntropyLoss(
            ignore_index=ignore_id, reduction="none"
        )
        self.ignore_id = ignore_id

    @property
    def compute_on_mask(self) -> bool:
        return self._compute_on_mask

    @property
    def mask_type(self) -> str:
        return self._mask_type

    def forward(self, ref, inf) -> torch.Tensor:
        """time-frequency cross-entropy loss.

        Args:
            ref: (Batch, T) or (Batch, T, C)
            inf: (Batch, T, nclass) or (Batch, T, C, nclass)
        Returns:
            loss: (Batch,)
        """
        assert ref.shape[0] == inf.shape[0] and ref.shape[1] == inf.shape[1], (
            ref.shape,
            inf.shape,
        )

        if ref.dim() == 2:
            loss = self.cross_entropy(inf.permute(0, 2, 1), ref).mean(dim=1)
        elif ref.dim() == 3:
            loss = self.cross_entropy(inf.permute(0, 3, 1, 2), ref).mean(dim=[1, 2])
        else:
            raise ValueError(
                "Invalid input shape: ref={}, inf={}".format(ref.shape, inf.shape)
            )

        with torch.no_grad():
            pred = inf.argmax(-1)
            valid_mask = ref != self.ignore_id
            correct = (pred == ref).masked_fill(~valid_mask, 0).float()
            if ref.dim() == 2:
                acc = correct.sum(dim=1) / valid_mask.sum(dim=1).float()
            elif ref.dim() == 3:
                acc = correct.sum(dim=[1, 2]) / valid_mask.sum(dim=[1, 2]).float()
            self.stats = {"acc": acc.cpu() * 100}

        return loss
