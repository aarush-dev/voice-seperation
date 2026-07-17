"""SpecAugment-style masking along the time or frequency axis.

Randomly zeroes (or mean-fills) contiguous stripes along the time or
frequency axis of spectral features, as a data-augmentation step applied
during training of the separation/ASR frontend. Mirrors the "time masking"
/ "frequency masking" components of SpecAugment.
"""
import math
from typing import Optional, Sequence, Tuple, Union

import torch
from typeguard import check_argument_types


def _resolve_mask_dim(dim: Union[int, str]) -> Tuple[int, str]:
    """Normalize a ``dim`` argument (int or "time"/"freq") to (index, name)."""
    if isinstance(dim, str):
        if dim == "time":
            dim = 1
        elif dim == "freq":
            dim = 2
        else:
            raise ValueError("dim must be int, 'time' or 'freq'")
    if dim == 1:
        mask_axis = "time"
    elif dim == 2:
        mask_axis = "freq"
    else:
        mask_axis = "unknown"
    return dim, mask_axis


def _generate_mask(
    spec: torch.Tensor,
    mask_width_range: Sequence[int],
    dim: int,
    num_mask: int,
) -> torch.Tensor:
    """Sample random stripe masks along ``dim`` of a (Batch, ..., D) tensor.

    Args:
        spec: (Batch, Length, Freq), already flattened over any channel dim.
        mask_width_range: Stripe width is drawn uniformly from this range.
        dim: Axis (1 or 2) that the stripes run along.
        num_mask: Number of stripes to sample per batch element.

    Returns:
        mask: Broadcastable boolean mask, (Batch, Length, 1) if dim == 1,
            or (Batch, 1, Freq) if dim == 2.
    """
    B = spec.shape[0]
    # D = Length or Freq
    D = spec.shape[dim]
    # mask_length: (B, num_mask, 1)
    mask_length = torch.randint(
        mask_width_range[0],
        mask_width_range[1],
        (B, num_mask),
        device=spec.device,
    ).unsqueeze(2)

    # mask_pos: (B, num_mask, 1)
    mask_pos = torch.randint(
        0, max(1, D - mask_length.max()), (B, num_mask), device=spec.device
    ).unsqueeze(2)

    # aran: (1, 1, D)
    aran = torch.arange(D, device=spec.device)[None, None, :]
    # mask: (Batch, num_mask, D)
    mask = (mask_pos <= aran) * (aran < (mask_pos + mask_length))
    # Multiply masks: (Batch, num_mask, D) -> (Batch, D)
    mask = mask.any(dim=1)
    if dim == 1:
        # mask: (Batch, Length, 1)
        mask = mask.unsqueeze(2)
    elif dim == 2:
        # mask: (Batch, 1, Freq)
        mask = mask.unsqueeze(1)
    return mask


def mask_along_axis(
    spec: torch.Tensor,
    spec_lengths: torch.Tensor,
    mask_width_range: Sequence[int] = (0, 30),
    dim: int = 1,
    num_mask: int = 2,
    replace_with_zero: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply mask along the specified direction.

    Args:
        spec: (Batch, Length, Freq)
        spec_lengths: (Length): Not using lengths in this implementation
        mask_width_range: Select the width randomly between this range
    """

    org_size = spec.size()
    if spec.dim() == 4:
        # spec: (Batch, Channel, Length, Freq) -> (Batch * Channel, Length, Freq)
        spec = spec.view(-1, spec.size(2), spec.size(3))

    mask = _generate_mask(spec, mask_width_range, dim, num_mask)

    if replace_with_zero:
        value = 0.0
    else:
        value = spec.mean()

    if spec.requires_grad:
        spec = spec.masked_fill(mask, value)
    else:
        spec = spec.masked_fill_(mask, value)
    spec = spec.view(*org_size)
    return spec, spec_lengths


class MaskAlongAxis(torch.nn.Module):
    """Mask a spectral feature along the time or frequency axis.

    Draws ``num_mask`` stripes of width sampled uniformly from
    ``mask_width_range`` and fills them with zero (or the tensor mean).

    Attributes:
        mask_width_range: (min, max) stripe width.
        num_mask: Number of stripes applied per forward call.
        dim: Axis index (1=time, 2=freq) the stripes run along.
        replace_with_zero: If True, fill masked regions with 0, else with
            the mean of the input.
    """

    def __init__(
        self,
        mask_width_range: Union[int, Sequence[int]] = (0, 30),
        num_mask: int = 2,
        dim: Union[int, str] = "time",
        replace_with_zero: bool = True,
    ):
        assert check_argument_types()
        if isinstance(mask_width_range, int):
            mask_width_range = (0, mask_width_range)
        if len(mask_width_range) != 2:
            raise TypeError(
                f"mask_width_range must be a tuple of int and int values: "
                f"{mask_width_range}",
            )

        assert mask_width_range[1] > mask_width_range[0]
        dim, mask_axis = _resolve_mask_dim(dim)
        self.mask_axis = mask_axis

        super().__init__()
        self.mask_width_range = mask_width_range
        self.num_mask = num_mask
        self.dim = dim
        self.replace_with_zero = replace_with_zero

    def extra_repr(self) -> str:
        return (
            f"mask_width_range={self.mask_width_range}, "
            f"num_mask={self.num_mask}, axis={self.mask_axis}"
        )

    def forward(
        self, spec: torch.Tensor, spec_lengths: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward function.

        Args:
            spec: (Batch, Length, Freq)
        """

        return mask_along_axis(
            spec,
            spec_lengths,
            mask_width_range=self.mask_width_range,
            dim=self.dim,
            num_mask=self.num_mask,
            replace_with_zero=self.replace_with_zero,
        )


class MaskAlongAxisVariableMaxWidth(torch.nn.Module):
    """Mask input spec along a specified axis with variable maximum width.

    Formula:
        max_width = max_width_ratio * seq_len
    """

    def __init__(
        self,
        mask_width_ratio_range: Union[float, Sequence[float]] = (0.0, 0.05),
        num_mask: int = 2,
        dim: Union[int, str] = "time",
        replace_with_zero: bool = True,
    ):
        assert check_argument_types()
        if isinstance(mask_width_ratio_range, float):
            mask_width_ratio_range = (0.0, mask_width_ratio_range)
        if len(mask_width_ratio_range) != 2:
            raise TypeError(
                f"mask_width_ratio_range must be a tuple of float and float values: "
                f"{mask_width_ratio_range}",
            )

        assert mask_width_ratio_range[1] > mask_width_ratio_range[0]
        dim, mask_axis = _resolve_mask_dim(dim)
        self.mask_axis = mask_axis

        super().__init__()
        self.mask_width_ratio_range = mask_width_ratio_range
        self.num_mask = num_mask
        self.dim = dim
        self.replace_with_zero = replace_with_zero

    def extra_repr(self) -> str:
        return (
            f"mask_width_ratio_range={self.mask_width_ratio_range}, "
            f"num_mask={self.num_mask}, axis={self.mask_axis}"
        )

    def forward(
        self, spec: torch.Tensor, spec_lengths: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward function.

        Args:
            spec: (Batch, Length, Freq)
        """

        max_seq_len = spec.shape[self.dim]
        min_mask_width = math.floor(max_seq_len * self.mask_width_ratio_range[0])
        min_mask_width = max([0, min_mask_width])
        max_mask_width = math.floor(max_seq_len * self.mask_width_ratio_range[1])
        max_mask_width = min([max_seq_len, max_mask_width])

        if max_mask_width > min_mask_width:
            return mask_along_axis(
                spec,
                spec_lengths,
                mask_width_range=(min_mask_width, max_mask_width),
                dim=self.dim,
                num_mask=self.num_mask,
                replace_with_zero=self.replace_with_zero,
            )
        return spec, spec_lengths
