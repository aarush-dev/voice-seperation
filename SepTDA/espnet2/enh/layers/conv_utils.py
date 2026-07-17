# noqa: E501 ported from https://discuss.pytorch.org/t/utility-function-for-calculating-the-shape-of-a-conv-output/11173/7
"""Helpers for computing the output spatial shape of 2-D (transposed) convolutions.

Used by the complex-valued conv layers in ``complexnn.py`` to work out padding
and output sizes without running the convolution.
"""
import math
from typing import Tuple, Union


def num2tuple(num: Union[int, Tuple[int, int]]) -> Tuple[int, int]:
    """Broadcast a scalar to a ``(num, num)`` pair; pass tuples through unchanged."""
    return num if isinstance(num, tuple) else (num, num)


def conv2d_output_shape(h_w, kernel_size=1, stride=1, pad=0, dilation=1):
    """Compute the (height, width) output shape of ``nn.Conv2d``.

    Mirrors the shape formula from the PyTorch ``Conv2d`` docs. Each argument
    may be a scalar (applied to both dimensions) or a ``(height, width)``
    tuple; `pad` may additionally be per-side, i.e. a tuple of tuples.
    """
    h_w, kernel_size, stride, pad, dilation = (
        num2tuple(h_w),
        num2tuple(kernel_size),
        num2tuple(stride),
        num2tuple(pad),
        num2tuple(dilation),
    )
    pad = num2tuple(pad[0]), num2tuple(pad[1])

    h = math.floor(
        (h_w[0] + sum(pad[0]) - dilation[0] * (kernel_size[0] - 1) - 1) / stride[0] + 1
    )
    w = math.floor(
        (h_w[1] + sum(pad[1]) - dilation[1] * (kernel_size[1] - 1) - 1) / stride[1] + 1
    )

    return h, w


def convtransp2d_output_shape(
    h_w, kernel_size=1, stride=1, pad=0, dilation=1, out_pad=0
):
    """Compute the (height, width) output shape of ``nn.ConvTranspose2d``.

    Mirrors the shape formula from the PyTorch ``ConvTranspose2d`` docs. Each
    argument may be a scalar (applied to both dimensions) or a
    ``(height, width)`` tuple; `pad` may additionally be per-side, i.e. a
    tuple of tuples.
    """
    h_w, kernel_size, stride, pad, dilation, out_pad = (
        num2tuple(h_w),
        num2tuple(kernel_size),
        num2tuple(stride),
        num2tuple(pad),
        num2tuple(dilation),
        num2tuple(out_pad),
    )
    pad = num2tuple(pad[0]), num2tuple(pad[1])

    h = (
        (h_w[0] - 1) * stride[0]
        - sum(pad[0])
        + dilation[0] * (kernel_size[0] - 1)
        + out_pad[0]
        + 1
    )
    w = (
        (h_w[1] - 1) * stride[1]
        - sum(pad[1])
        + dilation[1] * (kernel_size[1] - 1)
        + out_pad[1]
        + 1
    )

    return h, w
