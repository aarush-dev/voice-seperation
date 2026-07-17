# Implementation of the TCN proposed in
# Luo. et al.  "Conv-tasnet: Surpassing ideal time-frequency
# magnitude masking for speech separation."
#
# The code is based on:
# https://github.com/kaituoxu/Conv-TasNet/blob/master/src/conv_tasnet.py
#
"""Temporal Convolutional Network (TCN) encoder, without the final mask head.

This produces the bottleneck feature that a separate mask module (see
:mod:`espnet2.diar.layers.multi_mask`) turns into per-speaker masks; unlike
the original Conv-TasNet, the final 1x1 mask conv is not included here.

Shape legend used throughout this module: M = batch size, N = number of
autoencoder filters, B = bottleneck channels, H = conv-block channels,
K = sequence length.
"""

import torch
import torch.nn as nn

EPS = torch.finfo(torch.get_default_dtype()).eps


class TemporalConvNet(nn.Module):
    """Stack of dilated residual conv blocks, repeated ``R`` times.

    Applies channel-wise layer norm, a bottleneck 1x1 conv, then ``R``
    repeats of ``X`` :class:`TemporalBlock` s with exponentially increasing
    dilation (2**0, 2**1, ..., 2**(X-1)) within each repeat.
    """

    def __init__(
        self,
        N: int,
        B: int,
        H: int,
        P: int,
        X: int,
        R: int,
        norm_type: str = "gLN",
        causal: bool = False,
    ):
        """Basic Module of tasnet.

        Args:
            N: Number of filters in autoencoder.
            B: Number of channels in bottleneck 1 * 1-conv block.
            H: Number of channels in convolutional blocks.
            P: Kernel size in convolutional blocks.
            X: Number of convolutional blocks in each repeat.
            R: Number of repeats.
            norm_type: BN, gLN, cLN.
            causal: causal or non-causal.
        """
        super().__init__()
        # Components
        # [M, N, K] -> [M, N, K]
        layer_norm = ChannelwiseLayerNorm(N)
        # [M, N, K] -> [M, B, K]
        bottleneck_conv1x1 = nn.Conv1d(N, B, 1, bias=False)
        # [M, B, K] -> [M, B, K]
        repeats = []
        for _ in range(R):
            blocks = []
            for block_idx in range(X):
                dilation = 2**block_idx
                padding = (
                    (P - 1) * dilation if causal else (P - 1) * dilation // 2
                )
                blocks += [
                    TemporalBlock(
                        B,
                        H,
                        P,
                        stride=1,
                        padding=padding,
                        dilation=dilation,
                        norm_type=norm_type,
                        causal=causal,
                    )
                ]
            repeats += [nn.Sequential(*blocks)]
        temporal_conv_net = nn.Sequential(*repeats)
        # Put together (except mask_conv1x1, modified from the original code)
        self.network = nn.Sequential(layer_norm, bottleneck_conv1x1, temporal_conv_net)

    def forward(self, mixture_w: torch.Tensor) -> torch.Tensor:
        """Keep this API same with TasNet.

        Args:
            mixture_w: encoder output, shape [M, N, K]. M is batch size.

        Returns:
            bottleneck_feature: shape [M, B, K].
        """
        return self.network(mixture_w)  # [M, N, K] -> [M, B, K]


class TemporalBlock(nn.Module):
    """1x1 conv -> PReLU -> norm -> depthwise-separable conv, with a residual add."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        padding: int,
        dilation: int,
        norm_type: str = "gLN",
        causal: bool = False,
    ):
        super().__init__()
        # [M, B, K] -> [M, H, K]
        conv1x1 = nn.Conv1d(in_channels, out_channels, 1, bias=False)
        prelu = nn.PReLU()
        norm = chose_norm(norm_type, out_channels)
        # [M, H, K] -> [M, B, K]
        dsconv = DepthwiseSeparableConv(
            out_channels,
            in_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            norm_type,
            causal,
        )
        # Put together
        self.net = nn.Sequential(conv1x1, prelu, norm, dsconv)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward.

        Args:
            x: shape [M, B, K].

        Returns:
            shape [M, B, K].
        """
        residual = x
        out = self.net(x)
        # TODO(Jing): when P = 3 here works fine, but when P = 2 maybe need to pad?
        return out + residual  # look like w/o F.relu is better than w/ F.relu


class DepthwiseSeparableConv(nn.Module):
    """Depthwise conv (per-channel) followed by a pointwise 1x1 conv."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        padding: int,
        dilation: int,
        norm_type: str = "gLN",
        causal: bool = False,
    ):
        super().__init__()
        # Use `groups` option to implement depthwise convolution
        # [M, H, K] -> [M, H, K]
        depthwise_conv = nn.Conv1d(
            in_channels,
            in_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=in_channels,
            bias=False,
        )
        if causal:
            chomp = Chomp1d(padding)
        prelu = nn.PReLU()
        norm = chose_norm(norm_type, in_channels)
        # [M, H, K] -> [M, B, K]
        pointwise_conv = nn.Conv1d(in_channels, out_channels, 1, bias=False)
        # Put together
        if causal:
            self.net = nn.Sequential(depthwise_conv, chomp, prelu, norm, pointwise_conv)
        else:
            self.net = nn.Sequential(depthwise_conv, prelu, norm, pointwise_conv)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward.

        Args:
            x: shape [M, H, K].

        Returns:
            result: shape [M, B, K].
        """
        return self.net(x)


class Chomp1d(nn.Module):
    """Trims trailing padding so causal-conv output length matches the input."""

    def __init__(self, chomp_size: int):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward.

        Args:
            x: shape [M, H, Kpad].

        Returns:
            shape [M, H, K].
        """
        return x[:, :, : -self.chomp_size].contiguous()


def chose_norm(norm_type: str, channel_size: int) -> nn.Module:
    """The input of normalization will be (M, C, K), where M is batch size.

    C is channel size and K is sequence length.
    """
    if norm_type == "gLN":
        return GlobalLayerNorm(channel_size)
    elif norm_type == "cLN":
        return ChannelwiseLayerNorm(channel_size)
    elif norm_type == "BN":
        # Given input (M, C, K), nn.BatchNorm1d(C) will accumulate statics
        # along M and K, so this BN usage is right.
        return nn.BatchNorm1d(channel_size)
    else:
        raise ValueError("Unsupported normalization type")


class ChannelwiseLayerNorm(nn.Module):
    """Channel-wise Layer Normalization (cLN): normalizes over the channel dim."""

    def __init__(self, channel_size: int):
        super().__init__()
        self.gamma = nn.Parameter(torch.Tensor(1, channel_size, 1))  # [1, N, 1]
        self.beta = nn.Parameter(torch.Tensor(1, channel_size, 1))  # [1, N, 1]
        self.reset_parameters()

    def reset_parameters(self):
        self.gamma.data.fill_(1)
        self.beta.data.zero_()

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        """Forward.

        Args:
            y: shape [M, N, K]. M is batch size, N is channel size, K is length.

        Returns:
            cLN_y: shape [M, N, K].
        """
        mean = torch.mean(y, dim=1, keepdim=True)  # [M, 1, K]
        var = torch.var(y, dim=1, keepdim=True, unbiased=False)  # [M, 1, K]
        cLN_y = self.gamma * (y - mean) / torch.pow(var + EPS, 0.5) + self.beta
        return cLN_y


class GlobalLayerNorm(nn.Module):
    """Global Layer Normalization (gLN): normalizes over both channel and time."""

    def __init__(self, channel_size: int):
        super().__init__()
        self.gamma = nn.Parameter(torch.Tensor(1, channel_size, 1))  # [1, N, 1]
        self.beta = nn.Parameter(torch.Tensor(1, channel_size, 1))  # [1, N, 1]
        self.reset_parameters()

    def reset_parameters(self):
        self.gamma.data.fill_(1)
        self.beta.data.zero_()

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        """Forward.

        Args:
            y: shape [M, N, K]. M is batch size, N is channel size, K is length.

        Returns:
            gLN_y: shape [M, N, K].
        """
        mean = y.mean(dim=1, keepdim=True).mean(dim=2, keepdim=True)  # [M, 1, 1]
        var = (
            (torch.pow(y - mean, 2)).mean(dim=1, keepdim=True).mean(dim=2, keepdim=True)
        )
        gLN_y = self.gamma * (y - mean) / torch.pow(var + EPS, 0.5) + self.beta
        return gLN_y
