"""Band-splitting front-end/back-end used by band-split RNN-style separators.

Splits a full-band spectral feature into a set of (possibly unequal-width)
sub-bands, projects each sub-band into a shared embedding dimension
(:class:`BandSplitEncoder`), and later maps the per-band embeddings back to
sub-band spectral features (:class:`BandSplitDecoder`).
"""

import torch
from beartype import beartype
from beartype.typing import Optional, Tuple
from torch import Tensor, nn
from torch.nn import Module, ModuleList
import torch.nn.functional as F


def exists(val) -> bool:
    return val is not None


def default(v, d):
    return v if exists(v) else d


class RMSNorm(Module):
    """Root-mean-square layer normalization (no mean-centering, no bias)."""

    def __init__(self, dim: int):
        super().__init__()
        self.scale = dim**0.5
        self.gamma = nn.Parameter(torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        return F.normalize(x, dim=-1) * self.scale * self.gamma


class BandSplitEncoder(Module):
    """Splits the frequency axis into bands and projects each to ``dim``.

    Args:
        dim: shared embedding dimension for every band.
        dim_inputs: width (number of frequency bins) of each band; the
            bands are concatenated to cover the full input frequency axis.
    """

    @beartype
    def __init__(self, dim: int, dim_inputs: Tuple[int, ...]):
        super().__init__()
        self.dim_inputs = dim_inputs
        self.to_features = ModuleList([])

        for dim_in in dim_inputs:
            net = nn.Sequential(RMSNorm(dim_in), nn.Linear(dim_in, dim))
            self.to_features.append(net)

    def forward(self, x: Tensor) -> Tensor:
        """(..., sum(dim_inputs)) -> (..., num_bands, dim)."""
        band_inputs = x.split(self.dim_inputs, dim=-1)

        band_embeddings = []
        for split_input, to_feature in zip(band_inputs, self.to_features):
            band_embeddings.append(to_feature(split_input))

        return torch.stack(band_embeddings, dim=-2)


def MLP(
    dim_in: int,
    dim_out: int,
    dim_hidden: Optional[int] = None,
    depth: int = 1,
    activation: type = nn.Tanh,
) -> nn.Sequential:
    """Build a plain feed-forward MLP with ``depth`` hidden layers."""
    dim_hidden = default(dim_hidden, dim_in)

    layers = []
    dims = (dim_in, *((dim_hidden,) * depth), dim_out)

    for ind, (layer_dim_in, layer_dim_out) in enumerate(zip(dims[:-1], dims[1:])):
        is_last = ind == (len(dims) - 2)

        layers.append(nn.Linear(layer_dim_in, layer_dim_out))

        if is_last:
            continue

        layers.append(activation())

    return nn.Sequential(*layers)


class BandSplitDecoder(Module):
    """Maps per-band embeddings back to sub-band spectral features.

    Each band has its own MLP (ending in a GLU) that projects the shared
    embedding dimension ``dim`` back to that band's original width.

    Args:
        dim: shared embedding dimension for every band (matches
            :class:`BandSplitEncoder`'s ``dim``).
        dim_inputs: width (number of frequency bins) of each band.
        depth: number of hidden layers in each band's MLP.
        mlp_expansion_factor: hidden-layer width of each MLP, as a multiple
            of ``dim``.
    """

    @beartype
    def __init__(
        self,
        dim: int,
        dim_inputs: Tuple[int, ...],
        depth: int,
        mlp_expansion_factor: int = 4,
    ):
        super().__init__()
        self.dim_inputs = dim_inputs
        self.to_freqs = ModuleList([])
        dim_hidden = dim * mlp_expansion_factor

        for dim_in in dim_inputs:
            mlp = nn.Sequential(
                MLP(dim, dim_in * 2, dim_hidden=dim_hidden, depth=depth),
                nn.GLU(dim=-1),
            )
            self.to_freqs.append(mlp)

    def forward(self, x: Tensor) -> Tensor:
        """(..., num_bands, dim) -> (..., sum(dim_inputs))."""
        band_embeddings = x.unbind(dim=-2)
        band_outputs = []

        for band_features, mlp in zip(band_embeddings, self.to_freqs):
            band_outputs.append(mlp(band_features))
        return torch.cat(band_outputs, dim=-1)
