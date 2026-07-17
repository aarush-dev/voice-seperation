# noqa E501: Ported from https://github.com/BUTSpeechFIT/speakerbeam/blob/main/src/models/adapt_layers.py
# Copyright (c) 2021 Brno University of Technology
# Copyright (c) 2021 Nippon Telegraph and Telephone corporation (NTT).
# All rights reserved
# By Katerina Zmolikova, August 2021.

"""Speaker adaptation layers for target-speaker extraction (SpeakerBeam-style).

Each layer fuses activations of the main separation network with an
enrollment (speaker) embedding, either by concatenation, multiplication/
addition, or attention, so that the network's output is conditioned on the
target speaker.
"""

from functools import partial
from typing import Any, Dict, List, Tuple, Union

import torch
import torch.nn as nn

TensorOrTensors = Union[torch.Tensor, Tuple[torch.Tensor, ...], List[torch.Tensor]]


def make_adapt_layer(
    type: str,
    indim: int,
    enrolldim: int,
    ninputs: int = 1,
    adapt_layer_kwargs: Dict[str, Any] = {},
) -> nn.Module:
    """Instantiate an adaptation layer by name.

    Args:
        type: key into ``adaptation_layer_types`` (e.g. "concat", "muladd",
            "mul", "attn").
        indim: hidden dimension of the main-network activations.
        enrolldim: hidden dimension of the enrollment embedding.
        ninputs: number of tensors to adapt at once (e.g. 2 when adapting
            both a residual and a skip-connection branch together).
        adapt_layer_kwargs: extra keyword arguments forwarded to the
            adaptation layer's constructor.
    """
    adapt_class = adaptation_layer_types.get(type)
    return adapt_class(indim, enrolldim, ninputs, **adapt_layer_kwargs)


def into_tuple(x: TensorOrTensors) -> Tuple[torch.Tensor, ...]:
    """Transforms tensor/list/tuple into tuple."""
    if isinstance(x, list):
        return tuple(x)
    elif isinstance(x, torch.Tensor):
        return (x,)
    elif isinstance(x, tuple):
        return x
    else:
        raise ValueError("x should be tensor, list of tuple")


def into_orig_type(x: Tuple[torch.Tensor, ...], orig_type: type) -> TensorOrTensors:
    """Inverts into_tuple function."""
    if orig_type is tuple:
        return x
    if orig_type is list:
        return list(x)
    if orig_type is torch.Tensor:
        return x[0]
    else:
        assert False


class ConcatAdaptLayer(nn.Module):
    """Adapt by concatenating the enrollment embedding along the channel axis.

    The enrollment embedding is broadcast over time and concatenated to the
    main activations, then projected back down to ``indim`` with a linear
    layer.
    """

    def __init__(self, indim: int, enrolldim: int, ninputs: int = 1):
        super().__init__()
        self.ninputs = ninputs
        self.transform = nn.ModuleList(
            [nn.Linear(indim + enrolldim, indim) for _ in range(ninputs)]
        )

    def forward(
        self, main: TensorOrTensors, enroll: TensorOrTensors
    ) -> TensorOrTensors:
        """ConcatAdaptLayer forward.

        Args:
            main: tensor or tuple or list
                  activations in the main neural network, which are adapted
                  tuple/list may be useful when we want to apply the adaptation
                    to both normal and skip connection at once
                  each tensor has shape (B, indim, T)
            enroll: tensor or tuple or list
                    embedding extracted from enrollment
                    tuple/list may be useful when we want to apply the adaptation
                      to both normal and skip connection at once
                    each tensor has shape (B, enrolldim)
        """
        assert type(main) == type(enroll)
        orig_type = type(main)
        main, enroll = into_tuple(main), into_tuple(enroll)
        assert len(main) == len(enroll) == self.ninputs

        out = []
        for transform, main0, enroll0 in zip(self.transform, main, enroll):
            # broadcast enroll0 (B, enrolldim) over time and concat on the
            # channel axis, then project (B, T, indim+enrolldim) -> (B, T, indim)
            out.append(
                transform(
                    torch.cat(
                        (main0, enroll0[:, :, None].expand(main0.shape)), dim=1
                    ).permute(0, 2, 1)
                ).permute(0, 2, 1)
            )
        return into_orig_type(tuple(out), orig_type)


class MulAddAdaptLayer(nn.Module):
    """Adapt by an elementwise multiply, optionally followed by an add.

    The enrollment embedding directly supplies the scale (and optionally the
    bias) that is applied to the main activations, so ``enrolldim`` must
    match ``indim`` (or ``2 * indim`` when addition is also used).
    """

    def __init__(
        self, indim: int, enrolldim: int, ninputs: int = 1, do_addition: bool = True
    ):
        super().__init__()
        self.ninputs = ninputs
        self.do_addition = do_addition

        if do_addition:
            assert enrolldim == 2 * indim, (enrolldim, indim)
        else:
            assert enrolldim == indim, (enrolldim, indim)

    def forward(
        self, main: TensorOrTensors, enroll: TensorOrTensors
    ) -> TensorOrTensors:
        """MulAddAdaptLayer Forward.

        Args:
            main: tensor or tuple or list
                  activations in the main neural network, which are adapted
                  tuple/list may be useful when we want to apply the adaptation
                    to both normal and skip connection at once
                  each tensor has shape (B, indim, T)
            enroll: tensor or tuple or list
                    embedding extracted from enrollment
                    tuple/list may be useful when we want to apply the adaptation
                      to both normal and skip connection at once
                    each tensor has shape (B, enrolldim)
        """
        assert type(main) == type(enroll)
        orig_type = type(main)
        main, enroll = into_tuple(main), into_tuple(enroll)
        assert len(main) == len(enroll) == self.ninputs, (
            len(main),
            len(enroll),
            self.ninputs,
        )
        out = []
        for main0, enroll0 in zip(main, enroll):
            if self.do_addition:
                enroll0_mul, enroll0_add = torch.chunk(enroll0, 2, dim=1)
                out.append(
                    enroll0_mul[:, :, None] * main0 + enroll0_add[:, :, None]
                )
            else:
                out.append(enroll0[:, :, None] * main0)
        return into_orig_type(tuple(out), orig_type)


class AttentionAdaptLayer(nn.Module):
    def __init__(
        self,
        indim: int,
        enrolldim: int,
        ninputs: int = 1,
        softmax_temp: float = 1,
        attention_dim: int = 200,
        hidden_dim: int = 200,
        is_dualpath_process: bool = False,
        return_attn: bool = False,
    ):
        """
        AttentionAdaptLayer for speaker selection in target speaker extraction.
        https://ieeexplore.ieee.org/abstract/document/8683448

        Args:
            indim: int,
                Input hidden dimension.
            enrolldim: int
                Hidden dimension of enrollment embedding.
            ninputs: int, optional
                The number of inputs (default: ``1``).
            softmax_temp: int, optional
                Temprature of softmax funcion (default: ``1``).
            attention_dim: int, optional
                Hidden dimension of attention (default: ``200``).
            hidden_dim: int, optional
                Hidden dimension in MLP layers (default: ``200``).
            is_dualpath_process: bool, optonal
                Whether the backbone model is dual-path model or not (default: ``False``).
            return_attn: bool, optional
                If ``True``, attention weight is also returned (default:``False``).
        """
        super().__init__()
        self.return_attn = return_attn
        self.mlp_v = nn.Sequential(
            nn.Linear(indim, hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, attention_dim),
        )
        self.mlp_iv = nn.Sequential(
            nn.Linear(indim, hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, attention_dim),
        )
        self.mlp_aux = nn.Sequential(
            nn.Linear(enrolldim, hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, attention_dim),
        )

        self.W_v = nn.Linear(attention_dim, attention_dim, bias=False)
        self.W_iv = nn.Linear(attention_dim, attention_dim, bias=False)
        self.W_aux = nn.Linear(attention_dim, attention_dim, bias=False)
        self.w = nn.Linear(attention_dim, 1, bias=False)
        self.b = nn.Parameter(torch.randn(attention_dim))

        self.attention_activation = nn.Tanh()
        self.softmax = nn.Softmax(dim=-2)
        self.alpha = softmax_temp
        self.indim = indim
        self.is_dualpath_process = is_dualpath_process

    def forward(
        self, main: TensorOrTensors, enroll: TensorOrTensors = None
    ) -> Union[TensorOrTensors, Tuple[TensorOrTensors, torch.Tensor]]:
        """AttentionAdaptLayer Forward. Variable names follow the paper.

        Args:
            main: tensor or tuple or list
                  activations in the main neural network, which are adapted
                  tuple/list may be useful when we want to apply the adaptation
                    to both normal and skip connection at once
            enroll: tensor or tuple or list
                    embedding extracted from enrollment
                    tuple/list may be useful when we want to apply the adaptation
                      to both normal and skip connection at once
        """
        outputs = []
        orig_type = type(main)
        is_tse = enroll is not None

        if not isinstance(main, tuple):
            main = (main,)
            enroll = (enroll,)
        for main0, enroll0 in zip(main, enroll):
            # non-dualpath case:
            #   main: (..., nspk, time, hidden)
            #   enroll: (..., time, hidden)
            # dual-path case:
            #   main: (..., nspk, chunk_size, num_chunk, hidden)
            #   enroll: (..., chunk_size, num_chunk, hidden)
            if is_tse:
                out, a = self._forward_tse(main0, enroll0)
            else:
                out, a = self._forward_no_enroll(main0)
            outputs.append(out)
        if self.return_attn:
            return into_orig_type(tuple(outputs), orig_type), a
        else:
            return into_orig_type(tuple(outputs), orig_type)

    def _forward_tse(
        self, main0: torch.Tensor, enroll0: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Attend over speakers using the enrollment embedding as the query."""
        assert type(main0) == type(enroll0)
        if self.is_dualpath_process:
            batch, num_spk, chunk_size, num_chunks, hidden = main0.shape
            main0 = main0.permute(0, 2, 3, 1, 4)
            mean_dim = (1, 2)
        else:
            batch, num_spk, time, hidden = main0.shape
            main0 = main0.transpose(-2, -3)
            mean_dim = (1,)
        s_v = self.mlp_v(main0)  # (..., time, nspk, hidden)
        s_iv = self.mlp_iv(main0).mean(
            dim=mean_dim, keepdim=True
        )  # (..., 1, nspk, hidden)
        s_aux = self.mlp_aux(enroll0).mean(dim=mean_dim, keepdim=True)

        # e: [batch, chunk_size, num_chunk, n_enroll, nspk, hidden]
        # a: [batch, chunk_size, num_chunk, n_enroll, nspk, 1]
        e_v = self.W_v(s_v)  # (batch, chunk_size, num_chunk, nspk, hidden)
        e_iv = self.W_iv(s_iv)  # (..., nspk, hidden)
        e_aux = self.W_aux(s_aux)  # (batch, 1, 1, hidden) / (batch, 1, hidden)
        e = (
            e_v[..., None, :, :]
            + e_iv[..., None, :, :]
            + e_aux[..., None, None, :, :]
            + self.b
        )
        a = self.w(self.attention_activation(e))
        a = self.softmax(a * self.alpha)
        out = (a * main0[..., None, :, :]).sum(dim=-2)[..., 0, :]
        return out, a

    def _forward_no_enroll(
        self, main0: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Fallback path (no enrollment): each speaker attends only to itself.

        Builds a fixed identity-like attention (2x2 block-diagonal for 2
        speakers) so that the output equals the input unchanged.
        """
        if self.is_dualpath_process:
            batch, chunk_size, num_chunks, num_spk, hidden = main0.shape
            zeros = main0.new_zeros((batch, chunk_size, num_chunks, 1, 1, 1))
            ones = main0.new_ones((batch, chunk_size, num_chunks, 1, 1, 1))
        else:
            batch, time, num_spk, hidden = main0.shape
            zeros = main0.new_zeros((batch, time, 1, 1, 1))
            ones = main0.new_ones((batch, time, 1, 1, 1))
        a = torch.cat(
            (
                torch.cat((ones, zeros), dim=-2),
                torch.cat((zeros, ones), dim=-2),
            ),
            dim=-3,
        )
        out = (a * main0[..., None, :, :]).sum(dim=-2)
        return out, a


# aliases for possible adaptation layer types
adaptation_layer_types = {
    "concat": ConcatAdaptLayer,
    "muladd": MulAddAdaptLayer,
    "mul": partial(MulAddAdaptLayer, do_addition=False),
    "attn": AttentionAdaptLayer,
}
