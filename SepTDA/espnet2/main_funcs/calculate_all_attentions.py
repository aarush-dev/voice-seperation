"""Extracts attention weights from every attention layer of an ESPnet2 model.

Runs a batch through the model one sample at a time (attention modules across
the supported RNN/Transformer implementations don't share a single output
convention, so batched extraction isn't practical), collecting each layer's
attention weights via forward hooks. Used for attention-plot visualization
and debugging, not for training or inference.
"""

from collections import defaultdict
from typing import Any, Dict, List, Union

import torch
from torch import Tensor
from torch.utils.hooks import RemovableHandle

from espnet2.train.abs_espnet_model import AbsESPnetModel
from espnet.nets.pytorch_backend.rnn.attentions import (
    AttAdd,
    AttCov,
    AttCovLoc,
    AttDot,
    AttForward,
    AttForwardTA,
    AttLoc,
    AttLoc2D,
    AttLocRec,
    AttMultiHeadAdd,
    AttMultiHeadDot,
    AttMultiHeadLoc,
    AttMultiHeadMultiResLoc,
    NoAtt,
)
from espnet.nets.pytorch_backend.transformer.attention import MultiHeadedAttention

# Per-sample hook outputs: layer name -> either a single tensor
# (MultiHeadedAttention, called once per forward) or a list of per-decoding
# -step tensors/tensor-lists (RNN attentions, called once per output step).
HookOutputs = Dict[str, Union[Tensor, List[Any]]]


def _capture_attention_output(
    name: str, module: torch.nn.Module, output: Any, outputs: HookOutputs
) -> None:
    """Forward-hook body: pull attention weights out of `output` into `outputs[name]`.

    Each attention implementation returns its weights in a different shape or
    container, matching that module's own `forward()` return signature.
    `MultiHeadedAttention` computes attention for the whole sequence in one
    call, so it overwrites `outputs[name]` directly; the other (RNN-style)
    attentions are invoked once per decoding step, so their weights are
    appended to a list at `outputs[name]`.

    Args:
        name: Fully-qualified module name, used as the key into `outputs`.
        module: The attention submodule that produced `output`.
        output: The module's forward-pass return value.
        outputs: Mutated in place to record the extracted weights.
    """
    if isinstance(module, MultiHeadedAttention):
        # NOTE(kamo): MultiHeadedAttention doesn't return attention weight
        # attn: (B, Head, Tout, Tin)
        outputs[name] = module.attn.detach().cpu()
    elif isinstance(module, AttLoc2D):
        _context, w = output
        # w: previous concatenated attentions, (B, nprev, Tin)
        att_w = w[:, -1].detach().cpu()
        outputs.setdefault(name, []).append(att_w)
    elif isinstance(module, (AttCov, AttCovLoc)):
        _context, w = output
        assert isinstance(w, list), type(w)
        # w: list of previous attentions, nprev x (B, Tin)
        att_w = w[-1].detach().cpu()
        outputs.setdefault(name, []).append(att_w)
    elif isinstance(module, AttLocRec):
        # w: (B, Tin)
        _context, (w, (_att_h, _att_c)) = output
        att_w = w.detach().cpu()
        outputs.setdefault(name, []).append(att_w)
    elif isinstance(
        module,
        (
            AttMultiHeadDot,
            AttMultiHeadAdd,
            AttMultiHeadLoc,
            AttMultiHeadMultiResLoc,
        ),
    ):
        _context, w = output
        # w: nhead x (B, Tin)
        assert isinstance(w, list), type(w)
        att_w = [_w.detach().cpu() for _w in w]
        outputs.setdefault(name, []).append(att_w)
    elif isinstance(
        module,
        (
            AttAdd,
            AttDot,
            AttForward,
            AttForwardTA,
            AttLoc,
            NoAtt,
        ),
    ):
        _context, w = output
        att_w = w.detach().cpu()
        outputs.setdefault(name, []).append(att_w)


def _register_attention_hooks(
    model: torch.nn.Module, outputs: HookOutputs
) -> Dict[str, RemovableHandle]:
    """Attach a forward hook to every submodule that records into `outputs`.

    Returns:
        Mapping of module name -> hook handle, so callers can remove the
        hooks afterwards via `handle.remove()`.
    """
    handles = {}
    for name, submodule in model.named_modules():

        def hook(module, input, output, name=name):
            _capture_attention_output(name, module, output, outputs)

        handles[name] = submodule.register_forward_hook(hook)
    return handles


def _slice_sample(
    batch: Dict[str, Tensor], keys: List[str], ibatch: int
) -> Dict[str, Tensor]:
    """Slice out sample `ibatch` from a batch, trimming to its own length.

    Args:
        batch: Full batch, e.g. `{"speech": (B, L, ...), "speech_lengths": (B,)}`.
        keys: Non-length, non-"utt_id" keys of `batch` to slice.
        ibatch: Index of the sample to extract.

    Returns:
        A single-sample batch: for each key, shape `(B, L, ...) -> (1, L2, ...)`
        (trimmed to `batch[key + "_lengths"][ibatch]` when a lengths tensor
        exists) and `(B,) -> (1,)` for the corresponding lengths entries.
        Also carries through `"utt_id"` untouched, if present.
    """
    sample = {
        k: (
            batch[k][ibatch, None, : batch[k + "_lengths"][ibatch]]
            if k + "_lengths" in batch
            else batch[k][ibatch, None]
        )
        for k in keys
    }
    sample.update(
        {
            k + "_lengths": batch[k + "_lengths"][ibatch, None]
            for k in keys
            if k + "_lengths" in batch
        }
    )
    if "utt_id" in batch:
        sample["utt_id"] = batch["utt_id"]
    return sample


def _reduce_hook_output(output: Union[Tensor, List[Any]]) -> Tensor:
    """Normalize one sample's collected hook output to (Tout, Tin) or (NHead, Tout, Tin).

    Args:
        output: Either the single tensor recorded for `MultiHeadedAttention`
            (shape `(1, NHead, Tout, Tin)`), or a list accumulated over
            decoding steps -- either `Tout x (1, Tin)` tensors, or
            `Tout x (nhead x (1, Tin))` nested lists for multi-head RNN
            attentions.

    Returns:
        A single tensor of shape `(Tout, Tin)` or `(NHead, Tout, Tin)`.
    """
    if isinstance(output, list):
        if isinstance(output[0], list):
            # output: nhead x (Tout, Tin)
            return torch.stack(
                [
                    # Tout x (1, Tin) -> (Tout, Tin)
                    torch.cat([o[idx] for o in output], dim=0)
                    for idx in range(len(output[0]))
                ],
                dim=0,
            )
        else:
            # Tout x (1, Tin) -> (Tout, Tin)
            return torch.cat(output, dim=0)
    else:
        # (1, NHead, Tout, Tin) -> (NHead, Tout, Tin)
        return output.squeeze(0)


@torch.no_grad()
def calculate_all_attentions(
    model: AbsESPnetModel, batch: Dict[str, torch.Tensor]
) -> Dict[str, List[torch.Tensor]]:
    """Derive the outputs from the all attention layers

    Args:
        model:
        batch: same as forward
    Returns:
        return_dict: A dict of a list of tensor.
        key_names x batch x (D1, D2, ...)

    """
    batch_size = len(next(iter(batch.values())))
    assert all(len(v) == batch_size for v in batch.values()), {
        k: v.shape for k, v in batch.items()
    }

    # 1. Register forward_hook fn to save the output from specific layers
    outputs: HookOutputs = {}
    handles = _register_attention_hooks(model, outputs)

    # 2. Just forward one by one sample.
    # Batch-mode can't be used to keep requirements small for each models.
    keys = [k for k in batch if not (k.endswith("_lengths") or k in ["utt_id"])]

    return_dict = defaultdict(list)
    for ibatch in range(batch_size):
        model(**_slice_sample(batch, keys, ibatch))

        # Derive the attention results
        for name, output in outputs.items():
            return_dict[name].append(_reduce_hook_output(output))
        outputs.clear()

    # 3. Remove all hooks
    for handle in handles.values():
        handle.remove()

    return dict(return_dict)
