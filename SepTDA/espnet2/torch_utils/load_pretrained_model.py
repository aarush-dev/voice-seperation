"""Load a (possibly partial) pretrained checkpoint into a model.

Supports loading an entire checkpoint, or just a sub-tree of the checkpoint's
state dict into a sub-module of the target model, as specified by a compact
`<file_path>:<src_key>:<dst_key>:<exclude_keys>` string (see
`load_pretrained_model` below). This is the mechanism used to warm-start
training or fine-tuning from another run's checkpoint, so its key-matching
semantics must be preserved exactly.
"""
import logging
from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.nn
import torch.optim

StateDict = Dict[str, Union[float, torch.Tensor]]


def filter_state_dict(
    dst_state: StateDict,
    src_state: StateDict,
) -> StateDict:
    """Filter out entries of `src_state` that don't match `dst_state`.

    An entry is dropped (with a warning) if its key is absent from
    `dst_state`, or if the shapes of the corresponding tensors differ.

    Args:
        dst_state: reference state dict for filtering
        src_state: target state dict for filtering

    Returns:
        The subset of `src_state` whose keys exist in `dst_state` with a
        matching tensor size.
    """
    match_state = {}
    for key, value in src_state.items():
        if key in dst_state and (dst_state[key].size() == src_state[key].size()):
            match_state[key] = value
        else:
            if key not in dst_state:
                logging.warning(
                    f"Filter out {key} from pretrained dict"
                    + " because of name not found in target dict"
                )
            else:
                logging.warning(
                    f"Filter out {key} from pretrained dict"
                    + " because of size mismatch"
                    + f"({dst_state[key].size()}-{src_state[key].size()})"
                )
    return match_state


def _parse_init_param(
    init_param: str,
) -> Tuple[str, Optional[str], Optional[str], Optional[str]]:
    """Split an `init_param` spec into its path/src_key/dst_key/excludes parts.

    `init_param` follows the `<file_path>:<src_key>:<dst_key>:<exclude_keys>`
    format documented on `load_pretrained_model`; trailing fields may be
    omitted. An empty `src_key` or `dst_key` field (e.g. "path::decoder") is
    normalized to `None`, matching the original behavior.
    """
    sps = init_param.split(":", 4)
    if len(sps) == 4:
        path, src_key, dst_key, excludes = sps
    elif len(sps) == 3:
        path, src_key, dst_key = sps
        excludes = None
    elif len(sps) == 2:
        path, src_key = sps
        dst_key, excludes = None, None
    else:
        (path,) = sps
        src_key, dst_key, excludes = None, None, None
    if src_key == "":
        src_key = None
    if dst_key == "":
        dst_key = None
    return path, src_key, dst_key, excludes


def _get_nested_attr(obj: Any, key: str) -> Any:
    """Get a nested attribute of `obj` addressed by a dotted `key`.

    >>> class A(torch.nn.Module):
    ...     def __init__(self):
    ...         super().__init__()
    ...         self.linear = torch.nn.Linear(10, 10)
    >>> a = A()
    >>> assert A.linear.weight is get_attr(A, 'linear.weight')

    """
    if key.strip() == "":
        return obj
    for k in key.split("."):
        obj = getattr(obj, k)
    return obj


def _select_dst_module(model: torch.nn.Module, dst_key: Optional[str]) -> Any:
    """Return the sub-module of `model` addressed by `dst_key` (or `model`)."""
    if dst_key is None:
        return model
    return _get_nested_attr(model, dst_key)


def _load_src_state_dict(
    path: str,
    map_location: str,
    src_key: Optional[str],
    excludes: Optional[str],
) -> StateDict:
    """Load the checkpoint at `path` and apply exclude/src_key filtering.

    Args:
        path: Checkpoint file path.
        map_location: Passed through to `torch.load`.
        src_key: If given, only keys starting with `src_key + "."` are kept,
            and that prefix is stripped from the retained keys.
        excludes: Comma-separated key prefixes to drop from the checkpoint
            before `src_key` filtering is applied.

    Returns:
        The filtered checkpoint state dict.
    """
    src_state = torch.load(path, map_location=map_location)
    if excludes is not None:
        for e in excludes.split(","):
            src_state = {k: v for k, v in src_state.items() if not k.startswith(e)}

    if src_key is not None:
        src_state = {
            k[len(src_key) + 1 :]: v
            for k, v in src_state.items()
            if k.startswith(src_key)
        }
    return src_state


def load_pretrained_model(
    init_param: str,
    model: torch.nn.Module,
    ignore_init_mismatch: bool,
    map_location: str = "cpu",
) -> None:
    """Load a model state and set it to the model.

    Args:
        init_param: <file_path>:<src_key>:<dst_key>:<exclude_Keys>

    Examples:
        >>> load_pretrained_model("somewhere/model.pth", model)
        >>> load_pretrained_model("somewhere/model.pth:decoder:decoder", model)
        >>> load_pretrained_model("somewhere/model.pth:decoder:decoder:", model)
        >>> load_pretrained_model(
        ...     "somewhere/model.pth:decoder:decoder:decoder.embed", model
        ... )
        >>> load_pretrained_model("somewhere/decoder.pth::decoder", model)
    """
    path, src_key, dst_key, excludes = _parse_init_param(init_param)
    obj = _select_dst_module(model, dst_key)
    src_state = _load_src_state_dict(path, map_location, src_key, excludes)

    dst_state = obj.state_dict()
    if ignore_init_mismatch:
        src_state = filter_state_dict(dst_state, src_state)
    dst_state.update(src_state)
    obj.load_state_dict(dst_state)
