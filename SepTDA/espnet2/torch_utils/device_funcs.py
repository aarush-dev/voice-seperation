"""Recursive device/dtype-movement helpers for (possibly nested) tensor collections.

These utilities walk arbitrary Python containers (dict, list, tuple, dataclass,
namedtuple) that may hold ``torch.Tensor`` or ``numpy.ndarray`` leaves and apply
a device/dtype transfer to every leaf, preserving the original container
structure and type. They are used, e.g., to move a whole training batch to the
GPU (`to_device`) or to coerce collated outputs into a form that
``torch.nn.DataParallel`` can gather across devices (`force_gatherable`).
"""
import dataclasses
import warnings
from typing import Any, Optional

import numpy as np
import torch


def to_device(
    data: Any,
    device: Optional[str] = None,
    dtype: Optional[torch.dtype] = None,
    non_blocking: bool = False,
    copy: bool = False,
) -> Any:
    """Recursively move tensors (and numpy arrays) in `data` to `device`/`dtype`.

    Supports dicts, dataclasses, namedtuples, lists, tuples, numpy arrays and
    torch tensors; any other type is returned unchanged. Container types and
    keys/fields are preserved exactly.

    Args:
        data: Arbitrary (possibly nested) object to transfer.
        device: Target device, e.g. "cuda:0". `None` leaves the device as-is.
        dtype: Target dtype. `None` leaves the dtype as-is.
        non_blocking: Passed through to `torch.Tensor.to`.
        copy: Passed through to `torch.Tensor.to`.

    Returns:
        A new object with the same structure as `data` but with tensor leaves
        moved to the requested device/dtype.
    """
    if isinstance(data, dict):
        return {
            k: to_device(v, device, dtype, non_blocking, copy) for k, v in data.items()
        }
    elif dataclasses.is_dataclass(data) and not isinstance(data, type):
        return type(data)(
            *[
                to_device(v, device, dtype, non_blocking, copy)
                for v in dataclasses.astuple(data)
            ]
        )
    # maybe namedtuple. I don't know the correct way to judge namedtuple.
    elif isinstance(data, tuple) and type(data) is not tuple:
        return type(data)(
            *[to_device(o, device, dtype, non_blocking, copy) for o in data]
        )
    elif isinstance(data, (list, tuple)):
        return type(data)(to_device(v, device, dtype, non_blocking, copy) for v in data)
    elif isinstance(data, np.ndarray):
        return to_device(torch.from_numpy(data), device, dtype, non_blocking, copy)
    elif isinstance(data, torch.Tensor):
        return data.to(device, dtype, non_blocking, copy)
    else:
        return data


def force_gatherable(data: Any, device: Any) -> Any:
    """Recursively coerce `data` into a form gatherable by `torch.nn.DataParallel`.

    The difference from `to_device()` is that plain Python `float`/`int`
    leaves are converted into 1-dimensional tensors, since `DataParallel`
    can only gather:
        - `torch.cuda.Tensor` with 1 or more dimensions (a 0-dim tensor
          triggers a warning from DataParallel, so it is reshaped to 1-dim
          here), or
        - a list, tuple, dict of such tensors.

    Args:
        data: Arbitrary (possibly nested) object to coerce.
        device: Target device for tensor leaves.

    Returns:
        A new object with the same structure as `data`, with numeric leaves
        turned into 1-dim tensors on `device`.
    """
    if isinstance(data, dict):
        return {k: force_gatherable(v, device) for k, v in data.items()}
    # DataParallel can't handle NamedTuple well
    elif isinstance(data, tuple) and type(data) is not tuple:
        return type(data)(*[force_gatherable(o, device) for o in data])
    elif isinstance(data, (list, tuple, set)):
        return type(data)(force_gatherable(v, device) for v in data)
    elif isinstance(data, np.ndarray):
        return force_gatherable(torch.from_numpy(data), device)
    elif isinstance(data, torch.Tensor):
        if data.dim() == 0:
            # To 1-dim array
            data = data[None]
        return data.to(device)
    elif isinstance(data, float):
        return torch.tensor([data], dtype=torch.float, device=device)
    elif isinstance(data, int):
        return torch.tensor([data], dtype=torch.long, device=device)
    elif data is None:
        return None
    else:
        warnings.warn(f"{type(data)} may not be gatherable by DataParallel")
        return data
