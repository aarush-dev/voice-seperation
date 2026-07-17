"""Recursive weighted-sum/average ops over (possibly nested, possibly distributed)
tensor collections.

Used to combine per-utterance losses/stats (weighted by e.g. sequence length)
into a scalar, optionally reducing across a distributed data-parallel group
via all-gather/all-reduce. `obj`/`a` may be a tensor, or a nested list/tuple/
dict of tensors, mirroring the structure produced by the training loop.
"""
from typing import Any, Dict, List, Tuple, Union

import torch
from espnet2.train.distributed_utils import get_world_size

if torch.distributed.is_available():
    from torch.distributed import ReduceOp

# A (possibly nested) tensor container: a tensor, None, or a list/tuple/dict of these.
Nested = Union[torch.Tensor, None, List[Any], Tuple[Any, ...], Dict[Any, Any]]


def recursive_sum(
    obj: Nested, weight: torch.Tensor, distributed: bool = False
) -> Nested:
    """Recursively compute `sum(obj * weight)` over every tensor leaf of `obj`.

    Each tensor leaf must have the same shape as `weight`  # (B,).
    When `distributed` is True, the per-rank sums are all-gathered and then
    combined with `nansum` (falling back to plain `sum` if every rank's
    value is NaN), so that a NaN on one rank doesn't silently corrupt the
    reduction for the entire group.

    Args:
        obj: Tensor or nested list/tuple/dict of tensors to reduce.
        weight: 1-D per-sample weight tensor, e.g. sequence lengths. # (B,)
        distributed: Whether to all-gather partial sums across the
            distributed process group.

    Returns:
        The same nested structure as `obj`, with each tensor leaf replaced
        by its scalar weighted sum.

    Raises:
        ValueError: If a leaf of `obj` is not a Tensor, None, list, tuple,
            or dict.
    """
    assert weight.dim() == 1, weight.size()
    if isinstance(obj, (tuple, list)):
        return type(obj)(recursive_sum(v, weight, distributed) for v in obj)
    elif isinstance(obj, dict):
        return {k: recursive_sum(v, weight, distributed) for k, v in obj.items()}
    elif isinstance(obj, torch.Tensor):
        assert obj.size() == weight.size(), (obj.size(), weight.size())
        obj = (obj * weight.type(obj.dtype)).sum()  # scalar
        if distributed:
            gathered = [
                torch.empty_like(obj) for _ in range(torch.distributed.get_world_size())
            ]
            torch.distributed.all_gather(gathered, obj)
            if all([torch.isnan(o) for o in gathered]):
                obj = torch.sum(torch.stack(gathered))
            else:
                obj = torch.nansum(torch.stack(gathered))
        return obj
    elif obj is None:
        return None
    else:
        raise ValueError(type(obj))


def recursive_divide(a: Nested, b: torch.Tensor) -> Nested:
    """Recursively divide every tensor leaf of `a` by scalar tensor `b`.

    Args:
        a: Tensor or nested list/tuple/dict of tensors (the numerator).
        b: Scalar tensor divisor, broadcast against every leaf of `a`.

    Returns:
        The same nested structure as `a`, with each tensor leaf divided
        by `b`.

    Raises:
        ValueError: If a leaf of `a` is not a Tensor, None, list, tuple,
            or dict.
    """
    if isinstance(a, (tuple, list)):
        return type(a)(recursive_divide(v, b) for v in a)
    elif isinstance(a, dict):
        return {k: recursive_divide(v, b) for k, v in a.items()}
    elif isinstance(a, torch.Tensor):
        assert a.size() == b.size(), (a.size(), b.size())
        return a / b.type(a.dtype)
    elif a is None:
        return None
    else:
        raise ValueError(type(a))


def recursive_average(
    obj: Nested, weight: torch.Tensor, distributed: bool = False
) -> Tuple[Nested, torch.Tensor]:
    """Recursively compute the `weight`-weighted average of every tensor leaf.

    Equivalent to `recursive_sum(obj, weight) / weight.sum()`, with the
    weight's own sum optionally all-reduced across the distributed process
    group first so every rank divides by the same (global) total weight.

    Args:
        obj: Tensor or nested list/tuple/dict of tensors to average.
        weight: 1-D per-sample weight tensor, e.g. sequence lengths. # (B,)
        distributed: Whether to reduce sums/weight across the distributed
            process group.

    Returns:
        A tuple `(averaged_obj, total_weight)`, where `averaged_obj` has the
        same nested structure as `obj` and `total_weight` is the (globally
        reduced, if `distributed`) scalar sum of `weight`.
    """
    obj = recursive_sum(obj, weight, distributed)
    weight = weight.sum()  # scalar
    if distributed:
        torch.distributed.all_reduce(weight, op=ReduceOp.SUM)
    # Normalize weight to be sum-to-1
    obj = recursive_divide(obj, weight)
    return obj, weight
