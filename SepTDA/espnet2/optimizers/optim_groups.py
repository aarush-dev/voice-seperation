# noqa: E501 This code is modified from: https://github.com/HazyResearch/state-spaces/blob/main/src/utils/optim_groups.py
"""Helpers for building per-parameter-group optimizer configurations.

Used to exclude weight decay from selected parameters (e.g. biases and
normalization-layer parameters) when constructing an optimizer for training,
following the pattern described in
https://discuss.pytorch.org/t/weight-decay-only-for-weights-of-nn-linear-and-nn-conv/114348
"""

from typing import Any, Dict, Type

import torch.nn as nn
from torch.optim import Optimizer


def add_optimizer_hooks(
    model,
    bias_weight_decay=False,
    normalization_weight_decay=False,
):
    """Set zero weight decay for some params

    Set weight_decay=0.0 for parameters in model.no_weight_decay, for parameters with
    attribute _no_weight_decay==True, for bias parameters if bias_weight_decay==False,
    for normalization parameters if normalization_weight_decay==False

    See: https://discuss.pytorch.org/t/weight-decay-only-for-weights-of-nn-linear-and-nn-conv/114348 # noqa

    Args:
        model: The module whose parameters may be tagged for zero weight decay.
        bias_weight_decay: If False, bias parameters get weight_decay=0.0.
        normalization_weight_decay: If False, parameters belonging to
            normalization layers (BatchNorm, GroupNorm, LayerNorm, etc.) get
            weight_decay=0.0.

    Side Effects:
        For every matching parameter ``p``, sets ``p._optim = {"weight_decay": 0.0}``
        in place. Parameters that already carry a truthy ``_no_weight_decay``
        attribute are also tagged, regardless of the layer type they belong to.
    """
    # Separate out all parameters to those that will and won't experience regularizing
    # weight decay
    blacklist_weight_modules = (nn.Embedding,)
    if not normalization_weight_decay:
        blacklist_weight_modules += (
            nn.BatchNorm1d,
            nn.BatchNorm2d,
            nn.BatchNorm3d,
            nn.GroupNorm,
            nn.SyncBatchNorm,
            nn.InstanceNorm1d,
            nn.InstanceNorm2d,
            nn.InstanceNorm3d,
            nn.LayerNorm,
            nn.LocalResponseNorm,
        )
    for _module_name, module in model.named_modules():
        for param_name, param in module.named_parameters():
            if (
                (not bias_weight_decay and param_name.endswith("bias"))
                or getattr(param, "_no_weight_decay", False)
                or isinstance(module, blacklist_weight_modules)
            ):
                setattr(param, "_optim", {"weight_decay": 0.0})


def _unique_hyperparam_dicts(hyperparam_dicts: list) -> list:
    """Deduplicate a list of flat dicts while keeping the result deterministic.

    Each dict is converted to a ``frozenset`` of its items so that duplicate
    hyperparameter sets collapse to one entry; ``dict.fromkeys`` preserves
    first-seen order before the final ``sorted`` call imposes a canonical order.

    Args:
        hyperparam_dicts: Dicts such as ``{"weight_decay": 0.0}`` gathered from
            each parameter's ``_optim`` attribute.

    Returns:
        The unique dicts, sorted for deterministic ordering.
    """
    unique_itemsets = sorted(
        dict.fromkeys(frozenset(hp.items()) for hp in hyperparam_dicts)
    )
    return [dict(itemset) for itemset in unique_itemsets]


def configure_optimizer(
    model: nn.Module,
    optim_class: Type[Optimizer],
    optim_conf: Dict[str, Any],
    weight_decay_conf: Dict[str, Any],
) -> Optimizer:
    """Build an optimizer whose parameter groups honor per-parameter weight decay.

    Parameters tagged via `add_optimizer_hooks` (e.g. biases, normalization
    layers) are split into their own `add_param_group` calls with
    `weight_decay=0.0`, while all other parameters use `optim_conf` as-is.

    Args:
        model: The module to optimize.
        optim_class: Optimizer class to instantiate, e.g. `torch.optim.Adam`.
        optim_conf: Keyword arguments passed to `optim_class` (and used as the
            base config for every extra parameter group).
        weight_decay_conf: Keyword arguments forwarded to `add_optimizer_hooks`
            (`bias_weight_decay`, `normalization_weight_decay`).

    Returns:
        The constructed optimizer, with additional parameter groups added for
        any parameters that require special hyperparameters.
    """
    # Set zero weight decay for some params
    add_optimizer_hooks(
        model,
        **weight_decay_conf,
    )

    # Normal parameters
    all_params: list = list(model.parameters())
    params = [p for p in all_params if not hasattr(p, "_optim")]

    # Instantiate base optimizer
    optimizer = optim_class(params, **optim_conf)

    # Add parameters with special hyperparameters
    special_hyperparams = [getattr(p, "_optim") for p in all_params if hasattr(p, "_optim")]
    unique_hyperparam_dicts = _unique_hyperparam_dicts(special_hyperparams)
    for hyperparams in unique_hyperparam_dicts:
        group_params = [
            p for p in all_params if getattr(p, "_optim", None) == hyperparams
        ]
        optimizer.add_param_group({"params": group_params, **optim_conf, **hyperparams})

    return optimizer
