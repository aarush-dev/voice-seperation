"""Prints a human-readable model parameter-count / size / dtype summary.

Used for logging a quick overview of a model (structure, parameter counts,
trainable-parameter percentage, memory footprint) at the start of training.
"""
import humanfriendly
import numpy as np
import torch


def get_human_readable_count(number: int) -> str:
    """Return human_readable_count

    Originated from:
    https://github.com/PyTorchLightning/pytorch-lightning/blob/master/pytorch_lightning/core/memory.py

    Abbreviates an integer number with K, M, B, T for thousands, millions,
    billions and trillions, respectively.
    Examples:
        >>> get_human_readable_count(123)
        '123  '
        >>> get_human_readable_count(1234)  # (one thousand)
        '1 K'
        >>> get_human_readable_count(2e6)   # (two million)
        '2 M'
        >>> get_human_readable_count(3e9)   # (three billion)
        '3 B'
        >>> get_human_readable_count(4e12)  # (four trillion)
        '4 T'
        >>> get_human_readable_count(5e15)  # (more than trillion)
        '5,000 T'
    Args:
        number: a positive integer number
    Return:
        A string formatted according to the pattern described above.
    """
    assert number >= 0
    labels = [" ", "K", "M", "B", "T"]
    num_digits = int(np.floor(np.log10(number)) + 1 if number > 0 else 1)
    num_groups = int(np.ceil(num_digits / 3))
    num_groups = min(num_groups, len(labels))  # don't abbreviate beyond trillions
    shift = -3 * (num_groups - 1)
    number = number * (10**shift)
    index = num_groups - 1
    return f"{number:.2f} {labels[index]}"


def to_bytes(dtype: torch.dtype) -> int:
    """Return the size in bytes of one element of `dtype`.

    E.g. `torch.float16` -> 16 bits -> 2 bytes. Relies on the dtype's
    string representation ending in its bit-width (e.g. "torch.float16").
    """
    return int(str(dtype)[-2:]) // 8


def model_summary(model: torch.nn.Module) -> str:
    """Build a human-readable summary of `model`.

    Includes the module structure (`str(model)`), total and trainable
    parameter counts (abbreviated, e.g. "12.3 M"), the trainable-parameter
    percentage, the memory footprint of trainable parameters, and the dtype
    of the model's first parameter.

    Args:
        model: Model to summarize.

    Returns:
        A multi-line summary string.
    """
    message = "Model structure:\n"
    message += str(model)

    total_num_params = sum(p.numel() for p in model.parameters())
    trainable_num_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    percent_trainable = "{:.1f}".format(
        trainable_num_params * 100.0 / total_num_params
    )
    total_num_params_str = get_human_readable_count(total_num_params)
    trainable_num_params_str = get_human_readable_count(trainable_num_params)

    message += "\n\nModel summary:\n"
    message += f"    Class Name: {model.__class__.__name__}\n"
    message += f"    Total Number of model parameters: {total_num_params_str}\n"
    message += (
        f"    Number of trainable parameters: {trainable_num_params_str} "
        f"({percent_trainable}%)\n"
    )
    trainable_num_bytes = humanfriendly.format_size(
        sum(
            p.numel() * to_bytes(p.dtype) for p in model.parameters() if p.requires_grad
        )
    )
    message += f"    Size: {trainable_num_bytes}\n"
    dtype = next(iter(model.parameters())).dtype
    message += f"    Type: {dtype}"
    return message
