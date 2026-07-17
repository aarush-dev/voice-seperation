"""Convert a keyword-argument dict into a positional-argument tuple.

Used e.g. to adapt a model's forward-call kwargs into a positional tuple
suitable for tracing/graphing tools (such as tensorboard's `add_graph`)
that expect a plain example-input tuple rather than a kwargs dict.
"""

import inspect
from typing import Any, Callable, Dict, Tuple


def kwargs2args(func: Callable, kwargs: Dict[str, Any]) -> Tuple[Any, ...]:
    """Reorder ``kwargs`` into a positional tuple matching ``func``'s signature.

    Args:
        func: The function whose parameter order determines positions.
        kwargs: Argument values keyed by parameter name; keys not found
            in ``func``'s signature are ignored.

    Returns:
        A tuple of values in ``func``'s parameter order, truncated at the
        first parameter with no matching entry in ``kwargs``.
    """
    parameters = inspect.signature(func).parameters
    position_of = {name: i for i, name in enumerate(parameters)}
    args = [None for _ in range(len(parameters))]
    for k, v in kwargs.items():
        if k in position_of:
            args[position_of[k]] = v

    for i, v in enumerate(args):
        if v is None:
            break

    return tuple(args[:i])
