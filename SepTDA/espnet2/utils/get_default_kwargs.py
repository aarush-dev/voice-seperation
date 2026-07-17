"""Extract a function's default keyword-argument values as YAML-safe data.

Used to auto-populate config schemas/help text with the default values a
function (e.g. a model or loss constructor) would use if a given keyword
argument isn't explicitly configured.
"""

import inspect
from typing import Any, Callable, Dict


class Invalid:
    """Marker object for not serializable-object"""


def _yaml_serializable(value: Any) -> Any:
    """Recursively coerce ``value`` into a YAML-representable form.

    Tuples and sets are converted to lists; dicts are kept only if every
    key is a string, with each value recursively filtered; lists keep
    only fully-serializable elements (a single invalid element makes the
    whole list ``Invalid``); primitives are passed through unchanged.

    Args:
        value: Arbitrary Python value (typically a parameter default).

    Returns:
        A YAML-serializable equivalent of ``value``, or the ``Invalid``
        marker class if no such equivalent exists.
    """
    # isinstance(x, tuple) includes namedtuple, so type is used here
    if type(value) is tuple:
        return _yaml_serializable(list(value))
    elif isinstance(value, set):
        return _yaml_serializable(list(value))
    elif isinstance(value, dict):
        if not all(isinstance(k, str) for k in value):
            return Invalid
        retval = {}
        for k, v in value.items():
            v2 = _yaml_serializable(v)
            # Register only valid object
            if v2 not in (Invalid, inspect.Parameter.empty):
                retval[k] = v2
        return retval
    elif isinstance(value, list):
        retval = []
        for v in value:
            v2 = _yaml_serializable(v)
            # If any elements in the list are invalid,
            # the list also becomes invalid
            if v2 is Invalid:
                return Invalid
            else:
                retval.append(v2)
        return retval
    elif value in (inspect.Parameter.empty, None):
        return value
    elif isinstance(value, (float, int, complex, bool, str, bytes)):
        return value
    else:
        return Invalid


def get_default_kwargs(func: Callable) -> Dict[str, Any]:
    """Get the default values of the input function.

    Examples:
        >>> def func(a, b=3):  pass
        >>> get_default_kwargs(func)
        {'b': 3}

    """
    # params: An ordered mapping of inspect.Parameter
    params = inspect.signature(func).parameters
    data = {p.name: p.default for p in params.values()}
    # Remove not yaml-serializable object
    data = _yaml_serializable(data)
    return data
