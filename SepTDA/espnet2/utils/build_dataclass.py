"""Helper for constructing a dataclass instance from an argparse Namespace.

Used to bridge CLI/config arguments (parsed by argparse) into typed
dataclass configuration objects, validating each field's runtime type
against the dataclass's type annotations.
"""

import argparse
import dataclasses
from typing import Any, Type, TypeVar

from typeguard import check_type

_T = TypeVar("_T")


def build_dataclass(dataclass: Type[_T], args: argparse.Namespace) -> _T:
    """Build a dataclass instance by pulling matching fields off ``args``.

    Args:
        dataclass: A dataclass type whose fields should be populated.
        args: Parsed CLI/config namespace expected to have an attribute
            for every field of ``dataclass``.

    Returns:
        An instance of ``dataclass`` constructed from ``args``.

    Raises:
        ValueError: If ``args`` is missing an attribute required by a
            dataclass field.
        TypeError: If an attribute's runtime value doesn't match the
            field's declared type.
    """
    kwargs: dict[str, Any] = {}
    for field in dataclasses.fields(dataclass):
        if not hasattr(args, field.name):
            raise ValueError(
                f"args doesn't have {field.name}. You need to set it to ArgumentsParser"
            )
        check_type(field.name, getattr(args, field.name), field.type)
        kwargs[field.name] = getattr(args, field.name)
    return dataclass(**kwargs)
