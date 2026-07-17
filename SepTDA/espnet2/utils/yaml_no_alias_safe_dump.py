"""YAML dumping helper that disables anchor/alias output.

Used when writing config/checkpoint YAML files, so that repeated nested
values are rendered inline (fully expanded) rather than collapsed into
``&anchor``/``*alias`` references, which is easier for humans to read
and diff.
"""

from typing import Any, Optional

import yaml


class NoAliasSafeDumper(yaml.SafeDumper):
    """A `yaml.SafeDumper` that never emits anchors/aliases for shared objects."""

    # Disable anchor/alias in yaml because looks ugly
    def ignore_aliases(self, data: Any) -> bool:
        """Always report ``data`` as alias-ineligible, forcing full inline expansion."""
        return True


def yaml_no_alias_safe_dump(data: Any, stream: Optional[Any] = None, **kwargs):
    """Safe-dump in yaml with no anchor/alias"""
    return yaml.dump(
        data, stream, allow_unicode=True, Dumper=NoAliasSafeDumper, **kwargs
    )
