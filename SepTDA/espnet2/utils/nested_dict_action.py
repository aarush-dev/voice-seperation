"""Custom argparse Action that builds up a (possibly nested) dict value.

Lets a CLI flag be repeated to incrementally build a config dict, using
either dotted-key=yaml-value syntax (`--conf a.b=3`) or a literal
Python/YAML dict/mapping (`--conf {a: 3}`). Used for options like
`--conf` in ESPnet2 training scripts, driven from YAML configs.
"""

import argparse
import copy
from typing import Any, List, Optional, Sequence

import yaml


class NestedDictAction(argparse.Action):
    """Action class to append items to dict object.

    Examples:
        >>> parser = argparse.ArgumentParser()
        >>> _ = parser.add_argument('--conf', action=NestedDictAction,
        ...                         default={'a': 4})
        >>> parser.parse_args(['--conf', 'a=3', '--conf', 'c=4'])
        Namespace(conf={'a': 3, 'c': 4})
        >>> parser.parse_args(['--conf', 'c.d=4'])
        Namespace(conf={'a': 4, 'c': {'d': 4}})
        >>> parser.parse_args(['--conf', 'c.d=4', '--conf', 'c=2'])
        Namespace(conf={'a': 4, 'c': 2})
        >>> parser.parse_args(['--conf', '{d: 5, e: 9}'])
        Namespace(conf={'d': 5, 'e': 9})

    """

    _syntax = """Syntax:
  {op} <key>=<yaml-string>
  {op} <key>.<key2>=<yaml-string>
  {op} <python-dict>
  {op} <yaml-string>
e.g.
  {op} a=4
  {op} a.b={{c: true}}
  {op} {{"c": True}}
  {op} {{a: 34.5}}
"""

    def __init__(
        self,
        option_strings: List[str],
        dest: str,
        nargs=None,
        default: Any = None,
        choices=None,
        required: bool = False,
        help: Optional[str] = None,
        metavar: Optional[str] = None,
    ):
        """Register the action; ``type`` is forced to None (values are strings we parse ourselves)."""
        super().__init__(
            option_strings=option_strings,
            dest=dest,
            nargs=nargs,
            default=copy.deepcopy(default),
            type=None,
            choices=choices,
            required=required,
            help=help,
            metavar=metavar,
        )

    def __call__(
        self,
        parser: argparse.ArgumentParser,
        namespace: argparse.Namespace,
        values: Any,
        option_strings: Optional[Sequence[str]] = None,
    ) -> None:
        """Merge one occurrence of the flag's value into ``namespace``'s dict.

        Two syntaxes are supported:

        - ``key=<yaml-value>`` (optionally dotted, e.g. ``a.b.c=3``): merges
          the value at the given nested path into the existing dict.
        - A bare mapping literal (Python-dict or YAML), e.g. ``{a: 3}``:
          replaces or ``dict.update``s the existing dict wholesale.

        Raises:
            argparse.ArgumentTypeError: If a bare-mapping value doesn't
                evaluate to a dict via ``eval``.
            argparse.ArgumentError: If a bare-mapping value doesn't parse
                to a dict via YAML either.
        """
        # --{option} a.b=3 -> {'a': {'b': 3}}
        if "=" in values:
            indict = copy.deepcopy(getattr(namespace, self.dest, {}))
            key, value = values.split("=", maxsplit=1)
            if not value.strip() == "":
                value = yaml.load(value, Loader=yaml.Loader)
            if not isinstance(indict, dict):
                indict = {}

            keys = key.split(".")
            d = indict
            for idx, k in enumerate(keys):
                if idx == len(keys) - 1:
                    d[k] = value
                else:
                    if not isinstance(d.setdefault(k, {}), dict):
                        # Remove the existing value and recreates as empty dict
                        d[k] = {}
                    d = d[k]

            # Update the value
            setattr(namespace, self.dest, indict)
        else:
            try:
                # At the first, try eval(), i.e. Python syntax dict.
                # e.g. --{option} "{'a': 3}" -> {'a': 3}
                # This is workaround for internal behaviour of configargparse.
                value = eval(values, {}, {})
                if not isinstance(value, dict):
                    syntax = self._syntax.format(op=option_strings)
                    mes = f"must be interpreted as dict: but got {values}\n{syntax}"
                    raise argparse.ArgumentTypeError(self, mes)
            except Exception:
                # and the second, try yaml.load
                value = yaml.load(values, Loader=yaml.Loader)
                if not isinstance(value, dict):
                    syntax = self._syntax.format(op=option_strings)
                    mes = f"must be interpreted as dict: but got {values}\n{syntax}"
                    raise argparse.ArgumentError(self, mes)

            d = getattr(namespace, self.dest, None)
            if isinstance(d, dict):
                d.update(value)
            else:
                # Remove existing params, and overwrite
                setattr(namespace, self.dest, value)
