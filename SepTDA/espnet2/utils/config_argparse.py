"""argparse.ArgumentParser subclass with built-in YAML config-file support.

Adds a "--config" option so CLI entry points can load default argument
values from a YAML file, with values still overridable from the actual
command line (since config values are only applied via ``set_defaults``).
"""

import argparse
from pathlib import Path
from typing import Any, Optional, Sequence, Tuple

import yaml


class ArgumentParser(argparse.ArgumentParser):
    """Simple implementation of ArgumentParser supporting config file

    This class is originated from https://github.com/bw2/ConfigArgParse,
    but this class is lack of some features that it has.

    - Not supporting multiple config files
    - Automatically adding "--config" as an option.
    - Not supporting any formats other than yaml
    - Not checking argument type

    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.add_argument("--config", help="Give config file in yaml format")

    def parse_known_args(
        self,
        args: Optional[Sequence[str]] = None,
        namespace: Optional[argparse.Namespace] = None,
    ) -> Tuple[argparse.Namespace, list]:
        """Parse args, first loading defaults from --config if given.

        The command line is parsed once to discover "--config", the YAML
        file (if any) is loaded and applied via ``set_defaults`` so its
        values act as defaults, and then the command line is parsed again
        so explicit CLI arguments still take precedence over the config
        file.

        Raises:
            SystemExit: Via ``self.error()`` if the config file is
                missing, isn't a YAML mapping, or contains a key that
                doesn't correspond to any registered argument.
        """
        # Once parsing for setting from "--config"
        _args, _ = super().parse_known_args(args, namespace)
        if _args.config is not None:
            if not Path(_args.config).exists():
                self.error(f"No such file: {_args.config}")

            with open(_args.config, "r", encoding="utf-8") as f:
                config_dict: Any = yaml.safe_load(f)
            if not isinstance(config_dict, dict):
                self.error("Config file has non dict value: {_args.config}")

            for key in config_dict:
                for action in self._actions:
                    if key == action.dest:
                        break
                else:
                    self.error(f"unrecognized arguments: {key} (from {_args.config})")

            # NOTE(kamo): Ignore "--config" from a config file
            # NOTE(kamo): Unlike "configargparse", this module doesn't check type.
            #   i.e. We can set any type value regardless of argument type.
            self.set_defaults(**config_dict)
        return super().parse_known_args(args, namespace)
