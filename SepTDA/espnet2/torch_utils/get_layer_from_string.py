"""Instantiate a torch.nn layer *class* from its (case-insensitive) name.

Used when a config (e.g. YAML) specifies a layer type as a plain string
(such as "elu" or "ReLU") and the code needs the actual class object
(e.g. `torch.nn.ELU`) to construct it.
"""
import difflib
from types import ModuleType
from typing import List, Type

import torch


def get_layer(l_name: str, library: ModuleType = torch.nn) -> Type:
    """Return layer object handler from library e.g. from torch.nn

    E.g. if l_name=="elu", returns torch.nn.ELU.

    Args:
        l_name (string): Case insensitive name for layer in library (e.g. .'elu').
        library (module): Name of library/module where to search for object handler
        with l_name e.g. "torch.nn".

    Returns:
        layer_handler (object): handler for the requested layer e.g. (torch.nn.ELU)

    """
    all_torch_layers = [x for x in dir(torch.nn)]
    match = [x for x in all_torch_layers if l_name.lower() == x.lower()]
    if len(match) == 0:
        close_matches = _close_matches(l_name, all_torch_layers)
        raise NotImplementedError(
            "Layer with name {} not found in {}.\n Closest matches: {}".format(
                l_name, str(library), close_matches
            )
        )
    elif len(match) > 1:
        close_matches = _close_matches(l_name, all_torch_layers)
        raise NotImplementedError(
            "Multiple matchs for layer with name {} not found in {}.\n "
            "All matches: {}".format(l_name, str(library), close_matches)
        )
    else:
        # valid
        layer_handler = getattr(library, match[0])
        return layer_handler


def _close_matches(l_name: str, candidate_names: List[str]) -> List[str]:
    """Return names from `candidate_names` (lower-cased) close to `l_name`."""
    return difflib.get_close_matches(l_name, [x.lower() for x in candidate_names])
