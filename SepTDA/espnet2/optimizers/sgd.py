"""Thin subclass of `torch.optim.SGD` selected by name from training configs.

ESPnet2's task runner instantiates optimizers by looking up a class name
(here, "sgd") from the YAML training config and calling it with keyword
arguments only. `torch.optim.SGD` requires `lr` as a positional argument
with no default, which is incompatible with that instantiation pattern; this
subclass exists solely to give `lr` a default value.
"""

import torch
from typeguard import check_argument_types


class SGD(torch.optim.SGD):
    """Thin inheritance of torch.optim.SGD to bind the required arguments, 'lr'

    Note that
    the arguments of the optimizer invoked by AbsTask.main()
    must have default value except for 'param'.

    I can't understand why only SGD.lr doesn't have the default value.
    """

    def __init__(
        self,
        params,
        lr: float = 0.1,
        momentum: float = 0.0,
        dampening: float = 0.0,
        weight_decay: float = 0.0,
        nesterov: bool = False,
    ):
        """Construct the optimizer.

        Args:
            params: Iterable of parameters or named parameter groups to optimize.
            lr: Learning rate.
            momentum: Momentum factor.
            dampening: Dampening for momentum.
            weight_decay: L2 penalty coefficient.
            nesterov: Whether to enable Nesterov momentum.

        Raises:
            TypeError: If an argument does not match its declared type
                (enforced by `typeguard.check_argument_types`).
        """
        assert check_argument_types()
        super().__init__(
            params,
            lr=lr,
            momentum=momentum,
            dampening=dampening,
            weight_decay=weight_decay,
            nesterov=nesterov,
        )
