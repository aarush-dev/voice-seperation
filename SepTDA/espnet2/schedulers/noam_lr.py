"""Noam learning rate scheduler module."""
import warnings
from typing import List, Union

import torch
from torch.optim.lr_scheduler import _LRScheduler
from typeguard import check_argument_types

from espnet2.schedulers.abs_scheduler import AbsBatchStepScheduler


class NoamLR(_LRScheduler, AbsBatchStepScheduler):
    """The LR scheduler proposed by Noam.

    Selected via the "noamlr" name in YAML training configs
    (see ``scheduler_classes`` in ``espnet2/tasks/abs_task.py``).

    Ref:
        "Attention Is All You Need", https://arxiv.org/pdf/1706.03762.pdf

    The learning rate at step ``s`` (1-indexed) is:
        lr = base_lr * model_size ** -0.5 * min(s ** -0.5, s * warmup_steps ** -1.5)

    i.e. it increases linearly for the first ``warmup_steps`` steps and then
    decreases proportionally to the inverse square root of the step number.

    FIXME(kamo): PyTorch doesn't provide _LRScheduler as public class,
     thus the behaviour isn't guaranteed at forward PyTorch version.

    NOTE(kamo): The "model_size" in original implementation is derived from
     the model, but in this implementation, this parameter is a constant value.
     You need to change it if the model is changed.

    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        model_size: Union[int, float] = 320,
        warmup_steps: Union[int, float] = 25000,
        last_epoch: int = -1,
    ):
        """Initialize the scheduler.

        Args:
            optimizer: Optimizer whose learning rate is scheduled.
            model_size: Model dimensionality used to scale the base
                learning rate.
            warmup_steps: Number of steps over which the learning rate
                ramps up before it starts decaying.
            last_epoch: The index of the last step; -1 means training
                starts from scratch.
        """
        assert check_argument_types()
        self.model_size = model_size
        self.warmup_steps = warmup_steps

        base_lr = list(optimizer.param_groups)[0]["lr"]
        equivalent_warmuplr_lr = self.lr_for_WarmupLR(base_lr)
        warnings.warn(
            f"NoamLR is deprecated. "
            f"Use WarmupLR(warmup_steps={warmup_steps}) "
            f"with Optimizer(lr={equivalent_warmuplr_lr})",
        )

        # __init__() must be invoked before setting field
        # because step() is also invoked in __init__()
        super().__init__(optimizer, last_epoch)

    def lr_for_WarmupLR(self, lr: float) -> float:
        """Convert a NoamLR base learning rate to an equivalent WarmupLR one.

        Args:
            lr: Base learning rate as configured for NoamLR.

        Returns:
            The base learning rate that would make WarmupLR produce the
            same schedule.
        """
        return lr / self.model_size**0.5 / self.warmup_steps**0.5

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(model_size={self.model_size}, "
            f"warmup_steps={self.warmup_steps})"
        )

    def get_lr(self) -> List[float]:
        """Compute the learning rate for the current step, for each param group."""
        step_num = self.last_epoch + 1
        return [
            lr
            * self.model_size**-0.5
            * min(step_num**-0.5, step_num * self.warmup_steps**-1.5)
            for lr in self.base_lrs
        ]
