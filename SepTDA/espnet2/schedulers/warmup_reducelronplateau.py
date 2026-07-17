"""ReduceLROnPlateau (with Warm up) learning rate scheduler module."""
from typing import Any, Dict, List, Optional, Sequence, Union

import torch
from torch import inf
from typeguard import check_argument_types

from espnet2.schedulers.abs_scheduler import (
    AbsBatchStepScheduler,
    AbsValEpochStepScheduler,
)


class WarmupReduceLROnPlateau(AbsBatchStepScheduler, AbsValEpochStepScheduler):
    """The WarmupReduceLROnPlateau scheduler.

    Selected via the "warmupreducelr" name in YAML training configs
    (see ``scheduler_classes`` in ``espnet2/tasks/abs_task.py``).

    This scheduler is the combination of WarmupLR and ReduceLROnPlateau:

    WarmupLR:
        lr = optimizer.lr * warmup_step ** 0.5
             * min(step ** -0.5, step * warmup_step ** -1.5)
    WarmupReduceLROnPlateau:
        if step <= warmup_step:
            lr = optimizer.lr * warmup_step ** 0.5
                 * min(step ** -0.5, step * warmup_step ** -1.5)
        else:
            lr = (
                optimizer.lr * factor
                if no improvement for a 'patience' number of epochs
                else optimizer.lr
            )

    Note that the maximum lr equals to optimizer.lr in this scheduler.

    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        # for WarmupLR
        warmup_steps: Union[int, float] = 25000,
        # for ReduceLROnPlateau
        mode="min",
        factor=0.1,
        patience=10,
        threshold=1e-4,
        threshold_mode="rel",
        cooldown=0,
        min_lr=0,
        eps=1e-8,
        verbose=False,
    ):
        """Initialize the scheduler.

        Args:
            optimizer: Optimizer whose learning rate is scheduled.
            warmup_steps: Number of steps over which the learning rate
                ramps up before the ReduceLROnPlateau phase begins.
            mode: One of "min" or "max". In "min" mode, the lr is reduced
                once the monitored metric stops decreasing; in "max" mode,
                once it stops increasing.
            factor: Multiplicative factor by which the learning rate is
                reduced. Must be < 1.0.
            patience: Number of epochs with no improvement after which the
                learning rate is reduced.
            threshold: Threshold used together with `threshold_mode` to
                decide whether a metric value counts as an improvement.
            threshold_mode: One of "rel" or "abs", controlling how
                `threshold` is interpreted.
            cooldown: Number of epochs to wait before resuming normal
                operation after the learning rate has been reduced.
            min_lr: A scalar or list of per-group lower bounds on the
                learning rate.
            eps: Minimal decay applied to the learning rate; updates smaller
                than this are ignored.
            verbose: If True, print a message each time the learning rate is
                reduced.
        """
        assert check_argument_types()
        self.warmup_steps = warmup_steps
        self.step_num = 0
        self.lr_scale = warmup_steps**-1
        self.base_lrs = self._collect_base_lrs(optimizer)

        if factor >= 1.0:
            raise ValueError("Factor should be < 1.0.")
        self.factor = factor

        # Attach optimizer
        self.optimizer = optimizer

        self.min_lrs = self._normalize_min_lrs(min_lr, optimizer)

        self.patience = patience
        self.verbose = verbose
        self.cooldown = cooldown
        self.cooldown_counter = 0
        self.mode = mode
        self.threshold = threshold
        self.threshold_mode = threshold_mode
        self.best = None
        self.num_bad_epochs = None
        self.mode_worse = None  # the worse value for the chosen mode
        self.eps = eps
        self.last_epoch = 0
        self._init_is_better(
            mode=mode, threshold=threshold, threshold_mode=threshold_mode
        )
        self._reset()

    @staticmethod
    def _collect_base_lrs(optimizer: torch.optim.Optimizer) -> List[float]:
        """Ensure each param group has an `initial_lr` and collect them.

        Args:
            optimizer: Optimizer whose param groups provide the base
                learning rates.

        Returns:
            The `initial_lr` of each param group, in order.
        """
        for group in optimizer.param_groups:
            if "initial_lr" not in group:
                group.setdefault("initial_lr", group["lr"])
        return [group["initial_lr"] for group in optimizer.param_groups]

    @staticmethod
    def _normalize_min_lrs(
        min_lr: Union[float, Sequence[float]], optimizer: torch.optim.Optimizer
    ) -> List[float]:
        """Broadcast `min_lr` into a per-param-group list.

        Args:
            min_lr: A single lower bound applied to all param groups, or a
                sequence with one bound per param group.
            optimizer: Optimizer whose param groups determine the expected
                length of `min_lr` when it is a sequence.

        Returns:
            A list of lower bounds, one per param group.
        """
        if isinstance(min_lr, list) or isinstance(min_lr, tuple):
            if len(min_lr) != len(optimizer.param_groups):
                raise ValueError(
                    "expected {} min_lrs, got {}".format(
                        len(optimizer.param_groups), len(min_lr)
                    )
                )
            return list(min_lr)
        else:
            return [min_lr] * len(optimizer.param_groups)

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(warmup_steps={self.warmup_steps}, "
            f"mode={self.mode}, factor={self.factor}, patience={self.patience}"
        )

    def step(self, metrics: Optional[float] = None, epoch: Optional[int] = None):
        """Advance the scheduler by one step.

        If `metrics` is None, this behaves as a per-batch WarmupLR step
        (called during the warmup phase). Otherwise, this behaves as a
        per-epoch ReduceLROnPlateau step driven by the given metric.

        Args:
            metrics: The monitored validation metric for this epoch, or
                None during batch-wise warmup stepping.
            epoch: The epoch index, if known.
        """
        if metrics is None:
            # WarmupLR
            self.step_num += 1
            if self.step_num <= self.warmup_steps:
                for param_group, lr in zip(self.optimizer.param_groups, self.base_lrs):
                    param_group["lr"] = lr * self.lr_scale * self.step_num
        else:
            # ReduceLROnPlateau
            self._step_reducelronplateau(metrics, epoch=epoch)

    def _reset(self):
        """Resets num_bad_epochs counter and cooldown counter."""
        self.best = self.mode_worse
        self.cooldown_counter = 0
        self.num_bad_epochs = 0

    def _step_reducelronplateau(
        self, metrics: Optional[float] = None, epoch: Optional[int] = None
    ):
        """Update `num_bad_epochs`/cooldown state and reduce lr if needed.

        Args:
            metrics: The monitored validation metric for this epoch.
            epoch: The epoch index; if None, inferred from `last_epoch + 1`.
        """
        # convert `metrics` to float, in case it's a zero-dim Tensor
        current = float(metrics)
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch

        if self.is_better(current, self.best):
            self.best = current
            self.num_bad_epochs = 0
        else:
            self.num_bad_epochs += 1

        if self.in_cooldown:
            self.cooldown_counter -= 1
            self.num_bad_epochs = 0  # ignore any bad epochs in cooldown

        if self.num_bad_epochs > self.patience:
            self._reduce_lr(epoch)
            self.cooldown_counter = self.cooldown
            self.num_bad_epochs = 0

        self._last_lr = [group["lr"] for group in self.optimizer.param_groups]

    def _reduce_lr(self, epoch: int):
        """Multiply each param group's lr by `factor`, floored at `min_lrs`.

        Args:
            epoch: The current epoch index, used only for the verbose log
                message.
        """
        self.eps = 1e-8
        for group_idx, param_group in enumerate(self.optimizer.param_groups):
            old_lr = float(param_group["lr"])
            new_lr = max(old_lr * self.factor, self.min_lrs[group_idx])
            if old_lr - new_lr > self.eps:
                param_group["lr"] = new_lr
                if self.verbose:
                    epoch_str = ("%.2f" if isinstance(epoch, float) else "%.5d") % epoch
                    print(
                        "Epoch {}: reducing learning rate"
                        " of group {} to {:.4e}.".format(epoch_str, group_idx, new_lr)
                    )

    @property
    def in_cooldown(self) -> bool:
        """Whether the scheduler is currently within its cooldown period."""
        return self.cooldown_counter > 0

    def is_better(self, a: float, best: Optional[float]) -> bool:
        """Decide whether metric value `a` improves on `best`.

        Comparison depends on `self.mode` ("min"/"max") and
        `self.threshold_mode` ("rel"/"abs"):
            min + rel: a < best * (1 - threshold)
            min + abs: a < best - threshold
            max + rel: a > best * (1 + threshold)
            max + abs: a > best + threshold

        Args:
            a: Candidate metric value.
            best: Current best metric value.

        Returns:
            True if `a` counts as an improvement over `best`.
        """
        if self.mode == "min" and self.threshold_mode == "rel":
            rel_epsilon = 1.0 - self.threshold
            return a < best * rel_epsilon

        elif self.mode == "min" and self.threshold_mode == "abs":
            return a < best - self.threshold

        elif self.mode == "max" and self.threshold_mode == "rel":
            rel_epsilon = self.threshold + 1.0
            return a > best * rel_epsilon

        else:  # mode == 'max' and epsilon_mode == 'abs':
            return a > best + self.threshold

    def _init_is_better(self, mode: str, threshold: float, threshold_mode: str):
        """Validate mode/threshold_mode and set `mode_worse` accordingly.

        Args:
            mode: One of "min" or "max".
            threshold: Threshold value used by `is_better`.
            threshold_mode: One of "rel" or "abs".
        """
        if mode not in {"min", "max"}:
            raise ValueError("mode " + mode + " is unknown!")
        if threshold_mode not in {"rel", "abs"}:
            raise ValueError("threshold mode " + threshold_mode + " is unknown!")

        if mode == "min":
            self.mode_worse = inf
        else:  # mode == 'max':
            self.mode_worse = -inf

        self.mode = mode
        self.threshold = threshold
        self.threshold_mode = threshold_mode

    def state_dict(self) -> Dict[str, Any]:
        """Return the scheduler's state as a dict, excluding the optimizer."""
        return {
            key: value for key, value in self.__dict__.items() if key != "optimizer"
        }

    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Restore the scheduler's state from a dict produced by state_dict()."""
        self.__dict__.update(state_dict)
        self._init_is_better(
            mode=self.mode, threshold=self.threshold, threshold_mode=self.threshold_mode
        )
