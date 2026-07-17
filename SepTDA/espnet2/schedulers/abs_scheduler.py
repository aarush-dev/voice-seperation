"""Abstract base classes for learning rate schedulers.

These classes define the interfaces that espnet2 LR schedulers must follow,
grouped by when ``step()`` is expected to be called during training:

- ``AbsBatchStepScheduler``: ``step()`` is called once per training iteration
  (batch).
- ``AbsEpochStepScheduler``: ``step()`` is called once per epoch.
- ``AbsValEpochStepScheduler``: ``step()`` is called once per epoch and takes
  a validation metric (e.g. ``ReduceLROnPlateau``-style schedulers).

The built-in PyTorch schedulers in ``torch.optim.lr_scheduler`` are
retroactively registered as virtual subclasses of these ABCs below, since
PyTorch itself doesn't expose a common base class that distinguishes
batch-step from epoch-step schedulers.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict

import torch.optim.lr_scheduler as L


class AbsScheduler(ABC):
    """Common interface shared by all espnet2 LR schedulers."""

    @abstractmethod
    def step(self, epoch: int = None):
        """Advance the scheduler by one step (batch or epoch, per subclass)."""
        pass

    @abstractmethod
    def state_dict(self) -> Dict[str, Any]:
        """Return the scheduler's state as a dict, for checkpointing."""
        pass

    @abstractmethod
    def load_state_dict(self, state):
        """Restore the scheduler's state from a dict produced by state_dict()."""
        pass


# If you need to define custom scheduler, please inherit these classes
class AbsBatchStepScheduler(AbsScheduler):
    """Base class for schedulers whose step() is called once per batch."""

    @abstractmethod
    def step(self, epoch: int = None):
        """Advance the scheduler by one training iteration (batch)."""
        pass

    @abstractmethod
    def state_dict(self) -> Dict[str, Any]:
        """Return the scheduler's state as a dict, for checkpointing."""
        pass

    @abstractmethod
    def load_state_dict(self, state):
        """Restore the scheduler's state from a dict produced by state_dict()."""
        pass


class AbsEpochStepScheduler(AbsScheduler):
    """Base class for schedulers whose step() is called once per epoch."""

    @abstractmethod
    def step(self, epoch: int = None):
        """Advance the scheduler by one epoch."""
        pass

    @abstractmethod
    def state_dict(self) -> Dict[str, Any]:
        """Return the scheduler's state as a dict, for checkpointing."""
        pass

    @abstractmethod
    def load_state_dict(self, state):
        """Restore the scheduler's state from a dict produced by state_dict()."""
        pass


class AbsValEpochStepScheduler(AbsEpochStepScheduler):
    """Base class for epoch schedulers that additionally need a metric.

    Used by schedulers such as ``ReduceLROnPlateau`` that decide how to
    adjust the learning rate based on a monitored validation value.
    """

    @abstractmethod
    def step(self, val, epoch: int = None):
        """Advance the scheduler by one epoch given a validation metric.

        Args:
            val: The monitored validation metric for this epoch.
            epoch: The epoch index, if known.
        """
        pass

    @abstractmethod
    def state_dict(self) -> Dict[str, Any]:
        """Return the scheduler's state as a dict, for checkpointing."""
        pass

    @abstractmethod
    def load_state_dict(self, state):
        """Restore the scheduler's state from a dict produced by state_dict()."""
        pass


# Create alias type to check the type
# Note(kamo): Currently PyTorch doesn't provide the base class
# to judge these classes.
AbsValEpochStepScheduler.register(L.ReduceLROnPlateau)
for _epoch_step_scheduler_cls in [
    L.ReduceLROnPlateau,
    L.LambdaLR,
    L.StepLR,
    L.MultiStepLR,
    L.MultiStepLR,
    L.ExponentialLR,
    L.CosineAnnealingLR,
]:
    AbsEpochStepScheduler.register(_epoch_step_scheduler_cls)

AbsBatchStepScheduler.register(L.CyclicLR)
for _batch_step_scheduler_cls in [
    L.OneCycleLR,
    L.CosineAnnealingWarmRestarts,
]:
    AbsBatchStepScheduler.register(_batch_step_scheduler_cls)
