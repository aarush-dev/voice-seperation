from abc import ABC, abstractmethod
from typing import Dict, Tuple

import torch


class AbsESPnetModel(torch.nn.Module, ABC):
    """The common abstract class among each tasks

    "ESPnetModel" is referred to a class which inherits torch.nn.Module,
    and makes the dnn-models forward as its member field,
    a.k.a delegate pattern,
    and defines "loss", "stats", and "weight" for the task.

    If you intend to implement new task in ESPNet,
    the model must inherit this class.
    In other words, the "mediator" objects between
    our training system and the your task class are
    just only these three values, loss, stats, and weight.

    Example:
        >>> from espnet2.tasks.abs_task import AbsTask
        >>> class YourESPnetModel(AbsESPnetModel):
        ...     def forward(self, input, input_lengths):
        ...         ...
        ...         return loss, stats, weight
        >>> class YourTask(AbsTask):
        ...     @classmethod
        ...     def build_model(cls, args: argparse.Namespace) -> YourESPnetModel:
    """

    @abstractmethod
    def forward(
        self, **batch: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
        """Compute the training loss for one batch.

        Args:
            **batch: Model inputs, one tensor per collated field (e.g.
                ``speech``, ``speech_lengths``, ``text``, ``text_lengths``).

        Returns:
            (loss, stats, weight):
                * loss: Scalar loss tensor to backpropagate.
                * stats: Named values to log (e.g. sub-losses, accuracy).
                * weight: Batch weight (e.g. batch size) used by
                    :class:`espnet2.train.trainer.Trainer` to combine stats
                    across devices/accumulation steps.
        """
        raise NotImplementedError

    @abstractmethod
    def collect_feats(self, **batch: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Compute intermediate features for stats collection (no loss/backward).

        Used by the "collect_stats" stage to determine feature shapes/
        statistics (e.g. for normalization) before actual training starts.
        """
        raise NotImplementedError
