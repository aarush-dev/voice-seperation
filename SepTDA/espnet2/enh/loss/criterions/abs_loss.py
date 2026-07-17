"""Abstract base class for speech enhancement / separation loss criterions."""
from abc import ABC, abstractmethod

import torch

EPS = torch.finfo(torch.get_default_dtype()).eps


class AbsEnhLoss(torch.nn.Module, ABC):
    """Base class for all Enhancement loss modules.

    A concrete criterion computes a per-utterance scalar loss from a single
    reference/estimate pair. Combining several references and estimates
    (e.g. searching over speaker permutations) is the responsibility of the
    loss *wrappers* in ``espnet2.enh.loss.wrappers``, not of the criterion
    itself.
    """

    @property
    def name(self) -> str:
        """Key under which this criterion's value is logged by the reporter.

        This string is written into training logs and stats dictionaries,
        so subclasses must keep it stable across refactors.
        """
        return NotImplementedError

    @property
    def only_for_test(self) -> bool:
        """Whether this criterion should only be evaluated at inference/validation time."""
        return False

    @abstractmethod
    def forward(
        self,
        ref: torch.Tensor,
        inf: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the loss between a reference and an estimate.

        Args:
            ref: reference signal.
            inf: inferred (estimated) signal.

        Returns:
            loss: Tensor of shape (Batch,), one loss value per utterance.
        """
        raise NotImplementedError
