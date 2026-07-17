"""Abstract base class for loss wrappers (permutation solvers, etc.)."""
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple

import torch


class AbsLossWrapper(torch.nn.Module, ABC):
    """Base class for all Enhancement loss wrapper modules.

    A loss wrapper turns a single-pair :class:`AbsEnhLoss` criterion into a
    multi-speaker training loss. It decides *how* the list of references is
    matched against the list of estimates (e.g. via permutation search) and
    reduces the per-pair losses into one scalar per batch. Wrappers also
    report per-criterion statistics and may pass information (such as the
    chosen permutation) downstream via the ``others`` dict.
    """

    # The weight for the current loss in the multi-task learning.
    # The overall training target will be combined as:
    # loss = weight_1 * loss_1 + ... + weight_N * loss_N
    weight = 1.0

    @abstractmethod
    def forward(
        self,
        ref: List,
        inf: List,
        others: Dict,
    ) -> Tuple[torch.Tensor, Dict, Dict]:
        """Compute the wrapped loss.

        Args:
            ref: list of reference tensors, one per speaker/source.
            inf: list of estimated tensors, one per speaker/source.
            others: auxiliary data shared between loss wrappers, e.g. a
                permutation computed by a previous wrapper, or embeddings
                needed by clustering-based losses.

        Returns:
            loss: scalar Tensor, the batch-averaged loss.
            stats: dict of scalar Tensors for logging.
            others: dict of auxiliary outputs (e.g. ``{"perm": ...}``) to be
                forwarded to subsequent loss wrappers.
        """
        raise NotImplementedError
