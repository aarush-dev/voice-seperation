"""Loss wrapper for Deep Clustering-style embedding losses (no permutation search)."""
from typing import Dict, List, Tuple

import torch

from espnet2.enh.loss.criterions.abs_loss import AbsEnhLoss
from espnet2.enh.loss.wrappers.abs_wrapper import AbsLossWrapper


class DPCLSolver(AbsLossWrapper):
    """Wraps a Deep Clustering criterion (e.g. ``FrequencyDomainDPCL``).

    Deep Clustering losses compare the Gram matrix of a learned T-F
    embedding against an oracle affinity target built from all references
    at once (see :class:`~espnet2.enh.loss.criterions.tf_domain.FrequencyDomainDPCL`).
    Because the affinity target does not depend on any assumed
    speaker-to-output ordering, this loss is permutation-free by
    construction -- no permutation search is needed here, unlike the PIT
    solvers.
    """

    def __init__(self, criterion: AbsEnhLoss, weight: float = 1.0):
        super().__init__()
        self.criterion = criterion
        self.weight = weight

    def forward(
        self, ref: List[torch.Tensor], inf: List[torch.Tensor], others: Dict = {}
    ) -> Tuple[torch.Tensor, Dict, Dict]:
        """A naive DPCL solver

        Args:
            ref (List[torch.Tensor]): [(batch, ...), ...] x n_spk
            inf (List[torch.Tensor]): [(batch, ...), ...]
            others (List): other data included in this solver
                e.g. "tf_embedding" learned embedding of all T-F bins (B, T * F, D)

        Returns:
            loss: (torch.Tensor): minimum loss with the best permutation
            stats: (dict), for collecting training status
            others: reserved
        """
        assert "tf_embedding" in others

        loss = self.criterion(ref, others["tf_embedding"]).mean()

        stats = dict()
        stats[self.criterion.name] = loss.detach()

        return loss.mean(), stats, {}
