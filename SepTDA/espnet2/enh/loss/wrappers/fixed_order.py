"""Loss wrapper that pairs references and estimates in a fixed (given) order."""
from collections import defaultdict
from typing import Dict, List, Tuple

import torch

from espnet2.enh.loss.criterions.abs_loss import AbsEnhLoss
from espnet2.enh.loss.wrappers.abs_wrapper import AbsLossWrapper


class FixedOrderSolver(AbsLossWrapper):
    """Pairs ``ref[i]`` with ``inf[i]`` directly, with no permutation search.

    Useful when the separator's output order is already known to match the
    reference order (e.g. target-speaker extraction, or when a previous PIT
    wrapper in the pipeline already fixed the ordering), so re-running a
    permutation search would be wasteful or incorrect.
    """

    def __init__(self, criterion: AbsEnhLoss, weight: float = 1.0):
        super().__init__()
        self.criterion = criterion
        self.weight = weight

    def forward(
        self, ref: List[torch.Tensor], inf: List[torch.Tensor], others: Dict = {}
    ) -> Tuple[torch.Tensor, Dict, Dict]:
        """An naive fixed-order solver

        Args:
            ref (List[torch.Tensor]): [(batch, ...), ...] x n_spk
            inf (List[torch.Tensor]): [(batch, ...), ...]

        Returns:
            loss: (torch.Tensor): minimum loss with the best permutation
            stats: dict, for collecting training status
            others: reserved
        """
        assert len(ref) == len(inf), (len(ref), len(inf))
        num_spk = len(ref)

        loss = 0.0
        stats = defaultdict(list)
        for r, i in zip(ref, inf):
            loss += torch.mean(self.criterion(r, i)) / num_spk
            for k, v in getattr(self.criterion, "stats", {}).items():
                stats[k].append(v)

        for k, v in stats.items():
            stats[k] = torch.stack(v, dim=1).mean()
        stats[self.criterion.name] = loss.detach()

        # Identity permutation: "perm" is still reported so that downstream
        # wrappers (e.g. a following FixedOrderSolver for an auxiliary loss)
        # can rely on it being present in `others`.
        perm = torch.arange(num_spk).unsqueeze(0).repeat(ref[0].size(0), 1)
        return loss.mean(), dict(stats), {"perm": perm}
