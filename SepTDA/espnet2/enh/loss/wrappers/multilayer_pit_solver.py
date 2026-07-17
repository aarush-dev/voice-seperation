"""PIT loss wrapper for models that produce intermediate (multi-layer) estimates.

Some separators (e.g. iterative/recurrent refinement architectures) produce a
separate estimate at each of several internal layers/iterations, in addition
to (or instead of) a single final estimate. This wrapper applies
:class:`~espnet2.enh.loss.wrappers.pit_solver.PITSolver` to each layer's
estimate against the same reference, and combines the per-layer PIT losses
into a single scalar for backpropagation.
"""
from typing import Dict, List, Optional, Sequence, Tuple, Union

import torch

from espnet2.enh.loss.criterions.abs_loss import AbsEnhLoss
from espnet2.enh.loss.wrappers.abs_wrapper import AbsLossWrapper
from espnet2.enh.loss.wrappers.pit_solver import PITSolver


class MultiLayerPITSolver(AbsLossWrapper):
    """Multi-Layer Permutation Invariant Training Solver."""

    def __init__(
        self,
        criterion: AbsEnhLoss,
        weight: float = 1.0,
        independent_perm: bool = True,
        layer_weights: Optional[Sequence[float]] = None,
    ):
        """Multi-Layer Permutation Invariant Training Solver.

        Compute the PIT loss given inferences of multiple layers and a single reference.
        It also support single inference and single reference in evaluation stage.

        Args:
            criterion (AbsEnhLoss): an instance of AbsEnhLoss
            weight (float): weight (between 0 and 1) of current loss
                for multi-task learning.
            independent_perm (bool):
                If True, PIT will be performed in forward to find the best permutation;
                If False, the permutation from the last LossWrapper output will be
                inherited.
                Note: You should be careful about the ordering of loss
                wrappers defined in the yaml config, if this argument is False.
            layer_weights (Optional[List[float]]): weights for each layer
                If not None, the loss of each layer will be weighted-summed using the
                specified weights.
        """
        super().__init__()
        self.criterion = criterion
        self.weight = weight
        self.independent_perm = independent_perm
        self.solver = PITSolver(criterion, weight, independent_perm)
        self.layer_weights = layer_weights

    def forward(
        self,
        ref: List[torch.Tensor],
        infs: Union[List[torch.Tensor], List[List[torch.Tensor]]],
        others: Dict = {},
    ) -> Tuple[torch.Tensor, Dict, Dict]:
        """Permutation invariant training solver.

        Args:
            ref (List[torch.Tensor]): [(batch, ...), ...] x n_spk
            infs (Union[List[torch.Tensor], List[List[torch.Tensor]]]):
                [(batch, ...), ...]

        Returns:
            loss: (torch.Tensor): minimum loss with the best permutation
            stats: dict, for collecting training status
            others: dict, in this PIT solver, permutation order will be returned
        """
        # In single-layer case, the model only estimates waveforms in the last layer.
        # The shape of infs is List[torch.Tensor]
        if not isinstance(infs[0], (tuple, list)) and len(infs) == len(ref):
            loss, stats, others = self.solver(ref, infs, others)
            return loss, stats, others

        # In multi-layer case, weighted-sum the PIT loss of each layer
        # The shape of ins is List[List[torch.Tensor]]
        return self._weighted_sum_layer_losses(ref, infs, others)

    def _weighted_sum_layer_losses(
        self,
        ref: List[torch.Tensor],
        infs: List[List[torch.Tensor]],
        others: Dict,
    ) -> Tuple[torch.Tensor, Dict, Dict]:
        """Run PIT independently on each layer's estimate and combine the losses.

        Each layer is scored against the same ``ref`` via a fresh call to
        the shared :class:`PITSolver` (so each layer may pick its own best
        permutation, unless ``independent_perm=False``). The per-layer
        losses are then combined either with the user-provided
        ``layer_weights``, or -- if none were given -- with linearly
        increasing weights ``(idx + 1) / len(infs)`` that put more emphasis
        on later (presumably more refined) layers, before averaging over the
        number of layers.
        """
        total_loss = 0.0
        stats: Dict = {}
        for idx, inf in enumerate(infs):
            loss, stats, others = self.solver(ref, inf, others)
            if self.layer_weights is not None:
                layer_weight = self.layer_weights[idx]
            else:
                layer_weight = (idx + 1) * (1.0 / len(infs))
            total_loss = total_loss + loss * layer_weight
        total_loss = total_loss / len(infs)
        return total_loss, stats, others
