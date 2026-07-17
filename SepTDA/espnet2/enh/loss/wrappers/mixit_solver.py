"""Mixture Invariant Training (MixIT) loss wrapper.

MixIT (Wisdom et al., 2020) trains a separator to output more estimated
sources than there are references (``num_inf = 2 * num_ref``), typically by
mixing together two independent noisy mixtures and asking the model to
separate the combined signal into ``num_inf`` tracks. Since there is no
per-track reference to match against, the estimated tracks are instead
partitioned into ``num_ref`` groups (by summation) and each group is
compared against one reference. As in PIT, the partitioning is not known in
advance, so every possible way of assigning each of the ``num_inf`` estimated
tracks to one of the ``num_ref`` groups is tried, and the assignment that
minimizes the loss is kept -- this is a many-to-one generalization of the
one-to-one permutation search used in :class:`~espnet2.enh.loss.wrappers.pit_solver.PITSolver`.
"""
import itertools
from typing import Dict, List, Tuple, Union

import torch
from torch_complex.tensor import ComplexTensor

from espnet2.enh.layers.complex_utils import einsum as complex_einsum
from espnet2.enh.layers.complex_utils import stack as complex_stack
from espnet2.enh.loss.criterions.abs_loss import AbsEnhLoss
from espnet2.enh.loss.wrappers.abs_wrapper import AbsLossWrapper

SignalList = Union[List[torch.Tensor], List[ComplexTensor]]


class MixITSolver(AbsLossWrapper):
    """Mixture Invariant Training Solver."""

    def __init__(
        self,
        criterion: AbsEnhLoss,
        weight: float = 1.0,
    ):
        """Mixture Invariant Training Solver.

        Args:
            criterion (AbsEnhLoss): an instance of AbsEnhLoss
            weight (float): weight (between 0 and 1) of current loss
                for multi-task learning.
        """
        super().__init__()
        self.criterion = criterion
        self.weight = weight

    @property
    def name(self):
        return "mixit"

    def _complex_einsum(self, equation: str, *operands) -> ComplexTensor:
        """``torch.einsum``-equivalent for (mixed real/complex) ``ComplexTensor`` operands."""
        for op in operands:
            if not isinstance(op, ComplexTensor):
                op = ComplexTensor(op, torch.zeros_like(op))
        return complex_einsum(equation, *operands)

    def _stack_signals_and_pick_einsum(
        self, ref: SignalList, inf: SignalList, num_ref: int
    ):
        """Stack the per-track signal lists into batched tensors.

        Real- and complex-valued signals need different stacking/einsum
        implementations (``torch_complex`` tensors are not native PyTorch
        tensors), so this also selects the matching einsum function.

        Returns:
            ref_tensor: (Batch, num_ref, ...) the first ``num_ref`` references
                (the remaining references, if any, are unused padding).
            inf_tensor: (Batch, num_inf, ...) all estimated tracks.
            einsum_fn: einsum callable compatible with the chosen tensor type.
        """
        is_complex = isinstance(ref[0], ComplexTensor)
        assert is_complex == isinstance(inf[0], ComplexTensor)

        if not is_complex:
            ref_tensor = torch.stack(ref[:num_ref], dim=1)
            inf_tensor = torch.stack(inf, dim=1)
            einsum_fn = torch.einsum
        else:
            ref_tensor = complex_stack(ref[:num_ref], dim=1)
            inf_tensor = complex_stack(inf, dim=1)
            einsum_fn = self._complex_einsum
        return ref_tensor, inf_tensor, einsum_fn

    @staticmethod
    def _enumerate_mixture_matrices(
        num_ref: int, num_inf: int, device, dtype
    ) -> torch.Tensor:
        """Enumerate every way to partition ``num_inf`` tracks into ``num_ref`` groups.

        Each of the ``num_ref ^ num_inf`` candidate assignments maps every
        estimated track to exactly one of the ``num_ref`` groups; encoded as
        a 0/1 "mixture matrix" that sums the tracks belonging to each group.

        Returns:
            (num_ref ^ num_inf, num_ref, num_inf) stack of mixture matrices.
        """
        all_assignments = list(itertools.product(range(num_ref), repeat=num_inf))
        return torch.stack(
            [
                torch.nn.functional.one_hot(
                    torch.tensor(assignment, dtype=torch.int64, device=device),
                    num_classes=num_ref,
                ).transpose(1, 0)
                for assignment in all_assignments
            ],
            dim=0,
        ).to(dtype)

    def forward(
        self,
        ref: SignalList,
        inf: SignalList,
        others: Dict = {},
    ) -> Tuple[torch.Tensor, Dict, Dict]:
        """MixIT solver.

        Args:
            ref (List[torch.Tensor]): [(batch, ...), ...] x n_spk
            inf (List[torch.Tensor]): [(batch, ...), ...] x n_est
        Returns:
            loss: (torch.Tensor): minimum loss with the best permutation
            stats: dict, for collecting training status
            others: dict, in this PIT solver, permutation order will be returned
        """
        num_inf = len(inf)
        num_ref = num_inf // 2
        device = ref[0].device

        ref_tensor, inf_tensor, einsum_fn = self._stack_signals_and_pick_einsum(
            ref, inf, num_ref
        )

        # all permutation assignments:
        #   [(0, 0, 0, 0), (0, 0, 0, 1), (0, 0, 1, 0), ..., (1, 1, 1, 1)]
        all_mixture_matrix = self._enumerate_mixture_matrices(
            num_ref, num_inf, device, inf_tensor.dtype
        )  # (num_ref ^ num_inf, num_ref, num_inf)

        # For every candidate assignment, sum the estimated tracks that fall
        # into each of the num_ref groups.
        # (num_ref ^ num_inf, batch, num_ref, seq_len, ...)
        if inf_tensor.dim() == 3:
            est_sum_mixture = einsum_fn("ari,bil->abrl", all_mixture_matrix, inf_tensor)
        elif inf_tensor.dim() > 3:
            est_sum_mixture = einsum_fn(
                "ari,bil...->abrl...", all_mixture_matrix, inf_tensor
            )

        losses = []
        for i in range(all_mixture_matrix.shape[0]):
            losses.append(
                sum(
                    [
                        self.criterion(ref_tensor[:, s], est_sum_mixture[i, :, s])
                        for s in range(num_ref)
                    ]
                )
                / num_ref
            )
        losses = torch.stack(losses, dim=0)  # (num_ref ^ num_inf, batch)

        # Pick, independently for each utterance in the batch, the grouping
        # assignment that minimizes the average per-group loss.
        loss, best_assignment_index = torch.min(losses, dim=0)  # (batch)
        loss = loss.mean()
        perm = torch.index_select(all_mixture_matrix, 0, best_assignment_index)

        if perm.is_complex():
            perm = perm.real

        stats = dict()
        stats[f"{self.criterion.name}_{self.name}"] = loss.detach()

        return loss.mean(), stats, {"perm": perm}
