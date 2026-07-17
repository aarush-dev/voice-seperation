"""Permutation Invariant Training (PIT) loss wrapper.

Speech separation models output an unordered set of estimated signals: there
is nothing that forces "output track 0" to correspond to "reference speaker
0". Permutation Invariant Training (Yu et al., 2017; Kolbaek et al., 2017)
solves this by trying every possible assignment between reference and
estimated tracks, computing the criterion for each assignment, and training
on whichever assignment happens to give the lowest loss for each utterance.
Because the "correct" permutation is picked *after* seeing the losses, the
network is never penalized for choosing an arbitrary output ordering -- only
for how well its (best-matched) outputs reconstruct the references.
"""
from collections import defaultdict
from itertools import permutations
from typing import Callable, Dict, List, Tuple

import torch

from espnet2.enh.loss.criterions.abs_loss import AbsEnhLoss
from espnet2.enh.loss.wrappers.abs_wrapper import AbsLossWrapper


class PITSolver(AbsLossWrapper):
    """Permutation Invariant Training Solver."""

    def __init__(
        self,
        criterion: AbsEnhLoss,
        weight: float = 1.0,
        independent_perm: bool = True,
        flexible_numspk: bool = False,
    ):
        """Permutation Invariant Training Solver.

        Args:
            criterion (AbsEnhLoss): an instance of AbsEnhLoss
            weight (float): weight (between 0 and 1) of current loss
                for multi-task learning.
            independent_perm (bool):
                If True, PIT will be performed in forward to find the best permutation;
                If False, the permutation from the last LossWrapper output will be
                inherited.
                NOTE (wangyou): You should be careful about the ordering of loss
                    wrappers defined in the yaml config, if this argument is False.
            flexible_numspk (bool):
                If True, num_spk will be taken from inf to handle flexible numbers of
                speakers. This is because ref may include dummy data in this case.
        """
        super().__init__()
        self.criterion = criterion
        self.weight = weight
        self.independent_perm = independent_perm
        self.flexible_numspk = flexible_numspk

    def forward(
        self, ref: List[torch.Tensor], inf: List[torch.Tensor], others: Dict = {}
    ) -> Tuple[torch.Tensor, Dict, Dict]:
        """PITSolver forward.

        Args:
            ref (List[torch.Tensor]): [(batch, ...), ...] x n_spk
            inf (List[torch.Tensor]): [(batch, ...), ...]

        Returns:
            loss: (torch.Tensor): minimum loss with the best permutation
            stats: dict, for collecting training status
            others: dict, in this PIT solver, permutation order will be returned
        """
        perm = others["perm"] if "perm" in others else None

        if not self.flexible_numspk:
            assert len(ref) == len(inf), (len(ref), len(inf))
            num_spk = len(ref)
        else:
            num_spk = len(inf)

        stats = defaultdict(list)
        criterion_call = self._make_recording_criterion(stats)

        if self.independent_perm or perm is None:
            # No usable permutation was handed down from a previous loss
            # wrapper (or the caller explicitly wants a fresh search): find,
            # independently for this criterion, the best assignment between
            # references and estimates for every utterance in the batch.
            loss, perm = self._search_best_permutation(
                ref, inf, num_spk, criterion_call, stats
            )
        else:
            # Reuse the per-utterance permutation computed by an earlier
            # loss wrapper instead of searching again. This keeps multiple
            # loss terms consistent with a single choice of output ordering.
            loss = self._loss_for_given_permutation(ref, inf, perm, criterion_call)

        loss = loss.mean()

        for k, v in stats.items():
            stats[k] = torch.stack(v, dim=1).mean()
        stats[self.criterion.name] = loss.detach()

        return loss.mean(), dict(stats), {"perm": perm}

    def _make_recording_criterion(
        self, stats: Dict[str, list]
    ) -> Callable[..., torch.Tensor]:
        """Wrap ``self.criterion`` so every call also records its ``.stats``.

        Some criterions (e.g. :class:`FrequencyDomainCrossEntropy`) attach a
        ``stats`` dict of extra metrics (like accuracy) to themselves as a
        side effect of ``forward``. Since PIT evaluates the criterion once
        per (reference, estimate) pair for *every* candidate permutation, we
        must capture those side-channel stats at each call site so they can
        later be filtered down to only the winning permutation.
        """

        def call(*args, **kwargs):
            loss = self.criterion(*args, **kwargs)
            for k, v in getattr(self.criterion, "stats", {}).items():
                stats[k].append(v)
            return loss

        return call

    def _permutation_loss(
        self,
        ref: List[torch.Tensor],
        inf: List[torch.Tensor],
        permutation: Tuple[int, ...],
        criterion_call: Callable[..., torch.Tensor],
    ) -> torch.Tensor:
        """Average criterion value when assigning ``inf[t]`` to ``ref[s]``.

        Args:
            permutation: a candidate assignment, where the value at index
                ``s`` is the estimate index ``t`` matched to reference ``s``.
        Returns:
            (Batch,) loss averaged over the ``num_spk`` matched pairs.
        """
        return sum(
            criterion_call(ref[s], inf[t]) for s, t in enumerate(permutation)
        ) / len(permutation)

    def _search_best_permutation(
        self,
        ref: List[torch.Tensor],
        inf: List[torch.Tensor],
        num_spk: int,
        criterion_call: Callable[..., torch.Tensor],
        stats: Dict[str, list],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Try every permutation and keep the best one, per utterance.

        Evaluates the criterion for all ``num_spk!`` reference/estimate
        assignments, stacks the resulting per-utterance losses into a
        (Batch, num_spk!) matrix, and takes the per-row (per-utterance)
        minimum. The winning permutation may differ across utterances in
        the same batch.

        Returns:
            loss: (Batch,) loss under each utterance's best permutation.
            perm: (Batch, num_spk) the winning permutation per utterance.
        """
        device = ref[0].device
        all_permutations = list(permutations(range(num_spk)))
        # (Batch, num_permutations)
        losses = torch.stack(
            [
                self._permutation_loss(ref, inf, p, criterion_call)
                for p in all_permutations
            ],
            dim=1,
        )
        loss, best_perm_index = torch.min(losses, dim=1)
        perm = torch.index_select(
            torch.tensor(all_permutations, device=device, dtype=torch.long),
            0,
            best_perm_index,
        )
        self._select_stats_for_best_permutation(
            stats, num_spk, len(all_permutations), best_perm_index
        )
        return loss, perm

    @staticmethod
    def _select_stats_for_best_permutation(
        stats: Dict[str, list],
        num_spk: int,
        num_permutations: int,
        best_perm_index: torch.Tensor,
    ) -> None:
        """Discard side-channel stats recorded for non-winning permutations.

        ``stats[k]`` holds one entry per criterion call, i.e.
        ``num_spk * num_permutations`` entries in call order. This reshapes
        that flat list into (Batch, num_permutations, num_spk, ...),
        averages over the ``num_spk`` speaker slots to get one value per
        permutation, and gathers only the value belonging to
        ``best_perm_index`` for each utterance -- mirroring exactly which
        permutation was chosen for the loss itself.
        """
        for k, v in stats.items():
            # (B, num_spk * num_permutations, ...)
            stacked = torch.stack(v, dim=1)
            batch_size, length, *rest = stacked.shape
            assert length == num_spk * num_permutations, (length, num_spk)
            per_perm = stacked.view(
                batch_size, length // num_spk, num_spk, *rest
            ).mean(2)
            if per_perm.dim() > 2:
                broadcast_shape = [1 for _ in rest]
                index = best_perm_index.view(
                    best_perm_index.shape[0], 1, *broadcast_shape
                ).expand(-1, -1, *rest)
            else:
                index = best_perm_index.unsqueeze(1)
            stats[k] = per_perm.gather(1, index.to(device=per_perm.device)).unbind(1)

    def _loss_for_given_permutation(
        self,
        ref: List[torch.Tensor],
        inf: List[torch.Tensor],
        perm: torch.Tensor,
        criterion_call: Callable[..., torch.Tensor],
    ) -> torch.Tensor:
        """Evaluate the criterion under a permutation fixed per utterance.

        Unlike :meth:`_search_best_permutation`, no search happens here:
        ``perm[batch]`` was already decided (e.g. by an earlier loss
        wrapper) and is applied as-is, one utterance at a time.

        Args:
            perm: (Batch, num_spk) permutation to apply for each utterance.
        Returns:
            (Batch,) loss under the given permutation.
        """
        return torch.tensor(
            [
                torch.tensor(
                    [
                        criterion_call(
                            ref[s][batch].unsqueeze(0), inf[t][batch].unsqueeze(0)
                        )
                        for s, t in enumerate(p)
                    ]
                ).mean()
                for batch, p in enumerate(perm)
            ]
        )
