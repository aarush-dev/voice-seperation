import torch
import torch.nn as nn
import numpy as np

from math import ceil
from itertools import permutations
from dataclasses import dataclass, field, fields
from loguru import logger
from sr_corrnet.utils.decorators import logger_wraps
from sr_corrnet.utils import util_stft


# Utility functions
def l2norm(mat, keepdim=False):
    return torch.norm(mat, dim=-1, keepdim=keepdim)

def l1norm(mat, keepdim=False):
    return torch.norm(mat, dim=-1, keepdim=keepdim, p=1)


       
@logger_wraps()
class PIT_SISNR_mag(nn.Module):
    def __init__(self, scale_inv: bool, device: torch.device):
        super().__init__()
        self.device = device
        self.scale_inv = scale_inv
        self.stft = util_stft.STFT(frame_length=512, frame_shift=256, device=self.device, normalize=True)

    def forward(self, estims, targets, eps=1.0e-10, prior_idx=None):
        assert len(estims) == len(targets), "The number of estimated sources and target sources must be the same."
        n_spks = len(estims)

        def _SDR_loss(permute, batch_idx=None):
            loss_for_permute = []
            for s, t in enumerate(permute):
                est = estims[s]
                src = targets[t]

                # If batch_idx is provided, select specific batch samples
                if batch_idx is not None:
                    est = est[[batch_idx]]
                    src = src[[batch_idx]]

                est_zm = est - torch.mean(input=est, dim=-1, keepdim=True)
                src_zm = src - torch.mean(input=src, dim=-1, keepdim=True)

                if self.scale_inv:
                    scale_factor = torch.sum(est * src, dim=-1, keepdim=True) / (l2norm(l2norm(src, keepdim=True),keepdim=True)**2 + eps)
                    src_zm_scale = scale_factor * src_zm

                est_stft = self.stft(est_zm, cplx=True)
                src_stft = self.stft(src_zm_scale, cplx=True)
                est_mag = torch.sqrt(est_stft.real**2 + est_stft.imag**2)
                src_mag = torch.sqrt(src_stft.real**2 + src_stft.imag**2)

                utt_loss = - 20 * torch.log10(eps + l2norm(l2norm(src_mag)) / (l2norm(l2norm(est_mag - src_mag)) + eps))
                utt_loss = torch.clamp(utt_loss, min=-30)
                
                loss_for_permute.append(utt_loss)
            return sum(loss_for_permute)/n_spks
        if prior_idx is not None:
            # Handle both single permutation and batch-wise permutations
            if isinstance(prior_idx, list) and len(prior_idx) > 0 and isinstance(prior_idx[0], list):
                # Batch-wise permutations: compute loss for each batch with its specific permutation
                batch_losses = [_SDR_loss(perm, batch_idx=b) for b, perm in enumerate(prior_idx)]
                min_perutt = torch.stack(batch_losses)
            else:
                # Single permutation for all batches
                min_perutt = _SDR_loss(prior_idx)
        else:
            pscore = torch.cat([_SDR_loss(p) for p in permutations(range(n_spks))])
            min_perutt, _ = torch.min(pscore, dim=0)

        return torch.mean(min_perutt)


@logger_wraps()
class PIT_SISNR_time(nn.Module):
    def __init__(self, scale_inv: bool, device: torch.device):
        super().__init__()
        self.device = device
        self.scale_inv = scale_inv

    def forward(self, estims, targets, eps=1.0e-10, return_perm_idx=False):
        assert len(estims) == len(targets), "The number of estimated sources and target sources must be the same."
        n_spks = len(estims)
        # Pairwise [n_est, n_ref] SI-SNR matrix computed once; permutations are then
        # scored by indexing into it. O(N^2) heavy work instead of O(N!*N).
        est = torch.stack(list(estims), dim=0)      # [n, ..., T]
        src = torch.stack(list(targets), dim=0)     # [n, ..., T]
        est_zm = est - torch.mean(est, dim=-1, keepdim=True)
        src_zm = src - torch.mean(src, dim=-1, keepdim=True)
        e = est_zm.unsqueeze(1)                     # [n, 1, ..., T]
        r = src_zm.unsqueeze(0)                     # [1, n, ..., T]
        r_scale = r
        if self.scale_inv:
            scale_factor = torch.sum(e * r, dim=-1, keepdim=True) / (l2norm(r, keepdim=True)**2 + eps)
            r_scale = scale_factor * r
        pair_loss = - 20 * torch.log10(eps + l2norm(r_scale) / (l2norm(e - r_scale) + eps))
        pair_loss = torch.clamp(pair_loss, min=-30)  # [n_est, n_ref, ...]

        perms = torch.tensor(list(permutations(range(n_spks))), device=pair_loss.device)  # [n!, n]
        rows = torch.arange(n_spks, device=pair_loss.device).unsqueeze(0).expand_as(perms)
        pscore = pair_loss[rows, perms].mean(dim=1)  # [n!, ...]
        indices = [list(p) for p in permutations(range(n_spks))]
        min_perutt, min_idx = torch.min(pscore, dim=0)
        if return_perm_idx:
            # Handle batch-wise indexing for permutation indices
            if min_idx.dim() > 0:  # batch size > 1
                batch_indices = [indices[idx.item()] for idx in min_idx]
                return torch.mean(min_perutt), batch_indices
            else:  # batch size = 1
                return torch.mean(min_perutt), indices[min_idx.item()]
        else:
            return torch.mean(min_perutt)


@logger_wraps()
class PIT_SISNRi(nn.Module):
    def __init__(self, scale_inv: bool, device: torch.device):
        super().__init__()
        self.device = device
        self.scale_inv = scale_inv
    
    def forward(self, estims, targets, input, eps=1.0e-20):
        assert len(estims) == len(targets), "The number of estimated sources and target sources must be the same."
        n_spks = len(estims)
        input_zm = input - torch.mean(input, dim=-1, keepdim=True)

        # Same pairwise-matrix trick as PIT_SISNR_time. The mixture term depends only
        # on the reference index, so summed over any full permutation it is constant.
        est = torch.stack(list(estims), dim=0)      # [n, ..., T]
        src = torch.stack(list(targets), dim=0)     # [n, ..., T]
        est_zm = est - torch.mean(est, dim=-1, keepdim=True)
        src_zm = src - torch.mean(src, dim=-1, keepdim=True)
        e = est_zm.unsqueeze(1)                     # [n, 1, ..., T]
        r = src_zm.unsqueeze(0)                     # [1, n, ..., T]
        r_s = r
        if self.scale_inv:
            factor = torch.sum(e * r, dim=-1, keepdim=True) / (l2norm(r, keepdim=True)**2 + eps)
            r_s = factor * r
        pair_est = 20 * torch.log10(eps + l2norm(r_s) / (l2norm(e - r_s) + eps))  # [n_est, n_ref, ...]

        src_zm_x = src_zm
        if self.scale_inv:
            src_zm_x = torch.sum(input_zm.unsqueeze(0) * src_zm, dim=-1, keepdim=True) / (l2norm(src_zm, keepdim=True)**2 + eps) * src_zm
        loss_in = 20 * torch.log10(eps + l2norm(src_zm_x) / (l2norm(input_zm.unsqueeze(0) - src_zm_x) + eps))  # [n_ref, ...]

        perms = torch.tensor(list(permutations(range(n_spks))), device=pair_est.device)  # [n!, n]
        rows = torch.arange(n_spks, device=pair_est.device).unsqueeze(0).expand_as(perms)
        pscore = pair_est[rows, perms].mean(dim=1) - loss_in.mean(dim=0)  # [n!, ...]
        max_perutt, max_idx = torch.max(pscore, dim=0)
        return torch.mean(max_perutt)
