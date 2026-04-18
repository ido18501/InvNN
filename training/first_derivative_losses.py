from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


class FirstDerivativeLoss(nn.Module):
    """
    Local fine-tuning loss for Stage 1:

        L = ||pred - gt||^2 + alpha * (||pred|| - target_norm)^2

    Intended use:
    - keep the pretrained operator architecture fixed
    - remove contrastive / equivariant training terms entirely
    - calibrate local first-derivative magnitude while preserving direction
    """

    def __init__(self, *, alpha_norm: float = 1.0, target_norm: float = 1.0) -> None:
        super().__init__()
        self.alpha_norm = float(alpha_norm)
        self.target_norm = float(target_norm)

    def forward(
        self,
        *,
        pred: torch.Tensor,
        gt: torch.Tensor,
        return_stats: bool = False,
    ):
        mse = F.mse_loss(pred, gt)
        pred_norm = pred.norm(dim=-1)
        norm_error = pred_norm - self.target_norm
        norm_loss = torch.mean(norm_error.square())
        loss = mse + self.alpha_norm * norm_loss

        if not return_stats:
            return loss

        with torch.no_grad():
            gt_norm = gt.norm(dim=-1)
            pred_unit = pred / (pred_norm[:, None] + 1e-8)
            gt_unit = gt / (gt_norm[:, None] + 1e-8)
            cosine = (pred_unit * gt_unit).sum(dim=-1).clamp(-1.0, 1.0)
            angle_deg = torch.rad2deg(torch.acos(cosine))
            abs_norm_error = norm_error.abs()

            stats: Dict[str, float] = {
                'loss': float(loss.item()),
                'mse_loss': float(mse.item()),
                'norm_loss': float(norm_loss.item()),
                'cosine_mean': float(cosine.mean().item()),
                'angle_deg_mean': float(angle_deg.mean().item()),
                'pred_norm_mean': float(pred_norm.mean().item()),
                'pred_norm_median': float(pred_norm.median().item()),
                'gt_norm_mean': float(gt_norm.mean().item()),
                'norm_error_mean': float(abs_norm_error.mean().item()),
                'norm_error_median': float(abs_norm_error.median().item()),
            }
        return loss, stats
