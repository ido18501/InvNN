from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class CurvatureVectorLoss(nn.Module):
    """
    Direct supervision:
        W(X) ≈ x''(s)

    Terms:
    - MSE to second derivative (main signal)
    - optional cosine alignment
    - weight regularization
    - optional zero-sum constraint (derivative-like stencil)
    """

    def __init__(
        self,
        *,
        lambda_mse: float = 1.0,
        lambda_cos: float = 0.0,
        lambda_reg: float = 1e-4,
        lambda_weight_sum: float = 1e-3,
    ):
        super().__init__()
        self.lambda_mse = float(lambda_mse)
        self.lambda_cos = float(lambda_cos)
        self.lambda_reg = float(lambda_reg)
        self.lambda_weight_sum = float(lambda_weight_sum)

    def forward(
        self,
        *,
        pred: torch.Tensor,        # [B,2]
        gt_second: torch.Tensor,   # [B,2]
        weights: torch.Tensor,     # [B,K]
        return_stats: bool = False,
    ):
        # ----- main signal -----
        mse_loss = F.mse_loss(pred, gt_second)

        # ----- cosine alignment (optional) -----
        pred_n = pred / (pred.norm(dim=-1, keepdim=True) + 1e-8)
        gt_n = gt_second / (gt_second.norm(dim=-1, keepdim=True) + 1e-8)
        cos = (pred_n * gt_n).sum(dim=-1)
        cos_loss = 1.0 - cos.mean()

        # ----- regularization -----
        reg_loss = weights.pow(2).mean()

        # ----- derivative-like constraint -----
        # For second derivative stencil: sum(weights) ≈ 0
        weight_sum = weights.sum(dim=-1)
        weight_sum_loss = (weight_sum ** 2).mean()

        loss = (
            self.lambda_mse * mse_loss
            + self.lambda_cos * cos_loss
            + self.lambda_reg * reg_loss
            + self.lambda_weight_sum * weight_sum_loss
        )

        if not return_stats:
            return loss

        with torch.no_grad():
            angle = torch.rad2deg(torch.acos(torch.clamp(cos, -1.0, 1.0)))

            stats = {
                "loss": float(loss.item()),
                "mse": float(mse_loss.item()),
                "cosine_mean": float(cos.mean().item()),
                "angle_deg_mean": float(angle.mean().item()),
                "pred_norm_mean": float(pred.norm(dim=-1).mean().item()),
                "gt_norm_mean": float(gt_second.norm(dim=-1).mean().item()),
                "weight_l2_mean": float(weights.norm(dim=-1).mean().item()),
                "weight_sum_abs_mean": float(weight_sum.abs().mean().item()),
            }

        return loss, stats