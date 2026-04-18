from __future__ import annotations

import torch
import torch.nn as nn


class CurvatureVectorLoss(nn.Module):
    """
    Direct supervision:
        W(X) ≈ x''(s)

    Main idea:
    - use a relative error term so huge-curvature samples do not dominate
    - keep a small absolute MSE term so magnitude still matters
    - regularize weights more strongly
    - encourage sum(weights) ≈ 0, as expected for a second-derivative-like stencil
    """

    def __init__(
        self,
        *,
        lambda_rel: float = 1.0,
        lambda_abs: float = 0.05,
        lambda_cos: float = 0.0,
        lambda_reg: float = 1e-2,
        lambda_weight_sum: float = 1e-2,
        tau_scale: float = 1.0,
        tau_min: float = 1e-3,
    ):
        super().__init__()
        self.lambda_rel = float(lambda_rel)
        self.lambda_abs = float(lambda_abs)
        self.lambda_cos = float(lambda_cos)
        self.lambda_reg = float(lambda_reg)
        self.lambda_weight_sum = float(lambda_weight_sum)
        self.tau_scale = float(tau_scale)
        self.tau_min = float(tau_min)

    def forward(
        self,
        *,
        pred: torch.Tensor,        # [B,2]
        gt_second: torch.Tensor,   # [B,2]
        weights: torch.Tensor,     # [B,K]
        return_stats: bool = False,
    ):
        residual = pred - gt_second
        residual_sq = (residual ** 2).sum(dim=-1)          # [B]
        gt_norm = gt_second.norm(dim=-1)                   # [B]
        pred_norm = pred.norm(dim=-1)                      # [B]

        # Robust batch scale for relative weighting
        tau = torch.clamp(gt_norm.detach().median() * self.tau_scale, min=self.tau_min)

        # Main fix: relative error so high-curvature outliers do not dominate
        rel_loss_per_sample = residual_sq / (gt_norm ** 2 + tau ** 2)
        rel_loss = rel_loss_per_sample.mean()

        # Keep some absolute signal for real magnitude fitting
        abs_mse = residual_sq.mean()

        # Optional direction term
        pred_n = pred / (pred_norm.unsqueeze(-1) + 1e-8)
        gt_n = gt_second / (gt_norm.unsqueeze(-1) + 1e-8)
        cos = (pred_n * gt_n).sum(dim=-1).clamp(-1.0, 1.0)
        cos_loss = 1.0 - cos.mean()

        # Regularization
        reg_loss = weights.pow(2).mean()

        # Second-derivative stencil should roughly sum to zero
        weight_sum = weights.sum(dim=-1)
        weight_sum_loss = (weight_sum ** 2).mean()

        loss = (
            self.lambda_rel * rel_loss
            + self.lambda_abs * abs_mse
            + self.lambda_cos * cos_loss
            + self.lambda_reg * reg_loss
            + self.lambda_weight_sum * weight_sum_loss
        )

        if not return_stats:
            return loss

        with torch.no_grad():
            angle = torch.rad2deg(torch.acos(cos))
            norm_ratio = pred_norm / (gt_norm + 1e-8)

            stats = {
                "loss": float(loss.item()),
                "rel_loss": float(rel_loss.item()),
                "abs_mse": float(abs_mse.item()),
                "cosine_mean": float(cos.mean().item()),
                "angle_deg_mean": float(angle.mean().item()),
                "pred_norm_mean": float(pred_norm.mean().item()),
                "gt_norm_mean": float(gt_norm.mean().item()),
                "gt_norm_median": float(gt_norm.median().item()),
                "gt_norm_p90": float(torch.quantile(gt_norm, 0.90).item()),
                "gt_norm_p99": float(torch.quantile(gt_norm, 0.99).item()),
                "gt_norm_max": float(gt_norm.max().item()),
                "pred_norm_ratio_mean": float(norm_ratio.mean().item()),
                "weight_l2_mean": float(weights.norm(dim=-1).mean().item()),
                "weight_sum_abs_mean": float(weight_sum.abs().mean().item()),
                "tau": float(tau.item()),
            }

        return loss, stats