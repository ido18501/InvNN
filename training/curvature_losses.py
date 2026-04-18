from __future__ import annotations

import torch
import torch.nn as nn


class CurvatureVectorLoss(nn.Module):
    """
    Direct supervision:
        W(X) ≈ x''(s)

    Current strategy:
    - relative residual loss: prevents huge-curvature outliers from dominating
    - log-norm matching: pushes the model away from tiny-output collapse
    - optional cosine term
    - mild L2 on weights
    - zero-sum stencil penalty: sum(weights) ≈ 0
    """

    def __init__(
        self,
        *,
        lambda_rel: float = 1.0,
        lambda_mag: float = 0.05,
        lambda_cos: float = 0.0,
        lambda_reg: float = 1e-3,
        lambda_weight_sum: float = 1e-2,
        tau_scale: float = 1.0,
        tau_min: float = 1e-3,
        norm_eps: float = 1e-8,
    ):
        super().__init__()
        self.lambda_rel = float(lambda_rel)
        self.lambda_mag = float(lambda_mag)
        self.lambda_cos = float(lambda_cos)
        self.lambda_reg = float(lambda_reg)
        self.lambda_weight_sum = float(lambda_weight_sum)
        self.tau_scale = float(tau_scale)
        self.tau_min = float(tau_min)
        self.norm_eps = float(norm_eps)

    def forward(
        self,
        *,
        pred: torch.Tensor,        # [B, 2]
        gt_second: torch.Tensor,   # [B, 2]
        weights: torch.Tensor,     # [B, K]
        return_stats: bool = False,
    ):
        pred_norm = pred.norm(dim=-1)                 # [B]
        gt_norm = gt_second.norm(dim=-1)              # [B]
        residual = pred - gt_second
        residual_sq = (residual ** 2).sum(dim=-1)     # [B]

        # Robust batch scale for relative weighting
        tau = torch.clamp(gt_norm.detach().median() * self.tau_scale, min=self.tau_min)

        # Relative residual term
        rel_loss_per_sample = residual_sq / (gt_norm ** 2 + tau ** 2)
        rel_loss = rel_loss_per_sample.mean()

        # Magnitude matching in log-space
        log_pred_norm = torch.log(pred_norm + self.norm_eps)
        log_gt_norm = torch.log(gt_norm + self.norm_eps)
        mag_loss = ((log_pred_norm - log_gt_norm) ** 2).mean()

        # Optional directional term
        pred_n = pred / (pred_norm.unsqueeze(-1) + self.norm_eps)
        gt_n = gt_second / (gt_norm.unsqueeze(-1) + self.norm_eps)
        cos = (pred_n * gt_n).sum(dim=-1).clamp(-1.0, 1.0)
        cos_loss = 1.0 - cos.mean()

        # Mild weight regularization
        reg_loss = weights.pow(2).mean()

        # Second-derivative-like stencil should sum to ~0
        weight_sum = weights.sum(dim=-1)
        weight_sum_loss = (weight_sum ** 2).mean()

        loss = (
            self.lambda_rel * rel_loss
            + self.lambda_mag * mag_loss
            + self.lambda_cos * cos_loss
            + self.lambda_reg * reg_loss
            + self.lambda_weight_sum * weight_sum_loss
        )

        if not return_stats:
            return loss

        with torch.no_grad():
            angle = torch.rad2deg(torch.acos(cos))
            norm_ratio = pred_norm / (gt_norm + self.norm_eps)

            stats = {
                "loss": float(loss.item()),
                "rel_loss": float(rel_loss.item()),
                "mag_loss": float(mag_loss.item()),
                "cosine_mean": float(cos.mean().item()),
                "angle_deg_mean": float(angle.mean().item()),
                "pred_norm_mean": float(pred_norm.mean().item()),
                "pred_norm_median": float(pred_norm.median().item()),
                "gt_norm_mean": float(gt_norm.mean().item()),
                "gt_norm_median": float(gt_norm.median().item()),
                "gt_norm_p90": float(torch.quantile(gt_norm, 0.90).item()),
                "gt_norm_p99": float(torch.quantile(gt_norm, 0.99).item()),
                "gt_norm_max": float(gt_norm.max().item()),
                "pred_norm_ratio_mean": float(norm_ratio.mean().item()),
                "pred_norm_ratio_median": float(norm_ratio.median().item()),
                "weight_l2_mean": float(weights.norm(dim=-1).mean().item()),
                "weight_sum_abs_mean": float(weight_sum.abs().mean().item()),
                "tau": float(tau.item()),
            }

        return loss, stats