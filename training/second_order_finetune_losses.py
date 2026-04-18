from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class SecondOrderLossBreakdown:
    loss: torch.Tensor
    stats: dict[str, float]


class RobustOperatorVectorLoss(nn.Module):
    """
    Main objective: robust vector residuals on first- and second-order operator outputs.

    Hierarchy:
      1) robust vector error on pred1 vs gt1 and pred2 vs gt2
      2) small directional / magnitude stabilizers
      3) optional structural regularization

    The robust term is pseudo-Huber on the per-sample L2 residual:
        delta^2 * (sqrt(1 + ||r||^2 / delta^2) - 1)
    which behaves like MSE near zero and like L1 on large residuals.
    """

    def __init__(
        self,
        *,
        lambda1_vec: float = 1.0,
        lambda2_vec: float = 1.0,
        lambda1_cos: float = 0.05,
        lambda2_cos: float = 0.05,
        lambda2_log: float = 0.02,
        lambda_rowsum: float = 0.0,
        lambda_weight_l2: float = 0.0,
        huber_delta1: float = 0.05,
        huber_delta2: float = 0.05,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        self.lambda1_vec = float(lambda1_vec)
        self.lambda2_vec = float(lambda2_vec)
        self.lambda1_cos = float(lambda1_cos)
        self.lambda2_cos = float(lambda2_cos)
        self.lambda2_log = float(lambda2_log)
        self.lambda_rowsum = float(lambda_rowsum)
        self.lambda_weight_l2 = float(lambda_weight_l2)
        self.huber_delta1 = float(huber_delta1)
        self.huber_delta2 = float(huber_delta2)
        self.eps = float(eps)

    def _pseudo_huber_vector(self, pred: torch.Tensor, target: torch.Tensor, delta: float) -> torch.Tensor:
        residual = pred - target
        r = torch.linalg.norm(residual, dim=-1)
        delta_t = pred.new_tensor(delta)
        return (delta_t ** 2) * (torch.sqrt(1.0 + (r / delta_t) ** 2) - 1.0)

    def _cosine_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_norm = torch.linalg.norm(pred, dim=-1)
        target_norm = torch.linalg.norm(target, dim=-1)
        valid = (pred_norm > self.eps) & (target_norm > self.eps)
        if not torch.any(valid):
            return pred.new_zeros(())
        cos = F.cosine_similarity(pred[valid], target[valid], dim=-1)
        return (1.0 - cos).mean()

    def _log_norm_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_n = torch.linalg.norm(pred, dim=-1)
        target_n = torch.linalg.norm(target, dim=-1)
        return F.smooth_l1_loss(torch.log1p(pred_n), torch.log1p(target_n))

    def forward(
        self,
        *,
        pred1: torch.Tensor,
        gt1: torch.Tensor,
        pred2: torch.Tensor,
        gt2: torch.Tensor,
        effective_weights: torch.Tensor,
        return_stats: bool = False,
    ):
        vec1 = self._pseudo_huber_vector(pred1, gt1, self.huber_delta1).mean()
        vec2 = self._pseudo_huber_vector(pred2, gt2, self.huber_delta2).mean()

        cos1 = self._cosine_loss(pred1, gt1)
        cos2 = self._cosine_loss(pred2, gt2)
        log2 = self._log_norm_loss(pred2, gt2)

        row_sum = effective_weights.sum(dim=-1)
        row_sum_loss = torch.mean(row_sum ** 2)
        weight_l2 = torch.mean(effective_weights ** 2)

        loss = (
            self.lambda1_vec * vec1
            + self.lambda2_vec * vec2
            + self.lambda1_cos * cos1
            + self.lambda2_cos * cos2
            + self.lambda2_log * log2
            + self.lambda_rowsum * row_sum_loss
            + self.lambda_weight_l2 * weight_l2
        )

        if not return_stats:
            return loss

        with torch.no_grad():
            pred1_norm = torch.linalg.norm(pred1, dim=-1)
            gt1_norm = torch.linalg.norm(gt1, dim=-1)
            pred2_norm = torch.linalg.norm(pred2, dim=-1)
            gt2_norm = torch.linalg.norm(gt2, dim=-1)

            stats = {
                'loss': float(loss.item()),
                'vec1_loss': float(vec1.item()),
                'vec2_loss': float(vec2.item()),
                'cos1_loss': float(cos1.item()),
                'cos2_loss': float(cos2.item()),
                'log2_loss': float(log2.item()),
                'rowsum_loss': float(row_sum_loss.item()),
                'weight_l2_loss': float(weight_l2.item()),
                'pred1_norm_mean': float(pred1_norm.mean().item()),
                'gt1_norm_mean': float(gt1_norm.mean().item()),
                'pred2_norm_mean': float(pred2_norm.mean().item()),
                'gt2_norm_mean': float(gt2_norm.mean().item()),
                'pred1_norm_median': float(pred1_norm.median().item()),
                'gt1_norm_median': float(gt1_norm.median().item()),
                'pred2_norm_median': float(pred2_norm.median().item()),
                'gt2_norm_median': float(gt2_norm.median().item()),
                'row_sum_mean': float(row_sum.mean().item()),
                'row_sum_abs_mean': float(row_sum.abs().mean().item()),
            }
        return loss, stats
