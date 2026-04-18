from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class RobustSecondDerivativeLoss(nn.Module):
    """
    Direct second-derivative operator loss.

    Main signal:
        robust vector residual between D X and x''(s)

    Small stabilizers:
        - cosine loss on direction
        - log-norm loss on magnitude
        - optional row-sum / weight L2 regularization

    Robust term is pseudo-Huber on per-sample vector residual norm:
        delta^2 * (sqrt(1 + ||r||^2 / delta^2) - 1)
    """

    def __init__(
        self,
        *,
        lambda_vec: float = 1.0,
        lambda_cos: float = 0.02,
        lambda_log: float = 0.01,
        lambda_rowsum: float = 0.0,
        lambda_weight_l2: float = 0.0,
        huber_delta: float = 0.05,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        self.lambda_vec = float(lambda_vec)
        self.lambda_cos = float(lambda_cos)
        self.lambda_log = float(lambda_log)
        self.lambda_rowsum = float(lambda_rowsum)
        self.lambda_weight_l2 = float(lambda_weight_l2)
        self.huber_delta = float(huber_delta)
        self.eps = float(eps)

    def _pseudo_huber_vector(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        residual = pred - target
        r = torch.linalg.norm(residual, dim=-1)
        delta_t = pred.new_tensor(self.huber_delta)
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
        pred2: torch.Tensor,
        gt2: torch.Tensor,
        effective_weights: torch.Tensor,
        return_stats: bool = False,
    ):
        vec = self._pseudo_huber_vector(pred2, gt2).mean()
        cos_loss = self._cosine_loss(pred2, gt2)
        log_loss = self._log_norm_loss(pred2, gt2)

        row_sum = effective_weights.sum(dim=-1)
        row_sum_loss = torch.mean(row_sum ** 2)
        weight_l2 = torch.mean(effective_weights ** 2)

        loss = (
            self.lambda_vec * vec
            + self.lambda_cos * cos_loss
            + self.lambda_log * log_loss
            + self.lambda_rowsum * row_sum_loss
            + self.lambda_weight_l2 * weight_l2
        )

        if not return_stats:
            return loss

        with torch.no_grad():
            pred2_norm = torch.linalg.norm(pred2, dim=-1)
            gt2_norm = torch.linalg.norm(gt2, dim=-1)
            stats = {
                'loss': float(loss.item()),
                'vec2_loss': float(vec.item()),
                'cos2_loss': float(cos_loss.item()),
                'log2_loss': float(log_loss.item()),
                'rowsum_loss': float(row_sum_loss.item()),
                'weight_l2_loss': float(weight_l2.item()),
                'pred2_norm_mean': float(pred2_norm.mean().item()),
                'pred2_norm_median': float(pred2_norm.median().item()),
                'gt2_norm_mean': float(gt2_norm.mean().item()),
                'gt2_norm_median': float(gt2_norm.median().item()),
                'row_sum_mean': float(row_sum.mean().item()),
                'row_sum_abs_mean': float(row_sum.abs().mean().item()),
            }
        return loss, stats
