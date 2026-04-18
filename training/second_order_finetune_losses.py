from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class LossBreakdown:
    total: torch.Tensor
    first_dir: torch.Tensor
    first_mag: torch.Tensor
    second_dir: torch.Tensor
    second_mag: torch.Tensor
    second_log_mag: torch.Tensor
    zero_sum: torch.Tensor
    weight_drift: torch.Tensor


class RobustOperatorSupervisionLoss(nn.Module):
    def __init__(
        self,
        *,
        lambda_first_dir: float = 1.0,
        lambda_first_mag: float = 0.25,
        lambda_second_dir: float = 0.75,
        lambda_second_mag: float = 0.15,
        lambda_second_log_mag: float = 0.20,
        lambda_zero_sum: float = 0.0,
        lambda_weight_drift: float = 0.0,
        huber_delta_first_mag: float = 0.05,
        huber_delta_second_mag: float = 0.05,
        huber_delta_zero_sum: float = 0.02,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        self.lambda_first_dir = float(lambda_first_dir)
        self.lambda_first_mag = float(lambda_first_mag)
        self.lambda_second_dir = float(lambda_second_dir)
        self.lambda_second_mag = float(lambda_second_mag)
        self.lambda_second_log_mag = float(lambda_second_log_mag)
        self.lambda_zero_sum = float(lambda_zero_sum)
        self.lambda_weight_drift = float(lambda_weight_drift)
        self.huber_delta_first_mag = float(huber_delta_first_mag)
        self.huber_delta_second_mag = float(huber_delta_second_mag)
        self.huber_delta_zero_sum = float(huber_delta_zero_sum)
        self.eps = float(eps)

    def _direction_loss(self, pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        pred_n = pred / (pred.norm(dim=-1, keepdim=True) + self.eps)
        gt_n = gt / (gt.norm(dim=-1, keepdim=True) + self.eps)
        cos = (pred_n * gt_n).sum(dim=-1).clamp(-1.0, 1.0)
        return (1.0 - cos).mean()

    def _mag_huber(self, pred: torch.Tensor, gt: torch.Tensor, delta: float) -> torch.Tensor:
        pred_norm = pred.norm(dim=-1)
        gt_norm = gt.norm(dim=-1)
        return F.huber_loss(pred_norm, gt_norm, reduction='mean', delta=delta)

    def _log_mag_loss(self, pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        pred_log = torch.log1p(pred.norm(dim=-1))
        gt_log = torch.log1p(gt.norm(dim=-1))
        return F.huber_loss(pred_log, gt_log, reduction='mean', delta=0.25)

    def _zero_sum_loss(self, row_sum: torch.Tensor) -> torch.Tensor:
        return F.huber_loss(row_sum, torch.zeros_like(row_sum), reduction='mean', delta=self.huber_delta_zero_sum)

    def _weight_drift_loss(
        self,
        current_params: list[torch.Tensor] | None,
        reference_params: list[torch.Tensor] | None,
    ) -> torch.Tensor:
        if not current_params or not reference_params:
            return torch.zeros((), device=reference_params[0].device if reference_params else 'cpu')
        vals = [(a - b).pow(2).mean() for a, b in zip(current_params, reference_params)]
        return torch.stack(vals).mean() if vals else torch.zeros((), device=current_params[0].device)

    def forward(
        self,
        *,
        pred1: torch.Tensor,
        gt1: torch.Tensor,
        pred2: torch.Tensor,
        gt2: torch.Tensor,
        row_sum: torch.Tensor,
        current_params: list[torch.Tensor] | None = None,
        reference_params: list[torch.Tensor] | None = None,
        return_stats: bool = False,
    ):
        first_dir = self._direction_loss(pred1, gt1)
        first_mag = self._mag_huber(pred1, gt1, self.huber_delta_first_mag)
        second_dir = self._direction_loss(pred2, gt2)
        second_mag = self._mag_huber(pred2, gt2, self.huber_delta_second_mag)
        second_log_mag = self._log_mag_loss(pred2, gt2)
        zero_sum = self._zero_sum_loss(row_sum)
        weight_drift = self._weight_drift_loss(current_params, reference_params)

        total = (
            self.lambda_first_dir * first_dir
            + self.lambda_first_mag * first_mag
            + self.lambda_second_dir * second_dir
            + self.lambda_second_mag * second_mag
            + self.lambda_second_log_mag * second_log_mag
            + self.lambda_zero_sum * zero_sum
            + self.lambda_weight_drift * weight_drift
        )

        if not return_stats:
            return total

        with torch.no_grad():
            pred1_norm = pred1.norm(dim=-1)
            gt1_norm = gt1.norm(dim=-1)
            pred2_norm = pred2.norm(dim=-1)
            gt2_norm = gt2.norm(dim=-1)
            cos1 = F.cosine_similarity(pred1, gt1, dim=-1)
            cos2 = F.cosine_similarity(pred2, gt2, dim=-1)
            stats = {
                'loss': float(total.item()),
                'first_dir_loss': float(first_dir.item()),
                'first_mag_loss': float(first_mag.item()),
                'second_dir_loss': float(second_dir.item()),
                'second_mag_loss': float(second_mag.item()),
                'second_log_mag_loss': float(second_log_mag.item()),
                'zero_sum_loss': float(zero_sum.item()),
                'weight_drift_loss': float(weight_drift.item()),
                'pred1_norm_mean': float(pred1_norm.mean().item()),
                'gt1_norm_mean': float(gt1_norm.mean().item()),
                'pred2_norm_mean': float(pred2_norm.mean().item()),
                'gt2_norm_mean': float(gt2_norm.mean().item()),
                'first_cos_mean': float(cos1.mean().item()),
                'second_cos_mean': float(cos2.mean().item()),
                'row_sum_abs_mean': float(row_sum.abs().mean().item()),
            }
        return total, stats
