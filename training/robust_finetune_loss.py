from __future__ import annotations

from typing import Iterable

import torch
import torch.nn as nn


class RobustDerivativeLoss(nn.Module):
    """
    Robust direct vector loss for derivative fine-tuning.

    Main objective:
      - pred1 vs gt1 : first derivative vector match
      - pred2 vs gt2 : second derivative vector match

    Robustification:
      pseudo-Huber on the vector residual norm

        rho_delta(r) = delta^2 * (sqrt(1 + (r/delta)^2) - 1)

    Why this choice:
      - quadratic near zero  -> behaves like MSE on the bulk
      - linear for large residuals -> outliers do not dominate
      - no division by GT norms -> stable, avoids NaNs from tiny denominators
    """

    def __init__(
        self,
        *,
        lambda_first: float = 1.0,
        lambda_second: float = 0.25,
        delta_first: float = 0.05,
        delta_second: float = 0.25,
        lambda_row_sum: float = 0.0,
        lambda_prox: float = 0.0,
    ) -> None:
        super().__init__()
        self.lambda_first = float(lambda_first)
        self.lambda_second = float(lambda_second)
        self.delta_first = float(delta_first)
        self.delta_second = float(delta_second)
        self.lambda_row_sum = float(lambda_row_sum)
        self.lambda_prox = float(lambda_prox)

        if self.delta_first <= 0.0:
            raise ValueError('delta_first must be positive.')
        if self.delta_second <= 0.0:
            raise ValueError('delta_second must be positive.')

    @staticmethod
    def _cosine_and_angle(pred: torch.Tensor, gt: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        pred_n = pred / (pred.norm(dim=-1, keepdim=True) + 1e-8)
        gt_n = gt / (gt.norm(dim=-1, keepdim=True) + 1e-8)
        cos = (pred_n * gt_n).sum(dim=-1).clamp(-1.0, 1.0)
        angle = torch.rad2deg(torch.acos(cos))
        return cos, angle

    @staticmethod
    def _pseudo_huber_from_vector_residual(
        pred: torch.Tensor,
        gt: torch.Tensor,
        delta: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        resid = pred - gt
        resid_norm = torch.linalg.vector_norm(resid, dim=-1)
        scaled = resid_norm / delta
        robust = (delta ** 2) * (torch.sqrt(1.0 + scaled * scaled) - 1.0)
        robust_mean = robust.mean()
        mse_mean = (resid * resid).mean()
        return robust_mean, mse_mean, resid_norm

    @staticmethod
    def _prox_penalty(
        current_params: Iterable[torch.Tensor] | None,
        reference_params: Iterable[torch.Tensor] | None,
    ) -> torch.Tensor | None:
        if current_params is None or reference_params is None:
            return None

        cur = list(current_params)
        ref = list(reference_params)
        if len(cur) != len(ref):
            raise ValueError('current_params and reference_params must have the same length.')
        if len(cur) == 0:
            return None

        vals = []
        for a, b in zip(cur, ref):
            vals.append(torch.mean((a - b.detach()) ** 2))
        return torch.stack(vals).mean()

    def forward(
        self,
        *,
        pred1: torch.Tensor,
        gt1: torch.Tensor,
        pred2: torch.Tensor,
        gt2: torch.Tensor,
        row_sum: torch.Tensor | None = None,
        current_params: Iterable[torch.Tensor] | None = None,
        reference_params: Iterable[torch.Tensor] | None = None,
        return_stats: bool = False,
    ):
        robust1, mse1, resid1 = self._pseudo_huber_from_vector_residual(pred1, gt1, self.delta_first)
        robust2, mse2, resid2 = self._pseudo_huber_from_vector_residual(pred2, gt2, self.delta_second)

        loss = self.lambda_first * robust1 + self.lambda_second * robust2

        row_sum_loss = None
        if row_sum is not None:
            row_sum_loss = torch.mean(row_sum ** 2)
            loss = loss + self.lambda_row_sum * row_sum_loss

        prox_loss = self._prox_penalty(current_params=current_params, reference_params=reference_params)
        if prox_loss is not None:
            loss = loss + self.lambda_prox * prox_loss

        if not return_stats:
            return loss

        with torch.no_grad():
            cos1, ang1 = self._cosine_and_angle(pred1, gt1)
            cos2, ang2 = self._cosine_and_angle(pred2, gt2)

            pred1_norm = pred1.norm(dim=-1)
            gt1_norm = gt1.norm(dim=-1)
            pred2_norm = pred2.norm(dim=-1)
            gt2_norm = gt2.norm(dim=-1)

            stats = {
                'loss': float(loss.item()),
                'robust_first_loss': float(robust1.item()),
                'robust_second_loss': float(robust2.item()),
                'mse1': float(mse1.item()),
                'mse2': float(mse2.item()),
                'cos1': float(cos1.mean().item()),
                'cos2': float(cos2.mean().item()),
                'ang1_deg': float(ang1.mean().item()),
                'ang2_deg': float(ang2.mean().item()),
                'pred1_norm_mean': float(pred1_norm.mean().item()),
                'gt1_norm_mean': float(gt1_norm.mean().item()),
                'pred2_norm_mean': float(pred2_norm.mean().item()),
                'gt2_norm_mean': float(gt2_norm.mean().item()),
                'mag1_ratio': float((pred1_norm / (gt1_norm + 1e-8)).mean().item()),
                'mag2_ratio': float((pred2_norm / (gt2_norm + 1e-8)).mean().item()),
                'resid1_norm_mean': float(resid1.mean().item()),
                'resid2_norm_mean': float(resid2.mean().item()),
            }
            if row_sum_loss is not None:
                stats['row_sum_loss'] = float(row_sum_loss.item())
                stats['row_sum_abs_mean'] = float(row_sum.abs().mean().item())
            if prox_loss is not None:
                stats['prox_loss'] = float(prox_loss.item())
        return loss, stats
