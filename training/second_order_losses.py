from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SecondOrderOperatorLoss(nn.Module):
    def __init__(
        self,
        *,
        lambda_mse: float = 1.0,
        lambda_cos: float = 0.1,
        lambda_log_norm: float = 0.1,
        lambda_rowsum: float = 1.0,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        self.lambda_mse = float(lambda_mse)
        self.lambda_cos = float(lambda_cos)
        self.lambda_log_norm = float(lambda_log_norm)
        self.lambda_rowsum = float(lambda_rowsum)
        self.eps = float(eps)

    def forward(
        self,
        *,
        pred_second: torch.Tensor,
        target_second: torch.Tensor,
        weights: torch.Tensor,
        return_stats: bool = False,
    ):
        mse = F.mse_loss(pred_second, target_second)

        pred_norm = pred_second.norm(dim=-1).clamp_min(self.eps)
        target_norm = target_second.norm(dim=-1).clamp_min(self.eps)

        cosine = F.cosine_similarity(pred_second, target_second, dim=-1, eps=self.eps)
        cos_loss = (1.0 - cosine).mean()

        log_norm_error = (torch.log(pred_norm) - torch.log(target_norm)).pow(2).mean()

        row_sum = weights.sum(dim=-1)
        rowsum_penalty = row_sum.pow(2).mean()

        loss = (
            self.lambda_mse * mse
            + self.lambda_cos * cos_loss
            + self.lambda_log_norm * log_norm_error
            + self.lambda_rowsum * rowsum_penalty
        )

        if not return_stats:
            return loss

        with torch.no_grad():
            angle = torch.rad2deg(torch.acos(cosine.clamp(-1.0, 1.0)))
            norm_ratio = pred_norm / target_norm
            sign_mse = torch.mean((pred_second + target_second) ** 2)

            stats = {
                "loss": float(loss.item()),
                "mse2": float(mse.item()),
                "cos_loss2": float(cos_loss.item()),
                "log_norm_mse2": float(log_norm_error.item()),
                "rowsum_penalty": float(rowsum_penalty.item()),
                "cos2_mean": float(cosine.mean().item()),
                "abs_cos2_mean": float(cosine.abs().mean().item()),
                "angle2_deg_mean": float(angle.mean().item()),
                "norm_ratio_mean": float(norm_ratio.mean().item()),
                "pred_norm2_mean": float(pred_norm.mean().item()),
                "target_norm2_mean": float(target_norm.mean().item()),
                "weight_l2_mean": float(weights.norm(dim=-1).mean().item()),
                "sign_mse2": float(sign_mse.item()),
            }

        return loss, stats
