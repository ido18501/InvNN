from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class EquivariantVectorLoss(nn.Module):
    """
    Objective for a patch-conditioned weight vector w(X) in R^K.

    Optimized terms:
    - equivariance of the predicted vector: f(TX) ~= A f(X)
    - contrastive InfoNCE on the predicted operator action itself
    - optional L2 regularization on the predicted weights
    - a small norm-floor penalty to avoid trivial zero-output collapse
    """

    def __init__(
        self,
        *,
        temperature: float = 0.1,
        lambda_nce: float = 1.0,
        lambda_eq: float = 1.0,
        lambda_reg: float = 1e-4,
        lambda_pred_norm: float = 0.05,
        pred_norm_target: float = 0.25,
    ) -> None:
        super().__init__()
        self.temperature = float(temperature)
        self.lambda_nce = float(lambda_nce)
        self.lambda_eq = float(lambda_eq)
        self.lambda_reg = float(lambda_reg)
        self.lambda_pred_norm = float(lambda_pred_norm)
        self.pred_norm_target = float(pred_norm_target)

    def _info_nce(
            self,
            pred_anchor_equivariant: torch.Tensor,
            pred_positive: torch.Tensor,
            pred_negatives: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        q = pred_anchor_equivariant
        p = pred_positive
        n = pred_negatives

        pos_logits = torch.sum(q * p, dim=-1, keepdim=True) / self.temperature
        neg_logits = torch.einsum('bd,bmd->bm', q, n) / self.temperature

        logits = torch.cat([pos_logits, neg_logits], dim=1)
        labels = torch.zeros(logits.shape[0], dtype=torch.long, device=logits.device)
        nce_loss = F.cross_entropy(logits, labels)

        pos_sim = torch.sum(q * p, dim=-1)
        neg_sim = torch.mean(torch.einsum('bd,bmd->bm', q, n), dim=-1)
        return nce_loss, pos_sim, neg_sim

    def forward(
        self,
        *,
        pred_anchor_equivariant: torch.Tensor,
        pred_positive: torch.Tensor,
        pred_negatives: torch.Tensor,
        weights: torch.Tensor,
        return_stats: bool = False,
    ):
        eq_raw_loss = F.mse_loss(pred_positive, pred_anchor_equivariant)

        nce_loss, pos_sim, neg_sim = self._info_nce(
            pred_anchor_equivariant=pred_anchor_equivariant,
            pred_positive=pred_positive,
            pred_negatives=pred_negatives,
        )

        reg_loss = weights.pow(2).mean()

        anchor_norm = pred_anchor_equivariant.norm(dim=-1)
        positive_norm = pred_positive.norm(dim=-1)
        pred_norm_loss = 0.5 * (
            F.relu(self.pred_norm_target - anchor_norm).mean()
            + F.relu(self.pred_norm_target - positive_norm).mean()
        )

        loss = (
            self.lambda_nce * nce_loss
            + self.lambda_eq * eq_raw_loss
            + self.lambda_reg * reg_loss
            + self.lambda_pred_norm * pred_norm_loss
        )

        if not return_stats:
            return loss

        with torch.no_grad():
            eq_cos = F.cosine_similarity(pred_positive, pred_anchor_equivariant, dim=-1)
            norm_mse = F.mse_loss(
                pred_positive.norm(dim=-1),
                pred_anchor_equivariant.norm(dim=-1),
            )
            weight_norm = weights.norm(dim=-1)

            stats = {
                'loss': float(loss.item()),
                'nce_loss': float(nce_loss.item()),
                'eq_raw_loss': float(eq_raw_loss.item()),
                'eq_norm_mse': float(norm_mse.item()),
                'eq_cos_mean': float(eq_cos.mean().item()),
                'positive_similarity_mean': float(pos_sim.mean().item()),
                'negative_similarity_mean': float(neg_sim.mean().item()),
                'weight_l2_mean': float(weight_norm.mean().item()),
                'pred_norm_floor_loss': float(pred_norm_loss.item()),
                'pred_anchor_norm_mean': float(anchor_norm.mean().item()),
                'pred_positive_norm_mean': float(positive_norm.mean().item()),
            }
        return loss, stats