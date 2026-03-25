from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class InvariantOperatorLoss(nn.Module):
    """
    Invariant-only loss.

    Components:
    - invariance: anchor and positive embeddings must match
    - variance: avoid collapse by keeping per-dimension std above gamma
    - covariance: decorrelate dimensions
    - negative separation: push negatives away from the anchor embedding
    - weight regularization: keep stencil from exploding

    No derivative label enters this loss.
    """

    def __init__(
        self,
        *,
        lambda_inv: float = 25.0,
        lambda_var: float = 25.0,
        lambda_cov: float = 1.0,
        lambda_neg: float = 1.0,
        lambda_reg: float = 1e-4,
        variance_target: float = 1.0,
        negative_margin: float = 0.25,
    ) -> None:
        super().__init__()
        self.lambda_inv = float(lambda_inv)
        self.lambda_var = float(lambda_var)
        self.lambda_cov = float(lambda_cov)
        self.lambda_neg = float(lambda_neg)
        self.lambda_reg = float(lambda_reg)
        self.variance_target = float(variance_target)
        self.negative_margin = float(negative_margin)

    @staticmethod
    def _variance_term(z: torch.Tensor, target: float) -> torch.Tensor:
        z = z - z.mean(dim=0, keepdim=True)
        std = torch.sqrt(z.var(dim=0, unbiased=False) + 1e-4)
        return F.relu(target - std).mean()

    @staticmethod
    def _covariance_term(z: torch.Tensor) -> torch.Tensor:
        z = z - z.mean(dim=0, keepdim=True)
        n, d = z.shape
        if n <= 1:
            return z.new_tensor(0.0)
        cov = (z.T @ z) / max(n - 1, 1)
        off_diag = cov - torch.diag(torch.diag(cov))
        return off_diag.pow(2).sum() / d

    def forward(
        self,
        *,
        embedding_anchor: torch.Tensor,
        embedding_positive: torch.Tensor,
        embedding_negatives: torch.Tensor,
        weights_anchor: torch.Tensor,
        return_stats: bool = False,
    ):
        inv_loss = F.mse_loss(embedding_anchor, embedding_positive)

        z_all = torch.cat([embedding_anchor, embedding_positive], dim=0)
        var_loss = self._variance_term(z_all, self.variance_target)
        cov_loss = self._covariance_term(z_all)

        anchor_expand = embedding_anchor.unsqueeze(1)
        neg_dist = torch.norm(anchor_expand - embedding_negatives, dim=-1)
        neg_loss = F.relu(self.negative_margin - neg_dist).mean()

        reg_loss = weights_anchor.pow(2).mean()

        loss = (
            self.lambda_inv * inv_loss
            + self.lambda_var * var_loss
            + self.lambda_cov * cov_loss
            + self.lambda_neg * neg_loss
            + self.lambda_reg * reg_loss
        )

        if not return_stats:
            return loss

        with torch.no_grad():
            stats = {
                'loss': float(loss.item()),
                'inv_loss': float(inv_loss.item()),
                'var_loss': float(var_loss.item()),
                'cov_loss': float(cov_loss.item()),
                'neg_loss': float(neg_loss.item()),
                'reg_loss': float(reg_loss.item()),
                'embedding_norm_mean': float(embedding_anchor.norm(dim=-1).mean().item()),
                'negative_distance_mean': float(neg_dist.mean().item()),
                'weights_norm_mean': float(weights_anchor.norm(dim=-1).mean().item()),
                'weights_sum_mean': float(weights_anchor.sum(dim=-1).mean().item()),
                'weights_abs_mean': float(weights_anchor.abs().mean().item()),
            }
        return loss, stats
