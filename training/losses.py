from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class EquivariantContrastiveOperatorLoss(nn.Module):
    """
    Train an operator through *equivariance*, not invariance.

    Positive pair:
        g(T(W(x)))   <->   g(W(Tx))
    not:
        g(W(x))      <->   g(W(Tx))

    The raw equivariance term uses full vector MSE, so magnitude matters.
    No analytic derivative enters this loss.
    """

    def __init__(
        self,
        *,
        temperature: float = 0.1,
        lambda_nce: float = 1.0,
        lambda_eq: float = 1.0,
        lambda_sum: float = 0.1,
        lambda_reg: float = 1e-4,
    ) -> None:
        super().__init__()
        self.temperature = float(temperature)
        self.lambda_nce = float(lambda_nce)
        self.lambda_eq = float(lambda_eq)
        self.lambda_sum = float(lambda_sum)
        self.lambda_reg = float(lambda_reg)

    @staticmethod
    def _apply_linear_map(transform_matrix: torch.Tensor, vectors: torch.Tensor) -> torch.Tensor:
        return torch.einsum('bij,bj->bi', transform_matrix, vectors)

    def _info_nce(
        self,
        proj_anchor_equivariant: torch.Tensor,
        proj_positive: torch.Tensor,
        proj_negatives: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        proj_anchor_equivariant = F.normalize(proj_anchor_equivariant, dim=-1)
        proj_positive = F.normalize(proj_positive, dim=-1)
        proj_negatives = F.normalize(proj_negatives, dim=-1)

        pos_logits = torch.sum(proj_anchor_equivariant * proj_positive, dim=-1, keepdim=True) / self.temperature
        neg_logits = torch.einsum('bd,bmd->bm', proj_anchor_equivariant, proj_negatives) / self.temperature
        logits = torch.cat([pos_logits, neg_logits], dim=1)
        labels = torch.zeros(logits.shape[0], dtype=torch.long, device=logits.device)
        nce_loss = F.cross_entropy(logits, labels)

        pos_sim = torch.sum(proj_anchor_equivariant * proj_positive, dim=-1)
        neg_sim = torch.mean(torch.einsum('bd,bmd->bm', proj_anchor_equivariant, proj_negatives), dim=-1)
        return nce_loss, pos_sim, neg_sim

    def forward(
        self,
        *,
        vector_anchor: torch.Tensor,
        vector_positive: torch.Tensor,
        proj_anchor_equivariant: torch.Tensor,
        proj_positive: torch.Tensor,
        proj_negatives: torch.Tensor,
        weights_anchor: torch.Tensor,
        transform_matrix: torch.Tensor,
        return_stats: bool = False,
    ):
        target_vector = self._apply_linear_map(transform_matrix, vector_anchor)

        # Full equivariance: direction + magnitude.
        eq_raw_loss = F.mse_loss(vector_positive, target_vector)

        nce_loss, pos_sim, neg_sim = self._info_nce(
            proj_anchor_equivariant=proj_anchor_equivariant,
            proj_positive=proj_positive,
            proj_negatives=proj_negatives,
        )

        sum_loss = weights_anchor.sum(dim=-1).pow(2).mean()
        reg_loss = weights_anchor.pow(2).mean()

        loss = (
            self.lambda_nce * nce_loss
            + self.lambda_eq * eq_raw_loss
            + self.lambda_sum * sum_loss
            + self.lambda_reg * reg_loss
        )

        if not return_stats:
            return loss

        with torch.no_grad():
            eq_cos = F.cosine_similarity(vector_positive, target_vector, dim=-1)
            target_norm = target_vector.norm(dim=-1)
            positive_norm = vector_positive.norm(dim=-1)
            norm_mse = F.mse_loss(positive_norm, target_norm)
            stats = {
                'loss': float(loss.item()),
                'nce_loss': float(nce_loss.item()),
                'eq_raw_loss': float(eq_raw_loss.item()),
                'eq_norm_mse': float(norm_mse.item()),
                'sum_loss': float(sum_loss.item()),
                'reg_loss': float(reg_loss.item()),
                'eq_cos_mean': float(eq_cos.mean().item()),
                'eq_angle_deg_mean': float(torch.rad2deg(torch.acos(eq_cos.clamp(-1.0, 1.0))).mean().item()),
                'proj_pos_sim_mean': float(pos_sim.mean().item()),
                'proj_neg_sim_mean': float(neg_sim.mean().item()),
                'target_vec_norm_mean': float(target_norm.mean().item()),
                'positive_vec_norm_mean': float(positive_norm.mean().item()),
                'weights_norm_mean': float(weights_anchor.norm(dim=-1).mean().item()),
                'weights_sum_mean': float(weights_anchor.sum(dim=-1).mean().item()),
                'weights_abs_mean': float(weights_anchor.abs().mean().item()),
            }
        return loss, stats
