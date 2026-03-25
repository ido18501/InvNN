from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class EquivariantMatrixOperatorLoss(nn.Module):
    """
    Objective for a patch-conditioned KxK operator W(X).

    Optimized terms:
    - full-field equivariance on the *action*: W(TX) TX ~= T(W(X) X)
      where only the linear part A of T(p)=Ap+b acts on the output field.
    - equivariant contrastive loss on field signatures
    - row-sum zero per predicted operator, to kill translation bias / constants
    - Frobenius regularization
    - optional locality penalty away from the diagonal

    No derivative labels enter this loss.
    """

    def __init__(
        self,
        *,
        temperature: float = 0.1,
        lambda_nce: float = 1.0,
        lambda_eq: float = 1.0,
        lambda_sum: float = 0.1,
        lambda_reg: float = 1e-4,
        lambda_loc: float = 0.0,
        locality_matrix: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.temperature = float(temperature)
        self.lambda_nce = float(lambda_nce)
        self.lambda_eq = float(lambda_eq)
        self.lambda_sum = float(lambda_sum)
        self.lambda_reg = float(lambda_reg)
        self.lambda_loc = float(lambda_loc)
        if locality_matrix is None:
            self.register_buffer('locality_matrix', torch.empty(0))
        else:
            self.register_buffer('locality_matrix', locality_matrix.float())

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
        field_anchor_equivariant: torch.Tensor,
        field_positive: torch.Tensor,
        proj_anchor_equivariant: torch.Tensor,
        proj_positive: torch.Tensor,
        proj_negatives: torch.Tensor,
        operator_matrix: torch.Tensor,
        return_stats: bool = False,
    ):
        eq_raw_loss = F.mse_loss(field_positive, field_anchor_equivariant)
        nce_loss, pos_sim, neg_sim = self._info_nce(
            proj_anchor_equivariant=proj_anchor_equivariant,
            proj_positive=proj_positive,
            proj_negatives=proj_negatives,
        )

        row_sum = operator_matrix.sum(dim=-1)
        sum_loss = row_sum.pow(2).mean()
        reg_loss = operator_matrix.pow(2).mean()

        if self.locality_matrix.numel() == 0 or self.lambda_loc == 0.0:
            loc_loss = operator_matrix.new_tensor(0.0)
        else:
            loc_penalty = self.locality_matrix.to(operator_matrix.device).unsqueeze(0)
            loc_loss = (loc_penalty * operator_matrix.pow(2)).mean()

        loss = (
            self.lambda_nce * nce_loss
            + self.lambda_eq * eq_raw_loss
            + self.lambda_sum * sum_loss
            + self.lambda_reg * reg_loss
            + self.lambda_loc * loc_loss
        )

        if not return_stats:
            return loss

        with torch.no_grad():
            eq_cos = F.cosine_similarity(
                field_positive.reshape(field_positive.shape[0], -1),
                field_anchor_equivariant.reshape(field_anchor_equivariant.shape[0], -1),
                dim=-1,
            )
            field_positive_norm = field_positive.norm(dim=(-1, -2))
            field_target_norm = field_anchor_equivariant.norm(dim=(-1, -2))
            norm_mse = F.mse_loss(field_positive_norm, field_target_norm)
            op_fro = operator_matrix.pow(2).sum(dim=(-1, -2)).sqrt()
            stats = {
                'loss': float(loss.item()),
                'nce_loss': float(nce_loss.item()),
                'eq_raw_loss': float(eq_raw_loss.item()),
                'eq_norm_mse': float(norm_mse.item()),
                'sum_loss': float(sum_loss.item()),
                'reg_loss': float(reg_loss.item()),
                'loc_loss': float(loc_loss.item()),
                'eq_cos_mean': float(eq_cos.mean().item()),
                'positive_similarity_mean': float(pos_sim.mean().item()),
                'negative_similarity_mean': float(neg_sim.mean().item()),
                'operator_fro_mean': float(op_fro.mean().item()),
                'operator_row_sum_abs_mean': float(row_sum.abs().mean().item()),
            }
        return loss, stats
