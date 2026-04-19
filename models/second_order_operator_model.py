from __future__ import annotations

import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dims: list[int], out_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        prev = in_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.GELU())
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            prev = h
        layers.append(nn.Linear(prev, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SecondOrderOperatorModel(nn.Module):
    """
    Predicts a local stencil w in R^K from a centered planar patch X in R^{Kx2},
    then applies it as:

        pred = sum_j w_j X_j

    so the model is explicitly learning a local discrete operator.
    """

    def __init__(
        self,
        patch_size: int,
        *,
        hidden_dims: list[int] | None = None,
        dropout: float = 0.0,
        enforce_zero_sum: bool = True,
        learn_output_scale: bool = False,
    ) -> None:
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [256, 256]

        self.patch_size = int(patch_size)
        self.enforce_zero_sum = bool(enforce_zero_sum)

        self.weight_head = MLP(
            in_dim=2 * self.patch_size,
            hidden_dims=hidden_dims,
            out_dim=self.patch_size,
            dropout=dropout,
        )

        self.output_scale = nn.Parameter(torch.tensor(1.0)) if learn_output_scale else None

    def get_weights(self, patch: torch.Tensor) -> torch.Tensor:
        if patch.ndim != 3 or patch.shape[-1] != 2:
            raise ValueError(f"Expected patch shape [B,K,2], got {tuple(patch.shape)}")
        if patch.shape[1] != self.patch_size:
            raise ValueError(f"Expected patch size {self.patch_size}, got {patch.shape[1]}")

        weights = self.weight_head(patch.reshape(patch.shape[0], -1))
        if self.enforce_zero_sum:
            weights = weights - weights.mean(dim=-1, keepdim=True)
        return weights

    def apply_weights(self, weights: torch.Tensor, patch: torch.Tensor) -> torch.Tensor:
        pred = torch.einsum("bk,bkd->bd", weights, patch)
        if self.output_scale is not None:
            pred = self.output_scale * pred
        return pred

    def forward(self, patch: torch.Tensor) -> dict[str, torch.Tensor]:
        weights = self.get_weights(patch)
        pred = self.apply_weights(weights, patch)
        return {
            "weights": weights,
            "pred_second": pred,
        }
