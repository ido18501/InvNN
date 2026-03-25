from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SharedMLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dims: list[int], use_batchnorm: bool = True, dropout: float = 0.0) -> None:
        super().__init__()
        if len(hidden_dims) == 0:
            raise ValueError('hidden_dims must contain at least one layer.')
        layers = []
        prev_dim = in_dim
        for dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, dim))
            if use_batchnorm:
                layers.append(nn.BatchNorm1d(dim))
            layers.append(nn.ReLU())
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            prev_dim = dim
        self.layers = nn.ModuleList(layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f'Expected input shape (B, P, C), got {tuple(x.shape)}')
        B, P, _ = x.shape
        out = x
        for layer in self.layers:
            if isinstance(layer, nn.BatchNorm1d):
                out = out.reshape(B * P, -1)
                out = layer(out)
                out = out.reshape(B, P, -1)
            else:
                out = layer(out)
        return out


class MLPHead(nn.Module):
    def __init__(self, in_dim: int, hidden_dims: list[int], out_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        layers = []
        prev = in_dim
        for dim in hidden_dims:
            layers.append(nn.Linear(prev, dim))
            layers.append(nn.ReLU())
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            prev = dim
        layers.append(nn.Linear(prev, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TangentOperatorModel(nn.Module):
    """
    Invariant-first model.

    - encode patch points
    - pool to patch descriptor
    - produce invariant code z
    - predict one scalar stencil W from z
    - apply W once and twice to centered patch coordinates

    The loss only acts on invariant embeddings and on regularity of the learned code.
    Derivative comparison is external diagnostics only.
    """

    def __init__(
        self,
        patch_size: int,
        point_dim: int = 2,
        point_mlp_dims: list[int] | None = None,
        projector_dims: list[int] | None = None,
        head_dims: list[int] | None = None,
        invariant_dim: int = 64,
        use_batchnorm: bool = True,
        point_dropout: float = 0.0,
        head_dropout: float = 0.0,
        normalize_embedding: bool = True,
        center_weights: bool = True,
    ) -> None:
        super().__init__()
        if point_mlp_dims is None:
            point_mlp_dims = [64, 64, 128]
        if projector_dims is None:
            projector_dims = [128]
        if head_dims is None:
            head_dims = [128, 64]

        self.patch_size = patch_size
        self.normalize_embedding = normalize_embedding
        self.center_weights = center_weights

        self.point_encoder = SharedMLP(
            in_dim=point_dim,
            hidden_dims=point_mlp_dims,
            use_batchnorm=use_batchnorm,
            dropout=point_dropout,
        )
        feature_dim = point_mlp_dims[-1]
        pooled_dim = 2 * feature_dim

        self.projector = MLPHead(
            in_dim=pooled_dim,
            hidden_dims=projector_dims,
            out_dim=invariant_dim,
            dropout=head_dropout,
        )
        self.operator_head = MLPHead(
            in_dim=invariant_dim,
            hidden_dims=head_dims,
            out_dim=patch_size,
            dropout=head_dropout,
        )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        if x.ndim != 3 or x.shape[-1] != 2:
            raise ValueError(f'Expected input shape (B, P, 2), got {tuple(x.shape)}')

        point_features = self.point_encoder(x)
        mean_feat = point_features.mean(dim=1)
        max_feat = point_features.max(dim=1).values
        patch_feature = torch.cat([mean_feat, max_feat], dim=-1)

        invariant_code = self.projector(patch_feature)
        embedding = F.normalize(invariant_code, dim=-1) if self.normalize_embedding else invariant_code

        weights = self.operator_head(invariant_code)
        if self.center_weights:
            weights = weights - weights.mean(dim=-1, keepdim=True)

        vector_first = torch.einsum('bp,bpd->bd', weights, x)
        weighted_patch = weights.unsqueeze(-1) * x
        vector_second = torch.einsum('bp,bpd->bd', weights, weighted_patch)

        return {
            'embedding': embedding,
            'invariant_code': invariant_code,
            'weights': weights,
            'weights_first': weights,
            'weights_second': weights,
            'vector': vector_first,
            'vector_first': vector_first,
            'vector_second': vector_second,
        }
