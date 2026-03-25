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
    Equivariant-contrastive operator model.

    The model predicts a single scalar stencil over the patch. That stencil is the
    object we hope will converge toward a derivative-like operator.

    Training loss sees only:
    - first-order operator output equivariance under the transformation family
    - contrastive separation of anchor / positive / negatives in a projection space
    - simple structural regularization on the stencil

    Analytic derivatives remain diagnostics only.
    """

    def __init__(
        self,
        patch_size: int,
        point_dim: int = 2,
        point_mlp_dims: list[int] | None = None,
        trunk_dims: list[int] | None = None,
        head_dims: list[int] | None = None,
        projector_dims: list[int] | None = None,
        projector_out_dim: int = 64,
        use_batchnorm: bool = True,
        point_dropout: float = 0.0,
        head_dropout: float = 0.0,
        normalize_projector: bool = True,
        center_weights: bool = True,
    ) -> None:
        super().__init__()
        if point_mlp_dims is None:
            point_mlp_dims = [64, 64, 128]
        if trunk_dims is None:
            trunk_dims = [128]
        if head_dims is None:
            head_dims = [128, 64]
        if projector_dims is None:
            projector_dims = [64, 64]

        self.patch_size = int(patch_size)
        self.normalize_projector = bool(normalize_projector)
        self.center_weights = bool(center_weights)

        self.point_encoder = SharedMLP(
            in_dim=point_dim,
            hidden_dims=point_mlp_dims,
            use_batchnorm=use_batchnorm,
            dropout=point_dropout,
        )
        feature_dim = point_mlp_dims[-1]
        pooled_dim = 2 * feature_dim

        self.trunk = MLPHead(
            in_dim=pooled_dim,
            hidden_dims=trunk_dims,
            out_dim=trunk_dims[-1],
            dropout=head_dropout,
        )
        trunk_dim = trunk_dims[-1]
        self.operator_head = MLPHead(
            in_dim=trunk_dim,
            hidden_dims=head_dims,
            out_dim=self.patch_size,
            dropout=head_dropout,
        )
        self.projector = MLPHead(
            in_dim=2,
            hidden_dims=projector_dims,
            out_dim=projector_out_dim,
            dropout=head_dropout,
        )

    def _pool_patch(self, x: torch.Tensor) -> torch.Tensor:
        point_features = self.point_encoder(x)
        mean_feat = point_features.mean(dim=1)
        max_feat = point_features.max(dim=1).values
        return torch.cat([mean_feat, max_feat], dim=-1)

    @staticmethod
    def _apply_weights(weights: torch.Tensor, patch: torch.Tensor) -> torch.Tensor:
        return torch.einsum('bp,bpd->bd', weights, patch)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        if x.ndim != 3 or x.shape[-1] != 2:
            raise ValueError(f'Expected input shape (B, P, 2), got {tuple(x.shape)}')

        pooled = self._pool_patch(x)
        trunk_feature = self.trunk(pooled)

        weights = self.operator_head(trunk_feature)
        if self.center_weights:
            weights = weights - weights.mean(dim=-1, keepdim=True)

        vector_first = self._apply_weights(weights, x)
        weighted_patch = weights.unsqueeze(-1) * x
        vector_second = self._apply_weights(weights, weighted_patch)

        projection = self.projector(vector_first)
        if self.normalize_projector:
            projection = F.normalize(projection, dim=-1)

        return {
            'trunk_feature': trunk_feature,
            'weights': weights,
            'vector': vector_first,
            'vector_first': vector_first,
            'vector_second': vector_second,
            'projection': projection,
        }
