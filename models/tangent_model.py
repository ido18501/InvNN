from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class MLPHead(nn.Module):
    def __init__(self, in_dim: int, hidden_dims: list[int], out_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        layers: list[nn.Module] = []
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
    Predict a KxK operator *per patch*.

    For an input patch X with shape [B, K, 2], the model predicts
        W(X) in R^{B x K x K}
    and applies it on the sample-index dimension:
        Y1 = W(X) X
        Y2 = W(X) Y1

    The same predicted operator is applied to the x- and y-coordinate channels
    separately; there is no x/y channel mixing inside W.

    Equivariance is enforced on the *action* of the predicted operator:
        W(TX) TX  ~=  T(W(X) X)
    where T(p) = A p + b, and only the linear part A acts on the output field.
    The translation bias b is intentionally ignored there, just like a true
    derivative operator would do once row sums are near zero.
    """

    def __init__(
        self,
        patch_size: int,
        *,
        operator_hidden_dims: list[int] | None = None,
        signature_hidden_dims: list[int] | None = None,
        signature_out_dim: int = 64,
        signature_center_radius: int = 0,
        head_dropout: float = 0.0,
        normalize_projector: bool = True,
        init_scale: float = 0.05,
        center_operator: bool = True,
        operator_bandwidth: int | None = None,
        learn_scale: bool = False,
        centered_input_for_operator: bool = True,
    ) -> None:
        super().__init__()
        if operator_hidden_dims is None:
            operator_hidden_dims = [256, 256]
        if signature_hidden_dims is None:
            signature_hidden_dims = [128, 64]

        self.patch_size = int(patch_size)
        self.center_index = self.patch_size // 2
        self.signature_center_radius = int(signature_center_radius)
        self.normalize_projector = bool(normalize_projector)
        self.center_operator = bool(center_operator)
        self.centered_input_for_operator = bool(centered_input_for_operator)

        rows = torch.arange(self.patch_size).view(-1, 1)
        cols = torch.arange(self.patch_size).view(1, -1)
        if operator_bandwidth is not None:
            bw = int(operator_bandwidth)
            op_mask = ((rows - cols).abs() <= bw).float()
        else:
            op_mask = torch.ones(self.patch_size, self.patch_size)
        self.register_buffer('operator_mask', op_mask)

        operator_in_dim = self.patch_size * 2
        self.operator_head = MLPHead(
            in_dim=operator_in_dim,
            hidden_dims=operator_hidden_dims,
            out_dim=self.patch_size * self.patch_size,
            dropout=head_dropout,
        )
        self.operator_init_scale = float(init_scale)

        signature_rows = 2 * self.signature_center_radius + 1
        signature_in_dim = signature_rows * 2
        self.signature_head = MLPHead(
            in_dim=signature_in_dim,
            hidden_dims=signature_hidden_dims,
            out_dim=signature_out_dim,
            dropout=head_dropout,
        )

        self.output_scale = nn.Parameter(torch.tensor(1.0)) if learn_scale else None
        self._init_last_layer()

    def _init_last_layer(self) -> None:
        last = self.operator_head.net[-1]
        if isinstance(last, nn.Linear):
            nn.init.normal_(last.weight, mean=0.0, std=self.operator_init_scale)
            nn.init.zeros_(last.bias)

    def _prepare_operator_input(self, x: torch.Tensor) -> torch.Tensor:
        if self.centered_input_for_operator:
            x = x - x.mean(dim=1, keepdim=True)
        return x.reshape(x.shape[0], -1)

    def get_operator(self, x: torch.Tensor) -> torch.Tensor:
        flat = self._prepare_operator_input(x)
        op = self.operator_head(flat).view(x.shape[0], self.patch_size, self.patch_size)
        op = op * self.operator_mask.unsqueeze(0)
        if self.center_operator:
            op = op - op.mean(dim=-1, keepdim=True)
        return op

    def _apply_operator(self, operator: torch.Tensor, patch: torch.Tensor) -> torch.Tensor:
        # operator: [B,K,K], patch: [B,K,2] -> [B,K,2]
        out = torch.einsum('bij,bjd->bid', operator, patch)
        if self.output_scale is not None:
            out = self.output_scale * out
        return out

    def get_center_vectors(self, field: torch.Tensor) -> torch.Tensor:
        return field[:, self.center_index, :]

    def _extract_signature_slice(self, field: torch.Tensor) -> torch.Tensor:
        left = max(0, self.center_index - self.signature_center_radius)
        right = min(self.patch_size, self.center_index + self.signature_center_radius + 1)
        window = field[:, left:right, :]
        return window.reshape(field.shape[0], -1)

    def project_field(self, field: torch.Tensor) -> torch.Tensor:
        projection = self.signature_head(self._extract_signature_slice(field))
        if self.normalize_projector:
            projection = F.normalize(projection, dim=-1)
        return projection

    @staticmethod
    def apply_linear_map_to_field(transform_matrix: torch.Tensor, field: torch.Tensor) -> torch.Tensor:
        # field [B,K,2], transform_matrix [B,2,2] => [B,K,2]
        return torch.einsum('bkd,bed->bke', field, transform_matrix)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        if x.ndim != 3 or x.shape[-1] != 2:
            raise ValueError(f'Expected input shape (B, K, 2), got {tuple(x.shape)}')
        if x.shape[1] != self.patch_size:
            raise ValueError(f'Expected patch size {self.patch_size}, got {x.shape[1]}')

        operator = self.get_operator(x)
        field_first = self._apply_operator(operator, x)
        field_second = self._apply_operator(operator, field_first)
        center_first = self.get_center_vectors(field_first)
        center_second = self.get_center_vectors(field_second)
        projection = self.project_field(field_first)

        return {
            'operator': operator,
            'field_first': field_first,
            'field_second': field_second,
            'center_first': center_first,
            'center_second': center_second,
            'projection': projection,
        }
