from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from tangent_dataset import PregeneratedCurveBank
from tangent_model import TangentOperatorModel
from derivatives import compute_arc_length_derivatives_from_parameter_derivatives
from curve_generation import BasisExpansionCurveCoeffs


Array = np.ndarray
Tensor = torch.Tensor


@dataclass
class FullCurveSample:
    curve_points: Tensor            # [N, 2]
    gt_first: Tensor               # [N, 2]
    gt_second: Tensor              # [N, 2]
    t_grid: Tensor                 # [N]
    has_analytic: Tensor           # [] bool


class FullCurveBankDataset(Dataset):
    """
    Full-curve dataset for operator-level training.

    Uses ONLY PregeneratedCurveBank, as requested.
    """

    def __init__(self, bank_path: str | Path, family: str = "euclidean", dtype: torch.dtype = torch.float32) -> None:
        self.bank = PregeneratedCurveBank(bank_path)
        self.family = str(family)
        self.dtype = dtype

    def __len__(self) -> int:
        return len(self.bank)

    def __getitem__(self, idx: int) -> FullCurveSample:
        curve_points, coeffs, t_grid = self.bank.get(idx)
        curve_points = np.asarray(curve_points, dtype=np.float64)
        n = len(curve_points)

        if coeffs is not None and t_grid is not None:
            gt_first, gt_second = _analytic_arc_length_derivatives_from_bank(
                coeffs=coeffs,
                t_grid=np.asarray(t_grid, dtype=np.float64),
                family=self.family,
            )
            has_analytic = True
            t_out = np.asarray(t_grid, dtype=np.float64)
        else:
            gt_first, gt_second = _finite_difference_arc_length_derivatives(curve_points)
            has_analytic = False
            t_out = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False, dtype=np.float64)

        return FullCurveSample(
            curve_points=torch.as_tensor(curve_points, dtype=self.dtype),
            gt_first=torch.as_tensor(gt_first, dtype=self.dtype),
            gt_second=torch.as_tensor(gt_second, dtype=self.dtype),
            t_grid=torch.as_tensor(t_out, dtype=self.dtype),
            has_analytic=torch.tensor(has_analytic, dtype=torch.bool),
        )


def parse_int_list(text: str | Iterable[int]) -> list[int]:
    if isinstance(text, (list, tuple)):
        return [int(x) for x in text]
    text = str(text).strip()
    if not text:
        return []
    return [int(x.strip()) for x in text.split(",") if x.strip()]



def load_model_from_config(checkpoint_path: str | Path, config_path: str | Path, device: str | torch.device = "cpu") -> TangentOperatorModel:
    config = json.loads(Path(config_path).read_text())
    model = TangentOperatorModel(
        patch_size=int(config["patch_size"]),
        operator_hidden_dims=parse_int_list(config["operator_hidden_dims"]),
        signature_hidden_dims=parse_int_list(config["signature_hidden_dims"]),
        signature_out_dim=int(config["signature_out_dim"]),
        signature_center_radius=int(config["signature_center_radius"]),
        head_dropout=float(config["head_dropout"]),
        normalize_projector=not bool(config.get("disable_normalize_projector", False)),
        init_scale=float(config.get("operator_init_scale", 0.05)),
        learn_scale=bool(config.get("learn_output_scale", False)),
        centered_input_for_operator=not bool(config.get("disable_centered_input_for_operator", False)),
    )
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model



def maybe_fix_global_sign(
    model: TangentOperatorModel,
    dataset: FullCurveBankDataset,
    device: str | torch.device = "cpu",
    max_curves: int = 8,
) -> int:
    """
    Detect whether the first-order operator is globally sign-flipped.
    Returns +1 or -1.
    """
    device = torch.device(device)
    vals: list[float] = []
    with torch.no_grad():
        for i in range(min(max_curves, len(dataset))):
            sample = dataset[i]
            curve = sample.curve_points.to(device)
            gt_first = sample.gt_first.to(device)
            patches = extract_cyclic_patches(curve, model.patch_size)
            raw_weights = model.get_weights(patches)
            proj_weights, _ = project_first_derivative_stencils(raw_weights, patches)
            global1 = apply_local_stencils_to_curve(proj_weights, patches)
            cos = cosine_mean(global1, gt_first)
            vals.append(float(cos.item()))
    mean_cos = float(np.mean(vals)) if vals else 1.0
    return -1 if mean_cos < -0.25 else 1



def extract_cyclic_patches(curve_points: Tensor, patch_size: int) -> Tensor:
    """
    curve_points: [N, 2]
    returns: [N, K, 2]
    """
    if curve_points.ndim != 2 or curve_points.shape[-1] != 2:
        raise ValueError(f"Expected [N,2], got {tuple(curve_points.shape)}")
    n = curve_points.shape[0]
    k = int(patch_size)
    r = k // 2
    offsets = torch.arange(-r, r + 1, device=curve_points.device)
    centers = torch.arange(n, device=curve_points.device).unsqueeze(1)
    idx = (centers + offsets.unsqueeze(0)) % n
    patches = curve_points[idx]
    return patches



def local_signed_arc_offsets(patches: Tensor, eps: float = 1e-8) -> Tensor:
    """
    patches: [N, K, 2]
    returns ds offsets relative to center: [N, K]

    Uses cumulative chord length on each side of the center.
    """
    n, k, _ = patches.shape
    c = k // 2

    seg = patches[:, 1:, :] - patches[:, :-1, :]
    seg_len = torch.linalg.norm(seg, dim=-1).clamp_min(eps)  # [N, K-1]

    ds = patches.new_zeros((n, k))

    # Right side: positive cumulative arc length from center.
    if c + 1 < k:
        right_lens = seg_len[:, c:]  # center->center+1, ...
        ds[:, c + 1:] = torch.cumsum(right_lens, dim=1)

    # Left side: negative cumulative arc length from center moving left.
    if c > 0:
        left_lens = seg_len[:, :c]  # 0->1, ..., c-1->c
        left_from_center = torch.flip(left_lens, dims=[1])
        ds[:, :c] = -torch.flip(torch.cumsum(left_from_center, dim=1), dims=[1])

    return ds



def project_first_derivative_stencils(
    raw_weights: Tensor,
    patches: Tensor,
    ridge: float = 1e-6,
    enforce_sum_zero: bool = True,
    enforce_first_moment: bool = True,
) -> tuple[Tensor, Tensor]:
    """
    Project each local stencil onto the affine constraint set:
      sum_i w_i = 0
      sum_i w_i ds_i = 1

    Returns:
      projected_weights: [N, K]
      ds_offsets: [N, K]
    """
    ds = local_signed_arc_offsets(patches)
    device = raw_weights.device
    dtype = raw_weights.dtype
    n, k = raw_weights.shape

    rows = []
    bvals = []
    if enforce_sum_zero:
        rows.append(torch.ones((n, k), device=device, dtype=dtype))
        bvals.append(torch.zeros((n, 1), device=device, dtype=dtype))
    if enforce_first_moment:
        rows.append(ds)
        bvals.append(torch.ones((n, 1), device=device, dtype=dtype))

    if not rows:
        return raw_weights, ds

    A = torch.stack(rows, dim=1)          # [N, M, K]
    b = torch.cat(bvals, dim=1)           # [N, M]

    # Euclidean projection onto affine constraints:
    # w* = w - A^T (A A^T + ridge I)^-1 (A w - b)
    Aw_minus_b = torch.einsum("nmk,nk->nm", A, raw_weights) - b
    AAT = torch.einsum("nmk,nlk->nml", A, A)
    eye = torch.eye(AAT.shape[-1], device=device, dtype=dtype).unsqueeze(0).expand(n, -1, -1)
    AAT_reg = AAT + ridge * eye
    lam = torch.linalg.solve(AAT_reg, Aw_minus_b.unsqueeze(-1)).squeeze(-1)
    correction = torch.einsum("nmk,nm->nk", A, lam)
    projected = raw_weights - correction
    return projected, ds



def assemble_cyclic_matrix_from_stencils(weights: Tensor) -> Tensor:
    """
    weights: [N, K] with center at K//2.
    returns W: [N, N]
    """
    n, k = weights.shape
    device = weights.device
    c = k // 2
    cols = (torch.arange(n, device=device).unsqueeze(1) + torch.arange(-c, c + 1, device=device).unsqueeze(0)) % n
    W = torch.zeros((n, n), dtype=weights.dtype, device=device)
    rows = torch.arange(n, device=device).unsqueeze(1).expand(-1, k)
    W[rows.reshape(-1), cols.reshape(-1)] = weights.reshape(-1)
    return W



def apply_local_stencils_to_curve(weights: Tensor, patches: Tensor, sign: int = 1) -> Tensor:
    out = torch.einsum("nk,nkd->nd", weights, patches)
    if sign not in (-1, 1):
        raise ValueError("sign must be ±1")
    return float(sign) * out



def curve_spacing(curve_points: Tensor, eps: float = 1e-8) -> Tensor:
    next_pts = torch.roll(curve_points, shifts=-1, dims=0)
    prev_pts = torch.roll(curve_points, shifts=1, dims=0)
    return 0.5 * (torch.linalg.norm(next_pts - curve_points, dim=-1) + torch.linalg.norm(curve_points - prev_pts, dim=-1)).clamp_min(eps)



def second_order_confidence(gt_second: Tensor, curve_points: Tensor, quantile_cap: float | None = 0.99, eps: float = 1e-8) -> tuple[Tensor, Tensor, dict[str, float]]:
    """
    Returns:
      conf: [N] in [0,1]
      valid: [N] bool
      stats: dict

    v1 implementation: quantile-based cap using gt-second norm combined with spacing-aware rho.
    """
    gt_norm = torch.linalg.norm(gt_second, dim=-1).clamp_min(eps)
    ds = curve_spacing(curve_points, eps=eps)
    rho = gt_norm * ds

    if quantile_cap is None:
        cap = gt_norm.max().detach()
    else:
        cap = torch.quantile(gt_norm.detach(), quantile_cap)
    rho_cap = torch.quantile(rho.detach(), 0.99)

    conf_mag = torch.clamp(cap / gt_norm, max=1.0)
    conf_rho = torch.clamp(rho_cap / rho, max=1.0)
    conf = conf_mag * conf_rho
    valid = conf > 0.5

    stats = {
        "gt2_norm_mean": float(gt_norm.mean().item()),
        "gt2_norm_q99": float(torch.quantile(gt_norm.detach(), 0.99).item()),
        "rho_mean": float(rho.mean().item()),
        "rho_q99": float(torch.quantile(rho.detach(), 0.99).item()),
        "valid2_fraction": float(valid.float().mean().item()),
    }
    return conf, valid, stats



def cosine_mean(pred: Tensor, target: Tensor, mask: Tensor | None = None, eps: float = 1e-8) -> Tensor:
    pred_n = pred / pred.norm(dim=-1, keepdim=True).clamp_min(eps)
    target_n = target / target.norm(dim=-1, keepdim=True).clamp_min(eps)
    cos = (pred_n * target_n).sum(dim=-1).clamp(-1.0, 1.0)
    if mask is not None:
        cos = cos[mask]
    return cos.mean() if cos.numel() > 0 else pred.new_tensor(float("nan"))



def masked_huber(x: Tensor, mask: Tensor | None = None, delta: float = 1.0) -> Tensor:
    if mask is not None:
        x = x[mask]
    if x.numel() == 0:
        return x.new_tensor(0.0)
    abs_x = x.abs()
    quad = torch.minimum(abs_x, x.new_tensor(delta))
    lin = abs_x - quad
    return (0.5 * quad.pow(2) + delta * lin).mean()



def fit_slope_intercept(x: Tensor, y: Tensor, mask: Tensor | None = None) -> tuple[float, float]:
    if mask is not None:
        x = x[mask]
        y = y[mask]
    if x.numel() < 2:
        return float("nan"), float("nan")
    xm = x.mean()
    ym = y.mean()
    var = ((x - xm) ** 2).mean()
    if float(var.item()) < 1e-12:
        return float("nan"), float("nan")
    slope = (((x - xm) * (y - ym)).mean() / var).item()
    intercept = (ym - slope * xm).item()
    return float(slope), float(intercept)



def pearson_corr(x: Tensor, y: Tensor, mask: Tensor | None = None, eps: float = 1e-8) -> float:
    if mask is not None:
        x = x[mask]
        y = y[mask]
    if x.numel() < 2:
        return float("nan")
    xm = x - x.mean()
    ym = y - y.mean()
    denom = torch.sqrt((xm.pow(2).mean()) * (ym.pow(2).mean())).clamp_min(eps)
    return float((xm * ym).mean().div(denom).item())



def operator_losses(
    global1: Tensor,
    global2: Tensor,
    gt_first: Tensor,
    gt_second: Tensor,
    conf2: Tensor,
    valid2: Tensor,
    *,
    alpha_cos1: float = 0.25,
    alpha_vec1: float = 1.0,
    beta_cos2: float = 1.0,
    beta_log2: float = 1.0,
    beta_lin2: float = 0.15,
    eps: float = 1e-6,
) -> tuple[Tensor, dict[str, float]]:
    # First order preservation.
    cos1_loss = 1.0 - cosine_mean(global1, gt_first, eps=eps)
    vec1_loss = masked_huber(global1 - gt_first)

    # Second order.
    cos2_loss = 1.0 - cosine_mean(global2, gt_second, eps=eps)
    pred2_norm = torch.linalg.norm(global2, dim=-1).clamp_min(eps)
    gt2_norm = torch.linalg.norm(gt_second, dim=-1).clamp_min(eps)
    log2_loss = masked_huber(torch.log(pred2_norm) - torch.log(gt2_norm))
    lin2_loss = (conf2 * F.smooth_l1_loss(pred2_norm, gt2_norm, reduction="none")).mean()

    total = (
        alpha_cos1 * cos1_loss
        + alpha_vec1 * vec1_loss
        + beta_cos2 * cos2_loss
        + beta_log2 * log2_loss
        + beta_lin2 * lin2_loss
    )

    slope_valid, intercept_valid = fit_slope_intercept(gt2_norm, pred2_norm, valid2)
    stats = {
        "loss": float(total.item()),
        "cos1": float(cosine_mean(global1, gt_first, eps=eps).item()),
        "cos2_full": float(cosine_mean(global2, gt_second, eps=eps).item()),
        "vec1_huber": float(vec1_loss.item()),
        "log2_huber": float(log2_loss.item()),
        "lin2_huber": float(lin2_loss.item()),
        "pearson2_valid": pearson_corr(gt2_norm, pred2_norm, valid2),
        "slope2_valid": slope_valid,
        "intercept2_valid": intercept_valid,
        "valid2_fraction": float(valid2.float().mean().item()),
        "pred2_norm_mean": float(pred2_norm.mean().item()),
        "gt2_norm_mean": float(gt2_norm.mean().item()),
    }
    return total, stats



def _analytic_arc_length_derivatives_from_bank(
    coeffs: BasisExpansionCurveCoeffs,
    t_grid: Array,
    family: str,
) -> tuple[Array, Array]:
    x_coeffs = np.asarray(coeffs.x_coeffs, dtype=np.float64)
    y_coeffs = np.asarray(coeffs.y_coeffs, dtype=np.float64)
    if x_coeffs.shape != y_coeffs.shape:
        raise ValueError("Coefficient shapes do not match")
    max_freq = len(x_coeffs) // 2
    t = np.asarray(t_grid, dtype=np.float64)
    n = len(t)

    first_dt = np.zeros((n, 2), dtype=np.float64)
    second_dt = np.zeros((n, 2), dtype=np.float64)
    third_dt = np.zeros((n, 2), dtype=np.float64)
    for k in range(1, max_freq + 1):
        xc = x_coeffs[2 * (k - 1)]
        xs = x_coeffs[2 * (k - 1) + 1]
        yc = y_coeffs[2 * (k - 1)]
        ys = y_coeffs[2 * (k - 1) + 1]
        ck = np.cos(k * t)
        sk = np.sin(k * t)
        first_dt[:, 0] += -k * xc * sk + k * xs * ck
        first_dt[:, 1] += -k * yc * sk + k * ys * ck
        second_dt[:, 0] += -(k ** 2) * xc * ck - (k ** 2) * xs * sk
        second_dt[:, 1] += -(k ** 2) * yc * ck - (k ** 2) * ys * sk
        third_dt[:, 0] += (k ** 3) * xc * sk - (k ** 3) * xs * ck
        third_dt[:, 1] += (k ** 3) * yc * sk - (k ** 3) * ys * ck

    first_ds, second_ds = compute_arc_length_derivatives_from_parameter_derivatives(
        family=family,
        first_dt=first_dt,
        second_dt=second_dt,
        third_dt=third_dt,
    )
    return first_ds.astype(np.float64), second_ds.astype(np.float64)



def _finite_difference_arc_length_derivatives(curve_points: Array, eps: float = 1e-8) -> tuple[Array, Array]:
    pts = np.asarray(curve_points, dtype=np.float64)
    next_pts = np.roll(pts, -1, axis=0)
    prev_pts = np.roll(pts, 1, axis=0)
    ds_next = np.linalg.norm(next_pts - pts, axis=-1)
    ds_prev = np.linalg.norm(pts - prev_pts, axis=-1)
    ds = 0.5 * (ds_next + ds_prev)
    ds = np.clip(ds, eps, None)

    first = (next_pts - prev_pts) / (2.0 * ds[:, None])
    first_norm = np.linalg.norm(first, axis=-1, keepdims=True)
    first = first / np.clip(first_norm, eps, None)
    second = (next_pts - 2.0 * pts + prev_pts) / (ds[:, None] ** 2)
    return first.astype(np.float64), second.astype(np.float64)
