from __future__ import annotations

import argparse
import heapq
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from datasets.tangent_dataset import TangentDataset
from models.tangent_model import TangentOperatorModel
from training.collate import tangent_collate_fn
from utils.derivatives import compute_single_anchor_fourier_arc_length_derivatives


# ============================================================
# generic helpers
# ============================================================

def parse_int_list(text: str) -> list[int]:
    text = str(text).strip()
    if not text:
        return []
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def summarize_array(x: np.ndarray) -> dict:
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        return {
            "count": 0,
            "mean": None,
            "std": None,
            "median": None,
            "min": None,
            "max": None,
            "p05": None,
            "p25": None,
            "p75": None,
            "p95": None,
        }
    return {
        "count": int(x.size),
        "mean": float(np.mean(x)),
        "std": float(np.std(x)),
        "median": float(np.median(x)),
        "min": float(np.min(x)),
        "max": float(np.max(x)),
        "p05": float(np.percentile(x, 5)),
        "p25": float(np.percentile(x, 25)),
        "p75": float(np.percentile(x, 75)),
        "p95": float(np.percentile(x, 95)),
    }


def circular_summary_deg(angles_deg: np.ndarray) -> dict:
    angles_deg = np.asarray(angles_deg, dtype=np.float64)
    if angles_deg.size == 0:
        return {"count": 0, "circular_mean_deg": None, "circular_std_deg": None, "resultant_length": None}
    rad = np.deg2rad(angles_deg)
    C = np.mean(np.cos(rad))
    S = np.mean(np.sin(rad))
    mean_rad = np.arctan2(S, C)
    R = np.sqrt(C * C + S * S)
    circ_std = np.sqrt(max(0.0, -2.0 * np.log(max(R, 1e-12))))
    return {
        "count": int(angles_deg.size),
        "circular_mean_deg": float(np.rad2deg(mean_rad)),
        "circular_std_deg": float(np.rad2deg(circ_std)),
        "resultant_length": float(R),
    }


def plot_hist(values: np.ndarray, title: str, xlabel: str, save_path: Path, bins: int = 60):
    values = np.asarray(values)
    if values.size == 0:
        return
    plt.figure(figsize=(6, 4))
    plt.hist(values, bins=bins)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("count")
    plt.tight_layout()
    plt.savefig(save_path, dpi=180)
    plt.close()


def plot_scatter(x: np.ndarray, y: np.ndarray, title: str, xlabel: str, ylabel: str, save_path: Path):
    x = np.asarray(x)
    y = np.asarray(y)
    if x.size == 0 or y.size == 0:
        return
    plt.figure(figsize=(6, 5))
    plt.scatter(x, y, s=6, alpha=0.25)
    plt.xscale("log")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(save_path, dpi=180)
    plt.close()


def move_batch(batch, device):
    batch.anchor = batch.anchor.to(device)
    batch.positive = batch.positive.to(device)
    batch.negatives = batch.negatives.to(device)
    batch.transform_matrix = batch.transform_matrix.to(device)
    batch.gt_first_anchor = batch.gt_first_anchor.to(device)
    batch.gt_second_anchor = batch.gt_second_anchor.to(device)
    batch.has_analytic_derivatives = batch.has_analytic_derivatives.to(device)
    return batch


def apply_W(W: torch.Tensor, X: torch.Tensor) -> torch.Tensor:
    return torch.einsum("bij,bjd->bid", W, X)


def flatten_cos(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    a_flat = a.reshape(a.shape[0], -1)
    b_flat = b.reshape(b.shape[0], -1)
    return F.cosine_similarity(a_flat, b_flat, dim=-1)

def apply_linear_map_to_vecs(T: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """
    T: (B,2,2)
    v: (B,2)
    returns: (B,2)
    """
    return torch.einsum("bij,bj->bi", T, v)

def affine_det_scale(T: torch.Tensor, power: float, eps: float = 1e-12) -> torch.Tensor:
    """
    T: (..., 2, 2)
    returns: (...) scalar scale = |det(T)|^power
    """
    det = T[..., 0, 0] * T[..., 1, 1] - T[..., 0, 1] * T[..., 1, 0]
    return torch.clamp(det.abs(), min=eps).pow(power)


def apply_affine_map_to_vecs(T: torch.Tensor, v: torch.Tensor, derivative_order: int) -> torch.Tensor:
    """
    T: (B,2,2)
    v: (B,2)
    derivative_order:
        0 -> A v
        1 -> |det A|^(-1/3) A v
        2 -> |det A|^(-2/3) A v
    """
    out = torch.einsum("bij,bj->bi", T, v)
    if derivative_order == 0:
        return out
    if derivative_order == 1:
        scale = affine_det_scale(T, power=-1.0 / 3.0).unsqueeze(-1)
        return scale * out
    if derivative_order == 2:
        scale = affine_det_scale(T, power=-2.0 / 3.0).unsqueeze(-1)
        return scale * out
    raise ValueError(f"Unsupported derivative_order: {derivative_order}")


def apply_affine_map_to_field(T: torch.Tensor, field: torch.Tensor, derivative_order: int) -> torch.Tensor:
    """
    T: (B,2,2)
    field: (B,K,2)
    """
    out = torch.einsum("bij,bkj->bki", T, field)
    if derivative_order == 0:
        return out
    if derivative_order == 1:
        scale = affine_det_scale(T, power=-1.0 / 3.0).view(-1, 1, 1)
        return scale * out
    if derivative_order == 2:
        scale = affine_det_scale(T, power=-2.0 / 3.0).view(-1, 1, 1)
        return scale * out
    raise ValueError(f"Unsupported derivative_order: {derivative_order}")

def frame_residual_angles_deg(pred: torch.Tensor, gt_dir: torch.Tensor) -> np.ndarray:
    pred_u = pred / (pred.norm(dim=-1, keepdim=True) + 1e-12)
    gt_u = gt_dir / (gt_dir.norm(dim=-1, keepdim=True) + 1e-12)
    gt_perp = torch.stack([-gt_u[:, 1], gt_u[:, 0]], dim=-1)
    x = (pred_u * gt_u).sum(dim=-1)
    y = (pred_u * gt_perp).sum(dim=-1)
    ang = torch.rad2deg(torch.atan2(y, x))
    return ang.detach().cpu().numpy()


# ============================================================
# exact reconstruction from config
# ============================================================

def build_dataset_from_config(cfg: dict, split: str) -> TangentDataset:
    seed_offset = {"train": 0, "val": 10000, "test": 20000}[split]
    return TangentDataset(
        length=int(cfg[f"{split}_length"]),
        family=cfg["family"],
        source=cfg[f"{split}_source"],
        bank_path=cfg[f"{split}_bank"],
        num_curve_points=int(cfg["num_curve_points"]),
        fourier_max_freq=int(cfg["fourier_max_freq"]),
        fourier_scale=float(cfg["fourier_scale"]),
        fourier_decay_power=float(cfg["fourier_decay_power"]),
        patch_size=int(cfg["patch_size"]),
        half_width=int(cfg["half_width"]),
        num_negatives=int(cfg["num_negatives"]),
        negative_min_offset=int(cfg["negative_min_offset"]),
        negative_max_offset=int(cfg["negative_max_offset"]),
        negative_other_curve_fraction=float(cfg["negative_other_curve_fraction"]),
        patch_mode=cfg["patch_mode"],
        jitter_fraction=float(cfg["jitter_fraction"]),
        warp_sampling_prob=float(cfg["warp_sampling_prob"]),
        warp_sampling_strength=float(cfg["warp_sampling_strength"]),
        seed=int(cfg["seed"]) + seed_offset,
    )


def build_model_from_config(cfg: dict, device: torch.device) -> TangentOperatorModel:
    model = TangentOperatorModel(
        patch_size=int(cfg["patch_size"]),
        operator_hidden_dims=parse_int_list(cfg["operator_hidden_dims"]),
        signature_hidden_dims=parse_int_list(cfg["signature_hidden_dims"]),
        signature_out_dim=int(cfg["signature_out_dim"]),
        signature_center_radius=int(cfg["signature_center_radius"]),
        head_dropout=float(cfg["head_dropout"]),
        normalize_projector=not bool(cfg["disable_normalize_projector"]),
        center_operator=not bool(cfg["disable_center_operator"]),
        operator_bandwidth=cfg["operator_bandwidth"],
        init_scale=float(cfg["operator_init_scale"]),
        learn_scale=bool(cfg["learn_output_scale"]),
        centered_input_for_operator=not bool(cfg["disable_centered_input_for_operator"]),
    ).to(device)
    return model


# ============================================================
# geometry / numerical GT on ACTUAL sampled curve
# ============================================================

def resample_closed_curve_uniform_arc_length(points: np.ndarray, num_points: int) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("points must have shape (N,2)")
    if len(points) < 4:
        raise ValueError("Need at least 4 points")
    extended = np.vstack([points, points[:1]])
    seg = np.linalg.norm(np.diff(extended, axis=0), axis=1)
    total = float(seg.sum())
    if total <= 1e-12:
        raise ValueError("Degenerate curve")
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    targets = np.linspace(0.0, total, num_points, endpoint=False)

    out = np.empty((num_points, 2), dtype=np.float64)
    j = 0
    for i, s in enumerate(targets):
        while j + 1 < len(cum) and cum[j + 1] <= s:
            j += 1
        if j >= len(seg):
            j = len(seg) - 1
        local_len = seg[j]
        if local_len <= 1e-12:
            out[i] = extended[j]
        else:
            alpha = (s - cum[j]) / local_len
            out[i] = (1.0 - alpha) * extended[j] + alpha * extended[j + 1]
    return out


def nearest_index(points: np.ndarray, query: np.ndarray) -> int:
    d2 = ((points - query.reshape(1, 2)) ** 2).sum(axis=1)
    return int(np.argmin(d2))

def det2d(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    return float(a[0] * b[1] - a[1] * b[0])


def numerical_arc_length_derivatives_on_actual_curve(
    curve_points: np.ndarray,
    anchor_index: int,
    dense_num_points: int = 4096,
    geometry: str = "euclidean",
    equiaffine_eps: float = 1e-8,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """
    Numerical derivatives on the ACTUAL sampled curve.

    geometry == "euclidean":
        returns:
            dx/ds_euc, d²x/ds_euc², dense_anchor_point, ds_euc

    geometry == "equiaffine":
        uses the dense Euclidean-arc-length parameter t := s_euc,
        estimates x_t, x_tt, then converts to equiaffine-arc-length derivatives:
            phi   = (|det(x_t, x_tt)| + eps)^(1/3)
            x_s   = x_t / phi
            x_ss  = x_tt / phi^2 - (phi_t / phi^3) x_t
        where phi_t is estimated by centered finite differences in t.

        returns:
            dx/ds_ea, d²x/ds_ea², dense_anchor_point, ds_euc
    """
    curve_points = np.asarray(curve_points, dtype=np.float64)
    dense = resample_closed_curve_uniform_arc_length(curve_points, num_points=dense_num_points)

    q = curve_points[anchor_index]
    k = nearest_index(dense, q)

    extended = np.vstack([curve_points, curve_points[:1]])
    total_length = float(np.linalg.norm(np.diff(extended, axis=0), axis=1).sum())
    ds = total_length / dense_num_points  # this is the dense Euclidean arc-length step

    def pt(idx: int) -> np.ndarray:
        return dense[idx % dense_num_points]

    curr_pt = pt(k)

    # centered derivatives at k with respect to dense Euclidean arc length t = s_euc
    prev_pt = pt(k - 1)
    next_pt = pt(k + 1)

    first_raw = (next_pt - prev_pt) / (2.0 * ds)
    second_raw = (next_pt - 2.0 * curr_pt + prev_pt) / (ds ** 2)

    if geometry == "euclidean":
        first = first_raw.copy()
        first_norm = np.linalg.norm(first)
        if first_norm > 1e-12:
            first = first / first_norm
        second = second_raw
        return first.astype(np.float64), second.astype(np.float64), curr_pt.astype(np.float64), float(ds)

    if geometry not in ("equiaffine", "affine"):
        raise ValueError(f"Unsupported geometry: {geometry}")

    # Need phi_t, so estimate phi at k-1, k, k+1
    def first_second_at(center_idx: int) -> tuple[np.ndarray, np.ndarray]:
        p_prev = pt(center_idx - 1)
        p_curr = pt(center_idx)
        p_next = pt(center_idx + 1)
        f1 = (p_next - p_prev) / (2.0 * ds)
        f2 = (p_next - 2.0 * p_curr + p_prev) / (ds ** 2)
        return f1, f2

    first_m1, second_m1 = first_second_at(k - 1)
    first_0, second_0 = first_raw, second_raw
    first_p1, second_p1 = first_second_at(k + 1)

    g_m1 = det2d(first_m1, second_m1)
    g_0 = det2d(first_0, second_0)
    g_p1 = det2d(first_p1, second_p1)

    phi_m1 = (abs(g_m1) + equiaffine_eps) ** (1.0 / 3.0)
    phi_0 = (abs(g_0) + equiaffine_eps) ** (1.0 / 3.0)
    phi_p1 = (abs(g_p1) + equiaffine_eps) ** (1.0 / 3.0)

    phi_t = (phi_p1 - phi_m1) / (2.0 * ds)

    first_ea = first_0 / phi_0
    second_ea = second_0 / (phi_0 ** 2) - (phi_t / (phi_0 ** 3)) * first_0

    return first_ea.astype(np.float64), second_ea.astype(np.float64), curr_pt.astype(np.float64), float(ds)


def heron_three_point_curvature_magnitude(
    curve_points: np.ndarray,
    anchor_index: int,
    dense_num_points: int = 4096,
) -> float:
    """
    Cross-check curvature magnitude using 3-point circumcircle / Heron formula
    on the ACTUAL curve after arc-length resampling.
    """
    dense = resample_closed_curve_uniform_arc_length(curve_points, num_points=dense_num_points)
    q = np.asarray(curve_points[anchor_index], dtype=np.float64)
    k = nearest_index(dense, q)

    a = dense[(k - 1) % dense_num_points]
    b = dense[k]
    c = dense[(k + 1) % dense_num_points]

    ab = np.linalg.norm(b - a)
    bc = np.linalg.norm(c - b)
    ca = np.linalg.norm(a - c)

    s = 0.5 * (ab + bc + ca)
    area_sq = max(0.0, s * (s - ab) * (s - bc) * (s - ca))
    if area_sq <= 1e-24:
        return 0.0
    area = np.sqrt(area_sq)

    denom = ab * bc * ca
    if denom <= 1e-18:
        return 0.0

    kappa = 4.0 * area / denom
    return float(kappa)


# ============================================================
# analytic GT mismatch audit
# ============================================================

def audit_analytic_gt_consistency(dataset: TangentDataset, num_dataset_indices: int = 64) -> dict:
    if dataset.source != "pregenerated" or dataset.bank is None or not dataset.bank.has_coeffs or dataset.bank.t_grid is None:
        return {"available": False, "reason": "pregenerated coeff/t_grid not available"}

    point_errs = []
    analytic_first_norms = []
    analytic_orthogonality = []

    n = min(num_dataset_indices, len(dataset))
    for idx in range(n):
        rng = dataset._make_rng(idx)
        curve_points, coeffs, t_grid = dataset._get_curve(rng, idx)
        if coeffs is None or t_grid is None:
            continue

        curve_points = np.asarray(curve_points, dtype=np.float64)
        t_grid = np.asarray(t_grid, dtype=np.float64)

        test_centers = np.linspace(0, len(curve_points) - 1, num=min(5, len(curve_points)), dtype=int)
        for c in test_centers:
            analytic_pt, analytic_first, analytic_second = compute_single_anchor_fourier_arc_length_derivatives(
                float(t_grid[c]), coeffs
            )
            point_errs.append(float(np.linalg.norm(curve_points[c] - analytic_pt)))
            analytic_first_norms.append(float(np.linalg.norm(analytic_first)))
            denom = np.linalg.norm(analytic_first) * np.linalg.norm(analytic_second) + 1e-12
            analytic_orthogonality.append(float(np.dot(analytic_first, analytic_second) / denom))

    return {
        "available": True,
        "num_checks": len(point_errs),
        "point_match_error": summarize_array(np.asarray(point_errs)),
        "analytic_first_norm": summarize_array(np.asarray(analytic_first_norms)),
        "analytic_first_vs_second_cos": summarize_array(np.asarray(analytic_orthogonality)),
        "warning": (
            "If point_match_error is not ~0, analytic derivatives do not match the actual sampled curve geometry. "
            "That usually means warp/resampling/noise changed the curve after loading coeffs,t_grid."
        ),
    }


# ============================================================
# visualization
# ============================================================

def plot_patch_with_field(ax, pts: np.ndarray, field: np.ndarray, title: str):
    ax.plot(pts[:, 0], pts[:, 1], "-o", ms=3)
    ax.quiver(pts[:, 0], pts[:, 1], field[:, 0], field[:, 1], angles="xy", scale_units="xy", scale=1.0)
    ax.set_title(title)
    ax.axis("equal")

def get_patch_center_point(pts: np.ndarray) -> np.ndarray:
    pts = np.asarray(pts, dtype=np.float64)
    return pts[len(pts) // 2]


def draw_center_vector(ax, center: np.ndarray, vec: np.ndarray, color: str, label: str, scale: float = 1.0, lw: float = 2.0):
    vec = np.asarray(vec, dtype=np.float64)
    center = np.asarray(center, dtype=np.float64)
    ax.arrow(
        center[0], center[1],
        scale * vec[0], scale * vec[1],
        color=color,
        width=0.0,
        head_width=0.0,
        length_includes_head=True,
        linewidth=lw,
        label=label,
    )


def robust_vector_scale(pred: np.ndarray, gt: np.ndarray, pts: np.ndarray, frac: float = 0.35) -> float:
    """
    Choose a visualization scale so overlaid center vectors are visible
    relative to the patch size, while preserving direction.
    """
    pts = np.asarray(pts, dtype=np.float64)
    pred = np.asarray(pred, dtype=np.float64)
    gt = np.asarray(gt, dtype=np.float64)

    mins = pts.min(axis=0)
    maxs = pts.max(axis=0)
    patch_extent = float(np.linalg.norm(maxs - mins))
    patch_extent = max(patch_extent, 1e-8)

    ref = max(float(np.linalg.norm(pred)), float(np.linalg.norm(gt)), 1e-8)
    return frac * patch_extent / ref


def plot_patch_with_center_derivative_overlay(
    ax,
    pts: np.ndarray,
    pred_vec: np.ndarray,
    gt_vec: np.ndarray,
    title: str,
    show_neg_gt: bool = False,
):
    pts = np.asarray(pts, dtype=np.float64)
    pred_vec = np.asarray(pred_vec, dtype=np.float64)
    gt_vec = np.asarray(gt_vec, dtype=np.float64)

    center = get_patch_center_point(pts)
    scale = robust_vector_scale(pred_vec, gt_vec, pts)

    ax.plot(pts[:, 0], pts[:, 1], "-o", ms=3, color="tab:blue")
    draw_center_vector(ax, center, pred_vec, color="black", label="pred", scale=scale, lw=2.0)
    draw_center_vector(ax, center, gt_vec, color="tab:green", label="gt", scale=scale, lw=2.0)

    if show_neg_gt:
        draw_center_vector(ax, center, -gt_vec, color="tab:red", label="-gt", scale=scale, lw=1.6)

    ax.set_title(title)
    ax.axis("equal")
    ax.legend(loc="best", fontsize=8)

def save_example_figure(example: dict, out_path: Path, num_neg_vis: int):
    anchor_pts = example["anchor_pts"]
    pos_pts = example["pos_pts"]
    neg_pts = example["neg_pts"][:num_neg_vis]

    anchor_f1 = example["anchor_f1"]
    pos_f1 = example["pos_f1"]
    neg_f1 = example["neg_f1"][:num_neg_vis]

    anchor_f2 = example["anchor_f2"]
    pos_f2 = example["pos_f2"]
    neg_f2 = example["neg_f2"][:num_neg_vis]

    cols = 2 + len(neg_pts)
    fig, axes = plt.subplots(3, cols, figsize=(4 * cols, 11))

    plot_patch_with_field(axes[0, 0], anchor_pts, np.zeros_like(anchor_f1), "Anchor patch")
    plot_patch_with_field(axes[1, 0], anchor_pts, anchor_f1, "Anchor: W(X)X")
    plot_patch_with_field(axes[2, 0], anchor_pts, anchor_f2, "Anchor: W²(X)X")

    plot_patch_with_field(axes[0, 1], pos_pts, np.zeros_like(pos_f1), "Positive patch")
    plot_patch_with_field(axes[1, 1], pos_pts, pos_f1, "Positive: W(TX)TX")
    plot_patch_with_field(axes[2, 1], pos_pts, pos_f2, "Positive: W²(TX)TX")

    for j in range(len(neg_pts)):
        plot_patch_with_field(axes[0, 2 + j], neg_pts[j], np.zeros_like(neg_f1[j]), f"Negative {j}")
        plot_patch_with_field(axes[1, 2 + j], neg_pts[j], neg_f1[j], f"Negative {j}: W(X⁻)X⁻")
        plot_patch_with_field(axes[2, 2 + j], neg_pts[j], neg_f2[j], f"Negative {j}: W²(X⁻)X⁻")

    fig.suptitle(
        f"dataset_idx={example['dataset_idx']} | heron_kappa={example['gt_second_norm_heron']:.4f} | "
        f"ea_fd_norm={example['gt_second_norm_fd']:.4f}",
        fontsize=14
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def save_center_overlay_figure(example: dict, out_path: Path, num_neg_vis: int):
    anchor_pts = example["anchor_pts"]
    pos_pts = example["pos_pts"]
    neg_pts = example["neg_pts"][:num_neg_vis]

    cols = 2 + len(neg_pts)
    fig, axes = plt.subplots(2, cols, figsize=(4 * cols, 7.5))

    # ===== FIRST DERIVATIVE =====
    plot_patch_with_center_derivative_overlay(
        axes[0, 0],
        anchor_pts,
        example["anchor_center_pred_f1"],
        example["anchor_center_gt_f1"],
        "Anchor center: pred/gt first",
        show_neg_gt=False,
    )

    plot_patch_with_center_derivative_overlay(
        axes[0, 1],
        pos_pts,
        example["pos_center_pred_f1"],
        example["pos_center_gt_f1"],
        "Positive center: pred/gt first",
        show_neg_gt=False,
    )

    for j in range(len(neg_pts)):
        plot_patch_with_center_derivative_overlay(
            axes[0, 2 + j],
            neg_pts[j],
            example["neg_center_pred_f1"][j],
            example["neg_center_gt_f1"][j],
            f"Negative {j} center: pred/gt first",
            show_neg_gt=False,
        )

    # ===== SECOND DERIVATIVE =====
    plot_patch_with_center_derivative_overlay(
        axes[1, 0],
        anchor_pts,
        example["anchor_center_pred_f2"],
        example["anchor_center_gt_f2"],
        "Anchor center: pred/gt second",
        show_neg_gt=True,
    )

    plot_patch_with_center_derivative_overlay(
        axes[1, 1],
        pos_pts,
        example["pos_center_pred_f2"],
        example["pos_center_gt_f2"],
        "Positive center: pred/gt second",
        show_neg_gt=True,
    )

    for j in range(len(neg_pts)):
        plot_patch_with_center_derivative_overlay(
            axes[1, 2 + j],
            neg_pts[j],
            example["neg_center_pred_f2"][j],
            example["neg_center_gt_f2"][j],
            f"Negative {j} center: pred/gt second",
            show_neg_gt=True,
        )

    fig.suptitle(
        f"dataset_idx={example['dataset_idx']} | heron_kappa={example['gt_second_norm_heron']:.4f} | "
        f"ea_fd_norm={example['gt_second_norm_fd']:.4f}",
        fontsize=14
    )

    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()

def get_patch_center_point(pts: np.ndarray) -> np.ndarray:
    pts = np.asarray(pts, dtype=np.float64)
    return pts[len(pts) // 2]


def draw_center_vector(ax, center: np.ndarray, vec: np.ndarray, color: str, label: str, scale: float = 1.0, lw: float = 2.0):
    vec = np.asarray(vec, dtype=np.float64)
    center = np.asarray(center, dtype=np.float64)
    ax.arrow(
        center[0], center[1],
        scale * vec[0], scale * vec[1],
        color=color,
        width=0.0,
        head_width=0.0,
        length_includes_head=True,
        linewidth=lw,
        label=label,
    )


def robust_vector_scale(pred: np.ndarray, gt: np.ndarray, pts: np.ndarray, frac: float = 0.35) -> float:
    """
    Choose a visualization scale so overlaid center vectors are visible
    relative to the patch size, while preserving direction.
    """
    pts = np.asarray(pts, dtype=np.float64)
    pred = np.asarray(pred, dtype=np.float64)
    gt = np.asarray(gt, dtype=np.float64)

    mins = pts.min(axis=0)
    maxs = pts.max(axis=0)
    patch_extent = float(np.linalg.norm(maxs - mins))
    patch_extent = max(patch_extent, 1e-8)

    ref = max(float(np.linalg.norm(pred)), float(np.linalg.norm(gt)), 1e-8)
    return frac * patch_extent / ref


def plot_patch_with_center_derivative_overlay(
    ax,
    pts: np.ndarray,
    pred_vec: np.ndarray,
    gt_vec: np.ndarray,
    title: str,
    show_neg_gt: bool = False,
):
    pts = np.asarray(pts, dtype=np.float64)
    pred_vec = np.asarray(pred_vec, dtype=np.float64)
    gt_vec = np.asarray(gt_vec, dtype=np.float64)

    center = get_patch_center_point(pts)
    scale = robust_vector_scale(pred_vec, gt_vec, pts)

    ax.plot(pts[:, 0], pts[:, 1], "-o", ms=3, color="tab:blue")
    draw_center_vector(ax, center, pred_vec, color="black", label="pred", scale=scale, lw=2.0)
    draw_center_vector(ax, center, gt_vec, color="tab:green", label="gt", scale=scale, lw=2.0)

    if show_neg_gt:
        draw_center_vector(ax, center, -gt_vec, color="tab:red", label="-gt", scale=scale, lw=1.6)

    ax.set_title(title)
    ax.axis("equal")
    ax.legend(loc="best", fontsize=8)



# ============================================================
# main
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-dir", type=str, required=True)
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--out-dir", type=str, required=True)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num-viz", type=int, default=8)
    parser.add_argument("--num-viz-negatives", type=int, default=2)
    parser.add_argument("--audit-indices", type=int, default=64)
    parser.add_argument("--fd-dense-points", type=int, default=4096)
    parser.add_argument("--curvature-thresholds", type=float, nargs="*", default=[0.0, 0.05, 0.1, 0.25, 0.5, 1.0])
    parser.add_argument(
        "--geometry",
        type=str,
        default="euclidean",
        choices=["euclidean", "equiaffine", "affine"],
    )
    parser.add_argument("--equiaffine-eps", type=float, default=1e-8)
    args = parser.parse_args()

    checkpoint_dir = Path(args.checkpoint_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = json.loads((checkpoint_dir / "config.json").read_text())
    device = torch.device(args.device)
    derivative_geometry = args.geometry

    dataset = build_dataset_from_config(cfg, args.split)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=tangent_collate_fn,
        drop_last=False,
    )

    model = build_model_from_config(cfg, device)
    state_dict = torch.load(checkpoint_dir / "best_model.pt", map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    analytic_audit = audit_analytic_gt_consistency(dataset, num_dataset_indices=args.audit_indices)
    (out_dir / "analytic_gt_audit.json").write_text(json.dumps(analytic_audit, indent=2))

    # raw accumulators
    gt_first_num = []
    gt_second_num = []
    gt_second_norm_fd = []
    gt_second_norm_heron = []

    pred_first = []
    pred_second = []

    first_cos = []
    second_cos = []
    first_angle = []
    second_angle = []
    first_residual = []
    second_residual = []

    pred_first_norm = []
    pred_second_norm = []
    gt_first_norm = []
    gt_second_norm = []

    pos_eq1_mse = []
    pos_eq1_mae = []
    pos_eq1_rel = []
    pos_eq1_cos = []

    neg_eq1_mse = []
    neg_eq1_mae = []
    neg_eq1_rel = []
    neg_eq1_cos = []

    pos_eq2_mse = []
    pos_eq2_mae = []
    pos_eq2_rel = []
    pos_eq2_cos = []

    neg_eq2_mse = []
    neg_eq2_mae = []
    neg_eq2_rel = []
    neg_eq2_cos = []

    pos_proj1 = []
    neg_proj1 = []
    pos_proj2 = []
    neg_proj2 = []

    # top examples by numerical curvature
    top_examples = []

    global_dataset_index = 0

    with torch.no_grad():
        for batch in loader:
            batch = move_batch(batch, device)

            anchor_out = model(batch.anchor)
            positive_out = model(batch.positive)

            B, M, K, _ = batch.negatives.shape
            flat_neg = batch.negatives.view(B * M, K, 2)
            neg_out = model(flat_neg)

            anchor_f1 = anchor_out["field_first"]
            pos_f1 = positive_out["field_first"]
            neg_f1 = neg_out["field_first"].view(B, M, K, 2)

            anchor_op = anchor_out["operator"]
            pos_op = positive_out["operator"]
            neg_op = neg_out["operator"].view(B, M, K, K)

            anchor_f2 = apply_W(anchor_op, anchor_f1)
            pos_f2 = apply_W(pos_op, pos_f1)
            neg_f2 = apply_W(neg_op.view(B * M, K, K), neg_out["field_first"]).view(B, M, K, 2)

            if derivative_geometry == "euclidean":
                target_f1 = model.apply_linear_map_to_field(batch.transform_matrix, anchor_f1)
                target_f2 = model.apply_linear_map_to_field(batch.transform_matrix, anchor_f2)
            else:
                target_f1 = apply_affine_map_to_field(batch.transform_matrix, anchor_f1, derivative_order=1)
                target_f2 = apply_affine_map_to_field(batch.transform_matrix, anchor_f2, derivative_order=2)

            # ===== per-sample numerical GT from ACTUAL sampled curve =====
            gt1_list = []
            gt2_list = []
            gt2_norm_list = []
            gt2_heron_list = []

            anchor_center_idx_cpu = batch.anchor_center_index.detach().cpu().numpy()
            for b in range(B):
                dataset_idx = global_dataset_index + b

                # Reconstruct the ACTUAL curve_points used for this dataset index.
                rng = dataset._make_rng(dataset_idx)
                curve_points, _, _ = dataset._get_curve(rng, dataset_idx)

                gt1, gt2, _, _ = numerical_arc_length_derivatives_on_actual_curve(
                    curve_points=np.asarray(curve_points, dtype=np.float64),
                    anchor_index=int(anchor_center_idx_cpu[b]),
                    dense_num_points=args.fd_dense_points,
                    geometry=derivative_geometry,
                    equiaffine_eps=args.equiaffine_eps,
                )
                gt2_h = heron_three_point_curvature_magnitude(
                    curve_points=np.asarray(curve_points, dtype=np.float64),
                    anchor_index=int(anchor_center_idx_cpu[b]),
                    dense_num_points=args.fd_dense_points,
                )

                gt1_list.append(gt1)
                gt2_list.append(gt2)
                gt2_norm_list.append(np.linalg.norm(gt2))
                gt2_heron_list.append(gt2_h)

            gt1_t = torch.as_tensor(np.stack(gt1_list, axis=0), dtype=batch.anchor.dtype, device=device)
            gt2_t = torch.as_tensor(np.stack(gt2_list, axis=0), dtype=batch.anchor.dtype, device=device)
            gt2_norm_t = torch.as_tensor(np.asarray(gt2_norm_list), dtype=batch.anchor.dtype, device=device)

            gt_first_num.extend(gt1_list)
            gt_second_num.extend(gt2_list)
            gt_second_norm_fd.extend(gt2_norm_list)
            gt_second_norm_heron.extend(gt2_heron_list)

            p1 = anchor_out["center_first"]
            p2 = anchor_out["center_second"]

            pos_center_pred_f1 = positive_out["center_first"]
            pos_center_pred_f2 = positive_out["center_second"]

            neg_center_pred_f1 = neg_out["center_first"].view(B, M, 2)
            neg_center_pred_f2 = neg_out["center_second"].view(B, M, 2)

            if derivative_geometry == "euclidean":
                pos_center_gt_f1 = apply_linear_map_to_vecs(batch.transform_matrix, gt1_t)
                pos_center_gt_f2 = apply_linear_map_to_vecs(batch.transform_matrix, gt2_t)
            else:
                pos_center_gt_f1 = apply_affine_map_to_vecs(batch.transform_matrix, gt1_t, derivative_order=1)
                pos_center_gt_f2 = apply_affine_map_to_vecs(batch.transform_matrix, gt2_t, derivative_order=2)

            c1 = F.cosine_similarity(p1, gt1_t, dim=-1)
            c2 = F.cosine_similarity(p2, gt2_t, dim=-1)
            a1 = torch.rad2deg(torch.acos(torch.clamp(c1, -1.0, 1.0)))
            a2 = torch.rad2deg(torch.acos(torch.clamp(c2, -1.0, 1.0)))

            first_cos.extend(c1.cpu().numpy().tolist())
            second_cos.extend(c2.cpu().numpy().tolist())
            first_angle.extend(a1.cpu().numpy().tolist())
            second_angle.extend(a2.cpu().numpy().tolist())
            first_residual.extend(frame_residual_angles_deg(p1, gt1_t).tolist())
            second_residual.extend(frame_residual_angles_deg(p2, gt2_t).tolist())
            pred_first_norm.extend(p1.norm(dim=-1).cpu().numpy().tolist())
            pred_second_norm.extend(p2.norm(dim=-1).cpu().numpy().tolist())
            gt_first_norm.extend(gt1_t.norm(dim=-1).cpu().numpy().tolist())
            gt_second_norm.extend(gt2_t.norm(dim=-1).cpu().numpy().tolist())

            # ===== equivariance stats: W =====
            diff_pos1 = pos_f1 - target_f1
            pos_eq1_mse.extend(diff_pos1.pow(2).mean(dim=(1, 2)).cpu().numpy().tolist())
            pos_eq1_mae.extend(diff_pos1.abs().mean(dim=(1, 2)).cpu().numpy().tolist())
            pos_eq1_rel.extend(
                (diff_pos1.norm(dim=(1, 2)) / (target_f1.norm(dim=(1, 2)) + 1e-12)).cpu().numpy().tolist()
            )
            pos_eq1_cos.extend(flatten_cos(pos_f1, target_f1).cpu().numpy().tolist())

            target_f1_m = target_f1.unsqueeze(1).expand_as(neg_f1)
            diff_neg1 = neg_f1 - target_f1_m
            neg_eq1_mse.extend(diff_neg1.pow(2).mean(dim=(2, 3)).reshape(-1).cpu().numpy().tolist())
            neg_eq1_mae.extend(diff_neg1.abs().mean(dim=(2, 3)).reshape(-1).cpu().numpy().tolist())

            denom1 = target_f1.norm(dim=(1, 2)).unsqueeze(1).expand(B, M)
            neg_eq1_rel.extend(
                (diff_neg1.norm(dim=(2, 3)) / (denom1 + 1e-12)).reshape(-1).cpu().numpy().tolist()
            )
            neg_eq1_cos.extend(
                F.cosine_similarity(neg_f1.reshape(B * M, -1), target_f1_m.reshape(B * M, -1), dim=-1).cpu().numpy().tolist()
            )

            # ===== equivariance stats: W² =====
            diff_pos2 = pos_f2 - target_f2
            pos_eq2_mse.extend(diff_pos2.pow(2).mean(dim=(1, 2)).cpu().numpy().tolist())
            pos_eq2_mae.extend(diff_pos2.abs().mean(dim=(1, 2)).cpu().numpy().tolist())
            pos_eq2_rel.extend(
                (diff_pos2.norm(dim=(1, 2)) / (target_f2.norm(dim=(1, 2)) + 1e-12)).cpu().numpy().tolist()
            )
            pos_eq2_cos.extend(flatten_cos(pos_f2, target_f2).cpu().numpy().tolist())

            target_f2_m = target_f2.unsqueeze(1).expand_as(neg_f2)
            diff_neg2 = neg_f2 - target_f2_m
            neg_eq2_mse.extend(diff_neg2.pow(2).mean(dim=(2, 3)).reshape(-1).cpu().numpy().tolist())
            neg_eq2_mae.extend(diff_neg2.abs().mean(dim=(2, 3)).reshape(-1).cpu().numpy().tolist())

            denom2 = target_f2.norm(dim=(1, 2)).unsqueeze(1).expand(B, M)
            neg_eq2_rel.extend(
                (diff_neg2.norm(dim=(2, 3)) / (denom2 + 1e-12)).reshape(-1).cpu().numpy().tolist()
            )
            neg_eq2_cos.extend(
                F.cosine_similarity(neg_f2.reshape(B * M, -1), target_f2_m.reshape(B * M, -1), dim=-1).cpu().numpy().tolist()
            )

            # ===== projection similarities =====
            z_anchor1 = model.project_field(target_f1)
            z_pos1 = model.project_field(pos_f1)
            z_neg1 = model.project_field(neg_f1.view(B * M, K, 2)).view(B, M, -1)

            pos_proj1.extend((z_anchor1 * z_pos1).sum(dim=-1).cpu().numpy().tolist())
            neg_proj1.extend(torch.einsum("bd,bmd->bm", z_anchor1, z_neg1).reshape(-1).cpu().numpy().tolist())

            z_anchor2 = model.project_field(target_f2)
            z_pos2 = model.project_field(pos_f2)
            z_neg2 = model.project_field(neg_f2.view(B * M, K, 2)).view(B, M, -1)

            pos_proj2.extend((z_anchor2 * z_pos2).sum(dim=-1).cpu().numpy().tolist())
            neg_proj2.extend(torch.einsum("bd,bmd->bm", z_anchor2, z_neg2).reshape(-1).cpu().numpy().tolist())

            # ===== keep high-curvature examples for visualization =====
            for b in range(B):
                sample = {
                    "dataset_idx": global_dataset_index + b,
                    "gt_second_norm_fd": float(gt2_norm_list[b]),
                    "gt_second_norm_heron": float(gt2_heron_list[b]),

                    "anchor_pts": batch.anchor[b].cpu().numpy(),
                    "pos_pts": batch.positive[b].cpu().numpy(),
                    "neg_pts": batch.negatives[b, :args.num_viz_negatives].cpu().numpy(),

                    "anchor_f1": anchor_f1[b].cpu().numpy(),
                    "pos_f1": pos_f1[b].cpu().numpy(),
                    "neg_f1": neg_f1[b, :args.num_viz_negatives].cpu().numpy(),

                    "anchor_f2": anchor_f2[b].cpu().numpy(),
                    "pos_f2": pos_f2[b].cpu().numpy(),
                    "neg_f2": neg_f2[b, :args.num_viz_negatives].cpu().numpy(),

                    "anchor_center_pred_f1": p1[b].cpu().numpy(),
                    "anchor_center_gt_f1": gt1_t[b].cpu().numpy(),
                    "anchor_center_pred_f2": p2[b].cpu().numpy(),
                    "anchor_center_gt_f2": gt2_t[b].cpu().numpy(),

                    "pos_center_pred_f1": pos_center_pred_f1[b].cpu().numpy(),
                    "pos_center_gt_f1": pos_center_gt_f1[b].cpu().numpy(),
                    "pos_center_pred_f2": pos_center_pred_f2[b].cpu().numpy(),
                    "pos_center_gt_f2": pos_center_gt_f2[b].cpu().numpy(),

                    "neg_center_pred_f1": neg_center_pred_f1[b, :args.num_viz_negatives].cpu().numpy(),
                    "neg_center_gt_f1": np.repeat(
                        gt1_t[b].cpu().numpy()[None, :],
                        args.num_viz_negatives,
                        axis=0
                    ),
                    "neg_center_pred_f2": neg_center_pred_f2[b, :args.num_viz_negatives].cpu().numpy(),
                    "neg_center_gt_f2": np.repeat(
                        gt2_t[b].cpu().numpy()[None, :],
                        args.num_viz_negatives,
                        axis=0
                    ),
                }

                # Use Heron curvature as the ranking signal, not equiaffine FD magnitude.
                score = sample["gt_second_norm_heron"]

                # Skip almost-straight examples.
                if score < 1.0:
                    continue

                if len(top_examples) < args.num_viz:
                    heapq.heappush(top_examples, (score, sample))
                else:
                    if score > top_examples[0][0]:
                        heapq.heapreplace(top_examples, (score, sample))

            global_dataset_index += B

    # convert to arrays
    first_cos_arr = np.asarray(first_cos)
    second_cos_arr = np.asarray(second_cos)
    first_angle_arr = np.asarray(first_angle)
    second_angle_arr = np.asarray(second_angle)
    first_residual_arr = np.asarray(first_residual)
    second_residual_arr = np.asarray(second_residual)
    gt_second_norm_arr = np.asarray(gt_second_norm_fd)
    gt_second_norm_heron_arr = np.asarray(gt_second_norm_heron)
    pred_second_norm_arr = np.asarray(pred_second_norm)

    # thresholded second-derivative stats
    second_thresholded = {}
    for thr in args.curvature_thresholds:
        mask = gt_second_norm_arr > thr
        second_thresholded[str(thr)] = {
            "num_samples": int(mask.sum()),
            "fraction": float(mask.mean()),
            "second_cos": summarize_array(second_cos_arr[mask]),
            "second_angle_deg": summarize_array(second_angle_arr[mask]),
            "second_residual_angle_deg": {
                **summarize_array(second_residual_arr[mask]),
                **circular_summary_deg(second_residual_arr[mask]),
            },
            "pred_second_norm": summarize_array(pred_second_norm_arr[mask]),
            "gt_second_norm": summarize_array(gt_second_norm_arr[mask]),
        }

    summary = {
        "checkpoint_dir": str(checkpoint_dir),
        "split": args.split,
        "config_used": cfg,
        "derivative_geometry": derivative_geometry,
        "equiaffine_eps": float(args.equiaffine_eps),
        "analytic_gt_audit": analytic_audit,

        "curvature_actual_geometry": {
            "gt_second_norm_fd": summarize_array(gt_second_norm_arr),
            "gt_second_norm_heron": summarize_array(gt_second_norm_heron_arr),
        },

        "derivatives_numerical_gt": {
            "first_cos": summarize_array(first_cos_arr),
            "second_cos": summarize_array(second_cos_arr),
            "first_angle_deg": summarize_array(first_angle_arr),
            "second_angle_deg": summarize_array(second_angle_arr),
            "first_residual_angle_deg": {
                **summarize_array(first_residual_arr),
                **circular_summary_deg(first_residual_arr),
            },
            "second_residual_angle_deg": {
                **summarize_array(second_residual_arr),
                **circular_summary_deg(second_residual_arr),
            },
            "pred_first_norm": summarize_array(np.asarray(pred_first_norm)),
            "pred_second_norm": summarize_array(np.asarray(pred_second_norm)),
            "gt_first_norm": summarize_array(np.asarray(gt_first_norm)),
            "gt_second_norm": summarize_array(np.asarray(gt_second_norm)),
            "second_thresholded": second_thresholded,
        },

        "equivariance_W": {
            "positive_mse": summarize_array(np.asarray(pos_eq1_mse)),
            "positive_mae": summarize_array(np.asarray(pos_eq1_mae)),
            "positive_relative_l2": summarize_array(np.asarray(pos_eq1_rel)),
            "positive_cos": summarize_array(np.asarray(pos_eq1_cos)),
            "negative_mse": summarize_array(np.asarray(neg_eq1_mse)),
            "negative_mae": summarize_array(np.asarray(neg_eq1_mae)),
            "negative_relative_l2": summarize_array(np.asarray(neg_eq1_rel)),
            "negative_cos": summarize_array(np.asarray(neg_eq1_cos)),
        },

        "equivariance_W2": {
            "positive_mse": summarize_array(np.asarray(pos_eq2_mse)),
            "positive_mae": summarize_array(np.asarray(pos_eq2_mae)),
            "positive_relative_l2": summarize_array(np.asarray(pos_eq2_rel)),
            "positive_cos": summarize_array(np.asarray(pos_eq2_cos)),
            "negative_mse": summarize_array(np.asarray(neg_eq2_mse)),
            "negative_mae": summarize_array(np.asarray(neg_eq2_mae)),
            "negative_relative_l2": summarize_array(np.asarray(neg_eq2_rel)),
            "negative_cos": summarize_array(np.asarray(neg_eq2_cos)),
        },

        "projection_similarity": {
            "W_positive": summarize_array(np.asarray(pos_proj1)),
            "W_negative": summarize_array(np.asarray(neg_proj1)),
            "W2_positive": summarize_array(np.asarray(pos_proj2)),
            "W2_negative": summarize_array(np.asarray(neg_proj2)),
        },
    }

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    np.savez(
        out_dir / "raw_metrics.npz",
        first_cos=first_cos_arr,
        second_cos=second_cos_arr,
        first_angle_deg=first_angle_arr,
        second_angle_deg=second_angle_arr,
        first_residual_angle_deg=first_residual_arr,
        second_residual_angle_deg=second_residual_arr,
        gt_second_norm_fd=gt_second_norm_arr,
        gt_second_norm_heron=gt_second_norm_heron_arr,
        pred_second_norm=pred_second_norm_arr,
        pos_eq1_cos=np.asarray(pos_eq1_cos),
        neg_eq1_cos=np.asarray(neg_eq1_cos),
        pos_eq2_cos=np.asarray(pos_eq2_cos),
        neg_eq2_cos=np.asarray(neg_eq2_cos),
        pos_proj1=np.asarray(pos_proj1),
        neg_proj1=np.asarray(neg_proj1),
        pos_proj2=np.asarray(pos_proj2),
        neg_proj2=np.asarray(neg_proj2),
    )

    if derivative_geometry == "euclidean":
        deriv1_label = "dx/ds"
        deriv2_label = "d²x/ds²"
        gt2_norm_title = "GT second-derivative norm (Euclidean)"
    else:
        deriv1_label = "dx/ds_aff"
        deriv2_label = "d²x/ds_aff²"
        gt2_norm_title = f"GT second-derivative norm ({derivative_geometry})"
    gt2_norm_xlabel = f"||{deriv2_label}||"
    # plots
    plot_hist(gt_second_norm_arr, gt2_norm_title, gt2_norm_xlabel, out_dir / "hist_gt_second_norm_fd.png")
    plot_hist(gt_second_norm_heron_arr, "GT curvature magnitude (Heron on actual geometry)", "kappa", out_dir / "hist_gt_second_norm_heron.png")

    plot_hist(first_cos_arr, f"First derivative cosine ({deriv1_label})", "cos(pred_first, gt_first_num)",
              out_dir / "hist_first_cos.png")
    plot_hist(second_cos_arr, f"Second derivative cosine ({deriv2_label})", "cos(pred_second, gt_second_num)",
              out_dir / "hist_second_cos.png")

    plot_hist(first_residual_arr, "First residual angle", "deg", out_dir / "hist_first_residual_angle.png")
    plot_hist(second_residual_arr, "Second residual angle", "deg", out_dir / "hist_second_residual_angle.png")

    plot_hist(np.asarray(pos_eq1_cos), "W equivariance: positive cosine", "cos(target_W, positive_W)", out_dir / "hist_W_pos_cos.png")
    plot_hist(np.asarray(neg_eq1_cos), "W equivariance: negative cosine", "cos(target_W, negative_W)", out_dir / "hist_W_neg_cos.png")
    plot_hist(np.asarray(pos_eq2_cos), "W² equivariance: positive cosine", "cos(target_W2, positive_W2)", out_dir / "hist_W2_pos_cos.png")
    plot_hist(np.asarray(neg_eq2_cos), "W² equivariance: negative cosine", "cos(target_W2, negative_W2)", out_dir / "hist_W2_neg_cos.png")

    plot_hist(np.asarray(pos_proj1), "Projection similarity on W: positives", "dot(z_anchor_W, z_pos_W)", out_dir / "hist_proj_W_pos.png")
    plot_hist(np.asarray(neg_proj1), "Projection similarity on W: negatives", "dot(z_anchor_W, z_neg_W)", out_dir / "hist_proj_W_neg.png")
    plot_hist(np.asarray(pos_proj2), "Projection similarity on W²: positives", "dot(z_anchor_W2, z_pos_W2)", out_dir / "hist_proj_W2_pos.png")
    plot_hist(np.asarray(neg_proj2), "Projection similarity on W²: negatives", "dot(z_anchor_W2, z_neg_W2)", out_dir / "hist_proj_W2_neg.png")

    # scatter plots for your hypothesis
    gt_second_norm_safe = np.maximum(gt_second_norm_arr, 1e-6)
    plot_scatter(
        gt_second_norm_safe,
        second_cos_arr,
        f"Second cosine vs GT second-derivative norm ({derivative_geometry})",
        gt2_norm_xlabel,
        "cos(pred_second, gt_second)",
        out_dir / "scatter_second_cos_vs_gt_curvature.png",
    )
    plot_scatter(
        gt_second_norm_safe,
        pred_second_norm_arr,
        f"Pred second norm vs GT second-derivative norm ({derivative_geometry})",
        gt2_norm_xlabel,
        "||pred_second||",
        out_dir / "scatter_pred_second_norm_vs_gt_curvature.png",
    )

    # save top high-curvature examples
    top_examples_sorted = sorted(top_examples, key=lambda x: x[0], reverse=True)
    for rank, (_, ex) in enumerate(top_examples_sorted):
        save_example_figure(ex, out_dir / f"example_{rank:02d}.png", args.num_viz_negatives)
        save_center_overlay_figure(ex, out_dir / f"example_center_overlay_{rank:02d}.png", args.num_viz_negatives)

    # quick text report
    with open(out_dir / "quick_report.txt", "w") as f:
        f.write("=== IMPORTANT ===\n")
        f.write(f"Derivative geometry mode: {derivative_geometry}\n")
        if derivative_geometry in ("equiaffine", "affine"):
            f.write(
                "Equiaffine mode uses the dense Euclidean-arc-length parameter as a temporary parameter t, "
                "then converts x_t, x_tt to x_{s_ea}, x_{s_ea s_ea}. "
                "The Heron curvature quantity remains a Euclidean curvature cross-check only.\n"
            )
        f.write("Use numerical_gt results below for derivative interpretation.\n")
        f.write("If analytic_gt_audit.point_match_error is not ~0, the old analytic derivative evaluation is invalid.\n\n")
        f.write("=== analytic gt audit ===\n")
        f.write(json.dumps(analytic_audit, indent=2))
        f.write("\n\n=== curvature on actual geometry ===\n")
        f.write(json.dumps(summary["curvature_actual_geometry"], indent=2))
        f.write("\n\n=== derivatives (numerical GT on actual geometry) ===\n")
        f.write(json.dumps(summary["derivatives_numerical_gt"], indent=2))
        f.write("\n\n=== W equivariance ===\n")
        f.write(json.dumps(summary["equivariance_W"], indent=2))
        f.write("\n\n=== W^2 equivariance ===\n")
        f.write(json.dumps(summary["equivariance_W2"], indent=2))
        f.write("\n\n=== projection similarity ===\n")
        f.write(json.dumps(summary["projection_similarity"], indent=2))

    print(f"Saved analysis to: {out_dir}")


if __name__ == "__main__":
    main()
