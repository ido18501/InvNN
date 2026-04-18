
from __future__ import annotations

import argparse
import json
import math
import os
import random
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from datasets.tangent_dataset import PregeneratedCurveBank
from models.tangent_model import TangentOperatorModel
from utils.derivatives import compute_fourier_arc_length_derivatives


# -----------------------------
# Utilities
# -----------------------------

EPS = 1e-12


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def parse_int_list(text: str) -> list[int]:
    return [int(x.strip()) for x in str(text).split(",") if x.strip()]


def parse_curve_indices(text: str | None) -> list[int] | None:
    if text is None:
        return None
    vals = [x.strip() for x in text.split(",") if x.strip()]
    return [int(v) for v in vals] if vals else None


def trimmed_mean(arr: np.ndarray, proportion_to_cut: float) -> float:
    arr = np.asarray(arr, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan")
    if proportion_to_cut <= 0.0:
        return float(arr.mean())
    if proportion_to_cut >= 0.5:
        raise ValueError("proportion_to_cut must be < 0.5")
    arr = np.sort(arr)
    k = int(math.floor(proportion_to_cut * arr.size))
    if 2 * k >= arr.size:
        return float(arr.mean())
    return float(arr[k: arr.size - k].mean())


def mad(arr: np.ndarray) -> float:
    arr = np.asarray(arr, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan")
    med = np.median(arr)
    return float(np.median(np.abs(arr - med)))


def summarize_array(arr: np.ndarray) -> dict[str, float]:
    arr = np.asarray(arr, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {k: float("nan") for k in [
            "count", "mean", "std", "median", "mad",
            "p01", "p05", "p10", "p25", "p50", "p75", "p90", "p95", "p99",
            "trimmed_mean_90", "trimmed_mean_95",
        ]}
    q = np.percentile(arr, [1, 5, 10, 25, 50, 75, 90, 95, 99])
    return {
        "count": float(arr.size),
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=0)),
        "median": float(np.median(arr)),
        "mad": mad(arr),
        "p01": float(q[0]),
        "p05": float(q[1]),
        "p10": float(q[2]),
        "p25": float(q[3]),
        "p50": float(q[4]),
        "p75": float(q[5]),
        "p90": float(q[6]),
        "p95": float(q[7]),
        "p99": float(q[8]),
        "trimmed_mean_90": trimmed_mean(arr, 0.05),
        "trimmed_mean_95": trimmed_mean(arr, 0.025),
    }


def flatten_summary(prefix: str, stats: dict[str, float]) -> dict[str, float]:
    return {f"{prefix}__{k}": v for k, v in stats.items()}


def save_json(obj: Any, path: str | Path) -> None:
    Path(path).write_text(json.dumps(obj, indent=2, sort_keys=True))


def make_intrinsic_patch(curve: np.ndarray, center: int, patch_size: int) -> np.ndarray:
    r = patch_size // 2
    idx = [(center + off) % len(curve) for off in range(-r, r + 1)]
    patch = curve[idx].copy()
    patch -= patch[r]
    return patch.astype(np.float32)


def batched_model_inference(
    model: TangentOperatorModel,
    curve: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    patches = np.stack(
        [make_intrinsic_patch(curve, i, model.patch_size) for i in range(len(curve))],
        axis=0,
    )
    preds = []
    weights_list = []
    with torch.no_grad():
        for s in range(0, len(patches), batch_size):
            x = torch.tensor(patches[s:s + batch_size], dtype=torch.float32, device=device)
            out = model(x)
            preds.append(out["pred"].detach().cpu().numpy())
            weights_list.append(out["weights"].detach().cpu().numpy())
    return np.concatenate(preds, axis=0), np.concatenate(weights_list, axis=0)


def build_cyclic_matrix(weights: np.ndarray) -> np.ndarray:
    N, K = weights.shape
    r = K // 2
    W = np.zeros((N, N), dtype=np.float64)
    for i in range(N):
        for j, off in enumerate(range(-r, r + 1)):
            col = (i + off) % N
            W[i, col] = float(weights[i, j])
    return W


def compute_curvature(gt1: np.ndarray, gt2: np.ndarray) -> np.ndarray:
    fn = np.maximum(np.linalg.norm(gt1, axis=1), EPS)
    cross = gt1[:, 0] * gt2[:, 1] - gt1[:, 1] * gt2[:, 0]
    kappa = np.abs(cross) / (fn ** 3 + EPS)
    return kappa


def compute_step_size(curve: np.ndarray) -> np.ndarray:
    return np.linalg.norm(np.roll(curve, -1, axis=0) - curve, axis=1)


def pointwise_pair_metrics(pred: np.ndarray, gt: np.ndarray) -> dict[str, np.ndarray]:
    pred = np.asarray(pred, dtype=np.float64)
    gt = np.asarray(gt, dtype=np.float64)

    pred_norm = np.maximum(np.linalg.norm(pred, axis=-1), EPS)
    gt_norm = np.maximum(np.linalg.norm(gt, axis=-1), EPS)
    cos = np.sum(pred * gt, axis=-1) / (pred_norm * gt_norm)
    cos = np.clip(cos, -1.0, 1.0)
    angle_deg = np.degrees(np.arccos(cos))
    abs_cos = np.abs(cos)
    norm_ratio = pred_norm / gt_norm
    sqerr = np.mean((pred - gt) ** 2, axis=-1)
    sqerr_sign_corrected = np.minimum(
        np.mean((pred - gt) ** 2, axis=-1),
        np.mean((pred + gt) ** 2, axis=-1),
    )
    return {
        "cosine": cos,
        "abs_cosine": abs_cos,
        "angle_deg": angle_deg,
        "norm_ratio": norm_ratio,
        "mse": sqerr,
        "mse_sign_corrected": sqerr_sign_corrected,
        "pred_norm": pred_norm,
        "gt_norm": gt_norm,
    }


def matrix_row_pairwise_cosines(mat: np.ndarray, max_rows: int = 512, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = mat.shape[0]
    if n == 0:
        return np.zeros((0,), dtype=np.float64)
    idx = np.arange(n)
    if n > max_rows:
        idx = rng.choice(idx, size=max_rows, replace=False)
    rows = mat[idx]
    norms = np.maximum(np.linalg.norm(rows, axis=1, keepdims=True), EPS)
    rows = rows / norms
    gram = rows @ rows.T
    tri = np.triu_indices_from(gram, k=1)
    return gram[tri]


def stencil_stats(weights: np.ndarray) -> dict[str, Any]:
    N, K = weights.shape
    c = K // 2
    left = weights[:, :c]
    center = weights[:, c]
    right = weights[:, c + 1:]
    right_flip = np.flip(right, axis=1) if right.shape[1] > 0 else np.zeros_like(left)
    anti_err = np.mean(np.abs(left + right_flip), axis=1) if right.shape[1] == left.shape[1] else np.full((N,), np.nan)
    return {
        "row_l2_norm": np.linalg.norm(weights, axis=1),
        "center_weight": center,
        "left_abs_mass": np.sum(np.abs(left), axis=1),
        "right_abs_mass": np.sum(np.abs(right), axis=1),
        "left_sum": np.sum(left, axis=1),
        "right_sum": np.sum(right, axis=1),
        "antisymmetry_error": anti_err,
        "pairwise_row_cosine": matrix_row_pairwise_cosines(weights, max_rows=min(512, N)),
    }


def matrix_spectral_stats(M: np.ndarray) -> dict[str, Any]:
    sv = np.linalg.svd(M, compute_uv=False)
    eig = np.linalg.eigvals(M)
    return {
        "fro_norm": float(np.linalg.norm(M, ord="fro")),
        "trace": float(np.trace(M).real),
        "singular_values": sv,
        "eigenvalues_real": eig.real,
        "eigenvalues_imag": eig.imag,
        "spectral_radius": float(np.max(np.abs(eig))) if eig.size else float("nan"),
    }


def band_mask(values: np.ndarray, band_name: str) -> np.ndarray:
    x = np.asarray(values, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.zeros((len(values),), dtype=bool)
    if band_name == "ALL":
        return np.isfinite(values)
    lo, hi = {
        "TRIM99": (0, 99),
        "TRIM95": (0, 95),
        "MID_2_98": (2, 98),
        "MID_10_90": (10, 90),
    }[band_name]
    qlo, qhi = np.percentile(x, [lo, hi])
    vals = np.asarray(values, dtype=np.float64)
    return np.isfinite(vals) & (vals >= qlo) & (vals <= qhi)


def maybe_downsample_indices(n: int, target: int) -> np.ndarray:
    if target <= 0 or target >= n:
        return np.arange(n)
    return np.unique(np.linspace(0, n - 1, target).astype(int))


def plot_hist(data: np.ndarray, title: str, path: Path, bins: int = 80) -> None:
    data = np.asarray(data, dtype=np.float64)
    data = data[np.isfinite(data)]
    if data.size == 0:
        return
    plt.figure(figsize=(8, 5))
    plt.hist(data, bins=bins)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def plot_ecdf(data: np.ndarray, title: str, path: Path) -> None:
    data = np.asarray(data, dtype=np.float64)
    data = data[np.isfinite(data)]
    if data.size == 0:
        return
    xs = np.sort(data)
    ys = np.arange(1, xs.size + 1) / xs.size
    plt.figure(figsize=(8, 5))
    plt.plot(xs, ys)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def plot_scatter_or_hexbin(x: np.ndarray, y: np.ndarray, xlabel: str, ylabel: str, title: str, path: Path, hexbin: bool = False) -> None:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if x.size == 0:
        return
    plt.figure(figsize=(7, 6))
    if hexbin and x.size > 1000:
        plt.hexbin(x, y, gridsize=50, mincnt=1)
        plt.colorbar()
    else:
        plt.scatter(x, y, s=5, alpha=0.4)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def plot_binned_summary(x: np.ndarray, y: np.ndarray, xlabel: str, ylabel: str, title: str, path: Path, nbins: int = 20) -> None:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if x.size == 0:
        return
    edges = np.quantile(x, np.linspace(0, 1, nbins + 1))
    edges = np.unique(edges)
    if edges.size < 3:
        return
    mids = []
    means = []
    meds = []
    for a, b in zip(edges[:-1], edges[1:]):
        m = (x >= a) & (x <= b if b == edges[-1] else x < b)
        if m.sum() == 0:
            continue
        mids.append(0.5 * (a + b))
        means.append(np.mean(y[m]))
        meds.append(np.median(y[m]))
    if not mids:
        return
    plt.figure(figsize=(8, 5))
    plt.plot(mids, means, label="mean")
    plt.plot(mids, meds, label="median")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def plot_heatmap(mat: np.ndarray, title: str, path: Path, vmax: float | None = None) -> None:
    plt.figure(figsize=(8, 6))
    plt.imshow(mat, aspect="auto", interpolation="nearest", vmax=vmax, vmin=None if vmax is None else -vmax)
    plt.colorbar()
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def plot_singular_values(sv: np.ndarray, title: str, path: Path) -> None:
    sv = np.asarray(sv, dtype=np.float64)
    if sv.size == 0:
        return
    plt.figure(figsize=(8, 5))
    plt.plot(sv)
    plt.yscale("log")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def plot_eigenvalues(real: np.ndarray, imag: np.ndarray, title: str, path: Path) -> None:
    plt.figure(figsize=(6, 6))
    plt.scatter(real, imag, s=8, alpha=0.7)
    plt.axhline(0.0, linewidth=1)
    plt.axvline(0.0, linewidth=1)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def plot_curve_quivers(curve: np.ndarray, vectors: list[tuple[str, np.ndarray]], title: str, path: Path, num_points: int) -> None:
    idx = maybe_downsample_indices(len(curve), num_points)
    plt.figure(figsize=(7, 7))
    plt.plot(curve[:, 0], curve[:, 1], linewidth=1)
    for name, vec in vectors:
        plt.quiver(
            curve[idx, 0],
            curve[idx, 1],
            vec[idx, 0],
            vec[idx, 1],
            angles="xy",
            scale_units="xy",
            scale=1.0,
            width=0.003,
            alpha=0.7,
            label=name,
        )
    plt.axis("equal")
    plt.legend()
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def plot_origin_arrows(vec: np.ndarray, title: str, path: Path, normalized: bool) -> None:
    v = np.asarray(vec, dtype=np.float64)
    if normalized:
        n = np.maximum(np.linalg.norm(v, axis=1, keepdims=True), EPS)
        v = v / n
    idx = maybe_downsample_indices(len(v), min(400, len(v)))
    v = v[idx]
    plt.figure(figsize=(6, 6))
    for a in v:
        plt.arrow(0, 0, a[0], a[1], head_width=0.02 if normalized else 0.01, length_includes_head=True, alpha=0.2)
    if normalized:
        t = np.linspace(0, 2 * np.pi, 400)
        plt.plot(np.cos(t), np.sin(t), linewidth=1)
    plt.axis("equal")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def plot_tip_trace(vec: np.ndarray, title: str, path: Path, normalized: bool = False) -> None:
    v = np.asarray(vec, dtype=np.float64)
    if normalized:
        n = np.maximum(np.linalg.norm(v, axis=1, keepdims=True), EPS)
        v = v / n
    plt.figure(figsize=(6, 6))
    plt.plot(v[:, 0], v[:, 1], linewidth=1)
    plt.scatter(v[:, 0], v[:, 1], s=4)
    plt.axis("equal")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def plot_line_series(series: list[tuple[str, np.ndarray]], title: str, path: Path) -> None:
    plt.figure(figsize=(11, 5))
    for name, arr in series:
        plt.plot(arr, label=name, linewidth=1)
    plt.legend()
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def plot_average_stencils(weights: np.ndarray, values: np.ndarray | None, title: str, path: Path, nbins: int = 5) -> None:
    K = weights.shape[1]
    offs = np.arange(-(K // 2), K // 2 + 1)
    plt.figure(figsize=(8, 5))
    plt.plot(offs, weights.mean(axis=0), label="all")
    if values is not None:
        vals = np.asarray(values, dtype=np.float64)
        mask = np.isfinite(vals)
        vals = vals[mask]
        ww = weights[mask]
        if vals.size > 0:
            qs = np.quantile(vals, np.linspace(0, 1, nbins + 1))
            qs = np.unique(qs)
            for i, (a, b) in enumerate(zip(qs[:-1], qs[1:])):
                m = (vals >= a) & (vals <= b if b == qs[-1] else vals < b)
                if m.sum() == 0:
                    continue
                plt.plot(offs, ww[m].mean(axis=0), label=f"bin_{i}")
    plt.legend()
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


@dataclass
class PairBlock:
    name: str
    pred: np.ndarray
    gt: np.ndarray
    metrics: dict[str, np.ndarray]


@dataclass
class CurveAnalysis:
    curve_idx: int
    n_points: int
    patch_size: int
    relative_patch_size: float
    pair_blocks: dict[str, PairBlock]
    curvature: np.ndarray
    step_size: np.ndarray
    weights: np.ndarray
    W: np.ndarray
    W2: np.ndarray
    stencil: dict[str, Any]
    W_stats: dict[str, Any]
    W2_stats: dict[str, Any]
    metadata: dict[str, Any]


def load_model(checkpoint_path: Path, device: torch.device) -> tuple[TangentOperatorModel, dict[str, Any]]:
    cfg = json.loads((checkpoint_path.parent / "config.json").read_text())
    model = TangentOperatorModel(
        patch_size=cfg["patch_size"],
        operator_hidden_dims=[int(x.strip()) for x in cfg["operator_hidden_dims"].split(",") if x.strip()],
        signature_hidden_dims=[int(x.strip()) for x in cfg["signature_hidden_dims"].split(",") if x.strip()],
        signature_out_dim=cfg["signature_out_dim"],
        signature_center_radius=cfg["signature_center_radius"],
        head_dropout=cfg["head_dropout"],
        normalize_projector=not cfg["disable_normalize_projector"],
        init_scale=cfg["operator_init_scale"],
        learn_scale=cfg["learn_output_scale"],
        centered_input_for_operator=not cfg["disable_centered_input_for_operator"],
    )
    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model, cfg


def analyze_single_curve(
    model: TangentOperatorModel,
    curve: np.ndarray,
    coeffs: Any,
    t_grid: np.ndarray,
    batch_size: int,
    device: torch.device,
    curve_idx: int,
    run_metadata: dict[str, Any],
) -> CurveAnalysis:
    curve = np.asarray(curve, dtype=np.float64)
    _, gt1, gt2 = compute_fourier_arc_length_derivatives(
        t=np.asarray(t_grid, dtype=np.float64),
        coeffs=coeffs,
        family="euclidean",
    )
    direct1, weights = batched_model_inference(model, curve, batch_size=batch_size, device=device)
    W = build_cyclic_matrix(weights)
    X = curve.astype(np.float64)
    global1 = W @ X
    global2_from_global1 = W @ global1
    global2_from_direct1 = W @ direct1
    W2 = W @ W

    pair_blocks = {}
    for name, pred, gt in [
        ("direct1_vs_gt1", direct1, gt1),
        ("global1_vs_gt1", global1, gt1),
        ("global2_from_global1_vs_gt2", global2_from_global1, gt2),
        ("global2_from_direct1_vs_gt2", global2_from_direct1, gt2),
        ("global1_vs_direct1", global1, direct1),
    ]:
        pair_blocks[name] = PairBlock(name=name, pred=pred, gt=gt, metrics=pointwise_pair_metrics(pred, gt))

    curvature = compute_curvature(gt1, gt2)
    step_size = compute_step_size(curve)
    stencil = stencil_stats(weights)
    W_stats = matrix_spectral_stats(W)
    W2_stats = matrix_spectral_stats(W2)

    return CurveAnalysis(
        curve_idx=curve_idx,
        n_points=len(curve),
        patch_size=model.patch_size,
        relative_patch_size=float(model.patch_size / max(len(curve), 1)),
        pair_blocks=pair_blocks,
        curvature=curvature,
        step_size=step_size,
        weights=weights,
        W=W,
        W2=W2,
        stencil=stencil,
        W_stats=W_stats,
        W2_stats=W2_stats,
        metadata=run_metadata,
    )


def per_curve_row(curve_res: CurveAnalysis) -> dict[str, Any]:
    row = {
        "curve_idx": curve_res.curve_idx,
        "n_points": curve_res.n_points,
        "patch_size": curve_res.patch_size,
        "relative_patch_size": curve_res.relative_patch_size,
        "curvature_mean": float(np.mean(curve_res.curvature)),
        "curvature_p95": float(np.percentile(curve_res.curvature, 95)),
        "step_size_mean": float(np.mean(curve_res.step_size)),
        "W_fro_norm": curve_res.W_stats["fro_norm"],
        "W_trace": curve_res.W_stats["trace"],
        "W_spectral_radius": curve_res.W_stats["spectral_radius"],
        "W2_fro_norm": curve_res.W2_stats["fro_norm"],
        "W2_spectral_radius": curve_res.W2_stats["spectral_radius"],
        "stencil_antisymmetry_error_mean": float(np.nanmean(curve_res.stencil["antisymmetry_error"])),
        "stencil_center_weight_mean": float(np.mean(curve_res.stencil["center_weight"])),
    }
    for pair_name, block in curve_res.pair_blocks.items():
        for metric_name, arr in block.metrics.items():
            row[f"{pair_name}__{metric_name}__mean"] = float(np.mean(arr))
            row[f"{pair_name}__{metric_name}__median"] = float(np.median(arr))
    return row


def collect_global_pointwise(curves: list[CurveAnalysis]) -> dict[str, dict[str, np.ndarray]]:
    out: dict[str, dict[str, list[np.ndarray]]] = {}
    for curve in curves:
        for pair_name, block in curve.pair_blocks.items():
            out.setdefault(pair_name, {})
            for metric_name, arr in block.metrics.items():
                out[pair_name].setdefault(metric_name, [])
                out[pair_name][metric_name].append(np.asarray(arr))
    return {
        pair_name: {metric_name: np.concatenate(parts, axis=0) for metric_name, parts in metric_dict.items()}
        for pair_name, metric_dict in out.items()
    }


def summarize_curves(curves: list[CurveAnalysis]) -> dict[str, Any]:
    global_arrays = collect_global_pointwise(curves)
    out: dict[str, Any] = {"global_pointwise": {}, "per_curve_summary": {}, "matrix_summary": {}, "trimmed_by_curvature": {}}

    for pair_name, metric_dict in global_arrays.items():
        out["global_pointwise"][pair_name] = {
            metric_name: summarize_array(arr) for metric_name, arr in metric_dict.items()
        }

    per_curve_df = pd.DataFrame([per_curve_row(c) for c in curves])
    per_curve_summary = {}
    for col in per_curve_df.columns:
        if col == "curve_idx":
            continue
        if pd.api.types.is_numeric_dtype(per_curve_df[col]):
            per_curve_summary[col] = summarize_array(per_curve_df[col].to_numpy(dtype=np.float64))
    out["per_curve_summary"] = per_curve_summary

    matrix_fields = {
        "W_fro_norm": np.array([c.W_stats["fro_norm"] for c in curves], dtype=np.float64),
        "W_trace": np.array([c.W_stats["trace"] for c in curves], dtype=np.float64),
        "W_spectral_radius": np.array([c.W_stats["spectral_radius"] for c in curves], dtype=np.float64),
        "W2_fro_norm": np.array([c.W2_stats["fro_norm"] for c in curves], dtype=np.float64),
        "W2_spectral_radius": np.array([c.W2_stats["spectral_radius"] for c in curves], dtype=np.float64),
        "antisymmetry_error_mean": np.array([np.nanmean(c.stencil["antisymmetry_error"]) for c in curves], dtype=np.float64),
    }
    out["matrix_summary"] = {k: summarize_array(v) for k, v in matrix_fields.items()}

    curvature = np.concatenate([c.curvature for c in curves], axis=0)
    target_pairs = {
        "first_derivative": "global1_vs_gt1",
        "second_derivative": "global2_from_global1_vs_gt2",
        "direct_first_derivative": "direct1_vs_gt1",
        "second_from_direct": "global2_from_direct1_vs_gt2",
    }
    bands = ["ALL", "TRIM99", "TRIM95", "MID_2_98", "MID_10_90"]
    for band in bands:
        mask = band_mask(curvature, band)
        out["trimmed_by_curvature"][band] = {}
        for label, pair_name in target_pairs.items():
            out["trimmed_by_curvature"][band][label] = {}
            for metric_name in ["abs_cosine", "norm_ratio", "mse"]:
                arr = global_arrays[pair_name][metric_name]
                out["trimmed_by_curvature"][band][label][metric_name] = summarize_array(arr[mask])

    return out


def save_pointwise_npz(curves: list[CurveAnalysis], path: Path) -> None:
    arrays: dict[str, np.ndarray] = {}
    for pair_name, metrics in collect_global_pointwise(curves).items():
        for metric_name, arr in metrics.items():
            arrays[f"{pair_name}__{metric_name}"] = arr
    arrays["curvature"] = np.concatenate([c.curvature for c in curves], axis=0)
    arrays["step_size"] = np.concatenate([c.step_size for c in curves], axis=0)
    arrays["relative_patch_size"] = np.concatenate([np.full((c.n_points,), c.relative_patch_size) for c in curves], axis=0)
    arrays["curve_idx"] = np.concatenate([np.full((c.n_points,), c.curve_idx) for c in curves], axis=0)
    np.savez_compressed(path, **arrays)


def write_readme(summary: dict[str, Any], output_dir: Path, curves: list[CurveAnalysis]) -> None:
    lines = []
    lines.append("Invariant operator analysis summary")
    lines.append("=" * 40)
    lines.append("")
    lines.append(f"Analyzed curves: {len(curves)}")
    if curves:
        lines.append(f"Patch size: {curves[0].patch_size}")
        lines.append(f"Relative patch size range: {min(c.relative_patch_size for c in curves):.6f} .. {max(c.relative_patch_size for c in curves):.6f}")
    lines.append("")
    for pair in ["direct1_vs_gt1", "global1_vs_gt1", "global2_from_global1_vs_gt2", "global2_from_direct1_vs_gt2", "global1_vs_direct1"]:
        if pair in summary["global_pointwise"]:
            abs_cos = summary["global_pointwise"][pair]["abs_cosine"]["mean"]
            mse = summary["global_pointwise"][pair]["mse"]["mean"]
            norm_ratio = summary["global_pointwise"][pair]["norm_ratio"]["median"]
            lines.append(f"{pair}: mean abs cos={abs_cos:.6f}, mean mse={mse:.6e}, median norm ratio={norm_ratio:.6f}")
    lines.append("")
    lines.append("Curvature-trimmed second derivative quality:")
    for band, data in summary["trimmed_by_curvature"].items():
        s = data["second_derivative"]["abs_cosine"]["mean"]
        m = data["second_derivative"]["mse"]["mean"]
        lines.append(f"  {band}: abs_cos mean={s:.6f}, mse mean={m:.6e}")
    lines.append("")
    lines.append("Matrix diagnostics:")
    for k, v in summary["matrix_summary"].items():
        lines.append(f"  {k}: mean={v['mean']:.6f}, median={v['median']:.6f}")
    (output_dir / "README_summary.txt").write_text("\n".join(lines))


def save_curve_artifacts(curve_res: CurveAnalysis, curve_dir: Path, num_points_visualized: int) -> None:
    ensure_dir(curve_dir)

    np.savez_compressed(
        curve_dir / "arrays.npz",
        curve_idx=curve_res.curve_idx,
        curvature=curve_res.curvature,
        step_size=curve_res.step_size,
        weights=curve_res.weights,
        W=curve_res.W,
        W2=curve_res.W2,
        **{f"{pair_name}__pred": block.pred for pair_name, block in curve_res.pair_blocks.items()},
        **{f"{pair_name}__gt": block.gt for pair_name, block in curve_res.pair_blocks.items()},
        **{f"{pair_name}__{metric}": arr for pair_name, block in curve_res.pair_blocks.items() for metric, arr in block.metrics.items()},
    )

    direct1 = curve_res.pair_blocks["direct1_vs_gt1"].pred
    gt1 = curve_res.pair_blocks["direct1_vs_gt1"].gt
    global1 = curve_res.pair_blocks["global1_vs_gt1"].pred
    gt2 = curve_res.pair_blocks["global2_from_global1_vs_gt2"].gt
    global2 = curve_res.pair_blocks["global2_from_global1_vs_gt2"].pred
    global2b = curve_res.pair_blocks["global2_from_direct1_vs_gt2"].pred

    curve = curve_res.pair_blocks["global1_vs_gt1"].gt * 0.0  # placeholder to recover n only
    # better: reconstruct curve via W? not stored; use gt/pred lengths only.
    # We still want actual curve for quivers, so store it in metadata if available.
    curve_actual = np.asarray(curve_res.metadata["curve"], dtype=np.float64)

    plot_curve_quivers(
        curve_actual,
        [("gt1", gt1), ("direct1", direct1), ("global1", global1)],
        "First derivative quivers",
        curve_dir / "curve_quiver_first.png",
        num_points_visualized,
    )
    plot_curve_quivers(
        curve_actual,
        [("gt2", gt2), ("global2_from_global1", global2), ("global2_from_direct1", global2b)],
        "Second derivative quivers",
        curve_dir / "curve_quiver_second.png",
        num_points_visualized,
    )

    for name, vec in [
        ("gt1", gt1),
        ("direct1", direct1),
        ("global1", global1),
        ("gt2", gt2),
        ("global2_from_global1", global2),
        ("global2_from_direct1", global2b),
    ]:
        plot_origin_arrows(vec, f"{name} origin arrows raw", curve_dir / f"origin_arrows_raw_{name}.png", normalized=False)
        plot_origin_arrows(vec, f"{name} origin arrows normalized", curve_dir / f"origin_arrows_norm_{name}.png", normalized=True)
        plot_tip_trace(vec, f"{name} tip trace raw", curve_dir / f"tip_trace_raw_{name}.png", normalized=False)
        plot_tip_trace(vec, f"{name} tip trace normalized", curve_dir / f"tip_trace_norm_{name}.png", normalized=True)

    vmax = float(np.max(np.abs(curve_res.weights))) if curve_res.weights.size else None
    plot_heatmap(curve_res.weights, "weights matrix", curve_dir / "weights_heatmap.png", vmax=vmax)
    vmaxW = float(np.percentile(np.abs(curve_res.W), 99)) if curve_res.W.size else None
    vmaxW2 = float(np.percentile(np.abs(curve_res.W2), 99)) if curve_res.W2.size else None
    plot_heatmap(curve_res.W, "W heatmap", curve_dir / "W_heatmap.png", vmax=vmaxW)
    plot_heatmap(curve_res.W2, "W^2 heatmap", curve_dir / "W2_heatmap.png", vmax=vmaxW2)

    plot_singular_values(curve_res.W_stats["singular_values"], "W singular values", curve_dir / "W_singular_values.png")
    plot_singular_values(curve_res.W2_stats["singular_values"], "W^2 singular values", curve_dir / "W2_singular_values.png")
    plot_eigenvalues(curve_res.W_stats["eigenvalues_real"], curve_res.W_stats["eigenvalues_imag"], "W eigenvalues", curve_dir / "W_eigenvalues.png")
    plot_eigenvalues(curve_res.W2_stats["eigenvalues_real"], curve_res.W2_stats["eigenvalues_imag"], "W^2 eigenvalues", curve_dir / "W2_eigenvalues.png")

    plot_average_stencils(curve_res.weights, None, "Average stencil", curve_dir / "stencil_average.png")
    plot_average_stencils(curve_res.weights, curve_res.curvature, "Average stencil by curvature bins", curve_dir / "stencil_by_curvature.png")
    plot_average_stencils(curve_res.weights, curve_res.step_size, "Average stencil by step-size bins", curve_dir / "stencil_by_step_size.png")

    plot_line_series(
        [
            ("direct1_cos", curve_res.pair_blocks["direct1_vs_gt1"].metrics["cosine"]),
            ("global1_cos", curve_res.pair_blocks["global1_vs_gt1"].metrics["cosine"]),
            ("global2_cos", curve_res.pair_blocks["global2_from_global1_vs_gt2"].metrics["cosine"]),
            ("global2_alt_cos", curve_res.pair_blocks["global2_from_direct1_vs_gt2"].metrics["cosine"]),
        ],
        "Cosines over curve index",
        curve_dir / "lineplot_cosines.png",
    )
    plot_line_series(
        [
            ("direct1_angle", curve_res.pair_blocks["direct1_vs_gt1"].metrics["angle_deg"]),
            ("global1_angle", curve_res.pair_blocks["global1_vs_gt1"].metrics["angle_deg"]),
            ("global2_angle", curve_res.pair_blocks["global2_from_global1_vs_gt2"].metrics["angle_deg"]),
            ("global2_alt_angle", curve_res.pair_blocks["global2_from_direct1_vs_gt2"].metrics["angle_deg"]),
        ],
        "Angles over curve index",
        curve_dir / "lineplot_angles.png",
    )
    plot_line_series(
        [
            ("direct1_norm_ratio", curve_res.pair_blocks["direct1_vs_gt1"].metrics["norm_ratio"]),
            ("global1_norm_ratio", curve_res.pair_blocks["global1_vs_gt1"].metrics["norm_ratio"]),
            ("global2_norm_ratio", curve_res.pair_blocks["global2_from_global1_vs_gt2"].metrics["norm_ratio"]),
            ("global2_alt_norm_ratio", curve_res.pair_blocks["global2_from_direct1_vs_gt2"].metrics["norm_ratio"]),
        ],
        "Norm ratios over curve index",
        curve_dir / "lineplot_norm_ratios.png",
    )
    plot_line_series(
        [
            ("curvature", curve_res.curvature),
            ("step_size", curve_res.step_size),
            ("global2_mse", curve_res.pair_blocks["global2_from_global1_vs_gt2"].metrics["mse"]),
            ("global2_alt_mse", curve_res.pair_blocks["global2_from_direct1_vs_gt2"].metrics["mse"]),
        ],
        "Curvature / step / mse over curve index",
        curve_dir / "lineplot_curvature_step_mse.png",
    )


def create_global_plots(curves: list[CurveAnalysis], global_dir: Path) -> None:
    pointwise = collect_global_pointwise(curves)
    curvature = np.concatenate([c.curvature for c in curves], axis=0)
    step_size = np.concatenate([c.step_size for c in curves], axis=0)

    targets = [
        ("direct1_vs_gt1", "direct1"),
        ("global1_vs_gt1", "global1"),
        ("global2_from_global1_vs_gt2", "global2_from_global1"),
        ("global2_from_direct1_vs_gt2", "global2_from_direct1"),
    ]
    for pair_name, short in targets:
        for metric in ["cosine", "abs_cosine", "norm_ratio", "mse"]:
            plot_hist(pointwise[pair_name][metric], f"{short} {metric} histogram", global_dir / f"hist_{short}_{metric}.png")
            plot_ecdf(pointwise[pair_name][metric], f"{short} {metric} ecdf", global_dir / f"ecdf_{short}_{metric}.png")
    plot_hist(curvature, "curvature histogram", global_dir / "hist_curvature.png")
    plot_hist(step_size, "step size histogram", global_dir / "hist_step_size.png")
    plot_ecdf(curvature, "curvature ecdf", global_dir / "ecdf_curvature.png")
    plot_ecdf(step_size, "step size ecdf", global_dir / "ecdf_step_size.png")

    second_abs = pointwise["global2_from_global1_vs_gt2"]["abs_cosine"]
    second_nr = pointwise["global2_from_global1_vs_gt2"]["norm_ratio"]
    second_mse = pointwise["global2_from_global1_vs_gt2"]["mse"]
    first_abs = pointwise["global1_vs_gt1"]["abs_cosine"]
    direct_abs = pointwise["direct1_vs_gt1"]["abs_cosine"]
    consistency = pointwise["global1_vs_direct1"]["abs_cosine"]

    plot_scatter_or_hexbin(curvature, second_abs, "curvature", "abs cos(second)", "curvature vs abs cos(second)", global_dir / "scatter_curvature_vs_second_abs_cos.png", hexbin=False)
    plot_scatter_or_hexbin(curvature, second_abs, "curvature", "abs cos(second)", "curvature vs abs cos(second) hexbin", global_dir / "hexbin_curvature_vs_second_abs_cos.png", hexbin=True)
    plot_scatter_or_hexbin(curvature, second_nr, "curvature", "norm ratio(second)", "curvature vs norm ratio(second)", global_dir / "scatter_curvature_vs_second_norm_ratio.png", hexbin=False)
    plot_scatter_or_hexbin(step_size, second_abs, "step size", "abs cos(second)", "step size vs abs cos(second)", global_dir / "scatter_step_vs_second_abs_cos.png", hexbin=False)
    plot_scatter_or_hexbin(step_size, second_abs, "step size", "abs cos(second)", "step size vs abs cos(second) hexbin", global_dir / "hexbin_step_vs_second_abs_cos.png", hexbin=True)
    plot_scatter_or_hexbin(step_size, second_nr, "step size", "norm ratio(second)", "step size vs norm ratio(second)", global_dir / "scatter_step_vs_second_norm_ratio.png", hexbin=False)
    plot_scatter_or_hexbin(first_abs, second_abs, "abs cos(first)", "abs cos(second)", "first vs second abs cos", global_dir / "scatter_first_vs_second_abs_cos.png", hexbin=False)
    plot_scatter_or_hexbin(direct_abs, consistency, "direct abs cos", "global-direct consistency", "direct quality vs global consistency", global_dir / "scatter_direct_quality_vs_global_consistency.png", hexbin=False)

    plot_binned_summary(curvature, second_abs, "curvature", "abs cos(second)", "binned curvature vs abs cos(second)", global_dir / "binned_curvature_vs_second_abs_cos.png")
    plot_binned_summary(curvature, second_nr, "curvature", "norm ratio(second)", "binned curvature vs norm ratio(second)", global_dir / "binned_curvature_vs_second_norm_ratio.png")
    plot_binned_summary(curvature, second_mse, "curvature", "mse(second)", "binned curvature vs mse(second)", global_dir / "binned_curvature_vs_second_mse.png")
    plot_binned_summary(step_size, second_abs, "step size", "abs cos(second)", "binned step size vs abs cos(second)", global_dir / "binned_step_vs_second_abs_cos.png")
    plot_binned_summary(step_size, second_nr, "step size", "norm ratio(second)", "binned step size vs norm ratio(second)", global_dir / "binned_step_vs_second_norm_ratio.png")
    plot_binned_summary(step_size, second_mse, "step size", "mse(second)", "binned step size vs mse(second)", global_dir / "binned_step_vs_second_mse.png")

    per_curve = pd.DataFrame([per_curve_row(c) for c in curves])
    for metric_col, fname in [
        ("global2_from_global1_vs_gt2__abs_cosine__mean", "boxplot_per_curve_second_abs_cos.png"),
        ("global2_from_global1_vs_gt2__norm_ratio__mean", "boxplot_per_curve_second_norm_ratio.png"),
        ("global2_from_global1_vs_gt2__mse__mean", "boxplot_per_curve_second_mse.png"),
    ]:
        if metric_col in per_curve.columns:
            plt.figure(figsize=(10, 5))
            plt.boxplot(per_curve[metric_col].to_numpy(dtype=np.float64), vert=False)
            plt.title(metric_col)
            plt.tight_layout()
            plt.savefig(global_dir / fname, dpi=180)
            plt.close()


def choose_curve_indices(bank: PregeneratedCurveBank, curve_indices: list[int] | None, num_curves: int, seed: int) -> list[int]:
    if curve_indices is not None and len(curve_indices) > 0:
        return curve_indices
    rng = np.random.default_rng(seed)
    n = len(bank)
    k = min(num_curves, n)
    return sorted(rng.choice(np.arange(n), size=k, replace=False).tolist())


def run_single_analysis(
    checkpoint: Path,
    bank_path: Path,
    output_dir: Path,
    curve_indices: list[int] | None,
    num_curves: int,
    num_curves_visualized: int,
    num_points_visualized: int,
    batch_size: int,
    device: str,
    random_seed: int,
    extra_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    ensure_dir(output_dir)
    global_dir = ensure_dir(output_dir / "global")
    curves_dir = ensure_dir(output_dir / "curves")
    set_seed(random_seed)
    dev = torch.device(device)
    model, cfg = load_model(checkpoint, dev)
    bank = PregeneratedCurveBank(bank_path)
    selected = choose_curve_indices(bank, curve_indices, num_curves, random_seed)

    curves: list[CurveAnalysis] = []
    for idx in selected:
        curve, coeffs, t_grid = bank.get(idx)
        if coeffs is None or t_grid is None:
            raise ValueError("This analysis requires coeffs and t_grid from PregeneratedCurveBank for analytic derivatives.")
        meta = {
            "checkpoint": str(checkpoint),
            "bank": str(bank_path),
            "config": cfg,
            "curve": np.asarray(curve, dtype=np.float64),
        }
        if extra_metadata:
            meta.update(extra_metadata)
        curves.append(
            analyze_single_curve(
                model=model,
                curve=np.asarray(curve, dtype=np.float64),
                coeffs=coeffs,
                t_grid=np.asarray(t_grid, dtype=np.float64),
                batch_size=batch_size,
                device=dev,
                curve_idx=idx,
                run_metadata=meta,
            )
        )

    summary = summarize_curves(curves)
    save_json(summary, global_dir / "global_summary.json")
    per_curve_df = pd.DataFrame([per_curve_row(c) for c in curves]).sort_values("curve_idx")
    per_curve_df.to_csv(global_dir / "per_curve_stats.csv", index=False)

    trim_rows = []
    for band, data in summary["trimmed_by_curvature"].items():
        for label, label_data in data.items():
            row = {"band": band, "target": label}
            for metric_name, stat_dict in label_data.items():
                row.update(flatten_summary(metric_name, stat_dict))
            trim_rows.append(row)
    pd.DataFrame(trim_rows).to_csv(global_dir / "trimmed_global_stats.csv", index=False)

    save_pointwise_npz(curves, global_dir / "pointwise_arrays.npz")
    create_global_plots(curves, global_dir)
    write_readme(summary, output_dir, curves)

    ranked = []
    for c in curves:
        score = float(np.mean(c.pair_blocks["global2_from_global1_vs_gt2"].metrics["abs_cosine"]))
        ranked.append((score, c.curve_idx))
    ranked.sort()
    save_json(
        {
            "worst_curves_by_second_abs_cos": [{"curve_idx": idx, "mean_abs_cos_second": score} for score, idx in ranked[: min(10, len(ranked))]],
            "best_curves_by_second_abs_cos": [{"curve_idx": idx, "mean_abs_cos_second": score} for score, idx in ranked[-min(10, len(ranked)):][::-1]],
        },
        global_dir / "best_worst_curves.json",
    )

    vis_set = set(selected[: min(num_curves_visualized, len(selected))])
    for c in curves:
        if c.curve_idx in vis_set:
            save_curve_artifacts(c, curves_dir / f"curve_{c.curve_idx}", num_points_visualized)

    return {
        "summary": summary,
        "selected_curve_indices": selected,
        "per_curve_df": per_curve_df,
    }


DATASET_NAMES = [
    "data_complex_f20_250to180",
    "data_complex_f20_500to300",
    "data_complex_f20_1000to500",
    "data_complex_f20_2000to1000",
    "data_complex_f20_3000to1500",
    "data_complex_f20_4000to2000",
]
PATCH_SIZES = [3, 5, 9, 13, 17, 21, 25, 31]


def infer_bank_path(banks_root: Path, dataset_name: str) -> Path:
    candidates = [
        banks_root / f"{dataset_name}.npz",
        banks_root / dataset_name / "test.npz",
        banks_root / dataset_name / "val.npz",
        banks_root / dataset_name / "train.npz",
        banks_root / dataset_name / "bank.npz",
        banks_root / dataset_name,
    ]
    for c in candidates:
        if c.is_file():
            return c
    raise FileNotFoundError(f"Could not infer bank path for dataset {dataset_name} under {banks_root}")


def find_run_dir(checkpoints_root: Path, dataset_name: str, patch_size: int) -> Path | None:
    ps_dir = checkpoints_root / dataset_name / "rank1" / f"ps{patch_size}"
    if not ps_dir.exists():
        return None
    candidates = []
    for nce_dir in sorted(ps_dir.glob("nce_*")):
        for temp_dir in sorted(nce_dir.glob("temp_*")):
            if (temp_dir / "DONE").exists() and (temp_dir / "best_model.pt").exists() and (temp_dir / "config.json").exists():
                candidates.append(temp_dir)
    if len(candidates) == 0:
        return None
    if len(candidates) > 1:
        raise RuntimeError(f"Expected exactly one completed run for {dataset_name} ps={patch_size}, found {len(candidates)}")
    return candidates[0]


def aggregate_sweep_results(run_rows: list[dict[str, Any]], output_dir: Path) -> None:
    if not run_rows:
        return
    df = pd.DataFrame(run_rows)
    df.to_csv(output_dir / "sweep_run_summary.csv", index=False)

    metric_pairs = [
        ("direct1_vs_gt1__abs_cosine_mean", "Direct first abs cos"),
        ("global1_vs_gt1__abs_cosine_mean", "Global first abs cos"),
        ("global2_from_global1_vs_gt2__abs_cosine_mean", "Global second abs cos"),
        ("global2_from_global1_vs_gt2__mse_mean", "Global second mse"),
        ("global1_vs_direct1__abs_cosine_mean", "Global/direct consistency"),
    ]

    for col, title in metric_pairs:
        if col not in df.columns:
            continue
        pivot = df.pivot(index="dataset_name", columns="patch_size", values=col).sort_index()
        plt.figure(figsize=(10, 5))
        plt.imshow(pivot.to_numpy(), aspect="auto", interpolation="nearest")
        plt.xticks(range(len(pivot.columns)), pivot.columns)
        plt.yticks(range(len(pivot.index)), pivot.index)
        plt.colorbar()
        plt.title(title)
        plt.tight_layout()
        plt.savefig(output_dir / f"heatmap_{col}.png", dpi=180)
        plt.close()

    df["relative_patch_size"] = df["patch_size"] / df["n_points_mean"]
    for col, title in metric_pairs:
        if col not in df.columns:
            continue
        plt.figure(figsize=(8, 5))
        plt.scatter(df["patch_size"], df[col], label="absolute patch")
        plt.scatter(df["relative_patch_size"], df[col], label="relative patch")
        plt.title(title)
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_dir / f"scatter_patch_effect_{col}.png", dpi=180)
        plt.close()

    grouped_dataset = df.groupby("dataset_name", as_index=False).mean(numeric_only=True)
    grouped_patch = df.groupby("patch_size", as_index=False).mean(numeric_only=True)

    grouped_dataset.to_csv(output_dir / "dataset_aggregates.csv", index=False)
    grouped_patch.to_csv(output_dir / "patch_aggregates.csv", index=False)

    summary = {
        "num_runs": int(len(df)),
        "datasets_present": sorted(df["dataset_name"].unique().tolist()),
        "patch_sizes_present": sorted(df["patch_size"].unique().tolist()),
    }
    save_json(summary, output_dir / "sweep_summary.json")


def run_sweep(
    checkpoints_root: Path,
    banks_root: Path,
    output_dir: Path,
    num_curves: int,
    num_curves_visualized: int,
    num_points_visualized: int,
    batch_size: int,
    device: str,
    random_seed: int,
) -> None:
    ensure_dir(output_dir)
    run_rows: list[dict[str, Any]] = []

    for dataset_name in DATASET_NAMES:
        bank_path = infer_bank_path(banks_root, dataset_name)
        for patch_size in PATCH_SIZES:
            run_dir = find_run_dir(checkpoints_root, dataset_name, patch_size)
            if run_dir is None:
                continue
            single_output = output_dir / dataset_name / f"ps{patch_size}"
            result = run_single_analysis(
                checkpoint=run_dir / "best_model.pt",
                bank_path=bank_path,
                output_dir=single_output,
                curve_indices=None,
                num_curves=num_curves,
                num_curves_visualized=num_curves_visualized,
                num_points_visualized=num_points_visualized,
                batch_size=batch_size,
                device=device,
                random_seed=random_seed,
                extra_metadata={"dataset_name": dataset_name, "patch_size": patch_size, "run_dir": str(run_dir)},
            )
            g = result["summary"]["global_pointwise"]
            row = {
                "dataset_name": dataset_name,
                "patch_size": patch_size,
                "run_dir": str(run_dir),
                "n_curves_analyzed": len(result["selected_curve_indices"]),
                "n_points_mean": float(result["per_curve_df"]["n_points"].mean()),
                "relative_patch_size_mean": float(result["per_curve_df"]["relative_patch_size"].mean()),
                "direct1_vs_gt1__abs_cosine_mean": g["direct1_vs_gt1"]["abs_cosine"]["mean"],
                "global1_vs_gt1__abs_cosine_mean": g["global1_vs_gt1"]["abs_cosine"]["mean"],
                "global2_from_global1_vs_gt2__abs_cosine_mean": g["global2_from_global1_vs_gt2"]["abs_cosine"]["mean"],
                "global2_from_direct1_vs_gt2__abs_cosine_mean": g["global2_from_direct1_vs_gt2"]["abs_cosine"]["mean"],
                "global1_vs_direct1__abs_cosine_mean": g["global1_vs_direct1"]["abs_cosine"]["mean"],
                "global2_from_global1_vs_gt2__mse_mean": g["global2_from_global1_vs_gt2"]["mse"]["mean"],
                "global2_from_global1_vs_gt2__norm_ratio_median": g["global2_from_global1_vs_gt2"]["norm_ratio"]["median"],
                "trim99_second_abs_cos_mean": result["summary"]["trimmed_by_curvature"]["TRIM99"]["second_derivative"]["abs_cosine"]["mean"],
                "trim95_second_abs_cos_mean": result["summary"]["trimmed_by_curvature"]["TRIM95"]["second_derivative"]["abs_cosine"]["mean"],
                "W_fro_norm_mean": result["summary"]["matrix_summary"]["W_fro_norm"]["mean"],
                "W2_fro_norm_mean": result["summary"]["matrix_summary"]["W2_fro_norm"]["mean"],
                "antisymmetry_error_mean": result["summary"]["matrix_summary"]["antisymmetry_error_mean"]["mean"],
            }
            run_rows.append(row)

    aggregate_sweep_results(run_rows, output_dir)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Research-grade analysis for TangentOperatorModel / invariant operator learning."
    )
    p.add_argument("--checkpoint", type=str, default=None)
    p.add_argument("--bank", type=str, default=None)
    p.add_argument("--output-dir", type=str, required=True)
    p.add_argument("--curve-indices", type=str, default=None)
    p.add_argument("--num-curves", type=int, default=32)
    p.add_argument("--num-curves-visualized", type=int, default=8)
    p.add_argument("--num-points-visualized", type=int, default=128)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--random-seed", type=int, default=123)

    p.add_argument("--checkpoints-root", type=str, default=None)
    p.add_argument("--banks-root", type=str, default=None)
    p.add_argument("--run-sweep", action="store_true")
    return p


def main() -> None:
    args = build_parser().parse_args()
    output_dir = Path(args.output_dir)

    if args.run_sweep:
        if args.checkpoints_root is None or args.banks_root is None:
            raise ValueError("--run-sweep requires --checkpoints-root and --banks-root")
        run_sweep(
            checkpoints_root=Path(args.checkpoints_root),
            banks_root=Path(args.banks_root),
            output_dir=output_dir,
            num_curves=args.num_curves,
            num_curves_visualized=args.num_curves_visualized,
            num_points_visualized=args.num_points_visualized,
            batch_size=args.batch_size,
            device=args.device,
            random_seed=args.random_seed,
        )
        return

    if args.checkpoint is None or args.bank is None:
        raise ValueError("Single-run analysis requires --checkpoint and --bank")

    run_single_analysis(
        checkpoint=Path(args.checkpoint),
        bank_path=Path(args.bank),
        output_dir=output_dir,
        curve_indices=parse_curve_indices(args.curve_indices),
        num_curves=args.num_curves,
        num_curves_visualized=args.num_curves_visualized,
        num_points_visualized=args.num_points_visualized,
        batch_size=args.batch_size,
        device=args.device,
        random_seed=args.random_seed,
    )


if __name__ == "__main__":
    main()
