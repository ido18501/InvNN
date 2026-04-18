from __future__ import annotations

import argparse
import json
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from models.tangent_model import TangentOperatorModel
from utils.patch_sampling import sample_patch_around_index
from utils.derivatives import compute_fourier_arc_length_derivatives
from utils.curve_generation import BasisExpansionCurveCoeffs


# ============================================================
# Utilities
# ============================================================

def parse_int_list(text: str | list[int] | None) -> list[int]:
    if text is None:
        return []
    if isinstance(text, list):
        return [int(x) for x in text]
    text = str(text).strip()
    if not text:
        return []
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def safe_float(x: Any, default: float = float("nan")) -> float:
    try:
        return float(x)
    except Exception:
        return default

def correlation(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    if len(a) == 0 or len(b) == 0:
        return float("nan")
    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def load_metrics_json(run_dir: Path) -> dict[str, Any] | None:
    metrics_path = run_dir / "metrics.json"
    if not metrics_path.exists():
        return None
    try:
        return json.loads(metrics_path.read_text())
    except Exception:
        return None


def convergence_from_metrics(run_dir: Path) -> dict[str, float]:
    data = load_metrics_json(run_dir)
    if data is None:
        return {"conv_90": float("nan"), "conv_95": float("nan"), "conv_99": float("nan")}

    # support either val_history or history.val
    val_hist = data.get("val_history")
    if val_hist is None and "history" in data and isinstance(data["history"], dict):
        val_hist = data["history"].get("val")

    if not isinstance(val_hist, dict):
        return {"conv_90": float("nan"), "conv_95": float("nan"), "conv_99": float("nan")}

    cos1 = val_hist.get("first_cosine_mean")
    if cos1 is None:
        return {"conv_90": float("nan"), "conv_95": float("nan"), "conv_99": float("nan")}

    cos1 = np.asarray(cos1, dtype=np.float64)

    def first_epoch(th: float) -> float:
        idx = np.where(cos1 >= th)[0]
        return float(idx[0] + 1) if len(idx) > 0 else float("nan")

    return {
        "conv_90": first_epoch(0.90),
        "conv_95": first_epoch(0.95),
        "conv_99": first_epoch(0.99),
    }

def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def mse(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    return float(np.mean((a - b) ** 2))


def norms(v: np.ndarray) -> np.ndarray:
    return np.linalg.norm(np.asarray(v, dtype=np.float64), axis=-1)


def cosine_and_angle(pred: np.ndarray, gt: np.ndarray, eps: float = 1e-12) -> tuple[np.ndarray, np.ndarray]:
    pred = np.asarray(pred, dtype=np.float64)
    gt = np.asarray(gt, dtype=np.float64)
    pred_n = pred / np.clip(norms(pred)[..., None], eps, None)
    gt_n = gt / np.clip(norms(gt)[..., None], eps, None)
    cos = np.sum(pred_n * gt_n, axis=-1)
    cos = np.clip(cos, -1.0, 1.0)
    ang = np.degrees(np.arccos(cos))
    return cos, ang


def find_all_run_dirs(root: Path) -> list[Path]:
    return sorted([p for p in root.rglob("config.json") if p.is_file()])


def load_config(run_dir: Path) -> dict[str, Any]:
    return json.loads((run_dir / "config.json").read_text())


def find_checkpoint(run_dir: Path) -> Path | None:
    candidates = [
        run_dir / "best_model.pt",
        run_dir / "best.pt",
        run_dir / "model_best.pt",
        run_dir / "checkpoint_best.pt",
        run_dir / "last.pt",
        run_dir / "checkpoint.pt",
        run_dir / "init_model.pt",
    ]
    for p in candidates:
        if p.exists():
            return p
    pts = sorted(run_dir.glob("*.pt"))
    return pts[0] if pts else None


def load_curve_bank(npz_path: Path):
    data = np.load(npz_path, allow_pickle=False)
    curve_points = np.asarray(data["curve_points"], dtype=np.float64)
    x_coeffs = np.asarray(data["x_coeffs"], dtype=np.float64) if "x_coeffs" in data.files else None
    y_coeffs = np.asarray(data["y_coeffs"], dtype=np.float64) if "y_coeffs" in data.files else None
    t_grid = np.asarray(data["t_grid"], dtype=np.float64) if "t_grid" in data.files else None
    return curve_points, x_coeffs, y_coeffs, t_grid


# ============================================================
# Log parsing for convergence / scheduler traces
# ============================================================

RUN_RE = re.compile(r"^RUN:\s+(?P<tag>.+)$")
EPOCH_RE = re.compile(r"^Epoch\s+(?P<epoch>\d+)")
VAL_ANALYTIC_RE = re.compile(
    r"^val\s+analytic\s+\|\s+cos1=(?P<cos1>[-+0-9.eE]+)\s+ang1=(?P<ang1>[-+0-9.eE]+)°\s+mse1=(?P<mse1>[-+0-9.eE]+)"
)
VAL_RE = re.compile(
    r"^val\s+\|\s+loss=(?P<loss>[-+0-9.eE]+)\s+nce=(?P<nce>[-+0-9.eE]+)\s+eqmse=(?P<eqmse>[-+0-9.eE]+)\s+eqnorm=(?P<eqnorm>[-+0-9.eE]+)\s+eqcos=(?P<eqcos>[-+0-9.eE]+)"
)
LR_RE = re.compile(r"^lr=(?P<lr>[-+0-9.eE]+)")
BEST_EPOCH_RE = re.compile(r"^Best validation epoch:\s+(?P<epoch>\d+)")


@dataclass
class EpochRecord:
    epoch: int
    val_loss: float = float("nan")
    val_nce: float = float("nan")
    val_eqmse: float = float("nan")
    val_eqcos: float = float("nan")
    val_cos1: float = float("nan")
    val_ang1: float = float("nan")
    val_mse1: float = float("nan")
    lr: float = float("nan")


def parse_logs(logs_root: Path) -> dict[str, dict[str, Any]]:
    parsed: dict[str, dict[str, Any]] = {}
    for log_path in sorted(logs_root.glob("*.out")):
        current_tag: str | None = None
        current_epoch: int | None = None
        current_best_epoch: int | None = None

        with log_path.open("r", errors="ignore") as f:
            for raw_line in f:
                line = raw_line.strip()
                if not line:
                    continue

                m = RUN_RE.match(line)
                if m:
                    current_tag = m.group("tag")
                    parsed.setdefault(current_tag, {"epochs": {}, "best_epoch": None, "log_file": str(log_path)})
                    current_epoch = None
                    current_best_epoch = None
                    continue

                if current_tag is None:
                    continue

                m = EPOCH_RE.match(line)
                if m:
                    current_epoch = int(m.group("epoch"))
                    parsed[current_tag]["epochs"].setdefault(current_epoch, EpochRecord(epoch=current_epoch))
                    continue

                m = VAL_RE.match(line)
                if m and current_epoch is not None:
                    rec = parsed[current_tag]["epochs"].setdefault(current_epoch, EpochRecord(epoch=current_epoch))
                    rec.val_loss = safe_float(m.group("loss"))
                    rec.val_nce = safe_float(m.group("nce"))
                    rec.val_eqmse = safe_float(m.group("eqmse"))
                    rec.val_eqcos = safe_float(m.group("eqcos"))
                    continue

                m = VAL_ANALYTIC_RE.match(line)
                if m and current_epoch is not None:
                    rec = parsed[current_tag]["epochs"].setdefault(current_epoch, EpochRecord(epoch=current_epoch))
                    rec.val_cos1 = safe_float(m.group("cos1"))
                    rec.val_ang1 = safe_float(m.group("ang1"))
                    rec.val_mse1 = safe_float(m.group("mse1"))
                    continue

                m = LR_RE.match(line)
                if m and current_epoch is not None:
                    rec = parsed[current_tag]["epochs"].setdefault(current_epoch, EpochRecord(epoch=current_epoch))
                    rec.lr = safe_float(m.group("lr"))
                    continue

                m = BEST_EPOCH_RE.match(line)
                if m:
                    current_best_epoch = int(m.group("epoch"))
                    parsed[current_tag]["best_epoch"] = current_best_epoch
                    continue

    return parsed


# ============================================================
# Model / operator reconstruction
# ============================================================

def build_model_from_config(config: dict[str, Any], device: torch.device) -> TangentOperatorModel:
    model = TangentOperatorModel(
        patch_size=int(config["patch_size"]),
        operator_hidden_dims=parse_int_list(config.get("operator_hidden_dims", "256,256")),
        signature_hidden_dims=parse_int_list(config.get("signature_hidden_dims", "128,64")),
        signature_out_dim=int(config.get("signature_out_dim", 64)),
        signature_center_radius=int(config.get("signature_center_radius", 0)),
        head_dropout=float(config.get("head_dropout", 0.0)),
        normalize_projector=not bool(config.get("disable_normalize_projector", False)),
        init_scale=float(config.get("operator_init_scale", 0.05)),
        learn_scale=bool(config.get("learn_output_scale", False)),
        centered_input_for_operator=not bool(config.get("disable_centered_input_for_operator", False)),
    )
    model.to(device)
    model.eval()
    return model


def load_state_dict_into_model(model: torch.nn.Module, ckpt_path: Path, device: torch.device) -> None:
    payload = torch.load(ckpt_path, map_location=device)
    if isinstance(payload, dict):
        if "model_state_dict" in payload:
            state = payload["model_state_dict"]
        elif "state_dict" in payload:
            state = payload["state_dict"]
        else:
            state = payload
    else:
        raise ValueError(f"Unsupported checkpoint format in {ckpt_path}")

    cleaned = {}
    for k, v in state.items():
        nk = k[6:] if k.startswith("model.") else k
        cleaned[nk] = v

    model.load_state_dict(cleaned, strict=False)


def extract_patch(curve: np.ndarray, center_index: int, patch_size: int, half_width: int, patch_mode: str,
                  closed: bool, use_centered_patch: bool) -> tuple[np.ndarray, np.ndarray]:
    patch = sample_patch_around_index(
        curve_points=curve,
        center_index=center_index,
        patch_size=patch_size,
        half_width=half_width,
        mode=patch_mode,
        closed=closed,
        rng=np.random.default_rng(0),
        jitter_fraction=0.0,
    )
    pts = np.asarray(patch.centered_points if use_centered_patch else patch.points, dtype=np.float64)
    idx = np.asarray(patch.sample_indices, dtype=np.int64)
    return pts, idx


@torch.no_grad()
def build_operator_matrix_for_curve(
    model: TangentOperatorModel,
    curve: np.ndarray,
    patch_size: int,
    half_width: int,
    patch_mode: str,
    closed: bool,
    use_centered_patch: bool,
    device: torch.device,
) -> tuple[np.ndarray, list[np.ndarray]]:
    n = curve.shape[0]
    W = np.zeros((n, n), dtype=np.float64)
    local_weights: list[np.ndarray] = []

    for i in range(n):
        pts, sample_indices = extract_patch(
            curve,
            i,
            patch_size=patch_size,
            half_width=half_width,
            patch_mode=patch_mode,
            closed=closed,
            use_centered_patch=use_centered_patch,
        )
        x = torch.tensor(pts, dtype=torch.float32, device=device).unsqueeze(0)
        weights = model.get_weights(x).squeeze(0).detach().cpu().numpy().astype(np.float64)
        local_weights.append(weights.copy())
        for k, j in enumerate(sample_indices):
            W[i, int(j)] += float(weights[k])
    return W, local_weights


def get_gt_derivatives_for_curve(curve_idx: int, family: str, x_coeffs: np.ndarray | None, y_coeffs: np.ndarray | None,
                                 t_grid_all: np.ndarray | None) -> tuple[np.ndarray | None, np.ndarray | None]:
    if x_coeffs is None or y_coeffs is None or t_grid_all is None:
        return None, None
    coeffs = BasisExpansionCurveCoeffs(
        x_coeffs=x_coeffs[curve_idx].copy(),
        y_coeffs=y_coeffs[curve_idx].copy(),
    )
    t_grid = t_grid_all if t_grid_all.ndim == 1 else t_grid_all[curve_idx]
    _, gt_first, gt_second = compute_fourier_arc_length_derivatives(
        t=t_grid,
        coeffs=coeffs,
        family=family,
    )
    return np.asarray(gt_first, dtype=np.float64), np.asarray(gt_second, dtype=np.float64)


# ============================================================
# Plotting
# ============================================================

def plot_hist(values: np.ndarray, title: str, xlabel: str, out_path: Path, bins: int = 40) -> None:
    plt.figure(figsize=(6, 4))
    plt.hist(np.asarray(values).ravel(), bins=bins)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("count")
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def plot_along_curve_norms(pred: np.ndarray, gt: np.ndarray | None, out_path: Path, title: str) -> None:
    plt.figure(figsize=(7, 4))
    plt.plot(norms(pred), label="pred")
    if gt is not None:
        plt.plot(norms(gt), label="gt")
    plt.xlabel("curve index")
    plt.ylabel("norm")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def plot_norm_scatter(gt: np.ndarray, pred: np.ndarray, out_path: Path, title: str) -> None:
    plt.figure(figsize=(5, 5))
    plt.scatter(norms(gt), norms(pred), s=8, alpha=0.6)
    plt.xlabel("GT norm")
    plt.ylabel("Pred norm")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()

def plot_W_heatmap(W: np.ndarray, out_path: Path, title: str) -> None:
    plt.figure(figsize=(6, 5))
    plt.imshow(W, aspect="auto")
    plt.title(title)
    plt.xlabel("column j")
    plt.ylabel("row i")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def plot_curve_and_field(curve: np.ndarray, vecs: np.ndarray, gt: np.ndarray | None, out_path: Path, title: str) -> None:
    step = max(1, len(curve) // 60)
    plt.figure(figsize=(6, 6))
    plt.plot(curve[:, 0], curve[:, 1], linewidth=1.0, label="curve")
    plt.quiver(
        curve[::step, 0], curve[::step, 1],
        vecs[::step, 0], vecs[::step, 1],
        angles="xy", scale_units="xy", scale=1.0, width=0.003,
        label="prediction",
    )
    if gt is not None:
        plt.quiver(
            curve[::step, 0], curve[::step, 1],
            gt[::step, 0], gt[::step, 1],
            angles="xy", scale_units="xy", scale=1.0, width=0.002,
            label="gt",
        )
    plt.axis("equal")
    plt.legend()
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def plot_unit_circle_vectors(pred_first: np.ndarray, gt_first: np.ndarray | None, out_path: Path, title: str) -> None:
    plt.figure(figsize=(6, 6))
    theta = np.linspace(0, 2 * np.pi, 400)
    plt.plot(np.cos(theta), np.sin(theta), linewidth=1.0)

    pred_n = pred_first / np.clip(norms(pred_first)[:, None], 1e-12, None)
    plt.quiver(
        np.zeros(len(pred_n)), np.zeros(len(pred_n)),
        pred_n[:, 0], pred_n[:, 1],
        angles="xy", scale_units="xy", scale=1.0, width=0.002,
        alpha=0.35, label="pred unit",
    )
    if gt_first is not None:
        gt_n = gt_first / np.clip(norms(gt_first)[:, None], 1e-12, None)
        plt.quiver(
            np.zeros(len(gt_n)), np.zeros(len(gt_n)),
            gt_n[:, 0], gt_n[:, 1],
            angles="xy", scale_units="xy", scale=1.0, width=0.0015,
            alpha=0.25, label="gt unit",
        )
    plt.axis("equal")
    plt.xlim(-1.1, 1.1)
    plt.ylim(-1.1, 1.1)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def plot_selected_point_examples(curve: np.ndarray, pred_first: np.ndarray, gt_first: np.ndarray | None,
                                 pred_second: np.ndarray, gt_second: np.ndarray | None, idxs: np.ndarray,
                                 out_path: Path, title: str) -> None:
    plt.figure(figsize=(7, 7))
    plt.plot(curve[:, 0], curve[:, 1], linewidth=1.0, color="black", alpha=0.6)
    colors = plt.cm.tab10(np.linspace(0, 1, len(idxs)))
    for c, i in zip(colors, idxs):
        x, y = curve[i]
        plt.scatter([x], [y], s=30)
        pf = pred_first[i]
        ps = pred_second[i]
        plt.quiver([x], [y], [pf[0]], [pf[1]], angles="xy", scale_units="xy", scale=1.0, width=0.004)
        plt.quiver([x], [y], [ps[0]], [ps[1]], angles="xy", scale_units="xy", scale=1.0, width=0.004)
        if gt_first is not None:
            gf = gt_first[i]
            plt.quiver([x], [y], [gf[0]], [gf[1]], angles="xy", scale_units="xy", scale=1.0, width=0.003)
        if gt_second is not None:
            gs = gt_second[i]
            plt.quiver([x], [y], [gs[0]], [gs[1]], angles="xy", scale_units="xy", scale=1.0, width=0.003)
    plt.axis("equal")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def plot_progress_vs_ps(best_df: pd.DataFrame, out_dir: Path) -> None:
    if best_df.empty:
        return

    best_df = best_df.sort_values("patch_size")
    x = best_df["patch_size"].to_numpy()

    plt.figure(figsize=(7, 4))
    for col, label in [
        ("conv_epoch_90", "epoch cos1>=0.90"),
        ("conv_epoch_95", "epoch cos1>=0.95"),
        ("conv_epoch_99", "epoch cos1>=0.99"),
    ]:
        if col in best_df.columns:
            y = best_df[col].to_numpy(dtype=float)
            plt.plot(x, y, marker="o", label=label)
    plt.xlabel("patch size")
    plt.ylabel("epoch")
    plt.title("Convergence epoch vs patch size (best selected model)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "best_models_convergence_vs_patch_size.png", dpi=180)
    plt.close()

    plt.figure(figsize=(7, 4))
    for col, label in [
        ("global_first_cos_mean", "first cos mean"),
        ("global_second_cos_mean", "second cos mean"),
    ]:
        if col in best_df.columns:
            y = best_df[col].to_numpy(dtype=float)
            plt.plot(x, y, marker="o", label=label)
    plt.xlabel("patch size")
    plt.ylabel("cosine")
    plt.title("Cosine quality vs patch size (best selected model)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "best_models_cosine_vs_patch_size.png", dpi=180)
    plt.close()

    plt.figure(figsize=(7, 4))
    for col, label in [
        ("global_first_mse", "first mse"),
        ("global_second_mse", "second mse"),
    ]:
        if col in best_df.columns:
            y = best_df[col].to_numpy(dtype=float)
            plt.plot(x, y, marker="o", label=label)
    plt.xlabel("patch size")
    plt.ylabel("mse")
    plt.title("Errors vs patch size (best selected model)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "best_models_mse_vs_patch_size.png", dpi=180)
    plt.close()

def plot_second_order_comparison(best_df: pd.DataFrame, out_dir: Path) -> None:
    if best_df.empty:
        return

    best_df = best_df.sort_values("patch_size")
    x = best_df["patch_size"].to_numpy()

    metrics = [
        ("global_second_cos_mean", "Second cosine mean"),
        ("global_second_abs_cos_mean", "Second abs-cos mean"),
        ("global_second_corr_mean", "Second correlation mean"),
        ("global_second_scale_ratio_mean", "Second scale ratio mean"),
    ]

    plt.figure(figsize=(8, 5))
    for col, label in metrics:
        if col in best_df.columns:
            y = best_df[col].to_numpy(dtype=float)
            plt.plot(x, y, marker="o", label=label)
    plt.xlabel("patch size")
    plt.ylabel("value")
    plt.title("Second-order behavior vs patch size (best selected model)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "best_models_second_order_behavior_vs_patch_size.png", dpi=180)
    plt.close()
# ============================================================
# Analysis
# ============================================================

def convergence_epoch(log_info: dict[str, Any] | None, threshold: float) -> int | float:
    if not log_info:
        return float("nan")
    epochs = log_info.get("epochs", {})
    for ep in sorted(epochs.keys()):
        rec = epochs[ep]
        if np.isfinite(rec.val_cos1) and rec.val_cos1 >= threshold:
            return int(ep)
    return float("nan")


def compute_score(row: pd.Series) -> float:
    first_cos = safe_float(row.get("global_first_cos_mean"), -1.0)
    second_cos = safe_float(row.get("global_second_cos_mean"), -1.0)
    second_abs_cos = safe_float(row.get("global_second_abs_cos_mean"), -1.0)

    p1 = safe_float(row.get("global_pred_first_norm_mean"), float("nan"))
    g1 = safe_float(row.get("global_gt_first_norm_mean"), float("nan"))
    p2 = safe_float(row.get("global_pred_second_norm_mean"), float("nan"))
    g2 = safe_float(row.get("global_gt_second_norm_mean"), float("nan"))

    def log_ratio_penalty(a: float, b: float) -> float:
        if not np.isfinite(a) or not np.isfinite(b) or a <= 0 or b <= 0:
            return 10.0
        return abs(math.log(a / b))

    if first_cos < 0:
        return -100.0

    penalty = 0.15 * log_ratio_penalty(p1, g1) + 0.10 * log_ratio_penalty(p2, g2)
    score = 1.10 * second_cos + 0.20 * second_abs_cos + 0.45 * first_cos - penalty
    return float(score)


def analyze_one_run(
    run_dir: Path,
    analysis_root: Path,
    logs_by_tag: dict[str, dict[str, Any]],
    device: torch.device,
    num_eval_curves: int,
    num_plot_curves: int,
) -> dict[str, Any] | None:
    config = load_config(run_dir)
    ckpt_path = find_checkpoint(run_dir)
    if ckpt_path is None:
        return None

    run_tag = str(run_dir.relative_to(run_dir.parents[2])) if len(run_dir.parents) >= 3 else run_dir.name

    family = str(config.get("family", "euclidean"))
    if family == "affine":
        return None

    bank_path = config.get("val_bank") or config.get("test_bank") or config.get("train_bank")
    if bank_path is None:
        return None

    patch_size = int(config["patch_size"])
    half_width = int(config.get("half_width", 12))
    patch_mode = str(config.get("patch_mode", "intrinsic_ordered_stencil"))
    use_centered_patch = not bool(config.get("disable_return_centered", False))
    closed = True

    curve_points_all, x_coeffs, y_coeffs, t_grid_all = load_curve_bank(Path(bank_path))
    model = build_model_from_config(config, device)
    load_state_dict_into_model(model, ckpt_path, device)

    n_curves_total = curve_points_all.shape[0]
    curve_indices = np.linspace(0, n_curves_total - 1, min(num_eval_curves, n_curves_total), dtype=int)
    plot_curve_indices = curve_indices[: min(num_plot_curves, len(curve_indices))]

    model_analysis_dir = analysis_root / run_dir.relative_to(run_dir.parents[2])
    ensure_dir(model_analysis_dir)
    ensure_dir(model_analysis_dir / "curves")

    rows = []
    all_first_cos = []
    all_first_ang = []
    all_second_cos = []
    all_second_abs_cos = []
    all_second_ang = []
    all_first_pred_norm = []
    all_second_pred_norm = []
    all_first_gt_norm = []
    all_second_gt_norm = []
    all_second_corr = []
    all_second_scale_ratio = []
    row_sum_abs = []
    row_l1 = []

    for curve_idx in curve_indices:
        curve = curve_points_all[curve_idx]
        W, local_weights = build_operator_matrix_for_curve(
            model=model,
            curve=curve,
            patch_size=patch_size,
            half_width=half_width,
            patch_mode=patch_mode,
            closed=closed,
            use_centered_patch=use_centered_patch,
            device=device,
        )
        pred_first = W @ curve
        pred_second = W @ pred_first
        gt_first, gt_second = get_gt_derivatives_for_curve(curve_idx, family, x_coeffs, y_coeffs, t_grid_all)

        curve_stats: dict[str, Any] = {
            "curve_idx": int(curve_idx),
            "W_row_sum_mean": float(np.mean(np.sum(W, axis=1))),
            "W_row_sum_abs_mean": float(np.mean(np.abs(np.sum(W, axis=1)))),
            "W_row_l1_mean": float(np.mean(np.sum(np.abs(W), axis=1))),
            "W_abs_mean": float(np.mean(np.abs(W))),
            "pred_first_norm_mean": float(np.mean(norms(pred_first))),
            "pred_second_norm_mean": float(np.mean(norms(pred_second))),
        }

        row_sum_abs.append(curve_stats["W_row_sum_abs_mean"])
        row_l1.append(curve_stats["W_row_l1_mean"])
        all_first_pred_norm.extend(norms(pred_first).tolist())
        all_second_pred_norm.extend(norms(pred_second).tolist())

        if gt_first is not None:
            first_cos, first_ang = cosine_and_angle(pred_first, gt_first)
            curve_stats.update(
                {
                    "first_cos_mean": float(np.mean(first_cos)),
                    "first_angle_mean": float(np.mean(first_ang)),
                    "first_mse": mse(pred_first, gt_first),
                    "gt_first_norm_mean": float(np.mean(norms(gt_first))),
                }
            )
            all_first_cos.extend(first_cos.tolist())
            all_first_ang.extend(first_ang.tolist())
            all_first_gt_norm.extend(norms(gt_first).tolist())

        if gt_second is not None:
            second_cos, second_ang = cosine_and_angle(pred_second, gt_second)
            second_abs_cos = np.abs(second_cos)

            pred_second_norms = norms(pred_second)
            gt_second_norms = norms(gt_second)
            valid_scale = gt_second_norms > 1e-12
            second_scale_ratio = float(
                np.mean(pred_second_norms[valid_scale] / gt_second_norms[valid_scale])) if np.any(
                valid_scale) else float("nan")
            second_corr = correlation(pred_second, gt_second)

            curve_stats.update(
                {
                    "second_cos_mean": float(np.mean(second_cos)),
                    "second_abs_cos_mean": float(np.mean(second_abs_cos)),
                    "second_angle_mean": float(np.mean(second_ang)),
                    "second_mse": mse(pred_second, gt_second),
                    "second_corr": second_corr,
                    "second_scale_ratio": second_scale_ratio,
                    "gt_second_norm_mean": float(np.mean(gt_second_norms)),
                }
            )
            all_second_cos.extend(second_cos.tolist())
            all_second_abs_cos.extend(second_abs_cos.tolist())
            all_second_ang.extend(second_ang.tolist())
            all_second_gt_norm.extend(gt_second_norms.tolist())
            all_second_corr.append(second_corr)
            all_second_scale_ratio.append(second_scale_ratio)

        rows.append(curve_stats)

        if curve_idx in plot_curve_indices:
            cdir = model_analysis_dir / "curves" / f"curve_{curve_idx:04d}"
            ensure_dir(cdir)
            np.save(cdir / "W.npy", W)
            np.save(cdir / "pred_first.npy", pred_first)
            np.save(cdir / "pred_second.npy", pred_second)
            if gt_first is not None:
                np.save(cdir / "gt_first.npy", gt_first)
            if gt_second is not None:
                np.save(cdir / "gt_second.npy", gt_second)

            plot_W_heatmap(W, cdir / "W_heatmap.png", f"W heatmap - curve {curve_idx}")
            plot_curve_and_field(curve, pred_first, gt_first, cdir / "first_field.png",
                                 f"First derivative - curve {curve_idx}")
            plot_curve_and_field(curve, pred_second, gt_second, cdir / "second_field.png",
                                 f"Second derivative - curve {curve_idx}")
            plot_unit_circle_vectors(pred_first, gt_first, cdir / "unit_circle_first_vectors.png",
                                     f"Unit-circle view of first derivative - curve {curve_idx}")

            if gt_first is not None:
                plot_along_curve_norms(pred_first, gt_first, cdir / "first_norms_along_curve.png",
                                       f"First-derivative norms along curve - curve {curve_idx}")
            if gt_second is not None:
                plot_along_curve_norms(pred_second, gt_second, cdir / "second_norms_along_curve.png",
                                       f"Second-derivative norms along curve - curve {curve_idx}")
                plot_norm_scatter(gt_second, pred_second, cdir / "second_norm_scatter.png",
                                  f"Second norm scatter - curve {curve_idx}")
                sec_cos, _ = cosine_and_angle(pred_second, gt_second)
                plot_hist(sec_cos, f"Second cosine histogram - curve {curve_idx}", "cos(pred_second, gt_second)",
                          cdir / "second_cos_hist.png")
                plot_hist(np.abs(sec_cos), f"Second abs-cos histogram - curve {curve_idx}",
                          "|cos(pred_second, gt_second)|", cdir / "second_abs_cos_hist.png")

            idxs = np.linspace(0, len(curve) - 1, min(8, len(curve)), dtype=int)
            plot_selected_point_examples(
                curve, pred_first, gt_first, pred_second, gt_second, idxs,
                cdir / "selected_points_examples.png",
                f"Selected point examples - curve {curve_idx}"
            )

    curve_df = pd.DataFrame(rows)
    curve_df.to_csv(model_analysis_dir / "per_curve_metrics.csv", index=False)

    if len(all_first_cos) > 0:
        plot_hist(np.asarray(all_first_cos), "First-derivative cosine histogram", "cos(pred_first, gt_first)", model_analysis_dir / "hist_first_cos.png")
        plot_hist(np.asarray(all_first_ang), "First-derivative angle histogram", "angle(pred_first, gt_first)", model_analysis_dir / "hist_first_angle.png")
    if len(all_second_cos) > 0:
        plot_hist(np.asarray(all_second_cos), "Second-derivative cosine histogram", "cos(pred_second, gt_second)", model_analysis_dir / "hist_second_cos.png")
        plot_hist(np.asarray(all_second_ang), "Second-derivative angle histogram", "angle(pred_second, gt_second)", model_analysis_dir / "hist_second_angle.png")
    if len(all_first_pred_norm) > 0:
        plot_hist(np.asarray(all_first_pred_norm), "Predicted first norm histogram", "||pred_first||", model_analysis_dir / "hist_pred_first_norm.png")
    if len(all_second_pred_norm) > 0:
        plot_hist(np.asarray(all_second_pred_norm), "Predicted second norm histogram", "||pred_second||", model_analysis_dir / "hist_pred_second_norm.png")
    if len(all_first_gt_norm) > 0:
        plot_hist(np.asarray(all_first_gt_norm), "GT first norm histogram", "||gt_first||", model_analysis_dir / "hist_gt_first_norm.png")
    if len(all_second_gt_norm) > 0:
        plot_hist(np.asarray(all_second_gt_norm), "GT second norm histogram", "||gt_second||", model_analysis_dir / "hist_gt_second_norm.png")

    logs = logs_by_tag.get(str(run_dir.relative_to(run_dir.parents[2])), None)

    summary = {
        "run_dir": str(run_dir),
        "checkpoint": str(ckpt_path),
        "bank_path": str(bank_path),
        "family": family,
        "patch_size": patch_size,
        "lambda_eq": safe_float(config.get("lambda_eq")),
        "lambda_nce": safe_float(config.get("lambda_nce")),
        "temperature": safe_float(config.get("temperature")),
        "lr": safe_float(config.get("lr")),
        "lr_scheduler": config.get("lr_scheduler", "none"),
        "num_eval_curves": int(len(curve_indices)),
        "global_W_row_sum_abs_mean": float(np.mean(row_sum_abs)) if row_sum_abs else float("nan"),
        "global_W_row_l1_mean": float(np.mean(row_l1)) if row_l1 else float("nan"),
        "global_pred_first_norm_mean": float(np.mean(all_first_pred_norm)) if all_first_pred_norm else float("nan"),
        "global_pred_second_norm_mean": float(np.mean(all_second_pred_norm)) if all_second_pred_norm else float("nan"),
        "global_gt_first_norm_mean": float(np.mean(all_first_gt_norm)) if all_first_gt_norm else float("nan"),
        "global_gt_second_norm_mean": float(np.mean(all_second_gt_norm)) if all_second_gt_norm else float("nan"),
        "global_first_cos_mean": float(np.mean(all_first_cos)) if all_first_cos else float("nan"),
        "global_first_angle_mean": float(np.mean(all_first_ang)) if all_first_ang else float("nan"),
        "global_second_cos_mean": float(np.mean(all_second_cos)) if all_second_cos else float("nan"),
        "global_second_abs_cos_mean": float(np.mean(all_second_abs_cos)) if all_second_abs_cos else float("nan"),
        "global_second_angle_mean": float(np.mean(all_second_ang)) if all_second_ang else float("nan"),
        "global_second_corr_mean": float(np.nanmean(all_second_corr)) if all_second_corr else float("nan"),
        "global_second_scale_ratio_mean": float(np.nanmean(all_second_scale_ratio)) if all_second_scale_ratio else float("nan"),
        "global_first_mse": float(curve_df["first_mse"].mean()) if "first_mse" in curve_df else float("nan"),
        "global_second_mse": float(curve_df["second_mse"].mean()) if "second_mse" in curve_df else float("nan"),
        "best_epoch_from_log": logs.get("best_epoch") if logs else None,
        "conv_epoch_90": convergence_from_metrics(run_dir)["conv_90"],
        "conv_epoch_95": convergence_from_metrics(run_dir)["conv_95"],
        "conv_epoch_99": convergence_from_metrics(run_dir)["conv_99"],
        "log_file": logs.get("log_file") if logs else None,
    }
    summary["selection_score"] = compute_score(pd.Series(summary))

    (model_analysis_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary




def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoints-root", type=str, default="checkpoints_grid")
    parser.add_argument("--analysis-root", type=str, default="analysis_operator_grid")
    parser.add_argument("--logs-root", type=str, default="logs_grid")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num-eval-curves", type=int, default=8)
    parser.add_argument("--num-plot-curves", type=int, default=3)
    parser.add_argument("--model-filter", type=str, default="")
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()

    checkpoints_root = Path(args.checkpoints_root)
    analysis_root = Path(args.analysis_root)
    logs_root = Path(args.logs_root)
    ensure_dir(analysis_root)

    logs_by_tag = parse_logs(logs_root) if logs_root.exists() else {}
    device = torch.device(args.device)

    config_paths = find_all_run_dirs(checkpoints_root)
    summaries = []

    for cfg_path in config_paths:
        run_dir = cfg_path.parent
        rel = str(run_dir.relative_to(checkpoints_root))
        if args.model_filter and args.model_filter not in rel:
            continue

        out_dir = analysis_root / rel
        if args.skip_existing and (out_dir / "summary.json").exists():
            print(f"[skip] {rel}")
            try:
                summaries.append(json.loads((out_dir / "summary.json").read_text()))
            except Exception:
                pass
            continue

        print(f"[analyze] {rel}")
        try:
            summary = analyze_one_run(
                run_dir=run_dir,
                analysis_root=analysis_root,
                logs_by_tag=logs_by_tag,
                device=device,
                num_eval_curves=args.num_eval_curves,
                num_plot_curves=args.num_plot_curves,
            )
            if summary is not None:
                summaries.append(summary)
        except Exception as e:
            print(f"[error] {rel}: {e}")

    if not summaries:
        print("No completed/analyzable runs found.")
        return

    df = pd.DataFrame(summaries)
    df = df.sort_values(["patch_size", "selection_score"], ascending=[True, False])
    df.to_csv(analysis_root / "all_models_summary.csv", index=False)

    # Transparent leaderboards
    for metric, ascending in [
        ("selection_score", False),
        ("global_second_cos_mean", False),
        ("global_first_cos_mean", False),
        ("global_second_mse", True),
        ("global_first_mse", True),
    ]:
        sub = df.sort_values(metric, ascending=ascending)
        sub.to_csv(analysis_root / f"leaderboard_by_{metric}.csv", index=False)

    # Best model per patch size by selection score
    best_rows = []
    for ps, g in df.groupby("patch_size"):
        g = g.sort_values("selection_score", ascending=False)
        # Prefer models with strong first cosine when available.
        filt = g[g["global_first_cos_mean"] >= 0.95]
        best = filt.iloc[0] if len(filt) > 0 else g.iloc[0]
        best_rows.append(best)
    best_df = pd.DataFrame(best_rows).sort_values("patch_size")
    best_df.to_csv(analysis_root / "best_model_per_patch_size.csv", index=False)

    plot_progress_vs_ps(best_df, analysis_root)
    plot_second_order_comparison(best_df, analysis_root)

    # Also save a small human-readable markdown summary
    with (analysis_root / "README_summary.txt").open("w") as f:
        f.write("# Analysis summary\n\n")
        f.write("## Best model per patch size\n\n")
        f.write(best_df.to_string(index=False))
        f.write("\n\n## Top 20 by selection score\n\n")
        f.write(df.head(20).to_string(index=False))

    print(f"\nSaved analysis to: {analysis_root}")
    print(f"All-model summary: {analysis_root / 'all_models_summary.csv'}")
    print(f"Best per patch size: {analysis_root / 'best_model_per_patch_size.csv'}")


if __name__ == "__main__":
    main()
