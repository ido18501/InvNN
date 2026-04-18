from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

try:
    from models.tangent_model import TangentOperatorModel
except Exception:
    from tangent_model import TangentOperatorModel

try:
    from datasets.tangent_dataset import PregeneratedCurveBank
except Exception:
    from tangent_dataset import PregeneratedCurveBank

try:
    from utils.derivatives import compute_fourier_arc_length_derivatives
except Exception:
    from derivatives import compute_fourier_arc_length_derivatives


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compare operator behavior before/after fine-tuning.")
    p.add_argument("--checkpoint-a", type=str, required=True)
    p.add_argument("--checkpoint-b", type=str, required=True)
    p.add_argument("--label-a", type=str, default="before")
    p.add_argument("--label-b", type=str, default="after")
    p.add_argument("--config-a", type=str, required=True)
    p.add_argument("--config-b", type=str, default=None)
    p.add_argument("--bank", type=str, required=True)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--family", type=str, default="euclidean")
    p.add_argument("--patch-size", type=int, required=True)
    p.add_argument("--patch-mode", type=str, required=True)
    p.add_argument("--half-width", type=int, default=0)
    p.add_argument("--max-curves", type=int, default=None)
    p.add_argument("--flip-sign-a", action="store_true")
    p.add_argument("--flip-sign-b", action="store_true")
    p.add_argument("--out-json", type=str, default=None)
    return p.parse_args()


def load_json(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def parse_int_list(text_or_list: Any) -> list[int]:
    if isinstance(text_or_list, list):
        return [int(x) for x in text_or_list]
    text = str(text_or_list).strip()
    if not text:
        return []
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def build_model_from_config(config: dict[str, Any], patch_size_override: int | None = None) -> TangentOperatorModel:
    patch_size = int(config["patch_size"]) if patch_size_override is None else int(patch_size_override)
    return TangentOperatorModel(
        patch_size=patch_size,
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


def flip_operator_sign(model: torch.nn.Module) -> None:
    last_linear = None
    for module in reversed(model.operator_head.net):
        if isinstance(module, nn.Linear):
            last_linear = module
            break
    if last_linear is None:
        raise RuntimeError("Could not find final Linear layer in model.operator_head.net")
    with torch.no_grad():
        last_linear.weight.mul_(-1.0)
        if last_linear.bias is not None:
            last_linear.bias.mul_(-1.0)


def _rankdata_average_ties(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    order = np.argsort(x, kind="mergesort")
    sorted_x = x[order]
    ranks = np.empty(len(x), dtype=np.float64)
    i = 0
    while i < len(x):
        j = i + 1
        while j < len(x) and sorted_x[j] == sorted_x[i]:
            j += 1
        avg_rank = 0.5 * (i + j - 1) + 1.0
        ranks[order[i:j]] = avg_rank
        i = j
    return ranks


def spearman_corr(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    valid = np.isfinite(x) & np.isfinite(y)
    x = x[valid]
    y = y[valid]
    if len(x) < 2:
        return float("nan")
    rx = _rankdata_average_ties(x)
    ry = _rankdata_average_ties(y)
    rx = rx - rx.mean()
    ry = ry - ry.mean()
    denom = np.sqrt(np.sum(rx * rx) * np.sum(ry * ry))
    if denom <= 1e-12:
        return float("nan")
    return float(np.sum(rx * ry) / denom)


def _patch_offsets(patch_size: int, patch_mode: str, half_width: int) -> np.ndarray:
    if patch_mode == "intrinsic_ordered_stencil":
        r = patch_size // 2
        return np.arange(-r, r + 1, dtype=np.int64)
    if patch_mode == "uniform_symmetric":
        return np.rint(np.linspace(-half_width, half_width, patch_size, endpoint=True)).astype(np.int64)
    raise ValueError(
        "Deterministic global assembly is currently only supported for patch_mode in "
        "{'intrinsic_ordered_stencil', 'uniform_symmetric'}."
    )


def make_centered_cyclic_patches(curve_points: np.ndarray, offsets: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pts = np.asarray(curve_points, dtype=np.float32)
    n = len(pts)
    patches = []
    row_indices = []
    for i in range(n):
        idx = (i + offsets) % n
        patch = pts[idx]
        patches.append(patch - pts[i:i+1])
        row_indices.append(idx)
    return np.stack(patches, axis=0), np.stack(row_indices, axis=0)


def numeric_first_second_from_curve(curve_points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pts = np.asarray(curve_points, dtype=np.float64)
    prev_pt = np.roll(pts, 1, axis=0)
    next_pt = np.roll(pts, -1, axis=0)
    first_raw = next_pt - prev_pt
    first = first_raw / np.clip(np.linalg.norm(first_raw, axis=-1, keepdims=True), 1e-12, None)
    second = next_pt - 2.0 * pts + prev_pt
    return first, second


def cosine_and_angle(pred: np.ndarray, gt: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pred = np.asarray(pred, dtype=np.float64)
    gt = np.asarray(gt, dtype=np.float64)
    pred_u = pred / np.clip(np.linalg.norm(pred, axis=-1, keepdims=True), 1e-12, None)
    gt_u = gt / np.clip(np.linalg.norm(gt, axis=-1, keepdims=True), 1e-12, None)
    cos = np.sum(pred_u * gt_u, axis=-1)
    cos = np.clip(cos, -1.0, 1.0)
    ang = np.degrees(np.arccos(cos))
    return cos, ang


def vector_metrics(pred: np.ndarray, gt: np.ndarray, target_unit_norm: bool = False) -> dict[str, float]:
    cos, ang = cosine_and_angle(pred, gt)
    pred_norm = np.linalg.norm(pred, axis=-1)
    gt_norm = np.linalg.norm(gt, axis=-1)
    target_norm = np.ones_like(pred_norm) if target_unit_norm else gt_norm
    abs_norm_error = np.abs(pred_norm - target_norm)
    out = {
        "cosine_mean": float(np.mean(cos)),
        "abs_cosine_mean": float(np.mean(np.abs(cos))),
        "angle_deg_mean": float(np.mean(ang)),
        "mse": float(np.mean((pred - gt) ** 2)),
        "pred_norm_mean": float(np.mean(pred_norm)),
        "pred_norm_median": float(np.median(pred_norm)),
        "gt_norm_mean": float(np.mean(gt_norm)),
        "gt_norm_median": float(np.median(gt_norm)),
        "norm_error_mean": float(np.mean(abs_norm_error)),
        "norm_error_median": float(np.median(abs_norm_error)),
    }
    return out


@torch.no_grad()
def evaluate_checkpoint(
    *,
    checkpoint_path: str,
    config_path: str,
    bank_path: str,
    family: str,
    patch_size: int,
    patch_mode: str,
    half_width: int,
    device: str,
    max_curves: int | None,
    flip_sign: bool,
) -> dict[str, Any]:
    config = load_json(config_path)
    model = build_model_from_config(config, patch_size_override=patch_size)
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    if flip_sign:
        flip_operator_sign(model)
    device_t = torch.device(device)
    model.to(device_t)
    model.eval()

    bank = PregeneratedCurveBank(bank_path)
    offsets = _patch_offsets(patch_size=patch_size, patch_mode=patch_mode, half_width=half_width)

    num_curves = len(bank) if max_curves is None else min(len(bank), int(max_curves))
    per_curve: list[dict[str, Any]] = []
    W_fro_vals: list[float] = []
    W2_fro_vals: list[float] = []

    for i in range(num_curves):
        curve_points, coeffs, t_grid = bank.get(i)
        curve_points = np.asarray(curve_points, dtype=np.float64)
        n = len(curve_points)

        patches, row_indices = make_centered_cyclic_patches(curve_points.astype(np.float32), offsets)
        patch_t = torch.from_numpy(patches).to(device_t)
        out = model(patch_t)
        weights = out["weights"].detach().cpu().numpy().astype(np.float64)
        direct1 = out["pred"].detach().cpu().numpy().astype(np.float64)

        W = np.zeros((n, n), dtype=np.float64)
        rows = np.arange(n)[:, None]
        W[rows, row_indices] = weights
        global1 = W @ curve_points
        global2 = W @ global1
        global2_alt = W @ direct1

        if coeffs is not None and t_grid is not None:
            _, gt_first, gt_second = compute_fourier_arc_length_derivatives(
                t=np.asarray(t_grid, dtype=np.float64),
                coeffs=coeffs,
                family=family,
            )
            gt_first = np.asarray(gt_first, dtype=np.float64)
            gt_second = np.asarray(gt_second, dtype=np.float64)
        else:
            gt_first, gt_second = numeric_first_second_from_curve(curve_points)

        local1_metrics = vector_metrics(direct1, gt_first, target_unit_norm=True)
        global1_metrics = vector_metrics(global1, gt_first, target_unit_norm=True)
        global2_metrics = vector_metrics(global2, gt_second, target_unit_norm=False)
        global2_alt_metrics = vector_metrics(global2_alt, gt_second, target_unit_norm=False)

        gt2_norm = np.linalg.norm(gt_second, axis=-1)
        g2_norm = np.linalg.norm(global2, axis=-1)
        g2a_norm = np.linalg.norm(global2_alt, axis=-1)

        global2_metrics["norm_spearman_pointwise"] = spearman_corr(g2_norm, gt2_norm)
        global2_alt_metrics["norm_spearman_pointwise"] = spearman_corr(g2a_norm, gt2_norm)

        curve_entry = {
            "curve_index": i,
            "local1": local1_metrics,
            "global1": global1_metrics,
            "global2_from_global1": global2_metrics,
            "global2_from_direct1": global2_alt_metrics,
            "W_fro": float(np.linalg.norm(W, ord="fro")),
            "W2_fro": float(np.linalg.norm(W @ W, ord="fro")),
        }
        per_curve.append(curve_entry)
        W_fro_vals.append(curve_entry["W_fro"])
        W2_fro_vals.append(curve_entry["W2_fro"])

    def avg(section: str, key: str) -> float:
        vals = [entry[section][key] for entry in per_curve if np.isfinite(entry[section][key])]
        return float(np.mean(vals)) if vals else float("nan")

    summary = {
        "checkpoint": str(checkpoint_path),
        "config": str(config_path),
        "bank": str(bank_path),
        "family": family,
        "patch_size": int(patch_size),
        "patch_mode": patch_mode,
        "half_width": int(half_width),
        "flip_sign": bool(flip_sign),
        "num_curves": int(num_curves),
        "local1_vs_gt1": {
            k: avg("local1", k)
            for k in [
                "cosine_mean", "abs_cosine_mean", "angle_deg_mean", "mse",
                "pred_norm_mean", "pred_norm_median", "gt_norm_mean", "gt_norm_median",
                "norm_error_mean", "norm_error_median",
            ]
        },
        "global1_vs_gt1": {
            k: avg("global1", k)
            for k in [
                "cosine_mean", "abs_cosine_mean", "angle_deg_mean", "mse",
                "pred_norm_mean", "pred_norm_median", "gt_norm_mean", "gt_norm_median",
                "norm_error_mean", "norm_error_median",
            ]
        },
        "global2_from_global1_vs_gt2": {
            k: avg("global2_from_global1", k)
            for k in [
                "cosine_mean", "abs_cosine_mean", "angle_deg_mean", "mse",
                "pred_norm_mean", "pred_norm_median", "gt_norm_mean", "gt_norm_median",
                "norm_error_mean", "norm_error_median", "norm_spearman_pointwise",
            ]
        },
        "global2_from_direct1_vs_gt2": {
            k: avg("global2_from_direct1", k)
            for k in [
                "cosine_mean", "abs_cosine_mean", "angle_deg_mean", "mse",
                "pred_norm_mean", "pred_norm_median", "gt_norm_mean", "gt_norm_median",
                "norm_error_mean", "norm_error_median", "norm_spearman_pointwise",
            ]
        },
        "W_fro_mean": float(np.mean(W_fro_vals)) if W_fro_vals else float("nan"),
        "W2_fro_mean": float(np.mean(W2_fro_vals)) if W2_fro_vals else float("nan"),
        "per_curve": per_curve,
    }
    return summary


def print_comparison(label: str, result: dict[str, Any]) -> None:
    print(f"\n===== {label} =====")
    print(f"checkpoint: {result['checkpoint']}")
    if result.get("flip_sign", False):
        print("flip_sign: True")
    g1 = result["global1_vs_gt1"]
    g2 = result["global2_from_global1_vs_gt2"]
    g2a = result["global2_from_direct1_vs_gt2"]
    l1 = result["local1_vs_gt1"]
    print(
        "local1  | "
        f"cos={l1['cosine_mean']:.4f} abs_cos={l1['abs_cosine_mean']:.4f} "
        f"mse={l1['mse']:.6f} pred_norm={l1['pred_norm_mean']:.4f} norm_err={l1['norm_error_mean']:.4f}"
    )
    print(
        "global1 | "
        f"cos={g1['cosine_mean']:.4f} abs_cos={g1['abs_cosine_mean']:.4f} "
        f"mse={g1['mse']:.6f} pred_norm={g1['pred_norm_mean']:.4f} norm_err={g1['norm_error_mean']:.4f}"
    )
    print(
        "global2(W@global1) | "
        f"cos={g2['cosine_mean']:.4f} abs_cos={g2['abs_cosine_mean']:.4f} "
        f"mse={g2['mse']:.6f} norm_spear={g2['norm_spearman_pointwise']:.4f}"
    )
    print(
        "global2(W@direct1) | "
        f"cos={g2a['cosine_mean']:.4f} abs_cos={g2a['abs_cosine_mean']:.4f} "
        f"mse={g2a['mse']:.6f} norm_spear={g2a['norm_spearman_pointwise']:.4f}"
    )


def main() -> None:
    args = parse_args()
    config_b = args.config_b if args.config_b is not None else args.config_a

    result_a = evaluate_checkpoint(
        checkpoint_path=args.checkpoint_a,
        config_path=args.config_a,
        bank_path=args.bank,
        family=args.family,
        patch_size=args.patch_size,
        patch_mode=args.patch_mode,
        half_width=args.half_width,
        device=args.device,
        max_curves=args.max_curves,
        flip_sign=args.flip_sign_a,
    )
    result_b = evaluate_checkpoint(
        checkpoint_path=args.checkpoint_b,
        config_path=config_b,
        bank_path=args.bank,
        family=args.family,
        patch_size=args.patch_size,
        patch_mode=args.patch_mode,
        half_width=args.half_width,
        device=args.device,
        max_curves=args.max_curves,
        flip_sign=args.flip_sign_b,
    )

    print_comparison(args.label_a, result_a)
    print_comparison(args.label_b, result_b)

    comparison = {
        args.label_a: result_a,
        args.label_b: result_b,
    }
    if args.out_json is not None:
        out_path = Path(args.out_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(comparison, indent=2))
        print(f"\nSaved comparison JSON to: {out_path}")


if __name__ == "__main__":
    main()
