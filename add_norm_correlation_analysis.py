#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    from scipy.stats import spearmanr as scipy_spearmanr  # type: ignore
except Exception:
    scipy_spearmanr = None

PAIR_NAMES = [
    "direct1_vs_gt1",
    "global1_vs_gt1",
    "global2_from_global1_vs_gt2",
    "global2_from_direct1_vs_gt2",
    "global1_vs_direct1",
]

HEATMAP_PAIRS = [
    "global2_from_global1_vs_gt2",
    "global2_from_direct1_vs_gt2",
    "global1_vs_gt1",
    "direct1_vs_gt1",
]


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def sanitize(name: str) -> str:
    return name.replace("/", "_").replace(" ", "_")


def rankdata_average(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), dtype=np.float64)
    sorted_x = x[order]
    i = 0
    while i < len(x):
        j = i + 1
        while j < len(x) and sorted_x[j] == sorted_x[i]:
            j += 1
        avg_rank = 0.5 * (i + j - 1) + 1.0
        ranks[order[i:j]] = avg_rank
        i = j
    return ranks


def pearson_corr(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if len(x) < 2:
        return float("nan")
    x_std = np.std(x)
    y_std = np.std(y)
    if x_std < 1e-12 or y_std < 1e-12:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def spearman_corr(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if len(x) < 2:
        return float("nan")
    if scipy_spearmanr is not None:
        val = scipy_spearmanr(x, y).statistic
        return float(val)
    rx = rankdata_average(x)
    ry = rankdata_average(y)
    return pearson_corr(rx, ry)


def fit_stats(gt_norm: np.ndarray, pred_norm: np.ndarray) -> dict[str, float]:
    x = np.asarray(gt_norm, dtype=np.float64)
    y = np.asarray(pred_norm, dtype=np.float64)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if len(x) < 2:
        return {"slope": float("nan"), "intercept": float("nan"), "r2_pred": float("nan"), "r2_gt": float("nan")}
    if np.std(x) < 1e-12:
        return {"slope": float("nan"), "intercept": float("nan"), "r2_pred": float("nan"), "r2_gt": float("nan")}
    slope, intercept = np.polyfit(x, y, 1)
    y_fit = slope * x + intercept
    ss_res = float(np.sum((y - y_fit) ** 2))
    ss_tot_y = float(np.sum((y - np.mean(y)) ** 2))
    ss_tot_x = float(np.sum((x - np.mean(x)) ** 2))
    r2_pred = float("nan") if ss_tot_y < 1e-12 else 1.0 - ss_res / ss_tot_y
    # how much GT variance is explained if we invert the fit imperfectly is less meaningful, but keep optional
    r2_gt = float("nan") if ss_tot_x < 1e-12 else pearson_corr(x, y) ** 2
    return {"slope": float(slope), "intercept": float(intercept), "r2_pred": r2_pred, "r2_gt": r2_gt}


def summarize(arr: np.ndarray) -> dict[str, float]:
    arr = np.asarray(arr, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {k: float("nan") for k in ["mean", "std", "median", "p05", "p25", "p75", "p95"]}
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "median": float(np.median(arr)),
        "p05": float(np.percentile(arr, 5)),
        "p25": float(np.percentile(arr, 25)),
        "p75": float(np.percentile(arr, 75)),
        "p95": float(np.percentile(arr, 95)),
    }


def pair_norm_stats(pred: np.ndarray, gt: np.ndarray) -> dict[str, float]:
    pred_norm = np.linalg.norm(pred, axis=-1)
    gt_norm = np.linalg.norm(gt, axis=-1)
    out = {
        "pearson": pearson_corr(pred_norm, gt_norm),
        "spearman": spearman_corr(pred_norm, gt_norm),
    }
    out.update(fit_stats(gt_norm, pred_norm))
    out["pred_norm_mean"] = float(np.mean(pred_norm))
    out["gt_norm_mean"] = float(np.mean(gt_norm))
    out["pred_norm_std"] = float(np.std(pred_norm))
    out["gt_norm_std"] = float(np.std(gt_norm))
    return out


def collect_run_dirs(root: Path) -> list[Path]:
    candidates = []

    # Case 1: root itself is a single run
    if (root / "global").is_dir() and (root / "curves").is_dir():
        return [root]

    # Case 2: immediate children are runs
    for sub in sorted(root.iterdir()):
        if sub.is_dir() and (sub / "global").is_dir() and (sub / "curves").is_dir():
            candidates.append(sub)

    if candidates:
        return candidates

    # Case 3: dataset folders contain per-run subfolders
    for dataset_dir in sorted(root.iterdir()):
        if not dataset_dir.is_dir():
            continue
        for sub in sorted(dataset_dir.iterdir()):
            if sub.is_dir() and (sub / "global").is_dir() and (sub / "curves").is_dir():
                candidates.append(sub)

    if candidates:
        return candidates

    raise FileNotFoundError(f"Could not find run directories under {root}")


def parse_run_identity(run_dir: Path) -> dict[str, Any]:
    name = run_dir.name
    patch_size = None
    dataset_name = name
    if "_ps" in name:
        dataset_name, ps = name.rsplit("_ps", 1)
        try:
            patch_size = int(ps)
        except Exception:
            patch_size = None
    return {"run_name": name, "dataset_name": dataset_name, "patch_size": patch_size}


def analyze_run(run_dir: Path, scatter_pair: str | None, max_scatter_curves: int) -> dict[str, Any]:
    global_dir = run_dir / "global"
    curves_dir = run_dir / "curves"
    out_dir = global_dir / "norm_correlation"
    ensure_dir(out_dir)

    per_curve_rows: list[dict[str, Any]] = []
    pointwise_norms: dict[str, list[np.ndarray]] = {pair: [] for pair in PAIR_NAMES}
    pointwise_gt_norms: dict[str, list[np.ndarray]] = {pair: [] for pair in PAIR_NAMES}

    curve_paths = sorted([p for p in curves_dir.iterdir() if p.is_dir() and p.name.startswith("curve_")])
    for curve_path in curve_paths:
        arr_path = curve_path / "arrays.npz"
        if not arr_path.is_file():
            continue
        data = np.load(arr_path)
        row: dict[str, Any] = {"curve_dir": curve_path.name}
        try:
            row["curve_idx"] = int(np.asarray(data["curve_idx"]).item())
        except Exception:
            row["curve_idx"] = np.nan

        for pair in PAIR_NAMES:
            pred_key = f"{pair}__pred"
            gt_key = f"{pair}__gt"
            if pred_key not in data or gt_key not in data:
                continue
            pred = np.asarray(data[pred_key], dtype=np.float64)
            gt = np.asarray(data[gt_key], dtype=np.float64)
            stats = pair_norm_stats(pred, gt)
            for k, v in stats.items():
                row[f"{pair}__norm_{k}"] = v
            pointwise_norms[pair].append(np.linalg.norm(pred, axis=-1))
            pointwise_gt_norms[pair].append(np.linalg.norm(gt, axis=-1))
        per_curve_rows.append(row)

    per_curve_df = pd.DataFrame(per_curve_rows)
    per_curve_csv = out_dir / "per_curve_norm_correlations.csv"
    per_curve_df.to_csv(per_curve_csv, index=False)

    global_summary: dict[str, Any] = {
        "run_dir": str(run_dir),
        "pairs": {},
    }
    sweep_row = parse_run_identity(run_dir)

    for pair in PAIR_NAMES:
        if not pointwise_norms[pair]:
            continue
        pred_norm = np.concatenate(pointwise_norms[pair], axis=0)
        gt_norm = np.concatenate(pointwise_gt_norms[pair], axis=0)
        pointwise = pair_norm_stats(np.stack([pred_norm, np.zeros_like(pred_norm)], axis=-1), np.stack([gt_norm, np.zeros_like(gt_norm)], axis=-1))
        # per-curve summaries of metrics already in df
        curve_metrics = {}
        for metric in ["pearson", "spearman", "slope", "intercept", "r2_pred", "r2_gt", "pred_norm_mean", "gt_norm_mean", "pred_norm_std", "gt_norm_std"]:
            col = f"{pair}__norm_{metric}"
            if col in per_curve_df.columns:
                curve_metrics[metric] = summarize(per_curve_df[col].to_numpy(dtype=np.float64))
        global_summary["pairs"][pair] = {
            "pointwise": pointwise,
            "per_curve": curve_metrics,
        }
        for metric in ["pearson", "spearman", "slope", "r2_pred"]:
            sweep_row[f"{pair}__norm_{metric}_pointwise"] = pointwise.get(metric, float("nan"))
            if metric in curve_metrics:
                sweep_row[f"{pair}__norm_{metric}_per_curve_mean"] = curve_metrics[metric]["mean"]

    with open(out_dir / "norm_correlation_summary.json", "w") as f:
        json.dump(global_summary, f, indent=2)

    if scatter_pair and scatter_pair in pointwise_norms and pointwise_norms[scatter_pair]:
        n_show = min(max_scatter_curves, len(pointwise_norms[scatter_pair]))
        pred_norm = np.concatenate(pointwise_norms[scatter_pair][:n_show], axis=0)
        gt_norm = np.concatenate(pointwise_gt_norms[scatter_pair][:n_show], axis=0)
        plt.figure(figsize=(6, 6))
        plt.scatter(gt_norm, pred_norm, s=6, alpha=0.25)
        if np.std(gt_norm) > 1e-12:
            slope, intercept = np.polyfit(gt_norm, pred_norm, 1)
            xline = np.linspace(np.min(gt_norm), np.max(gt_norm), 200)
            plt.plot(xline, slope * xline + intercept, linewidth=2)
        plt.xlabel("GT norm")
        plt.ylabel("Pred norm")
        plt.title(f"Norm scatter: {scatter_pair}")
        plt.tight_layout()
        plt.savefig(out_dir / f"scatter_{sanitize(scatter_pair)}_norms.png", dpi=180)
        plt.close()

    return sweep_row


def make_heatmap(df: pd.DataFrame, value_col: str, out_path: Path, title: str) -> None:
    sub = df[["dataset_name", "patch_size", value_col]].dropna()
    if sub.empty:
        return
    pivot = sub.pivot(index="dataset_name", columns="patch_size", values=value_col)
    pivot = pivot.sort_index().reindex(sorted(pivot.columns), axis=1)
    plt.figure(figsize=(12, 6))
    im = plt.imshow(pivot.values, aspect="auto")
    plt.xticks(range(len(pivot.columns)), [str(c) for c in pivot.columns])
    plt.yticks(range(len(pivot.index)), pivot.index)
    plt.title(title)
    plt.colorbar(im)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def main() -> None:
    p = argparse.ArgumentParser(description="Add norm-correlation analysis from saved invariant-operator artifacts.")
    p.add_argument("--analysis-root", type=str, required=True, help="Sweep root or single run directory.")
    p.add_argument("--scatter-pair", type=str, default="global2_from_global1_vs_gt2", choices=PAIR_NAMES)
    p.add_argument("--max-scatter-curves", type=int, default=8)
    args = p.parse_args()

    root = Path(args.analysis_root)
    run_dirs = collect_run_dirs(root)

    sweep_rows = []
    for run_dir in run_dirs:
        print(f"Processing {run_dir}")
        sweep_rows.append(analyze_run(run_dir, args.scatter_pair, args.max_scatter_curves))

    if len(run_dirs) > 1:
        sweep_df = pd.DataFrame(sweep_rows)
        sweep_csv = root / "norm_correlation_sweep_summary.csv"
        sweep_df.to_csv(sweep_csv, index=False)
        print(f"Saved {sweep_csv}")

        for pair in HEATMAP_PAIRS:
            for metric in ["pearson", "spearman", "r2_pred"]:
                col = f"{pair}__norm_{metric}_pointwise"
                if col in sweep_df.columns:
                    make_heatmap(
                        sweep_df,
                        col,
                        root / f"heatmap_{sanitize(pair)}__norm_{metric}_pointwise.png",
                        f"{pair} norm {metric} (pointwise)",
                    )


if __name__ == "__main__":
    main()
