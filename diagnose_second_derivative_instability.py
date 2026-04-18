from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from datasets.tangent_dataset import PregeneratedCurveBank
from utils.derivatives import (
    evaluate_fourier_curve_and_parameter_derivatives,
    compute_arc_length_derivatives_from_parameter_derivatives,
)

try:
    from scipy.signal import savgol_filter
    HAVE_SCIPY = True
except Exception:
    HAVE_SCIPY = False


EPS = 1e-12


@dataclass
class CurveDiagnostics:
    per_point: pd.DataFrame
    per_curve: pd.DataFrame


def wrap_idx(i: np.ndarray | int, n: int):
    return np.mod(i, n)


def angle_between(u: np.ndarray, v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    un = u / np.clip(np.linalg.norm(u, axis=-1, keepdims=True), eps, None)
    vn = v / np.clip(np.linalg.norm(v, axis=-1, keepdims=True), eps, None)
    c = np.sum(un * vn, axis=-1)
    c = np.clip(c, -1.0, 1.0)
    return np.degrees(np.arccos(c))


def signed_turning_angle(prev_vec: np.ndarray, next_vec: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    # angle from prev_vec to next_vec in (-pi, pi]
    prev_n = prev_vec / np.clip(np.linalg.norm(prev_vec, axis=-1, keepdims=True), eps, None)
    next_n = next_vec / np.clip(np.linalg.norm(next_vec, axis=-1, keepdims=True), eps, None)
    cross = prev_n[:, 0] * next_n[:, 1] - prev_n[:, 1] * next_n[:, 0]
    dot = np.sum(prev_n * next_n, axis=-1)
    return np.arctan2(cross, np.clip(dot, -1.0, 1.0))


def rankdata_average(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x, kind='mergesort')
    ranks = np.empty(len(x), dtype=np.float64)
    xs = x[order]
    i = 0
    while i < len(x):
        j = i + 1
        while j < len(x) and xs[j] == xs[i]:
            j += 1
        avg_rank = 0.5 * (i + j - 1) + 1.0
        ranks[order[i:j]] = avg_rank
        i = j
    return ranks


def pearsonr_np(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    x = x - x.mean()
    y = y - y.mean()
    denom = np.sqrt(np.sum(x * x) * np.sum(y * y))
    if denom <= 0:
        return np.nan
    return float(np.sum(x * y) / denom)


def spearmanr_np(x: np.ndarray, y: np.ndarray) -> float:
    rx = rankdata_average(np.asarray(x))
    ry = rankdata_average(np.asarray(y))
    return pearsonr_np(rx, ry)


def linear_fit(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    x = np.asarray(y_true, dtype=np.float64)
    y = np.asarray(y_pred, dtype=np.float64)
    A = np.stack([x, np.ones_like(x)], axis=1)
    sol, *_ = np.linalg.lstsq(A, y, rcond=None)
    slope, intercept = sol
    return {"slope": float(slope), "intercept": float(intercept)}


def exact_curve_quantities(coeffs, t_grid: np.ndarray):
    pts, dt1, dt2, dt3 = evaluate_fourier_curve_and_parameter_derivatives(t_grid, coeffs)
    ds1, ds2 = compute_arc_length_derivatives_from_parameter_derivatives(
        family="euclidean",
        first_dt=dt1,
        second_dt=dt2,
        third_dt=dt3,
    )
    sigma = np.linalg.norm(dt1, axis=-1)
    sigma = np.clip(sigma, EPS, None)
    sigma_t = np.sum(dt1 * dt2, axis=-1) / sigma

    A = dt2 / (sigma[:, None] ** 2)
    B = dt1 * (sigma_t[:, None] / (sigma[:, None] ** 3))
    cancel_ratio = (np.linalg.norm(A, axis=-1) + np.linalg.norm(B, axis=-1)) / np.clip(np.linalg.norm(ds2, axis=-1), EPS, None)

    return {
        "points": pts,
        "dt1": dt1,
        "dt2": dt2,
        "dt3": dt3,
        "ds1": ds1,
        "ds2": ds2,
        "kappa": np.linalg.norm(ds2, axis=-1),
        "sigma": sigma,
        "sigma_t": sigma_t,
        "A": A,
        "B": B,
        "cancel_ratio": cancel_ratio,
    }


def polygon_local_geometry(points: np.ndarray) -> Dict[str, np.ndarray]:
    n = len(points)
    p_prev = points[wrap_idx(np.arange(n) - 1, n)]
    p = points
    p_next = points[wrap_idx(np.arange(n) + 1, n)]

    v_prev = p - p_prev
    v_next = p_next - p
    chord = p_next - p_prev

    h_prev = np.linalg.norm(v_prev, axis=-1)
    h_next = np.linalg.norm(v_next, axis=-1)
    h_chord = np.linalg.norm(chord, axis=-1)
    h_avg = 0.5 * (h_prev + h_next)
    irregularity_ratio = np.maximum(h_prev, h_next) / np.clip(np.minimum(h_prev, h_next), EPS, None)
    turning = np.abs(signed_turning_angle(v_prev, v_next))

    tangent_chord = chord / np.clip(h_chord[:, None], EPS, None)
    normal_left = np.stack([-tangent_chord[:, 1], tangent_chord[:, 0]], axis=1)

    # simple turning-angle curvature magnitude proxy: |
    # delta theta / average local arc step |
    kappa_turn = turning / np.clip(h_avg, EPS, None)
    vec_turn = kappa_turn[:, None] * normal_left

    # circumcircle-based curvature magnitude: 4 area / (abc)
    a = h_prev
    b = h_next
    c = h_chord
    cross = np.abs(v_prev[:, 0] * v_next[:, 1] - v_prev[:, 1] * v_next[:, 0])
    area2 = cross  # = 2 * area
    kappa_circ = 2.0 * area2 / np.clip(a * b * c, EPS, None)
    vec_circ = kappa_circ[:, None] * normal_left

    return {
        "h_prev": h_prev,
        "h_next": h_next,
        "h_avg": h_avg,
        "h_chord": h_chord,
        "irregularity_ratio": irregularity_ratio,
        "turning_angle_abs": turning,
        "normal_left": normal_left,
        "kappa_turn": kappa_turn,
        "vec_turn": vec_turn,
        "kappa_circ": kappa_circ,
        "vec_circ": vec_circ,
    }


def estimate_second_from_savgol(points: np.ndarray, window: int, degree: int) -> Dict[str, np.ndarray]:
    if not HAVE_SCIPY:
        raise RuntimeError("SciPy not available; cannot run Savitzky-Golay baseline.")
    n = len(points)
    if window % 2 == 0:
        window += 1
    if window >= n:
        window = n - 1 if (n - 1) % 2 == 1 else n - 2
    if window < degree + 2:
        window = degree + 3 if (degree + 3) % 2 == 1 else degree + 4

    ext = np.concatenate([points[-window:], points, points[:window]], axis=0)
    x = ext[:, 0]
    y = ext[:, 1]

    dx = savgol_filter(x, window_length=window, polyorder=degree, deriv=1, delta=1.0, mode='interp')[window:-window]
    dy = savgol_filter(y, window_length=window, polyorder=degree, deriv=1, delta=1.0, mode='interp')[window:-window]
    ddx = savgol_filter(x, window_length=window, polyorder=degree, deriv=2, delta=1.0, mode='interp')[window:-window]
    ddy = savgol_filter(y, window_length=window, polyorder=degree, deriv=2, delta=1.0, mode='interp')[window:-window]

    d1 = np.stack([dx, dy], axis=1)
    d2 = np.stack([ddx, ddy], axis=1)
    speed = np.linalg.norm(d1, axis=-1)
    speed = np.clip(speed, EPS, None)
    speed_t = np.sum(d1 * d2, axis=-1) / speed
    ds2 = d2 / (speed[:, None] ** 2) - d1 * (speed_t[:, None] / (speed[:, None] ** 3))
    return {"vec_savgol": ds2, "kappa_savgol": np.linalg.norm(ds2, axis=-1)}


def summarize_method(df: pd.DataFrame, pred_vec_col: str, pred_norm_col: str, prefix: str) -> Dict[str, float]:
    gt_vec = np.stack(df["gt_vec"].to_numpy())
    pred_vec = np.stack(df[pred_vec_col].to_numpy())
    gt_norm = df["gt_norm"].to_numpy()
    pred_norm = df[pred_norm_col].to_numpy()

    cos = np.sum(
        pred_vec / np.clip(np.linalg.norm(pred_vec, axis=-1, keepdims=True), EPS, None)
        * gt_vec / np.clip(np.linalg.norm(gt_vec, axis=-1, keepdims=True), EPS, None),
        axis=-1,
    )
    cos = np.clip(cos, -1.0, 1.0)
    angle = np.degrees(np.arccos(cos))

    fit = linear_fit(gt_norm, pred_norm)
    return {
        f"{prefix}_cosine_mean": float(np.mean(cos)),
        f"{prefix}_angle_mean": float(np.mean(angle)),
        f"{prefix}_mse": float(np.mean((pred_vec - gt_vec) ** 2)),
        f"{prefix}_norm_spearman": float(spearmanr_np(gt_norm, pred_norm)),
        f"{prefix}_norm_pearson": float(pearsonr_np(gt_norm, pred_norm)),
        f"{prefix}_log1p_norm_pearson": float(pearsonr_np(np.log1p(gt_norm), np.log1p(pred_norm))),
        f"{prefix}_norm_fit_slope": fit["slope"],
        f"{prefix}_norm_fit_intercept": fit["intercept"],
    }


def make_plots(df: pd.DataFrame, outdir: Path, methods: List[str]) -> None:
    outdir.mkdir(parents=True, exist_ok=True)

    # 1) target tail
    plt.figure(figsize=(7, 5))
    plt.hist(df["gt_norm"], bins=120)
    plt.xlabel("true ||x''(s)||")
    plt.ylabel("count")
    plt.title("Distribution of true curvature magnitude")
    plt.tight_layout()
    plt.savefig(outdir / "gt_norm_hist.png", dpi=180)
    plt.close()

    plt.figure(figsize=(7, 5))
    vals = np.sort(df["gt_norm"].to_numpy())
    ccdf = 1.0 - np.arange(1, len(vals) + 1) / len(vals)
    plt.loglog(np.maximum(vals, 1e-12), np.maximum(ccdf, 1e-12))
    plt.xlabel("true ||x''(s)||")
    plt.ylabel("CCDF")
    plt.title("Tail of true curvature magnitude")
    plt.tight_layout()
    plt.savefig(outdir / "gt_norm_ccdf.png", dpi=180)
    plt.close()

    # 2) instability diagnostics
    for xcol, name in [
        ("irregularity_ratio", "error_vs_irregularity"),
        ("cancel_ratio", "error_vs_cancel_ratio"),
        ("gt_norm", "error_vs_true_norm"),
        ("turning_angle_abs", "error_vs_turning_angle"),
    ]:
        plt.figure(figsize=(7, 5))
        x = df[xcol].to_numpy()
        for m in methods:
            y = np.abs(df[f"{m}_norm"] - df["gt_norm"])
            plt.scatter(x, y, s=4, alpha=0.15, label=m)
        if len(methods) <= 4:
            plt.legend()
        plt.xlabel(xcol)
        plt.ylabel("absolute norm error")
        plt.title(name)
        if xcol in {"gt_norm", "cancel_ratio"}:
            plt.xscale("log")
            plt.yscale("log")
        plt.tight_layout()
        plt.savefig(outdir / f"{name}.png", dpi=180)
        plt.close()

    # 3) predicted vs true on log-scale
    plt.figure(figsize=(7, 5))
    gt = np.log1p(df["gt_norm"].to_numpy())
    for m in methods:
        pred = np.log1p(df[f"{m}_norm"].to_numpy())
        plt.scatter(gt, pred, s=4, alpha=0.15, label=m)
    lim0 = min(gt.min(), *(np.log1p(df[f"{m}_norm"]).min() for m in methods))
    lim1 = max(gt.max(), *(np.log1p(df[f"{m}_norm"]).max() for m in methods))
    plt.plot([lim0, lim1], [lim0, lim1], 'k--', linewidth=1)
    plt.xlabel("log1p true ||x''||")
    plt.ylabel("log1p predicted ||x''||")
    plt.title("Log-scale calibration")
    if len(methods) <= 4:
        plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / "log_calibration_scatter.png", dpi=180)
    plt.close()


def run_bank(bank_path: str, output_dir: str, max_curves: int | None, savgol_window: int, savgol_degree: int) -> None:
    bank = PregeneratedCurveBank(bank_path)
    outdir = Path(output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    n_curves = len(bank) if max_curves is None else min(len(bank), max_curves)
    per_point_rows = []
    per_curve_rows = []

    for curve_idx in range(n_curves):
        points, coeffs, t_grid = bank.get(curve_idx)
        if coeffs is None or t_grid is None:
            raise RuntimeError(
                "This diagnostic needs coeffs and t_grid in the bank so we can compute exact ground truth and isolate numerical instability."
            )

        exact = exact_curve_quantities(coeffs, t_grid)
        geom = polygon_local_geometry(points)
        n = len(points)

        curve_df = pd.DataFrame({
            "curve_idx": curve_idx,
            "point_idx": np.arange(n),
            "gt_norm": exact["kappa"],
            "cancel_ratio": exact["cancel_ratio"],
            "sigma": exact["sigma"],
            "sigma_t": exact["sigma_t"],
            "h_prev": geom["h_prev"],
            "h_next": geom["h_next"],
            "h_avg": geom["h_avg"],
            "irregularity_ratio": geom["irregularity_ratio"],
            "turning_angle_abs": geom["turning_angle_abs"],
        })
        curve_df["gt_vec"] = list(exact["ds2"])
        curve_df["turn_vec"] = list(geom["vec_turn"])
        curve_df["circ_vec"] = list(geom["vec_circ"])
        curve_df["turn_norm"] = geom["kappa_turn"]
        curve_df["circ_norm"] = geom["kappa_circ"]

        if HAVE_SCIPY:
            sg = estimate_second_from_savgol(points, window=savgol_window, degree=savgol_degree)
            curve_df["savgol_vec"] = list(sg["vec_savgol"])
            curve_df["savgol_norm"] = sg["kappa_savgol"]

        # curve-level summaries
        q90 = np.quantile(exact["kappa"], 0.90)
        q99 = np.quantile(exact["kappa"], 0.99)
        per_curve_rows.append({
            "curve_idx": curve_idx,
            "num_points": n,
            "gt_norm_mean": float(np.mean(exact["kappa"])),
            "gt_norm_median": float(np.median(exact["kappa"])),
            "gt_norm_q90": float(q90),
            "gt_norm_q99": float(q99),
            "gt_norm_max": float(np.max(exact["kappa"])),
            "irregularity_mean": float(np.mean(geom["irregularity_ratio"])),
            "irregularity_max": float(np.max(geom["irregularity_ratio"])),
            "cancel_ratio_mean": float(np.mean(exact["cancel_ratio"])),
            "cancel_ratio_q95": float(np.quantile(exact["cancel_ratio"], 0.95)),
        })

        per_point_rows.append(curve_df)

    df = pd.concat(per_point_rows, ignore_index=True)
    per_curve = pd.DataFrame(per_curve_rows)

    methods = ["turn", "circ"] + (["savgol"] if HAVE_SCIPY and "savgol_norm" in df.columns else [])
    summary = {}
    for m in methods:
        summary.update(summarize_method(df, pred_vec_col=f"{m}_vec", pred_norm_col=f"{m}_norm", prefix=m))

    # quantile-conditioned summaries
    gt = df["gt_norm"].to_numpy()
    deciles = pd.qcut(gt, q=10, duplicates="drop")
    cond_rows = []
    for q, sub in df.groupby(deciles, observed=False):
        row = {
            "bin": str(q),
            "count": len(sub),
            "gt_norm_mean": float(sub["gt_norm"].mean()),
            "irregularity_mean": float(sub["irregularity_ratio"].mean()),
            "cancel_ratio_mean": float(sub["cancel_ratio"].mean()),
        }
        for m in methods:
            row[f"{m}_abs_norm_err_mean"] = float(np.mean(np.abs(sub[f"{m}_norm"] - sub["gt_norm"])))
            row[f"{m}_norm_pearson"] = float(pearsonr_np(sub["gt_norm"].to_numpy(), sub[f"{m}_norm"].to_numpy()))
        cond_rows.append(row)
    quantile_df = pd.DataFrame(cond_rows)

    # regressions against instability factors
    factor_summary = {}
    for factor in ["irregularity_ratio", "cancel_ratio", "gt_norm", "turning_angle_abs"]:
        for m in methods:
            err = np.abs(df[f"{m}_norm"].to_numpy() - gt)
            factor_summary[f"corr_{m}_abs_err__{factor}"] = pearsonr_np(np.log1p(df[factor].to_numpy()), np.log1p(err))

    # save
    df.to_parquet(outdir / "per_point.parquet")
    per_curve.to_csv(outdir / "per_curve.csv", index=False)
    quantile_df.to_csv(outdir / "error_by_true_norm_quantile.csv", index=False)
    with open(outdir / "summary.json", "w") as f:
        json.dump({**summary, **factor_summary, "num_points_total": int(len(df)), "num_curves": int(n_curves)}, f, indent=2)

    make_plots(df, outdir, methods)

    print(json.dumps({**summary, **factor_summary, "num_points_total": int(len(df)), "num_curves": int(n_curves)}, indent=2))
    print(f"Saved outputs to: {outdir}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--bank", type=str, required=True, help="Path to pregenerated .npz bank")
    p.add_argument("--output-dir", type=str, required=True)
    p.add_argument("--max-curves", type=int, default=None)
    p.add_argument("--savgol-window", type=int, default=21)
    p.add_argument("--savgol-degree", type=int, default=5)
    args = p.parse_args()

    run_bank(
        bank_path=args.bank,
        output_dir=args.output_dir,
        max_curves=args.max_curves,
        savgol_window=args.savgol_window,
        savgol_degree=args.savgol_degree,
    )
