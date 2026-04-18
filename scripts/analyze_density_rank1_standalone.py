#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import json
import math
import os
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# -----------------------------------------------------------------------------
# Standalone analyzer for density x patch-size rank-1 experiments.
#
# It scans checkpoint directories like:
#   checkpoints_density/<dataset>/rank1/ps<PS>/nce_<LNCE>/temp_<TEMP>/...
#
# For each run, it tries to recover metrics from files already written by the
# project (json/csv/txt). It is intentionally defensive because exact filenames
# can vary. Once metrics are found, it builds a summary table and a compact but
# informative plot set.
# -----------------------------------------------------------------------------

EXPECTED_KEYS = [
    "global_first_cos_mean",
    "global_first_mse_mean",
    "global_first_norm_mean",
    "global_second_cos_mean",
    "global_second_abs_cos_mean",
    "global_second_corr_mean",
    "global_second_norm_corr_mean",
    "global_second_scale_ratio_mean",
    "global_second_mse_mean",
    "best_val_loss",
    "best_epoch",
    "epochs_ran",
]

KEY_ALIASES = {
    "first_cos": "global_first_cos_mean",
    "first_cos_mean": "global_first_cos_mean",
    "mean_first_cos": "global_first_cos_mean",
    "global_first_cos": "global_first_cos_mean",
    "first_mse": "global_first_mse_mean",
    "first_mse_mean": "global_first_mse_mean",
    "global_first_mse": "global_first_mse_mean",
    "first_norm": "global_first_norm_mean",
    "first_norm_mean": "global_first_norm_mean",
    "second_cos": "global_second_cos_mean",
    "second_cos_mean": "global_second_cos_mean",
    "second_abs_cos": "global_second_abs_cos_mean",
    "second_abs_cos_mean": "global_second_abs_cos_mean",
    "global_second_abs_cos": "global_second_abs_cos_mean",
    "second_corr": "global_second_corr_mean",
    "second_corr_mean": "global_second_corr_mean",
    "vector_corr": "global_second_corr_mean",
    "second_vector_corr": "global_second_corr_mean",
    "second_norm_corr": "global_second_norm_corr_mean",
    "second_norm_corr_mean": "global_second_norm_corr_mean",
    "norm_corr": "global_second_norm_corr_mean",
    "second_scale_ratio": "global_second_scale_ratio_mean",
    "second_scale_ratio_mean": "global_second_scale_ratio_mean",
    "scale_ratio": "global_second_scale_ratio_mean",
    "second_mse": "global_second_mse_mean",
    "second_mse_mean": "global_second_mse_mean",
    "global_second_mse": "global_second_mse_mean",
    "val_loss": "best_val_loss",
    "best_validation_loss": "best_val_loss",
    "epoch": "best_epoch",
}

PREFERRED_METRICS = [
    "global_first_cos_mean",
    "global_first_mse_mean",
    "global_second_abs_cos_mean",
    "global_second_corr_mean",
    "global_second_norm_corr_mean",
    "global_second_scale_ratio_mean",
    "global_second_mse_mean",
    "best_epoch",
]


def canonicalize_key(key: str) -> str:
    k = key.strip().lower()
    return KEY_ALIASES.get(k, key.strip())


def to_float_if_possible(x: Any) -> Any:
    if isinstance(x, (int, float, np.integer, np.floating)):
        return float(x)
    if isinstance(x, str):
        s = x.strip()
        try:
            return float(s)
        except Exception:
            return x
    return x


def flatten_dict(d: Dict[str, Any], prefix: str = "") -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in d.items():
        key = f"{prefix}.{k}" if prefix else str(k)
        if isinstance(v, dict):
            out.update(flatten_dict(v, key))
        else:
            out[key] = v
    return out


def extract_metrics_from_mapping(mapping: Dict[str, Any]) -> Dict[str, float]:
    found: Dict[str, float] = {}
    flat = flatten_dict(mapping)
    for k, v in flat.items():
        ck = canonicalize_key(k.split(".")[-1])
        vv = to_float_if_possible(v)
        if isinstance(vv, float) and math.isfinite(vv):
            if ck in EXPECTED_KEYS or ck in PREFERRED_METRICS:
                found[ck] = vv
    return found


def extract_metrics_from_json(path: Path) -> Dict[str, float]:
    try:
        data = json.loads(path.read_text())
    except Exception:
        return {}
    if isinstance(data, dict):
        return extract_metrics_from_mapping(data)
    if isinstance(data, list) and data and isinstance(data[0], dict):
        best: Dict[str, float] = {}
        for item in data:
            best.update(extract_metrics_from_mapping(item))
        return best
    return {}


def extract_metrics_from_csv(path: Path) -> Dict[str, float]:
    try:
        df = pd.read_csv(path)
    except Exception:
        return {}
    if df.empty:
        return {}

    # Try last row, best row, and per-column scalar style.
    candidates: List[Dict[str, Any]] = []
    try:
        candidates.append(df.iloc[-1].to_dict())
    except Exception:
        pass
    if "best_epoch" in df.columns:
        try:
            idx = int(df["best_epoch"].dropna().index[-1])
            candidates.append(df.loc[idx].to_dict())
        except Exception:
            pass
    if len(df) == 1:
        candidates.append(df.iloc[0].to_dict())

    merged: Dict[str, float] = {}
    for cand in candidates:
        merged.update(extract_metrics_from_mapping(cand))

    for col in df.columns:
        ck = canonicalize_key(col)
        if ck in EXPECTED_KEYS or ck in PREFERRED_METRICS:
            series = pd.to_numeric(df[col], errors="coerce").dropna()
            if len(series) > 0:
                merged[ck] = float(series.iloc[-1])
    return merged


def extract_metrics_from_txt(path: Path) -> Dict[str, float]:
    text = path.read_text(errors="ignore")
    found: Dict[str, float] = {}
    # key=value or key: value
    patt = re.compile(r"([A-Za-z0-9_\.\-/]+)\s*[:=]\s*(-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)")
    for m in patt.finditer(text):
        ck = canonicalize_key(m.group(1).split("/")[-1])
        if ck in EXPECTED_KEYS or ck in PREFERRED_METRICS:
            found[ck] = float(m.group(2))
    return found


def parse_run_info(path: Path) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    if not path.exists():
        return out
    for line in path.read_text(errors="ignore").splitlines():
        if "=" not in line:
            continue
        k, v = line.split("=", 1)
        out[k.strip()] = v.strip()
    return out


def parse_dataset_name(name: str) -> Tuple[Optional[int], Optional[int]]:
    m = re.search(r"_(\d+)to(\d+)(?:_filt)?$", name)
    if not m:
        return None, None
    return int(m.group(1)), int(m.group(2))


def list_candidate_metric_files(run_dir: Path) -> List[Path]:
    patterns = [
        "**/*metrics*.json",
        "**/*summary*.json",
        "**/*results*.json",
        "**/*eval*.json",
        "**/*analysis*.json",
        "**/*metrics*.csv",
        "**/*summary*.csv",
        "**/*results*.csv",
        "**/*eval*.csv",
        "**/*metrics*.txt",
        "**/*summary*.txt",
        "**/*results*.txt",
        "**/*eval*.txt",
        "**/run_info.txt",
    ]
    files: List[Path] = []
    for pat in patterns:
        files.extend(run_dir.glob(pat))
    # Deduplicate while preserving order.
    seen = set()
    uniq = []
    for f in files:
        if f in seen or not f.is_file():
            continue
        seen.add(f)
        uniq.append(f)
    return uniq


def collect_one_run(run_dir: Path) -> Dict[str, Any]:
    # Canonical path shape: base/dataset/rank1/psX/nce_Y/temp_Z
    # We intentionally ignore backup dirs like temp_0.20_partial_YYYYmmdd_HHMMSS.
    parts = run_dir.parts
    ps = None
    lnce = None
    temp = None
    dataset_name = None
    rank = None
    for i, p in enumerate(parts):
        if p.startswith("ps") and p[2:].isdigit():
            ps = int(p[2:])
        elif p.startswith("nce_"):
            try:
                lnce = float(p[len("nce_"):])
            except ValueError:
                pass
        elif p.startswith("temp_"):
            m = re.fullmatch(r"temp_(-?\d+(?:\.\d+)?)", p)
            if m:
                temp = float(m.group(1))
        elif re.fullmatch(r"rank\d+", p):
            rank = int(p[len("rank"):])
            if i >= 1:
                dataset_name = parts[i - 1]

    rec: Dict[str, Any] = {
        "run_dir": str(run_dir),
        "dataset": dataset_name,
        "rank": rank,
        "patch_size": ps,
        "lambda_nce": lnce,
        "temperature": temp,
        "done": (run_dir / "DONE").exists(),
    }

    raw_pts, final_pts = parse_dataset_name(dataset_name or "")
    rec["raw_points"] = raw_pts
    rec["num_curve_points"] = final_pts
    if raw_pts and final_pts:
        rec["keep_ratio"] = final_pts / raw_pts
        rec["point_density_multiplier_vs_300"] = final_pts / 300.0
        rec["relative_patch_fraction"] = ps / final_pts if ps else np.nan

    # Start with manifest info if present.
    manifest = Path("manifests_density") / (Path(*run_dir.parts[-5:])) / "run_info.txt"
    info = parse_run_info(manifest)
    rec.update({k: to_float_if_possible(v) for k, v in info.items()})

    # Search metric files inside the checkpoint dir.
    metrics: Dict[str, float] = {}
    metric_files = list_candidate_metric_files(run_dir)
    rec["metric_files_found"] = len(metric_files)
    rec["metric_sources"] = ";".join(str(p.relative_to(run_dir)) for p in metric_files[:20])

    for p in metric_files:
        ext = p.suffix.lower()
        cur: Dict[str, float] = {}
        if ext == ".json":
            cur = extract_metrics_from_json(p)
        elif ext == ".csv":
            cur = extract_metrics_from_csv(p)
        elif ext in {".txt", ".log", ".out", ".err"}:
            cur = extract_metrics_from_txt(p)
        metrics.update(cur)

    rec.update(metrics)
    rec["has_core_metrics"] = all(k in rec for k in [
        "global_first_cos_mean",
        "global_second_abs_cos_mean",
        "global_second_corr_mean",
    ])
    return rec


def scan_runs(base_dir: Path, rank: int) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    rank_pat = f"rank{rank}"
    canonical_temp_re = re.compile(r"temp_-?\d+(?:\.\d+)?$")
    for temp_dir in base_dir.glob(f"*/{rank_pat}/ps*/nce_*/temp_*"):
        if not temp_dir.is_dir():
            continue
        if not canonical_temp_re.fullmatch(temp_dir.name):
            # Skip backup/partial directories such as temp_0.20_partial_20260406_134151.
            continue
        rec = collect_one_run(temp_dir)
        if rec.get("patch_size") is None or rec.get("lambda_nce") is None or rec.get("temperature") is None:
            continue
        rows.append(rec)
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df = df.sort_values(["num_curve_points", "patch_size", "lambda_nce", "temperature"], na_position="last")
    return df.reset_index(drop=True)


def add_derived_scores(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "global_second_scale_ratio_mean" in out.columns:
        out["second_scale_closeness"] = 1.0 - (out["global_second_scale_ratio_mean"] - 1.0).abs()
    else:
        out["second_scale_closeness"] = np.nan

    for col in [
        "global_first_cos_mean",
        "global_second_abs_cos_mean",
        "global_second_corr_mean",
        "global_second_norm_corr_mean",
        "second_scale_closeness",
    ]:
        if col not in out.columns:
            out[col] = np.nan

    out["composite_score"] = (
        0.20 * out["global_first_cos_mean"].astype(float) +
        0.30 * out["global_second_abs_cos_mean"].astype(float) +
        0.30 * out["global_second_corr_mean"].astype(float) +
        0.10 * out["global_second_norm_corr_mean"].astype(float) +
        0.10 * out["second_scale_closeness"].astype(float)
    )
    return out


def save_heatmap(df: pd.DataFrame, value_col: str, out_path: Path, title: str, fmt: str = ".3f") -> None:
    if value_col not in df.columns:
        return
    piv = df.pivot_table(index="patch_size", columns="num_curve_points", values=value_col, aggfunc="mean")
    if piv.empty:
        return
    fig, ax = plt.subplots(figsize=(10, 6))
    arr = piv.values.astype(float)
    im = ax.imshow(arr, aspect="auto")
    ax.set_xticks(range(len(piv.columns)))
    ax.set_xticklabels([str(c) for c in piv.columns])
    ax.set_yticks(range(len(piv.index)))
    ax.set_yticklabels([str(i) for i in piv.index])
    ax.set_xlabel("Final sampled points")
    ax.set_ylabel("Patch size")
    ax.set_title(title)
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            v = arr[i, j]
            if np.isfinite(v):
                ax.text(j, i, format(v, fmt), ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def save_line_by_density(df: pd.DataFrame, value_col: str, out_path: Path, title: str) -> None:
    if value_col not in df.columns:
        return
    tmp = df.dropna(subset=["num_curve_points", "patch_size", value_col]).copy()
    if tmp.empty:
        return
    fig, ax = plt.subplots(figsize=(10, 6))
    for density, sub in sorted(tmp.groupby("num_curve_points"), key=lambda x: x[0]):
        sub = sub.sort_values("patch_size")
        ax.plot(sub["patch_size"], sub[value_col], marker="o", label=f"N={int(density)}")
    ax.set_xlabel("Patch size")
    ax.set_ylabel(value_col)
    ax.set_title(title)
    ax.legend(ncol=2, fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def save_line_by_patch(df: pd.DataFrame, value_col: str, out_path: Path, title: str) -> None:
    if value_col not in df.columns:
        return
    tmp = df.dropna(subset=["num_curve_points", "patch_size", value_col]).copy()
    if tmp.empty:
        return
    fig, ax = plt.subplots(figsize=(10, 6))
    for ps, sub in sorted(tmp.groupby("patch_size"), key=lambda x: x[0]):
        sub = sub.sort_values("num_curve_points")
        ax.plot(sub["num_curve_points"], sub[value_col], marker="o", label=f"PS={int(ps)}")
    ax.set_xlabel("Final sampled points")
    ax.set_ylabel(value_col)
    ax.set_title(title)
    ax.legend(ncol=2, fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def save_relative_patch_plot(df: pd.DataFrame, value_col: str, out_path: Path, title: str) -> None:
    if value_col not in df.columns or "relative_patch_fraction" not in df.columns:
        return
    tmp = df.dropna(subset=["relative_patch_fraction", value_col]).copy()
    if tmp.empty:
        return
    fig, ax = plt.subplots(figsize=(10, 6))
    for density, sub in sorted(tmp.groupby("num_curve_points"), key=lambda x: x[0]):
        sub = sub.sort_values("relative_patch_fraction")
        ax.plot(sub["relative_patch_fraction"], sub[value_col], marker="o", label=f"N={int(density)}")
    ax.set_xlabel("Relative patch fraction = patch_size / num_curve_points")
    ax.set_ylabel(value_col)
    ax.set_title(title)
    ax.legend(ncol=2, fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def save_scatter(df: pd.DataFrame, x: str, y: str, out_path: Path, title: str) -> None:
    if x not in df.columns or y not in df.columns:
        return
    tmp = df.dropna(subset=[x, y, "num_curve_points", "patch_size"]).copy()
    if tmp.empty:
        return
    fig, ax = plt.subplots(figsize=(8, 6))
    sc = ax.scatter(tmp[x], tmp[y], c=tmp["num_curve_points"], s=35 + 2 * tmp["patch_size"], alpha=0.8)
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    cb = fig.colorbar(sc, ax=ax)
    cb.set_label("Final sampled points")
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def write_report(df: pd.DataFrame, out_path: Path) -> None:
    lines: List[str] = []
    lines.append("Density x Patch Rank Analysis")
    lines.append("=" * 40)
    lines.append(f"Runs discovered: {len(df)}")
    if len(df) == 0:
        out_path.write_text("\n".join(lines))
        return

    lines.append(f"Datasets: {sorted(df['dataset'].dropna().unique().tolist())}")
    lines.append(f"Patch sizes: {sorted(df['patch_size'].dropna().astype(int).unique().tolist())}")
    lines.append(f"Final sampled points: {sorted(df['num_curve_points'].dropna().astype(int).unique().tolist())}")
    lines.append(f"Done runs: {int(df['done'].fillna(False).sum())}")
    lines.append(f"Runs with core metrics: {int(df['has_core_metrics'].fillna(False).sum())}")
    lines.append("")

    usable = df[df["has_core_metrics"].fillna(False)].copy()
    if usable.empty:
        lines.append("No runs with core metrics were found inside checkpoint directories.")
        lines.append("This usually means training completed but per-run evaluation artifacts were not written yet.")
        out_path.write_text("\n".join(lines))
        return

    metrics = [
        "global_first_cos_mean",
        "global_second_abs_cos_mean",
        "global_second_corr_mean",
        "global_second_norm_corr_mean",
        "global_second_scale_ratio_mean",
        "global_first_mse_mean",
        "global_second_mse_mean",
        "best_epoch",
        "composite_score",
    ]

    lines.append("Metric ranges across usable runs:")
    for m in metrics:
        if m in usable.columns:
            s = pd.to_numeric(usable[m], errors="coerce").dropna()
            if len(s) > 0:
                lines.append(f"  {m}: min={s.min():.4f}  mean={s.mean():.4f}  max={s.max():.4f}")
    lines.append("")

    def best_by(group_col: str, score_col: str = "composite_score") -> List[str]:
        rows: List[str] = []
        if score_col not in usable.columns:
            return rows
        for g, sub in usable.groupby(group_col):
            sub = sub.sort_values(score_col, ascending=False)
            r = sub.iloc[0]
            rows.append(
                f"  {group_col}={g}: dataset={r['dataset']} ps={int(r['patch_size'])} "
                f"nce={r['lambda_nce']} temp={r['temperature']} score={r[score_col]:.4f} "
                f"first_cos={r['global_first_cos_mean']:.4f} second_abs_cos={r['global_second_abs_cos_mean']:.4f} "
                f"second_corr={r['global_second_corr_mean']:.4f}"
            )
        return rows

    lines.append("Best run per density by composite score:")
    lines.extend(best_by("num_curve_points"))
    lines.append("")
    lines.append("Best run per patch size by composite score:")
    lines.extend(best_by("patch_size"))
    lines.append("")
    out_path.write_text("\n".join(lines))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-dir", type=Path, default=Path("checkpoints_density"))
    ap.add_argument("--rank", type=int, default=1)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--require-done", action="store_true", help="Only analyze runs that have a DONE marker.")
    args = ap.parse_args()

    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    df = scan_runs(args.base_dir, args.rank)
    if df.empty:
        raise SystemExit(f"No rank{args.rank} runs found under {args.base_dir}")

    if args.require_done:
        df = df[df["done"].fillna(False)].copy()

    df = add_derived_scores(df)

    df.to_csv(out_dir / "all_discovered_runs.csv", index=False)

    coverage_cols = [
        "dataset", "patch_size", "num_curve_points", "lambda_nce", "temperature",
        "done", "has_core_metrics", "metric_files_found",
    ]
    df[coverage_cols].sort_values(["num_curve_points", "patch_size"]).to_csv(out_dir / "coverage.csv", index=False)

    usable = df[df["has_core_metrics"].fillna(False)].copy()
    usable.to_csv(out_dir / "rank_summary.csv", index=False)

    if not usable.empty:
        by_density = usable.sort_values("composite_score", ascending=False).groupby("num_curve_points").head(1)
        by_patch = usable.sort_values("composite_score", ascending=False).groupby("patch_size").head(1)
        by_density.to_csv(out_dir / "best_per_density.csv", index=False)
        by_patch.to_csv(out_dir / "best_per_patch.csv", index=False)

        heatmap_specs = [
            ("global_first_cos_mean", "First derivative cosine heatmap"),
            ("global_second_abs_cos_mean", "Second derivative |cos| heatmap"),
            ("global_second_corr_mean", "Second derivative vector correlation heatmap"),
            ("global_second_norm_corr_mean", "Second derivative norm correlation heatmap"),
            ("second_scale_closeness", "Second derivative scale closeness heatmap"),
            ("global_first_mse_mean", "First derivative MSE heatmap"),
            ("global_second_mse_mean", "Second derivative MSE heatmap"),
            ("best_epoch", "Best epoch heatmap"),
            ("composite_score", "Composite score heatmap"),
        ]
        for col, title in heatmap_specs:
            save_heatmap(usable, col, out_dir / f"heatmap_{col}.png", title)

        for col in [
            "global_first_cos_mean",
            "global_second_abs_cos_mean",
            "global_second_corr_mean",
            "global_second_norm_corr_mean",
            "second_scale_closeness",
            "global_first_mse_mean",
            "global_second_mse_mean",
            "best_epoch",
        ]:
            save_line_by_density(usable, col, out_dir / f"line_patch_effect_{col}.png", f"Patch-size effect on {col}")
            save_line_by_patch(usable, col, out_dir / f"line_density_effect_{col}.png", f"Density effect on {col}")

        for col in [
            "global_first_cos_mean",
            "global_second_abs_cos_mean",
            "global_second_corr_mean",
            "global_second_norm_corr_mean",
            "second_scale_closeness",
        ]:
            save_relative_patch_plot(usable, col, out_dir / f"relative_patch_fraction_{col}.png", f"Relative patch fraction vs {col}")

        save_scatter(
            usable,
            "global_second_abs_cos_mean",
            "global_second_corr_mean",
            out_dir / "scatter_second_abs_cos_vs_second_corr.png",
            "Second-order |cos| vs vector correlation",
        )
        save_scatter(
            usable,
            "global_first_cos_mean",
            "global_second_corr_mean",
            out_dir / "scatter_first_cos_vs_second_corr.png",
            "First-order cosine vs second-order vector correlation",
        )
        save_scatter(
            usable,
            "relative_patch_fraction",
            "global_second_corr_mean",
            out_dir / "scatter_relative_fraction_vs_second_corr.png",
            "Relative patch fraction vs second-order vector correlation",
        )

    write_report(df, out_dir / "report.txt")
    print(f"Saved analysis to: {out_dir}")


if __name__ == "__main__":
    main()
