#!/usr/bin/env python3
"""Comprehensive analysis/visualization for rank-1 density x patch-size sweeps.

Designed to consume a CSV summary with one row per evaluated run. It expects at least
these columns (matching the uploaded 500->300 grid summaries):

- run_dir
- patch_size
- lambda_nce
- temperature
- global_first_cos_mean
- global_first_mse
- global_second_cos_mean
- global_second_abs_cos_mean
- global_second_corr_mean
- global_second_scale_ratio_mean
- global_second_mse
- conv_epoch_90 / conv_epoch_95 / conv_epoch_99 (optional but recommended)

It filters to rank1 runs by default (run_dir containing '/rank1/').
Outputs:
- cleaned rank1 CSV
- coverage tables
- best-per-density and best-per-patch tables
- a compact report.txt
- a panel of readable plots / heatmaps

Usage example:
python /mnt/data/analyze_density_patch_rank1.py \
  --input-csv results_density/rank1_summary.csv \
  --output-dir analysis_density_rank1
"""
from __future__ import annotations

import argparse
import math
import os
import re
from pathlib import Path
from typing import Iterable, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# -----------------------------
# Parsing helpers
# -----------------------------
DATASET_RE = re.compile(r"data_complex_f20_(\d+)to(\d+)")


def parse_dataset_points(text: str) -> tuple[int | None, int | None]:
    if not isinstance(text, str):
        return None, None
    m = DATASET_RE.search(text)
    if not m:
        return None, None
    return int(m.group(1)), int(m.group(2))


def geometric_density_label(raw_n: int, final_n: int) -> str:
    return f"{raw_n}->{final_n}"


def safe_col(df: pd.DataFrame, name: str, default=np.nan):
    if name not in df.columns:
        df[name] = default
    return df[name]


# -----------------------------
# Derived metrics
# -----------------------------

def add_derived_columns(df: pd.DataFrame) -> pd.DataFrame:
    raw_points: List[int | None] = []
    final_points: List[int | None] = []
    density_labels: List[str | None] = []
    keep_ratios: List[float | None] = []
    patch_fracs: List[float | None] = []
    patch_frac_raws: List[float | None] = []
    approx_stride: List[float | None] = []

    for _, row in df.iterrows():
        text = row.get("bank_path", None)
        raw_n, final_n = parse_dataset_points(text)
        if raw_n is None or final_n is None:
            text = row.get("run_dir", None)
            raw_n, final_n = parse_dataset_points(text)

        raw_points.append(raw_n)
        final_points.append(final_n)
        density_labels.append(geometric_density_label(raw_n, final_n) if raw_n and final_n else None)
        keep_ratios.append((final_n / raw_n) if raw_n and final_n else np.nan)
        ps = row.get("patch_size", np.nan)
        patch_fracs.append((ps / final_n) if final_n and pd.notna(ps) else np.nan)
        patch_frac_raws.append((ps / raw_n) if raw_n and pd.notna(ps) else np.nan)
        approx_stride.append((raw_n / final_n) if raw_n and final_n else np.nan)

    df = df.copy()
    df["raw_points"] = raw_points
    df["final_points"] = final_points
    df["density_label"] = density_labels
    df["keep_ratio"] = keep_ratios
    df["patch_fraction_final"] = patch_fracs
    df["patch_fraction_raw"] = patch_frac_raws
    df["approx_stride"] = approx_stride

    # Stabilized scale-ratio closeness: 1 is best, decays symmetrically in log space.
    # e.g. ratio=1 => 1.0 ; ratio=2 or 0.5 => 0.5 ; ratio=4 or 0.25 => 0.25
    ratio = pd.to_numeric(df.get("global_second_scale_ratio_mean", np.nan), errors="coerce")
    with np.errstate(divide="ignore", invalid="ignore"):
        log2_abs = np.abs(np.log2(ratio))
        df["global_second_scale_closeness"] = 1.0 / (1.0 + log2_abs)

    # A compact composite for ranking/annotation only, not as the main scientific result.
    # Emphasize curvature structure and first-derivative correctness, lightly penalize MSE.
    first_cos = pd.to_numeric(df.get("global_first_cos_mean", np.nan), errors="coerce")
    second_abs = pd.to_numeric(df.get("global_second_abs_cos_mean", np.nan), errors="coerce")
    second_corr = pd.to_numeric(df.get("global_second_corr_mean", np.nan), errors="coerce")
    scale_close = pd.to_numeric(df.get("global_second_scale_closeness", np.nan), errors="coerce")
    first_mse = pd.to_numeric(df.get("global_first_mse", np.nan), errors="coerce")
    second_mse = pd.to_numeric(df.get("global_second_mse", np.nan), errors="coerce")

    df["analysis_score"] = (
        0.30 * first_cos
        + 0.30 * second_abs
        + 0.25 * second_corr
        + 0.15 * scale_close
        - 0.03 * np.log10(np.maximum(first_mse, 1e-12))
        - 0.04 * np.log10(np.maximum(second_mse, 1e-12))
    )
    return df


# -----------------------------
# Plot helpers
# -----------------------------

def save_fig(fig: plt.Figure, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def ordered_density_levels(df: pd.DataFrame) -> list[str]:
    levels = (
        df[["density_label", "final_points"]]
        .dropna()
        .drop_duplicates()
        .sort_values("final_points")
    )
    return levels["density_label"].tolist()


def ordered_patch_levels(df: pd.DataFrame) -> list[int]:
    return sorted(pd.to_numeric(df["patch_size"], errors="coerce").dropna().astype(int).unique().tolist())


def make_heatmap(df: pd.DataFrame, value_col: str, out_path: Path, title: str, fmt: str = ".3f"):
    density_order = ordered_density_levels(df)
    patch_order = ordered_patch_levels(df)
    piv = df.pivot_table(index="density_label", columns="patch_size", values=value_col, aggfunc="mean")
    piv = piv.reindex(index=density_order, columns=patch_order)

    fig, ax = plt.subplots(figsize=(1.3 * len(patch_order) + 3, 0.8 * len(density_order) + 2.2))
    im = ax.imshow(piv.values, aspect="auto")
    ax.set_title(title)
    ax.set_xlabel("Patch size")
    ax.set_ylabel("Sampling density")
    ax.set_xticks(range(len(patch_order)))
    ax.set_xticklabels(patch_order)
    ax.set_yticks(range(len(density_order)))
    ax.set_yticklabels(density_order)
    cbar = fig.colorbar(im, ax=ax)
    cbar.ax.set_ylabel(value_col)

    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            val = piv.iat[i, j]
            if pd.notna(val):
                ax.text(j, i, format(val, fmt), ha="center", va="center", fontsize=8)

    save_fig(fig, out_path)


def lineplot_by_density(df: pd.DataFrame, y_col: str, out_path: Path, title: str, ylabel: str | None = None):
    density_order = ordered_density_levels(df)
    patch_order = ordered_patch_levels(df)

    fig, ax = plt.subplots(figsize=(8.2, 5.2))
    for density in density_order:
        sub = df[df["density_label"] == density].sort_values("patch_size")
        ax.plot(sub["patch_size"], sub[y_col], marker="o", label=density)

    ax.set_title(title)
    ax.set_xlabel("Patch size")
    ax.set_ylabel(ylabel or y_col)
    ax.set_xticks(patch_order)
    ax.grid(True, alpha=0.3)
    ax.legend(title="Density", fontsize=8)
    save_fig(fig, out_path)


def lineplot_by_patch(df: pd.DataFrame, y_col: str, out_path: Path, title: str, ylabel: str | None = None):
    density_meta = (
        df[["density_label", "final_points"]]
        .dropna()
        .drop_duplicates()
        .sort_values("final_points")
    )
    density_order = density_meta["density_label"].tolist()
    xvals = density_meta["final_points"].tolist()
    patch_order = ordered_patch_levels(df)

    fig, ax = plt.subplots(figsize=(8.2, 5.2))
    for ps in patch_order:
        sub = df[df["patch_size"] == ps].copy()
        sub = sub.merge(density_meta, on=["density_label", "final_points"], how="inner")
        sub = sub.sort_values("final_points")
        ax.plot(sub["final_points"], sub[y_col], marker="o", label=f"ps={ps}")

    ax.set_title(title)
    ax.set_xlabel("Final sampled points")
    ax.set_ylabel(ylabel or y_col)
    ax.set_xscale("log")
    ax.set_xticks(xvals)
    ax.set_xticklabels(density_order, rotation=0)
    ax.grid(True, alpha=0.3)
    ax.legend(ncol=2, fontsize=8)
    save_fig(fig, out_path)


def scatter_two_metrics(df: pd.DataFrame, x_col: str, y_col: str, out_path: Path, title: str):
    fig, ax = plt.subplots(figsize=(7.2, 5.6))
    density_order = ordered_density_levels(df)
    marker_cycle = ["o", "s", "^", "D", "P", "X", "v", "<", ">"]
    density_to_marker = {d: marker_cycle[i % len(marker_cycle)] for i, d in enumerate(density_order)}

    for ps in ordered_patch_levels(df):
        sub_ps = df[df["patch_size"] == ps]
        for density in density_order:
            sub = sub_ps[sub_ps["density_label"] == density]
            if sub.empty:
                continue
            ax.scatter(
                sub[x_col],
                sub[y_col],
                marker=density_to_marker[density],
                s=70,
                alpha=0.9,
                label=None,
            )
            for _, r in sub.iterrows():
                ax.annotate(f"ps{int(r['patch_size'])}\n{density}", (r[x_col], r[y_col]), fontsize=7, alpha=0.75)

    ax.set_title(title)
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.grid(True, alpha=0.3)
    save_fig(fig, out_path)


def bar_best_per_density(best_df: pd.DataFrame, out_path: Path):
    fig, ax = plt.subplots(figsize=(8, 4.8))
    x = np.arange(len(best_df))
    ax.bar(x, best_df["analysis_score"])
    ax.set_xticks(x)
    ax.set_xticklabels([f"{d}\nps={p}" for d, p in zip(best_df["density_label"], best_df["patch_size"])])
    ax.set_ylabel("analysis_score")
    ax.set_title("Best patch size per density (rank-1 runs)")
    ax.grid(True, axis="y", alpha=0.3)
    save_fig(fig, out_path)


def write_report(df: pd.DataFrame, out_dir: Path):
    density_order = ordered_density_levels(df)
    patch_order = ordered_patch_levels(df)

    best_per_density = df.sort_values("analysis_score", ascending=False).groupby("density_label", as_index=False).first()
    best_per_patch = df.sort_values("analysis_score", ascending=False).groupby("patch_size", as_index=False).first()

    lines: list[str] = []
    lines.append("Rank-1 density x patch-size analysis")
    lines.append("=================================")
    lines.append("")
    lines.append(f"Rows analyzed: {len(df)}")
    lines.append(f"Densities: {', '.join(density_order)}")
    lines.append(f"Patch sizes: {', '.join(map(str, patch_order))}")
    lines.append("")

    lines.append("Best patch per density (by analysis_score)")
    lines.append("------------------------------------------")
    for _, r in best_per_density.sort_values("final_points").iterrows():
        lines.append(
            f"{r['density_label']}: ps={int(r['patch_size'])}, "
            f"first_cos={r['global_first_cos_mean']:.4f}, "
            f"second_abs_cos={r['global_second_abs_cos_mean']:.4f}, "
            f"second_corr={r['global_second_corr_mean']:.4f}, "
            f"scale_ratio={r['global_second_scale_ratio_mean']:.4f}, "
            f"score={r['analysis_score']:.4f}"
        )
    lines.append("")

    lines.append("Best density per patch (by analysis_score)")
    lines.append("-----------------------------------------")
    for _, r in best_per_patch.sort_values("patch_size").iterrows():
        lines.append(
            f"ps={int(r['patch_size'])}: {r['density_label']}, "
            f"first_cos={r['global_first_cos_mean']:.4f}, "
            f"second_abs_cos={r['global_second_abs_cos_mean']:.4f}, "
            f"second_corr={r['global_second_corr_mean']:.4f}, "
            f"scale_ratio={r['global_second_scale_ratio_mean']:.4f}, "
            f"score={r['analysis_score']:.4f}"
        )
    lines.append("")

    # Trend summaries via simple correlations, only as descriptive guides.
    tmp = df.dropna(subset=["patch_size", "final_points", "global_second_abs_cos_mean", "global_second_corr_mean", "global_first_cos_mean"]).copy()
    if len(tmp) >= 3:
        lines.append("Descriptive correlations")
        lines.append("------------------------")
        lines.append(
            f"corr(patch_size, first_cos) = {tmp['patch_size'].corr(tmp['global_first_cos_mean']):.4f}"
        )
        lines.append(
            f"corr(patch_size, second_abs_cos) = {tmp['patch_size'].corr(tmp['global_second_abs_cos_mean']):.4f}"
        )
        lines.append(
            f"corr(patch_size, second_corr) = {tmp['patch_size'].corr(tmp['global_second_corr_mean']):.4f}"
        )
        lines.append(
            f"corr(final_points, first_cos) = {tmp['final_points'].corr(tmp['global_first_cos_mean']):.4f}"
        )
        lines.append(
            f"corr(final_points, second_abs_cos) = {tmp['final_points'].corr(tmp['global_second_abs_cos_mean']):.4f}"
        )
        lines.append(
            f"corr(final_points, second_corr) = {tmp['final_points'].corr(tmp['global_second_corr_mean']):.4f}"
        )
        lines.append("")

    (out_dir / "report.txt").write_text("\n".join(lines))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input-csv", required=True, help="CSV with one row per evaluated run.")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--rank", default="rank1", help="Filter run_dir by this substring. Default: rank1")
    p.add_argument("--min-first-cos", type=float, default=-1.0, help="Optional filter to drop pathological runs.")
    p.add_argument("--min-second-abs-cos", type=float, default=-1.0, help="Optional filter to drop pathological runs.")
    args = p.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.input_csv)
    if "run_dir" not in df.columns:
        raise ValueError("input CSV must contain 'run_dir'")

    # rank filter
    df = df[df["run_dir"].astype(str).str.contains(fr"/{re.escape(args.rank)}/", regex=True)].copy()
    if df.empty:
        raise ValueError(f"No rows matched rank filter '{args.rank}'")

    # numeric conversions for core metrics
    numeric_cols = [
        "patch_size", "lambda_nce", "temperature",
        "global_first_cos_mean", "global_first_mse",
        "global_second_cos_mean", "global_second_abs_cos_mean",
        "global_second_corr_mean", "global_second_scale_ratio_mean",
        "global_second_mse", "conv_epoch_90", "conv_epoch_95", "conv_epoch_99",
    ]
    for col in numeric_cols:
        safe_col(df, col)
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = add_derived_columns(df)
    df = df[
        (df["global_first_cos_mean"] >= args.min_first_cos)
        & (df["global_second_abs_cos_mean"] >= args.min_second_abs_cos)
    ].copy()

    # Keep one row per (density, patch, lambda_nce, temperature) if duplicates appear.
    # Use best analysis_score among duplicates.
    dedup_keys = ["density_label", "patch_size", "lambda_nce", "temperature", "run_dir"]
    dedup_keys = [k for k in dedup_keys if k in df.columns]
    df = df.sort_values("analysis_score", ascending=False).drop_duplicates(subset=dedup_keys, keep="first")

    # Save cleaned analysis table.
    df.sort_values(["final_points", "patch_size"]).to_csv(out_dir / "rank1_cleaned.csv", index=False)

    # Coverage / pivot tables.
    coverage = (
        df.groupby(["density_label", "patch_size"], as_index=False)
        .size()
        .rename(columns={"size": "num_runs"})
        .sort_values(["density_label", "patch_size"])
    )
    coverage.to_csv(out_dir / "coverage.csv", index=False)

    metric_cols = [
        ("global_first_cos_mean", "first_cos_heatmap.png", "First derivative cosine mean"),
        ("global_second_abs_cos_mean", "second_abs_cos_heatmap.png", "Second derivative |cos| mean"),
        ("global_second_corr_mean", "second_vector_corr_heatmap.png", "Second derivative vector correlation mean"),
        ("global_second_scale_closeness", "second_scale_closeness_heatmap.png", "Second derivative scale closeness"),
        ("global_first_mse", "first_mse_heatmap.png", "First derivative MSE"),
        ("global_second_mse", "second_mse_heatmap.png", "Second derivative MSE"),
        ("conv_epoch_90", "conv90_heatmap.png", "Convergence epoch 90%"),
        ("analysis_score", "analysis_score_heatmap.png", "Composite analysis score"),
    ]
    for col, name, title in metric_cols:
        if col in df.columns:
            make_heatmap(df, col, out_dir / name, title)

    # Lines by density (patch effect).
    lineplot_by_density(df, "global_first_cos_mean", out_dir / "line_patch_effect_first_cos.png", "Patch-size effect on first-derivative cosine")
    lineplot_by_density(df, "global_second_abs_cos_mean", out_dir / "line_patch_effect_second_abs_cos.png", "Patch-size effect on second-derivative |cos|")
    lineplot_by_density(df, "global_second_corr_mean", out_dir / "line_patch_effect_second_corr.png", "Patch-size effect on second-derivative vector correlation")
    lineplot_by_density(df, "global_second_scale_closeness", out_dir / "line_patch_effect_scale_closeness.png", "Patch-size effect on second-derivative scale closeness")
    if "conv_epoch_90" in df.columns:
        lineplot_by_density(df, "conv_epoch_90", out_dir / "line_patch_effect_conv90.png", "Patch-size effect on convergence speed", ylabel="conv_epoch_90")

    # Lines by patch (density effect).
    lineplot_by_patch(df, "global_first_cos_mean", out_dir / "line_density_effect_first_cos.png", "Density effect on first-derivative cosine")
    lineplot_by_patch(df, "global_second_abs_cos_mean", out_dir / "line_density_effect_second_abs_cos.png", "Density effect on second-derivative |cos|")
    lineplot_by_patch(df, "global_second_corr_mean", out_dir / "line_density_effect_second_corr.png", "Density effect on second-derivative vector correlation")
    lineplot_by_patch(df, "global_second_scale_closeness", out_dir / "line_density_effect_scale_closeness.png", "Density effect on second-derivative scale closeness")
    if "conv_epoch_90" in df.columns:
        lineplot_by_patch(df, "conv_epoch_90", out_dir / "line_density_effect_conv90.png", "Density effect on convergence speed", ylabel="conv_epoch_90")

    # Relative-support plots: this makes density-vs-patch interpretation clearer.
    fig, ax = plt.subplots(figsize=(8.2, 5.2))
    for density in ordered_density_levels(df):
        sub = df[df["density_label"] == density].sort_values("patch_fraction_final")
        ax.plot(sub["patch_fraction_final"], sub["global_second_corr_mean"], marker="o", label=density)
    ax.set_title("Second-derivative vector correlation vs relative patch size")
    ax.set_xlabel("patch_size / final_points")
    ax.set_ylabel("global_second_corr_mean")
    ax.grid(True, alpha=0.3)
    ax.legend(title="Density", fontsize=8)
    save_fig(fig, out_dir / "relative_patch_fraction_vs_second_corr.png")

    fig, ax = plt.subplots(figsize=(8.2, 5.2))
    for density in ordered_density_levels(df):
        sub = df[df["density_label"] == density].sort_values("patch_fraction_final")
        ax.plot(sub["patch_fraction_final"], sub["global_second_abs_cos_mean"], marker="o", label=density)
    ax.set_title("Second-derivative |cos| vs relative patch size")
    ax.set_xlabel("patch_size / final_points")
    ax.set_ylabel("global_second_abs_cos_mean")
    ax.grid(True, alpha=0.3)
    ax.legend(title="Density", fontsize=8)
    save_fig(fig, out_dir / "relative_patch_fraction_vs_second_abs_cos.png")

    # Cross-metric scatterplots.
    scatter_two_metrics(df, "global_first_cos_mean", "global_second_abs_cos_mean", out_dir / "scatter_first_vs_second_abs_cos.png", "First- vs second-order directional quality")
    scatter_two_metrics(df, "global_second_abs_cos_mean", "global_second_corr_mean", out_dir / "scatter_second_abs_cos_vs_corr.png", "Second-order direction vs vector correlation")
    scatter_two_metrics(df, "global_second_corr_mean", "global_second_scale_closeness", out_dir / "scatter_second_corr_vs_scale.png", "Second-order vector correlation vs scale closeness")

    # Best tables.
    best_per_density = df.sort_values("analysis_score", ascending=False).groupby("density_label", as_index=False).first()
    best_per_patch = df.sort_values("analysis_score", ascending=False).groupby("patch_size", as_index=False).first()
    best_per_density.sort_values("final_points").to_csv(out_dir / "best_per_density.csv", index=False)
    best_per_patch.sort_values("patch_size").to_csv(out_dir / "best_per_patch.csv", index=False)
    bar_best_per_density(best_per_density.sort_values("final_points"), out_dir / "best_per_density_scorebar.png")

    # Pairwise delta tables: useful for later interpretation.
    rows = []
    density_order = ordered_density_levels(df)
    patch_order = ordered_patch_levels(df)
    metric = "global_second_corr_mean"
    piv = df.pivot_table(index="density_label", columns="patch_size", values=metric, aggfunc="mean")
    piv = piv.reindex(index=density_order, columns=patch_order)
    for density in density_order:
        vals = piv.loc[density]
        for a, b in zip(patch_order[:-1], patch_order[1:]):
            va, vb = vals[a], vals[b]
            if pd.notna(va) and pd.notna(vb):
                rows.append({
                    "density_label": density,
                    "from_patch": a,
                    "to_patch": b,
                    f"delta_{metric}": vb - va,
                })
    pd.DataFrame(rows).to_csv(out_dir / "adjacent_patch_deltas_second_corr.csv", index=False)

    write_report(df, out_dir)
    print(f"Wrote analysis to: {out_dir}")


if __name__ == "__main__":
    main()
