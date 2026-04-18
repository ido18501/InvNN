#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def parse_ratio_tag(tag: str) -> float:
    # rel_0p03 -> 0.03
    if tag.startswith("rel_"):
        tag = tag[len("rel_"):]
    return float(tag.replace("p", "."))


def infer_bank_path(banks_root: Path, dataset_name: str, split: str) -> Path:
    candidates = [
        banks_root / dataset_name / f"{split}.npz",
        banks_root / f"{dataset_name}.npz",
        banks_root / dataset_name / "bank.npz",
        banks_root / dataset_name,
    ]
    for c in candidates:
        if c.is_file():
            return c
    raise FileNotFoundError(f"Could not infer bank path for {dataset_name} split={split} under {banks_root}")


def find_completed_runs(checkpoints_root: Path) -> list[dict[str, Any]]:
    runs: list[dict[str, Any]] = []
    for dataset_dir in sorted(p for p in checkpoints_root.iterdir() if p.is_dir()):
        dataset_name = dataset_dir.name
        for rel_dir in sorted(p for p in dataset_dir.iterdir() if p.is_dir() and p.name.startswith("rel_")):
            ratio_tag = rel_dir.name
            ratio_value = parse_ratio_tag(ratio_tag)
            for ps_dir in sorted(p for p in rel_dir.iterdir() if p.is_dir() and p.name.startswith("ps")):
                try:
                    patch_size = int(ps_dir.name[2:])
                except Exception:
                    continue
                for nce_dir in sorted(p for p in ps_dir.iterdir() if p.is_dir() and p.name.startswith("nce_")):
                    for temp_dir in sorted(p for p in nce_dir.iterdir() if p.is_dir() and p.name.startswith("temp_")):
                        if (temp_dir / "DONE").exists() and (temp_dir / "best_model.pt").is_file() and (temp_dir / "config.json").is_file():
                            runs.append({
                                "dataset_name": dataset_name,
                                "ratio_tag": ratio_tag,
                                "target_ratio": ratio_value,
                                "patch_size": patch_size,
                                "run_dir": temp_dir,
                                "checkpoint": temp_dir / "best_model.pt",
                                "config": temp_dir / "config.json",
                            })
    return runs


def compute_norm_correlation_from_arrays(curves_dir: Path, pair: str) -> dict[str, float]:
    pred_norms = []
    gt_norms = []
    pred_key = f"{pair}__pred"
    gt_key = f"{pair}__gt"
    for curve_dir in sorted(p for p in curves_dir.iterdir() if p.is_dir() and p.name.startswith("curve_")):
        arr_path = curve_dir / "arrays.npz"
        if not arr_path.is_file():
            continue
        data = np.load(arr_path)
        if pred_key not in data or gt_key not in data:
            continue
        pred = np.asarray(data[pred_key], dtype=np.float64)
        gt = np.asarray(data[gt_key], dtype=np.float64)
        pred_norms.append(np.linalg.norm(pred, axis=-1))
        gt_norms.append(np.linalg.norm(gt, axis=-1))
    if not pred_norms:
        return {"pearson": float("nan"), "spearman": float("nan")}
    x = np.concatenate(gt_norms)
    y = np.concatenate(pred_norms)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if len(x) < 2 or np.std(x) < 1e-12 or np.std(y) < 1e-12:
        pearson = float("nan")
    else:
        pearson = float(np.corrcoef(x, y)[0, 1])
    # spearman without scipy
    xr = pd.Series(x).rank(method="average").to_numpy(dtype=np.float64)
    yr = pd.Series(y).rank(method="average").to_numpy(dtype=np.float64)
    if len(xr) < 2 or np.std(xr) < 1e-12 or np.std(yr) < 1e-12:
        spearman = float("nan")
    else:
        spearman = float(np.corrcoef(xr, yr)[0, 1])
    return {"pearson": pearson, "spearman": spearman}


def load_summary_row(run_output_dir: Path, meta: dict[str, Any]) -> dict[str, Any]:
    global_dir = run_output_dir / "global"
    summary = json.loads((global_dir / "global_summary.json").read_text())
    per_curve_df = pd.read_csv(global_dir / "per_curve_stats.csv")

    g = summary["global_pointwise"]
    row = {
        "dataset_name": meta["dataset_name"],
        "ratio_tag": meta["ratio_tag"],
        "target_ratio": float(meta["target_ratio"]),
        "patch_size": int(meta["patch_size"]),
        "run_dir": str(meta["run_dir"]),
        "analysis_dir": str(run_output_dir),
        "n_curves_analyzed": int(len(per_curve_df)),
        "n_points_mean": float(per_curve_df["n_points"].mean()),
        "relative_patch_size_mean": float(per_curve_df["relative_patch_size"].mean()),
        "direct1_vs_gt1__abs_cosine_mean": g["direct1_vs_gt1"]["abs_cosine"]["mean"],
        "global1_vs_gt1__abs_cosine_mean": g["global1_vs_gt1"]["abs_cosine"]["mean"],
        "global2_from_global1_vs_gt2__abs_cosine_mean": g["global2_from_global1_vs_gt2"]["abs_cosine"]["mean"],
        "global2_from_direct1_vs_gt2__abs_cosine_mean": g["global2_from_direct1_vs_gt2"]["abs_cosine"]["mean"],
        "global1_vs_direct1__abs_cosine_mean": g["global1_vs_direct1"]["abs_cosine"]["mean"],
        "global2_from_global1_vs_gt2__mse_mean": g["global2_from_global1_vs_gt2"]["mse"]["mean"],
        "global2_from_global1_vs_gt2__norm_ratio_median": g["global2_from_global1_vs_gt2"]["norm_ratio"]["median"],
        "trim99_second_abs_cos_mean": summary["trimmed_by_curvature"]["TRIM99"]["second_derivative"]["abs_cosine"]["mean"],
        "trim95_second_abs_cos_mean": summary["trimmed_by_curvature"]["TRIM95"]["second_derivative"]["abs_cosine"]["mean"],
        "W_fro_norm_mean": summary["matrix_summary"]["W_fro_norm"]["mean"],
        "W2_fro_norm_mean": summary["matrix_summary"]["W2_fro_norm"]["mean"],
        "antisymmetry_error_mean": summary["matrix_summary"]["antisymmetry_error_mean"]["mean"],
    }

    curves_dir = run_output_dir / "curves"
    n2 = compute_norm_correlation_from_arrays(curves_dir, "global2_from_global1_vs_gt2")
    n2d = compute_norm_correlation_from_arrays(curves_dir, "global2_from_direct1_vs_gt2")
    row.update({
        "global2_from_global1_vs_gt2__norm_pearson_pointwise": n2["pearson"],
        "global2_from_global1_vs_gt2__norm_spearman_pointwise": n2["spearman"],
        "global2_from_direct1_vs_gt2__norm_pearson_pointwise": n2d["pearson"],
        "global2_from_direct1_vs_gt2__norm_spearman_pointwise": n2d["spearman"],
    })
    return row


def make_heatmap(df: pd.DataFrame, value_col: str, output_path: Path, title: str) -> None:
    sub = df[["dataset_name", "ratio_tag", value_col]].dropna()
    if sub.empty:
        return
    ratio_order = sorted(sub["ratio_tag"].unique(), key=lambda s: parse_ratio_tag(s))
    pivot = sub.pivot(index="dataset_name", columns="ratio_tag", values=value_col)
    pivot = pivot.reindex(index=sorted(pivot.index), columns=ratio_order)
    plt.figure(figsize=(8, 5))
    im = plt.imshow(pivot.to_numpy(), aspect="auto", interpolation="nearest")
    plt.xticks(range(len(pivot.columns)), pivot.columns)
    plt.yticks(range(len(pivot.index)), pivot.index)
    plt.colorbar(im)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def make_lineplots(df: pd.DataFrame, value_col: str, output_path: Path, title: str) -> None:
    if value_col not in df.columns:
        return
    ratio_values = sorted(df["target_ratio"].unique())
    plt.figure(figsize=(8, 5))
    for dataset_name, sub in df.groupby("dataset_name"):
        sub = sub.sort_values("target_ratio")
        plt.plot(sub["target_ratio"], sub[value_col], marker="o", label=dataset_name)
    plt.xticks(ratio_values, [f"{x:.02f}" for x in ratio_values])
    plt.xlabel("Target relative patch ratio")
    plt.ylabel(value_col)
    plt.title(title)
    plt.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Analyze matched-relative-ratio experiment by wrapping invariant_operator_analysis.py per run.")
    p.add_argument("--checkpoints-root", type=str, default="checkpoints_relratio")
    p.add_argument("--banks-root", type=str, default=".")
    p.add_argument("--output-dir", type=str, required=True)
    p.add_argument("--analysis-split", type=str, default="test")
    p.add_argument("--num-curves", type=int, default=32)
    p.add_argument("--num-curves-visualized", type=int, default=6)
    p.add_argument("--num-points-visualized", type=int, default=128)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--random-seed", type=int, default=123)
    p.add_argument("--force-reanalyze", action="store_true")
    return p


def main() -> None:
    args = build_parser().parse_args()
    checkpoints_root = Path(args.checkpoints_root)
    banks_root = Path(args.banks_root)
    output_dir = ensure_dir(Path(args.output_dir))

    import sys
    sys.path.insert(0, str(Path.cwd()))
    try:
        import invariant_operator_analysis as base
    except Exception as e:
        raise RuntimeError("Could not import invariant_operator_analysis.py from current working directory.") from e

    runs = find_completed_runs(checkpoints_root)
    if not runs:
        raise FileNotFoundError(f"No completed runs found under {checkpoints_root}")

    run_rows: list[dict[str, Any]] = []
    for meta in runs:
        dataset_name = meta["dataset_name"]
        ratio_tag = meta["ratio_tag"]
        patch_size = meta["patch_size"]
        bank_path = infer_bank_path(banks_root, dataset_name, args.analysis_split)
        run_output_dir = output_dir / dataset_name / ratio_tag / f"ps{patch_size}"
        global_summary_path = run_output_dir / "global" / "global_summary.json"

        if args.force_reanalyze or not global_summary_path.is_file():
            print(f"[ANALYZE] {meta['run_dir']}")
            base.run_single_analysis(
                checkpoint=Path(meta["checkpoint"]),
                bank_path=bank_path,
                output_dir=run_output_dir,
                curve_indices=None,
                num_curves=args.num_curves,
                num_curves_visualized=args.num_curves_visualized,
                num_points_visualized=args.num_points_visualized,
                batch_size=args.batch_size,
                device=args.device,
                random_seed=args.random_seed,
                extra_metadata={
                    "dataset_name": dataset_name,
                    "ratio_tag": ratio_tag,
                    "target_ratio": meta["target_ratio"],
                    "patch_size": patch_size,
                    "source_run_dir": str(meta["run_dir"]),
                },
            )
        else:
            print(f"[REUSE] {run_output_dir}")

        row = load_summary_row(run_output_dir, meta)
        run_rows.append(row)

    df = pd.DataFrame(run_rows).sort_values(["dataset_name", "target_ratio", "patch_size"]).reset_index(drop=True)
    df.to_csv(output_dir / "relratio_sweep_summary.csv", index=False)

    ratio_agg = df.groupby(["ratio_tag", "target_ratio"], as_index=False).mean(numeric_only=True).sort_values("target_ratio")
    dataset_agg = df.groupby("dataset_name", as_index=False).mean(numeric_only=True)
    ratio_agg.to_csv(output_dir / "ratio_aggregates.csv", index=False)
    dataset_agg.to_csv(output_dir / "dataset_aggregates.csv", index=False)

    metrics = [
        ("global2_from_global1_vs_gt2__abs_cosine_mean", "Second derivative abs cosine"),
        ("global2_from_global1_vs_gt2__norm_spearman_pointwise", "Second derivative norm Spearman"),
        ("global2_from_global1_vs_gt2__norm_pearson_pointwise", "Second derivative norm Pearson"),
        ("direct1_vs_gt1__abs_cosine_mean", "Direct first abs cosine"),
        ("antisymmetry_error_mean", "Antisymmetry error"),
    ]
    for col, title in metrics:
        if col not in df.columns:
            continue
        make_heatmap(df, col, output_dir / f"heatmap_{col}_by_density_ratio.png", title)
        make_lineplots(df, col, output_dir / f"lineplot_{col}_by_ratio.png", title)

    summary = {
        "num_completed_runs": int(len(df)),
        "datasets_present": sorted(df["dataset_name"].unique().tolist()),
        "ratios_present": sorted(df["ratio_tag"].unique().tolist(), key=parse_ratio_tag),
        "best_by_second_abs_cos": df.sort_values("global2_from_global1_vs_gt2__abs_cosine_mean", ascending=False).head(10).to_dict(orient="records"),
    }
    (output_dir / "relratio_sweep_overview.json").write_text(json.dumps(summary, indent=2))
    print(f"Saved sweep outputs under {output_dir}")


if __name__ == "__main__":
    main()
