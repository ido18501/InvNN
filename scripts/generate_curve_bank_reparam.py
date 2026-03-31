from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from utils.curve_generation import (
    generate_random_reparameterized_fourier_curve,
    generate_random_simple_fourier_curve,
)


@dataclass
class CurveGenConfig:
    num_curves: int
    num_points: int
    max_freq: int
    scale: float
    decay_power: float
    min_size: float
    max_size: float
    max_tries: int
    intersection_check_points: int
    seed: int
    reparametrize_prob: float
    reparam_strength: float
    reparam_num_harmonics: int
    reparam_min_density: float
    reparam_max_density: float


def _generate_one(
    args: tuple[int, CurveGenConfig],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.int32]:
    idx, cfg = args
    rng = np.random.default_rng(cfg.seed + 104729 * idx)

    use_reparam = rng.random() < cfg.reparametrize_prob

    if use_reparam:
        curve_points, coeffs, _, t_warped = generate_random_reparameterized_fourier_curve(
            num_points=cfg.num_points,
            max_freq=cfg.max_freq,
            scale=cfg.scale,
            decay_power=cfg.decay_power,
            rng=rng,
            center=True,
            fit_to_canvas=True,
            min_size=cfg.min_size,
            max_size=cfg.max_size,
            reparam_strength=cfg.reparam_strength,
            reparam_num_harmonics=cfg.reparam_num_harmonics,
            reparam_min_density=cfg.reparam_min_density,
            reparam_max_density=cfg.reparam_max_density,
            max_tries=cfg.max_tries,
            enforce_simple=True,
            intersection_check_points=cfg.intersection_check_points,
        )
        t_grid = np.asarray(t_warped, dtype=np.float32)
        was_reparam = np.int32(1)
    else:
        t_uniform = np.linspace(0.0, 2.0 * np.pi, cfg.num_points, endpoint=False, dtype=np.float64)
        curve_points, coeffs = generate_random_simple_fourier_curve(
            t=t_uniform,
            max_freq=cfg.max_freq,
            scale=cfg.scale,
            decay_power=cfg.decay_power,
            rng=rng,
            max_tries=cfg.max_tries,
            center=True,
            fit_to_canvas=True,
            min_size=cfg.min_size,
            max_size=cfg.max_size,
            enforce_simple=True,
            intersection_check_points=cfg.intersection_check_points,
        )
        t_grid = t_uniform.astype(np.float32)
        was_reparam = np.int32(0)

    return (
        curve_points.astype(np.float32),
        coeffs.x_coeffs.astype(np.float32),
        coeffs.y_coeffs.astype(np.float32),
        t_grid,
        was_reparam,
    )


def _plot_contact_sheet(
    curves: np.ndarray,
    t_grids: np.ndarray,
    reparam_flags: np.ndarray,
    output_path: Path,
    cols: int = 5,
    point_stride: int = 25,
) -> None:
    n = len(curves)
    cols = max(1, cols)
    rows = math.ceil(n / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(3.2 * cols, 3.2 * rows))
    axes = np.array(axes).reshape(rows, cols)

    for ax in axes.flat:
        ax.axis("off")

    for i, (curve, t_grid, is_reparam) in enumerate(zip(curves, t_grids, reparam_flags)):
        ax = axes[i // cols, i % cols]

        # base curve outline
        closed_curve = np.vstack([curve, curve[:1]])
        ax.plot(closed_curve[:, 0], closed_curve[:, 1], lw=1.0, alpha=0.8)

        # show sampling / parameter order with colored points
        idxs = np.arange(0, len(curve), max(1, point_stride))
        pts = curve[idxs]
        colors = np.linspace(0.0, 1.0, len(pts))
        ax.scatter(pts[:, 0], pts[:, 1], c=colors, s=10, cmap="viridis")

        title = f"curve {i}"
        title += " | reparam" if int(is_reparam) == 1 else " | uniform"
        ax.set_title(title, fontsize=10)
        ax.set_aspect("equal")
        ax.axis("off")

    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()

    p.add_argument("--output", type=str, required=True)
    p.add_argument("--num-curves", type=int, required=True)

    p.add_argument("--num-points", type=int, default=1000)

    p.add_argument("--max-freq", type=int, default=5)
    p.add_argument("--scale", type=float, default=0.9)
    p.add_argument("--decay-power", type=float, default=2.0)
    p.add_argument("--min-size", type=float, default=0.45)
    p.add_argument("--max-size", type=float, default=0.75)
    p.add_argument("--max-tries", type=int, default=300)
    p.add_argument("--intersection-check-points", type=int, default=0)

    p.add_argument("--reparametrize-prob", type=float, default=0.85)
    p.add_argument("--reparam-strength", type=float, default=0.15)
    p.add_argument("--reparam-num-harmonics", type=int, default=2)
    p.add_argument("--reparam-min-density", type=float, default=0.7)
    p.add_argument("--reparam-max-density", type=float, default=1.5)

    p.add_argument("--workers", type=int, default=max(1, (mp.cpu_count() or 2) - 1))
    p.add_argument("--seed", type=int, default=123)

    p.add_argument("--preview-png", type=str, default="")
    p.add_argument("--preview-count", type=int, default=20)
    p.add_argument("--preview-point-stride", type=int, default=25)

    return p.parse_args()


def main() -> None:
    args = parse_args()

    actual_intersection_check_points = (
        args.intersection_check_points
        if args.intersection_check_points > 0
        else max(320, args.num_points // 2)
    )

    cfg = CurveGenConfig(
        num_curves=args.num_curves,
        num_points=args.num_points,
        max_freq=args.max_freq,
        scale=args.scale,
        decay_power=args.decay_power,
        min_size=args.min_size,
        max_size=args.max_size,
        max_tries=args.max_tries,
        intersection_check_points=actual_intersection_check_points,
        seed=args.seed,
        reparametrize_prob=args.reparametrize_prob,
        reparam_strength=args.reparam_strength,
        reparam_num_harmonics=args.reparam_num_harmonics,
        reparam_min_density=args.reparam_min_density,
        reparam_max_density=args.reparam_max_density,
    )

    work_items = [(i, cfg) for i in range(cfg.num_curves)]

    with mp.get_context("spawn").Pool(processes=args.workers) as pool:
        results = pool.map(_generate_one, work_items)

    curves = np.stack([r[0] for r in results], axis=0)
    x_coeffs = np.stack([r[1] for r in results], axis=0)
    y_coeffs = np.stack([r[2] for r in results], axis=0)
    t_grids = np.stack([r[3] for r in results], axis=0)
    reparam_flags = np.asarray([r[4] for r in results], dtype=np.int32)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    metadata = asdict(cfg)
    metadata["actual_intersection_check_points"] = actual_intersection_check_points
    metadata["reparam_fraction_realized"] = float(reparam_flags.mean())

    np.savez_compressed(
        output_path,
        curves=curves,
        curve_points=curves,
        points=curves,
        x_coeffs=x_coeffs,
        y_coeffs=y_coeffs,
        t_values=t_grids,
        t_grid=t_grids,
        reparameterized=reparam_flags,
        max_freq=np.asarray(cfg.max_freq, dtype=np.int32),
        metadata_json=np.asarray(json.dumps(metadata), dtype=object),
    )

    meta_path = output_path.with_suffix(".json")
    meta_path.write_text(json.dumps(metadata, indent=2))

    if args.preview_png:
        preview_count = min(args.preview_count, len(curves))
        _plot_contact_sheet(
            curves[:preview_count],
            t_grids[:preview_count],
            reparam_flags[:preview_count],
            Path(args.preview_png),
            point_stride=max(1, args.preview_point_stride),
        )

    print(f"saved {len(curves)} curves to {output_path}")
    print(f"num_points={cfg.num_points}")
    print(f"intersection_check_points={actual_intersection_check_points}")
    print(f"reparameterized_fraction={reparam_flags.mean():.3f}")
    print(f"metadata: {meta_path}")
    if args.preview_png:
        print(f"preview: {args.preview_png}")


if __name__ == "__main__":
    main()