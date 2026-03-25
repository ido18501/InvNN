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
    BasisExpansionCurveCoeffs,
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


def _generate_one(args: tuple[int, CurveGenConfig]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    idx, cfg = args
    rng = np.random.default_rng(cfg.seed + 104729 * idx)
    t = np.linspace(0.0, 2.0 * np.pi, cfg.num_points, endpoint=False)
    points, coeffs = generate_random_simple_fourier_curve(
        t=t,
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
    return points.astype(np.float32), coeffs.x_coeffs.astype(np.float32), coeffs.y_coeffs.astype(np.float32)


def _plot_contact_sheet(curves: np.ndarray, output_path: Path, cols: int = 5) -> None:
    n = len(curves)
    cols = max(1, cols)
    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(3.0 * cols, 3.0 * rows))
    axes = np.array(axes).reshape(rows, cols)
    for ax in axes.flat:
        ax.axis("off")
    for i, curve in enumerate(curves):
        ax = axes[i // cols, i % cols]
        ax.plot(curve[:, 0], curve[:, 1], lw=1.0)
        ax.set_aspect("equal")
        ax.set_title(f"curve {i}")
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=str, required=True)
    p.add_argument("--num-curves", type=int, required=True)
    p.add_argument("--num-points", type=int, default=1000)
    p.add_argument("--max-freq", type=int, default=7)
    p.add_argument("--scale", type=float, default=1.0)
    p.add_argument("--decay-power", type=float, default=1.65)
    p.add_argument("--min-size", type=float, default=0.30)
    p.add_argument("--max-size", type=float, default=0.90)
    p.add_argument("--max-tries", type=int, default=200)
    p.add_argument("--intersection-check-points", type=int, default=320)
    p.add_argument("--workers", type=int, default=max(1, (mp.cpu_count() or 2) - 1))
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--preview-png", type=str, default="")
    p.add_argument("--preview-count", type=int, default=20)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = CurveGenConfig(
        num_curves=args.num_curves,
        num_points=args.num_points,
        max_freq=args.max_freq,
        scale=args.scale,
        decay_power=args.decay_power,
        min_size=args.min_size,
        max_size=args.max_size,
        max_tries=args.max_tries,
        intersection_check_points=args.intersection_check_points,
        seed=args.seed,
    )

    work_items = [(i, cfg) for i in range(cfg.num_curves)]
    with mp.get_context("spawn").Pool(processes=args.workers) as pool:
        results = pool.map(_generate_one, work_items)

    curves = np.stack([r[0] for r in results], axis=0)
    x_coeffs = np.stack([r[1] for r in results], axis=0)
    y_coeffs = np.stack([r[2] for r in results], axis=0)
    t_values = np.linspace(0.0, 2.0 * np.pi, cfg.num_points, endpoint=False, dtype=np.float32)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        output_path,
        curves=curves,
        curve_points=curves,
        points=curves,
        x_coeffs=x_coeffs,
        y_coeffs=y_coeffs,
        t_values=t_values,
        t_grid=t_values,
        max_freq=np.asarray(cfg.max_freq, dtype=np.int32),
        metadata_json=np.asarray(json.dumps(asdict(cfg)), dtype=object),
    )

    meta_path = output_path.with_suffix(".json")
    meta_path.write_text(json.dumps(asdict(cfg), indent=2))

    if args.preview_png:
        preview_count = min(args.preview_count, len(curves))
        _plot_contact_sheet(curves[:preview_count], Path(args.preview_png))

    print(f"saved {len(curves)} curves to {output_path}")
    print(f"metadata: {meta_path}")
    if args.preview_png:
        print(f"preview: {args.preview_png}")


if __name__ == "__main__":
    main()
