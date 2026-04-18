
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
from tqdm.auto import tqdm

from utils.curve_generation import generate_random_reparameterized_fourier_curve


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate a pregenerated Fourier curve bank compatible with PregeneratedCurveBank."
    )
    p.add_argument("--output-dir", type=str, required=True)
    p.add_argument("--train-count", type=int, default=2000)
    p.add_argument("--val-count", type=int, default=300)
    p.add_argument("--test-count", type=int, default=300)

    # Geometry / sampling
    p.add_argument("--num-curve-points", type=int, default=2000,
                   help="Dense point count before optional downsampling.")
    p.add_argument("--downsample-to-points", type=int, default=1000,
                   help="Final sampled point count stored in the bank.")
    p.add_argument("--downsample-jitter", type=float, default=0.2)

    # Complexity (chosen as a useful bridge regime: harder than f3, much easier than f20)
    p.add_argument("--fourier-max-freq", type=int, default=7)
    p.add_argument("--fourier-scale", type=float, default=0.9)
    p.add_argument("--fourier-decay-power", type=float, default=2.5)

    # Reparameterization / irregular sampling
    p.add_argument("--reparam-strength", type=float, default=0.15)
    p.add_argument("--reparam-num-harmonics", type=int, default=2)
    p.add_argument("--reparam-min-density", type=float, default=0.7)
    p.add_argument("--reparam-max-density", type=float, default=1.5)

    # Shape cleanup / fitting
    p.add_argument("--curve-min-size", type=float, default=0.45)
    p.add_argument("--curve-max-size", type=float, default=0.75)
    p.add_argument("--curve-max-tries", type=int, default=1000)
    p.add_argument("--intersection-check-points", type=int, default=320)

    p.add_argument("--seed", type=int, default=123)
    return p.parse_args()


def generate_split(
    *,
    count: int,
    seed: int,
    num_curve_points: int,
    downsample_to_points: int | None,
    downsample_jitter: float,
    fourier_max_freq: int,
    fourier_scale: float,
    fourier_decay_power: float,
    reparam_strength: float,
    reparam_num_harmonics: int,
    reparam_min_density: float,
    reparam_max_density: float,
    curve_min_size: float,
    curve_max_size: float,
    curve_max_tries: int,
    intersection_check_points: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)

    curve_points_list: list[np.ndarray] = []
    x_coeffs_list: list[np.ndarray] = []
    y_coeffs_list: list[np.ndarray] = []
    t_grid_list: list[np.ndarray] = []

    for _ in tqdm(range(count), desc=f"generate[{count}]"):
        points, coeffs, _, t_warped = generate_random_reparameterized_fourier_curve(
            num_points=num_curve_points,
            max_freq=fourier_max_freq,
            scale=fourier_scale,
            decay_power=fourier_decay_power,
            rng=rng,
            center=True,
            fit_to_canvas=True,
            min_size=curve_min_size,
            max_size=curve_max_size,
            reparam_strength=reparam_strength,
            reparam_num_harmonics=reparam_num_harmonics,
            reparam_min_density=reparam_min_density,
            reparam_max_density=reparam_max_density,
            max_tries=curve_max_tries,
            enforce_simple=True,
            intersection_check_points=intersection_check_points,
            downsample_to_points=downsample_to_points,
            downsample_jitter=downsample_jitter,
        )
        curve_points_list.append(points.astype(np.float32, copy=False))
        x_coeffs_list.append(coeffs.x_coeffs.astype(np.float64, copy=False))
        y_coeffs_list.append(coeffs.y_coeffs.astype(np.float64, copy=False))
        t_grid_list.append(t_warped.astype(np.float64, copy=False))

    curve_points = np.stack(curve_points_list, axis=0)
    x_coeffs = np.stack(x_coeffs_list, axis=0)
    y_coeffs = np.stack(y_coeffs_list, axis=0)
    t_grid = np.stack(t_grid_list, axis=0)

    return curve_points, x_coeffs, y_coeffs, t_grid


def save_split(path: Path, curve_points: np.ndarray, x_coeffs: np.ndarray, y_coeffs: np.ndarray, t_grid: np.ndarray) -> None:
    np.savez_compressed(
        path,
        curve_points=curve_points,
        x_coeffs=x_coeffs,
        y_coeffs=y_coeffs,
        t_grid=t_grid,
    )


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    split_counts = {
        "train": args.train_count,
        "val": args.val_count,
        "test": args.test_count,
    }
    split_seed_offsets = {
        "train": 0,
        "val": 10_000,
        "test": 20_000,
    }

    for split, count in split_counts.items():
        curve_points, x_coeffs, y_coeffs, t_grid = generate_split(
            count=count,
            seed=args.seed + split_seed_offsets[split],
            num_curve_points=args.num_curve_points,
            downsample_to_points=args.downsample_to_points,
            downsample_jitter=args.downsample_jitter,
            fourier_max_freq=args.fourier_max_freq,
            fourier_scale=args.fourier_scale,
            fourier_decay_power=args.fourier_decay_power,
            reparam_strength=args.reparam_strength,
            reparam_num_harmonics=args.reparam_num_harmonics,
            reparam_min_density=args.reparam_min_density,
            reparam_max_density=args.reparam_max_density,
            curve_min_size=args.curve_min_size,
            curve_max_size=args.curve_max_size,
            curve_max_tries=args.curve_max_tries,
            intersection_check_points=args.intersection_check_points,
        )
        save_split(out_dir / f"{split}.npz", curve_points, x_coeffs, y_coeffs, t_grid)

    metadata = vars(args).copy()
    metadata["bank_type"] = "reparameterized_fourier_curve_bank"
    metadata["final_num_points"] = args.downsample_to_points if args.downsample_to_points is not None else args.num_curve_points
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
    print(f"Saved bank to: {out_dir}")


if __name__ == "__main__":
    main()
