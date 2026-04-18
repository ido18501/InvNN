#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Tuple

import numpy as np

from utils.curve_generation import (
    BasisExpansionCurveCoeffs,
    generate_random_reparameterized_fourier_curve,
    generate_random_simple_fourier_curve,
    sample_bounded_stride_indices,
)


REGIME_SPECS = {
    "f5": {"max_freq": 5, "decay_power": 3.0},
    "f7": {"max_freq": 7, "decay_power": 2.7},
    "f9": {"max_freq": 9, "decay_power": 2.5},
}

DENSITY_SPECS = {
    "2000to1000": {"num_curve_points": 2000, "downsample_to_points": 1000},
    "1000to500": {"num_curve_points": 1000, "downsample_to_points": 500},
    "500to300": {"num_curve_points": 500, "downsample_to_points": 300},
}

SPLITS = ("train", "val", "test")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate clean Fourier curve banks matching the project bank format.")
    p.add_argument("--regime", choices=sorted(REGIME_SPECS.keys()), required=True)
    p.add_argument("--density", choices=sorted(DENSITY_SPECS.keys()), required=True)
    p.add_argument("--output-root", type=Path, default=Path("."))
    p.add_argument("--reference-root", type=Path, default=Path("."))
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--reparametrize-prob", type=float, default=0.7)
    p.add_argument("--fourier-scale", type=float, default=0.9)
    p.add_argument("--curve-max-tries", type=int, default=300)
    p.add_argument("--curve-min-size", type=float, default=0.45)
    p.add_argument("--curve-max-size", type=float, default=0.75)
    p.add_argument("--reparam-strength", type=float, default=0.15)
    p.add_argument("--reparam-num-harmonics", type=int, default=2)
    p.add_argument("--reparam-min-density", type=float, default=0.7)
    p.add_argument("--reparam-max-density", type=float, default=1.5)
    p.add_argument("--downsample-jitter", type=float, default=0.2)
    p.add_argument("--force", action="store_true")
    return p.parse_args()


def load_reference_counts(reference_dir: Path) -> dict[str, int]:
    counts: dict[str, int] = {}
    for split in SPLITS:
        ref_file = reference_dir / f"{split}.npz"
        if not ref_file.exists():
            raise FileNotFoundError(f"Missing reference split: {ref_file}")
        with np.load(ref_file, allow_pickle=False) as data:
            counts[split] = int(np.asarray(data["curve_points"]).shape[0])
    return counts


def generate_one_curve(
    *,
    rng: np.random.Generator,
    num_curve_points: int,
    downsample_to_points: int,
    max_freq: int,
    decay_power: float,
    fourier_scale: float,
    curve_max_tries: int,
    curve_min_size: float,
    curve_max_size: float,
    reparametrize_prob: float,
    reparam_strength: float,
    reparam_num_harmonics: int,
    reparam_min_density: float,
    reparam_max_density: float,
    downsample_jitter: float,
) -> Tuple[np.ndarray, BasisExpansionCurveCoeffs, np.ndarray]:
    use_reparam = bool(rng.random() < reparametrize_prob)

    if use_reparam:
        curve_points, coeffs, _, t_warped = generate_random_reparameterized_fourier_curve(
            num_points=num_curve_points,
            max_freq=max_freq,
            scale=fourier_scale,
            decay_power=decay_power,
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
            intersection_check_points=max(320, num_curve_points // 2),
            downsample_to_points=downsample_to_points,
            downsample_jitter=downsample_jitter,
        )
        t_grid = t_warped
        return curve_points.astype(np.float32), coeffs, t_grid.astype(np.float64)

    t_grid = np.linspace(0.0, 2.0 * np.pi, num_curve_points, endpoint=False, dtype=np.float64)
    curve_points, coeffs = generate_random_simple_fourier_curve(
        t=t_grid,
        max_freq=max_freq,
        scale=fourier_scale,
        decay_power=decay_power,
        rng=rng,
        max_tries=curve_max_tries,
        center=True,
        fit_to_canvas=True,
        min_size=curve_min_size,
        max_size=curve_max_size,
        enforce_simple=False,
    )

    if downsample_to_points < len(curve_points):
        idxs = sample_bounded_stride_indices(
            len(curve_points),
            downsample_to_points,
            rng=rng,
            jitter=downsample_jitter,
        )
        curve_points = curve_points[idxs]
        t_grid = t_grid[idxs]

    return curve_points.astype(np.float32), coeffs, t_grid.astype(np.float64)


def save_split(
    *,
    out_file: Path,
    count: int,
    seed: int,
    num_curve_points: int,
    downsample_to_points: int,
    max_freq: int,
    decay_power: float,
    fourier_scale: float,
    curve_max_tries: int,
    curve_min_size: float,
    curve_max_size: float,
    reparametrize_prob: float,
    reparam_strength: float,
    reparam_num_harmonics: int,
    reparam_min_density: float,
    reparam_max_density: float,
    downsample_jitter: float,
) -> None:
    rng = np.random.default_rng(seed)
    curve_points_bank = []
    x_coeffs_bank = []
    y_coeffs_bank = []
    t_grid_bank = []

    for _ in range(count):
        curve_points, coeffs, t_grid = generate_one_curve(
            rng=rng,
            num_curve_points=num_curve_points,
            downsample_to_points=downsample_to_points,
            max_freq=max_freq,
            decay_power=decay_power,
            fourier_scale=fourier_scale,
            curve_max_tries=curve_max_tries,
            curve_min_size=curve_min_size,
            curve_max_size=curve_max_size,
            reparametrize_prob=reparametrize_prob,
            reparam_strength=reparam_strength,
            reparam_num_harmonics=reparam_num_harmonics,
            reparam_min_density=reparam_min_density,
            reparam_max_density=reparam_max_density,
            downsample_jitter=downsample_jitter,
        )
        curve_points_bank.append(curve_points)
        x_coeffs_bank.append(coeffs.x_coeffs.astype(np.float64))
        y_coeffs_bank.append(coeffs.y_coeffs.astype(np.float64))
        t_grid_bank.append(t_grid)

    out_file.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_file,
        curve_points=np.stack(curve_points_bank, axis=0).astype(np.float32),
        x_coeffs=np.stack(x_coeffs_bank, axis=0).astype(np.float64),
        y_coeffs=np.stack(y_coeffs_bank, axis=0).astype(np.float64),
        t_grid=np.stack(t_grid_bank, axis=0).astype(np.float64),
    )



def main() -> None:
    args = parse_args()
    regime_spec = REGIME_SPECS[args.regime]
    density_spec = DENSITY_SPECS[args.density]

    dataset_name = f"data_{args.regime}_{args.density}"
    reference_name = f"data_complex_f20_{args.density}"
    output_dir = args.output_root / dataset_name
    reference_dir = args.reference_root / reference_name

    if output_dir.exists() and not args.force:
        existing = [output_dir / f"{split}.npz" for split in SPLITS]
        if all(p.exists() for p in existing):
            print(f"Dataset already exists, skipping: {output_dir}")
            return

    counts = load_reference_counts(reference_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    split_seed_offsets = {"train": 0, "val": 100000, "test": 200000}
    for split in SPLITS:
        out_file = output_dir / f"{split}.npz"
        save_split(
            out_file=out_file,
            count=counts[split],
            seed=args.seed + split_seed_offsets[split],
            num_curve_points=density_spec["num_curve_points"],
            downsample_to_points=density_spec["downsample_to_points"],
            max_freq=regime_spec["max_freq"],
            decay_power=regime_spec["decay_power"],
            fourier_scale=args.fourier_scale,
            curve_max_tries=args.curve_max_tries,
            curve_min_size=args.curve_min_size,
            curve_max_size=args.curve_max_size,
            reparametrize_prob=args.reparametrize_prob,
            reparam_strength=args.reparam_strength,
            reparam_num_harmonics=args.reparam_num_harmonics,
            reparam_min_density=args.reparam_min_density,
            reparam_max_density=args.reparam_max_density,
            downsample_jitter=args.downsample_jitter,
        )
        print(f"Wrote {out_file} with {counts[split]} curves")

    metadata = {
        "dataset_name": dataset_name,
        "reference_name": reference_name,
        "seed": args.seed,
        "counts": counts,
        "regime": args.regime,
        "density": args.density,
        "fourier_max_freq": regime_spec["max_freq"],
        "fourier_decay_power": regime_spec["decay_power"],
        "fourier_scale": args.fourier_scale,
        "reparametrize_prob": args.reparametrize_prob,
        "reparam_strength": args.reparam_strength,
        "reparam_num_harmonics": args.reparam_num_harmonics,
        "reparam_min_density": args.reparam_min_density,
        "reparam_max_density": args.reparam_max_density,
        "num_curve_points": density_spec["num_curve_points"],
        "downsample_to_points": density_spec["downsample_to_points"],
        "downsample_jitter": args.downsample_jitter,
        "curve_min_size": args.curve_min_size,
        "curve_max_size": args.curve_max_size,
        "curve_max_tries": args.curve_max_tries,
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
