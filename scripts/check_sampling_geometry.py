from __future__ import annotations

import argparse
import numpy as np

from utils.curve_generation import (
    generate_random_basis_expansion_coeffs,
    make_fourier_basis_functions,
    make_fourier_coeff_std,
    evaluate_basis_expansion_curve,
    center_curve,
)


def compute_tangent(points: np.ndarray) -> np.ndarray:
    prev_pts = np.roll(points, 1, axis=0)
    next_pts = np.roll(points, -1, axis=0)
    d = next_pts - prev_pts
    n = np.linalg.norm(d, axis=1, keepdims=True)
    return d / np.clip(n, 1e-12, None)


def compute_second_difference(points: np.ndarray) -> np.ndarray:
    prev_pts = np.roll(points, 1, axis=0)
    next_pts = np.roll(points, -1, axis=0)
    return next_pts - 2.0 * points + prev_pts


def angle_deg(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    un = u / np.clip(np.linalg.norm(u, axis=1, keepdims=True), 1e-12, None)
    vn = v / np.clip(np.linalg.norm(v, axis=1, keepdims=True), 1e-12, None)
    cos = np.sum(un * vn, axis=1)
    cos = np.clip(cos, -1.0, 1.0)
    return np.degrees(np.arccos(cos))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--num-curves", type=int, default=10)
    p.add_argument("--num-curve-points", type=int, default=4000)
    p.add_argument("--reference-points", type=int, default=32000)
    p.add_argument("--max-freq", type=int, default=7)
    p.add_argument("--scale", type=float, default=1.0)
    p.add_argument("--decay-power", type=float, default=1.65)
    p.add_argument("--min-size", type=float, default=0.30)
    p.add_argument("--max-size", type=float, default=0.90)
    p.add_argument("--seed", type=int, default=123)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.reference_points % args.num_curve_points != 0:
        raise ValueError("--reference-points must be an integer multiple of --num-curve-points")

    stride = args.reference_points // args.num_curve_points
    rng = np.random.default_rng(args.seed)

    basis_functions = make_fourier_basis_functions(args.max_freq)
    coeff_std = make_fourier_coeff_std(
        max_freq=args.max_freq,
        scale=args.scale,
        decay_power=args.decay_power,
    )

    pos_errs = []
    tan_mean_errs = []
    tan_p95_errs = []
    sec_mean_errs = []
    sec_p95_errs = []

    for i in range(args.num_curves):
        coeffs = generate_random_basis_expansion_coeffs(
            num_basis_functions=len(basis_functions),
            coeff_std=coeff_std,
            rng=rng,
        )

        target_extent = float(rng.uniform(args.min_size, args.max_size))

        t_coarse = np.linspace(0.0, 2.0 * np.pi, args.num_curve_points, endpoint=False)
        t_fine = np.linspace(0.0, 2.0 * np.pi, args.reference_points, endpoint=False)

        coarse = evaluate_basis_expansion_curve(t_coarse, basis_functions, coeffs)
        fine = evaluate_basis_expansion_curve(t_fine, basis_functions, coeffs)

        coarse = center_curve(coarse)
        fine = center_curve(fine)

        coarse = coarse * (target_extent / max(np.max(np.abs(coarse)), 1e-12))
        fine = fine * (target_extent / max(np.max(np.abs(fine)), 1e-12))

        # exact same parameter locations
        fine_on_coarse_grid = fine[::stride]

        pos_err = np.linalg.norm(coarse - fine_on_coarse_grid, axis=1)

        tan_coarse = compute_tangent(coarse)
        tan_ref = compute_tangent(fine_on_coarse_grid)
        tan_ang = angle_deg(tan_coarse, tan_ref)

        sec_coarse = compute_second_difference(coarse)
        sec_ref = compute_second_difference(fine_on_coarse_grid)

        mask = (
            (np.linalg.norm(sec_coarse, axis=1) > 1e-8)
            & (np.linalg.norm(sec_ref, axis=1) > 1e-8)
        )
        sec_ang = angle_deg(sec_coarse[mask], sec_ref[mask]) if np.any(mask) else np.array([0.0])

        pos_errs.append(float(pos_err.mean()))
        tan_mean_errs.append(float(tan_ang.mean()))
        tan_p95_errs.append(float(np.percentile(tan_ang, 95)))
        sec_mean_errs.append(float(sec_ang.mean()))
        sec_p95_errs.append(float(np.percentile(sec_ang, 95)))

        print(
            f"[curve {i:02d}] "
            f"pos_mean={pos_err.mean():.6e} | "
            f"tan_mean={tan_ang.mean():.4f} deg | tan_p95={np.percentile(tan_ang,95):.4f} deg | "
            f"sec_mean={sec_ang.mean():.4f} deg | sec_p95={np.percentile(sec_ang,95):.4f} deg"
        )

    print("\n=== Summary over all curves ===")
    print(f"position mean error:      {np.mean(pos_errs):.6e}")
    print(f"tangent mean angle:       {np.mean(tan_mean_errs):.4f} deg")
    print(f"tangent 95th percentile:  {np.mean(tan_p95_errs):.4f} deg")
    print(f"second mean angle:        {np.mean(sec_mean_errs):.4f} deg")
    print(f"second 95th percentile:   {np.mean(sec_p95_errs):.4f} deg")


if __name__ == "__main__":
    main()
