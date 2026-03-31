from __future__ import annotations

import argparse
import numpy as np

from utils.curve_generation import (
    generate_random_basis_expansion_coeffs,
    make_fourier_basis_functions,
    make_fourier_coeff_std,
    evaluate_basis_expansion_curve,
    center_curve,
    sample_smooth_monotone_periodic_reparameterization,
)


TWO_PI = 2.0 * np.pi


def tangent(points: np.ndarray) -> np.ndarray:
    d = np.roll(points, -1, axis=0) - np.roll(points, 1, axis=0)
    n = np.linalg.norm(d, axis=1, keepdims=True)
    return d / np.clip(n, 1e-12, None)


def second_diff(points: np.ndarray) -> np.ndarray:
    return np.roll(points, -1, axis=0) - 2.0 * points + np.roll(points, 1, axis=0)


def angle_deg(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    un = u / np.clip(np.linalg.norm(u, axis=1, keepdims=True), 1e-12, None)
    vn = v / np.clip(np.linalg.norm(v, axis=1, keepdims=True), 1e-12, None)
    c = np.sum(un * vn, axis=1)
    c = np.clip(c, -1.0, 1.0)
    return np.degrees(np.arccos(c))


def circular_forward_differences(t_warped: np.ndarray) -> np.ndarray:
    """
    For t_warped in [0, 2pi), assumed sorted and periodic.
    Returns forward circular parameter increments, shape (N,).
    """
    t_warped = np.asarray(t_warped, dtype=np.float64)
    if t_warped.ndim != 1:
        raise ValueError("t_warped must be 1D")
    if len(t_warped) < 2:
        raise ValueError("Need at least 2 points")

    dt = np.empty_like(t_warped)
    dt[:-1] = t_warped[1:] - t_warped[:-1]
    dt[-1] = (t_warped[0] + TWO_PI) - t_warped[-1]
    return dt


def ratio_stats(x: np.ndarray, eps: float = 1e-12) -> tuple[float, float, float]:
    """
    Returns:
        max/min ratio,
        p95/p5 ratio,
        p90/p10 ratio
    """
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return float("nan"), float("nan"), float("nan")

    xmin = float(np.min(x))
    xmax = float(np.max(x))
    p05 = float(np.percentile(x, 5))
    p10 = float(np.percentile(x, 10))
    p90 = float(np.percentile(x, 90))
    p95 = float(np.percentile(x, 95))

    return (
        xmax / max(xmin, eps),
        p95 / max(p05, eps),
        p90 / max(p10, eps),
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()

    p.add_argument("--num-curves", type=int, default=10)
    p.add_argument("--num-points", type=int, default=4000)
    p.add_argument("--max-freq", type=int, default=7)
    p.add_argument("--scale", type=float, default=1.0)
    p.add_argument("--decay-power", type=float, default=1.65)
    p.add_argument("--strength", type=float, default=0.15)
    p.add_argument("--num-harmonics", type=int, default=2)
    p.add_argument("--min-density", type=float, default=0.7)
    p.add_argument("--max-density", type=float, default=1.5)
    p.add_argument("--seed", type=int, default=123)

    return p.parse_args()


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    basis_functions = make_fourier_basis_functions(args.max_freq)
    coeff_std = make_fourier_coeff_std(
        max_freq=args.max_freq,
        scale=args.scale,
        decay_power=args.decay_power,
    )

    dt_ratio_list = []
    dt_p95_p05_list = []
    dt_p90_p10_list = []

    geom_ratio_list = []
    geom_p95_p05_list = []
    geom_p90_p10_list = []

    tan_mean_list = []
    tan_p95_list = []
    sec_mean_list = []
    sec_p95_list = []

    for i in range(args.num_curves):
        coeffs = generate_random_basis_expansion_coeffs(
            num_basis_functions=len(basis_functions),
            coeff_std=coeff_std,
            rng=rng,
        )

        _, t_warped = sample_smooth_monotone_periodic_reparameterization(
            num_points=args.num_points,
            rng=rng,
            strength=args.strength,
            num_harmonics=args.num_harmonics,
            min_density=args.min_density,
            max_density=args.max_density,
        )

        sampled = evaluate_basis_expansion_curve(t_warped, basis_functions, coeffs)
        sampled = center_curve(sampled)

        # analytic self-check at the exact same t_warped points
        ref = evaluate_basis_expansion_curve(t_warped, basis_functions, coeffs)
        ref = center_curve(ref)

        tan_err = angle_deg(tangent(sampled), tangent(ref))

        sec_s = second_diff(sampled)
        sec_r = second_diff(ref)
        mask = (np.linalg.norm(sec_s, axis=1) > 1e-8) & (np.linalg.norm(sec_r, axis=1) > 1e-8)
        sec_err = angle_deg(sec_s[mask], sec_r[mask]) if np.any(mask) else np.array([0.0], dtype=np.float64)

        dt = circular_forward_differences(t_warped)
        dt_ratio, dt_p95_p05, dt_p90_p10 = ratio_stats(dt)

        seg = np.linalg.norm(np.roll(sampled, -1, axis=0) - sampled, axis=1)
        geom_ratio, geom_p95_p05, geom_p90_p10 = ratio_stats(seg)

        tan_mean = float(np.mean(tan_err))
        tan_p95 = float(np.percentile(tan_err, 95))
        sec_mean = float(np.mean(sec_err))
        sec_p95 = float(np.percentile(sec_err, 95))

        dt_ratio_list.append(dt_ratio)
        dt_p95_p05_list.append(dt_p95_p05)
        dt_p90_p10_list.append(dt_p90_p10)

        geom_ratio_list.append(geom_ratio)
        geom_p95_p05_list.append(geom_p95_p05)
        geom_p90_p10_list.append(geom_p90_p10)

        tan_mean_list.append(tan_mean)
        tan_p95_list.append(tan_p95)
        sec_mean_list.append(sec_mean)
        sec_p95_list.append(sec_p95)

        print(
            f"[curve {i:02d}] "
            f"tan_mean={tan_mean:.4f} deg | "
            f"tan_p95={tan_p95:.4f} deg | "
            f"sec_mean={sec_mean:.4f} deg | "
            f"sec_p95={sec_p95:.4f} deg | "
            f"dt_ratio={dt_ratio:.3f} | "
            f"dt_p95/p05={dt_p95_p05:.3f} | "
            f"dt_p90/p10={dt_p90_p10:.3f} | "
            f"geom_ratio={geom_ratio:.3f} | "
            f"geom_p95/p05={geom_p95_p05:.3f} | "
            f"geom_p90/p10={geom_p90_p10:.3f}"
        )

    print("\n=== Summary over all curves ===")
    print(f"tangent mean angle:         {np.mean(tan_mean_list):.4f} deg")
    print(f"tangent 95th percentile:    {np.mean(tan_p95_list):.4f} deg")
    print(f"second mean angle:          {np.mean(sec_mean_list):.4f} deg")
    print(f"second 95th percentile:     {np.mean(sec_p95_list):.4f} deg")
    print(f"mean dt_ratio:              {np.mean(dt_ratio_list):.3f}")
    print(f"mean dt_p95/p05:            {np.mean(dt_p95_p05_list):.3f}")
    print(f"mean dt_p90/p10:            {np.mean(dt_p90_p10_list):.3f}")
    print(f"mean geom_ratio:            {np.mean(geom_ratio_list):.3f}")
    print(f"mean geom_p95/p05:          {np.mean(geom_p95_p05_list):.3f}")
    print(f"mean geom_p90/p10:          {np.mean(geom_p90_p10_list):.3f}")


if __name__ == "__main__":
    main()
