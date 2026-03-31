from __future__ import annotations

import argparse
import numpy as np

from utils.curve_generation import generate_random_reparameterized_fourier_curve
from utils.derivatives import compute_fourier_arc_length_derivatives


Array = np.ndarray


def _normalize(v: Array, eps: float = 1e-12) -> Array:
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.clip(n, eps, None)


def _angle_deg(u: Array, v: Array, eps: float = 1e-12) -> Array:
    un = _normalize(u, eps=eps)
    vn = _normalize(v, eps=eps)
    c = np.sum(un * vn, axis=-1)
    c = np.clip(c, -1.0, 1.0)
    return np.degrees(np.arccos(c))


def _det2(a: Array, b: Array) -> Array:
    return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]


def _family_sigma_from_discrete_curve(
    points: Array,
    family: str,
    eps: float = 1e-12,
) -> Array:
    """
    Numerical ds/dt proxy on the sampled reparameterized curve.
    We interpret t as the original source parameter grid t_grid.
    """
    prev_pts = np.roll(points, 1, axis=0)
    next_pts = np.roll(points, -1, axis=0)

    # central first derivative wrt discrete parameter index; caller will divide by dt
    # here we only need ratios and consistent geometric formulas, so actual dt is applied outside
    raise RuntimeError("internal helper should not be called directly")


def _numeric_geometric_derivatives(
    points: Array,
    t_grid: Array,
    family: str,
    eps: float = 1e-12,
) -> tuple[Array, Array]:
    """
    Numerical geometric derivatives from the sampled curve points and the actual t_grid.

    Uses central differences in t, then converts to d/ds and d^2/ds^2 with the same
    geometric formulas as the analytical side, but numerically.
    """
    points = np.asarray(points, dtype=np.float64)
    t_grid = np.asarray(t_grid, dtype=np.float64)
    family = family.lower()

    n = len(points)
    if points.shape != (n, 2):
        raise ValueError("points must have shape (N, 2)")
    if t_grid.shape != (n,):
        raise ValueError("t_grid must have shape (N,)")

    t_prev = np.roll(t_grid, 1)
    t_next = np.roll(t_grid, -1)

    dt_back = t_grid - t_prev
    dt_fwd = t_next - t_grid

    # periodic unwrap
    dt_back[dt_back <= 0.0] += 2.0 * np.pi
    dt_fwd[dt_fwd <= 0.0] += 2.0 * np.pi

    prev_pts = np.roll(points, 1, axis=0)
    next_pts = np.roll(points, -1, axis=0)

    # Nonuniform central first derivative
    first_dt = (next_pts - prev_pts) / (dt_fwd[:, None] + dt_back[:, None])

    # Simple second derivative using local average spacing
    h = 0.5 * (dt_fwd + dt_back)
    second_dt = (next_pts - 2.0 * points + prev_pts) / np.clip(h[:, None] ** 2, eps, None)

    # Third derivative approximation for sigma_t in non-euclidean families
    prev2_pts = np.roll(points, 2, axis=0)
    next2_pts = np.roll(points, -2, axis=0)
    h3 = np.mean(h)
    third_dt = (next2_pts - 2.0 * next_pts + 2.0 * prev_pts - prev2_pts) / np.clip(
        2.0 * (h3 ** 3), eps, None
    )

    speed = np.linalg.norm(first_dt, axis=-1)
    speed_safe = np.clip(speed, eps, None)
    dot12 = np.sum(first_dt * second_dt, axis=-1)
    speed_t = dot12 / speed_safe

    det12 = _det2(first_dt, second_dt)
    det13 = _det2(first_dt, third_dt)

    if family == "euclidean":
        sigma = speed_safe
        sigma_t = speed_t

    elif family == "similarity":
        abs_det12 = np.clip(np.abs(det12), eps, None)
        sign_det12 = np.sign(det12)
        sigma = abs_det12 / (speed_safe ** 2)
        sigma_t = sign_det12 * det13 / (speed_safe ** 2) - 2.0 * abs_det12 * speed_t / (speed_safe ** 3)

    elif family == "equi_affine":
        abs_det12 = np.clip(np.abs(det12), eps, None)
        sign_det12 = np.sign(det12)
        sigma = abs_det12 ** (1.0 / 3.0)
        sigma_t = sign_det12 * det13 / (3.0 * (abs_det12 ** (2.0 / 3.0)))

    else:
        raise ValueError(f"Unsupported family: {family}")

    sigma = sigma[:, None]
    sigma_t = sigma_t[:, None]

    first_ds = first_dt / sigma
    second_ds = second_dt / (sigma ** 2) - first_dt * sigma_t / (sigma ** 3)
    return first_ds, second_ds


def _family_curvature_mask(first_analytic: Array, second_analytic: Array, family: str, eps: float = 1e-8) -> Array:
    family = family.lower()
    if family == "euclidean":
        mag = np.linalg.norm(second_analytic, axis=-1)
        return mag > eps
    if family == "similarity":
        mag = np.linalg.norm(second_analytic, axis=-1)
        return mag > eps
    if family == "equi_affine":
        mag = np.linalg.norm(second_analytic, axis=-1)
        return mag > eps
    return np.ones(len(first_analytic), dtype=bool)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--family", type=str, default="euclidean", choices=["euclidean", "similarity", "equi_affine"])
    p.add_argument("--num-curves", type=int, default=10)
    p.add_argument("--num-points", type=int, default=4000)
    p.add_argument("--max-freq", type=int, default=7)
    p.add_argument("--scale", type=float, default=1.0)
    p.add_argument("--decay-power", type=float, default=1.65)
    p.add_argument("--min-size", type=float, default=0.45)
    p.add_argument("--max-size", type=float, default=0.75)
    p.add_argument("--reparam-strength", type=float, default=0.15)
    p.add_argument("--reparam-num-harmonics", type=int, default=2)
    p.add_argument("--reparam-min-density", type=float, default=0.7)
    p.add_argument("--reparam-max-density", type=float, default=1.5)
    p.add_argument("--seed", type=int, default=123)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    first_angle_means = []
    first_angle_p95s = []
    second_angle_means = []
    second_angle_p95s = []
    first_rel_err_means = []
    second_rel_err_means = []

    for i in range(args.num_curves):
        curve_points, coeffs, _, t_warped = generate_random_reparameterized_fourier_curve(
            num_points=args.num_points,
            max_freq=args.max_freq,
            scale=args.scale,
            decay_power=args.decay_power,
            rng=rng,
            center=True,
            fit_to_canvas=True,
            min_size=args.min_size,
            max_size=args.max_size,
            reparam_strength=args.reparam_strength,
            reparam_num_harmonics=args.reparam_num_harmonics,
            reparam_min_density=args.reparam_min_density,
            reparam_max_density=args.reparam_max_density,
        )

        _, first_analytic, second_analytic = compute_fourier_arc_length_derivatives(
            t=t_warped,
            coeffs=coeffs,
            family=args.family,
        )

        first_numeric, second_numeric = _numeric_geometric_derivatives(
            points=curve_points,
            t_grid=t_warped,
            family=args.family,
        )

        first_ang = _angle_deg(first_analytic, first_numeric)

        mask = _family_curvature_mask(first_analytic, second_analytic, args.family)
        if np.any(mask):
            second_ang = _angle_deg(second_analytic[mask], second_numeric[mask])
        else:
            second_ang = np.array([0.0], dtype=np.float64)

        first_rel = np.linalg.norm(first_analytic - first_numeric, axis=-1) / np.clip(
            np.linalg.norm(first_analytic, axis=-1), 1e-12, None
        )
        second_rel = np.linalg.norm(second_analytic[mask] - second_numeric[mask], axis=-1) / np.clip(
            np.linalg.norm(second_analytic[mask], axis=-1), 1e-12, None
        ) if np.any(mask) else np.array([0.0], dtype=np.float64)

        first_angle_means.append(float(np.mean(first_ang)))
        first_angle_p95s.append(float(np.percentile(first_ang, 95)))
        second_angle_means.append(float(np.mean(second_ang)))
        second_angle_p95s.append(float(np.percentile(second_ang, 95)))
        first_rel_err_means.append(float(np.mean(first_rel)))
        second_rel_err_means.append(float(np.mean(second_rel)))

        print(
            f"[curve {i:02d}] "
            f"first_mean_ang={np.mean(first_ang):.4f} deg | "
            f"first_p95_ang={np.percentile(first_ang,95):.4f} deg | "
            f"second_mean_ang={np.mean(second_ang):.4f} deg | "
            f"second_p95_ang={np.percentile(second_ang,95):.4f} deg | "
            f"first_rel_mean={np.mean(first_rel):.4e} | "
            f"second_rel_mean={np.mean(second_rel):.4e}"
        )

    print("\n=== Summary ===")
    print(f"family:                    {args.family}")
    print(f"first mean angle:          {np.mean(first_angle_means):.4f} deg")
    print(f"first p95 angle:           {np.mean(first_angle_p95s):.4f} deg")
    print(f"second mean angle:         {np.mean(second_angle_means):.4f} deg")
    print(f"second p95 angle:          {np.mean(second_angle_p95s):.4f} deg")
    print(f"first mean relative error: {np.mean(first_rel_err_means):.4e}")
    print(f"second mean relative error:{np.mean(second_rel_err_means):.4e}")


if __name__ == "__main__":
    main()
