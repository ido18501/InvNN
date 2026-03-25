from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np


Array = np.ndarray
BasisFunction = Callable[[Array], Array]


@dataclass
class BasisExpansionCurveCoeffs:
    """
    Coefficients for a 2D parametric curve defined by a shared basis expansion:

        x(t) = sum_i x_coeffs[i] * phi_i(t)
        y(t) = sum_i y_coeffs[i] * phi_i(t)
    """
    x_coeffs: Array  # shape: (M,)
    y_coeffs: Array  # shape: (M,)


def generate_random_basis_expansion_coeffs(
    num_basis_functions: int,
    scale: float = 1.0,
    coeff_std: Array | None = None,
    rng: np.random.Generator | None = None,
) -> BasisExpansionCurveCoeffs:
    """
    Generate random coefficients for a generic 2D basis-expansion curve.
    """
    if num_basis_functions < 1:
        raise ValueError("num_basis_functions must be at least 1.")
    if scale <= 0:
        raise ValueError("scale must be positive.")

    if rng is None:
        rng = np.random.default_rng()

    if coeff_std is None:
        coeff_std = np.full(num_basis_functions, scale, dtype=np.float64)
    else:
        coeff_std = np.asarray(coeff_std, dtype=np.float64)
        if coeff_std.shape != (num_basis_functions,):
            raise ValueError("coeff_std must have shape (num_basis_functions,).")
        if np.any(coeff_std <= 0):
            raise ValueError("All entries of coeff_std must be positive.")

    x_coeffs = rng.normal(loc=0.0, scale=coeff_std, size=num_basis_functions)
    y_coeffs = rng.normal(loc=0.0, scale=coeff_std, size=num_basis_functions)

    return BasisExpansionCurveCoeffs(
        x_coeffs=x_coeffs,
        y_coeffs=y_coeffs,
    )


def evaluate_basis_expansion_curve(
    t: Array,
    basis_functions: Sequence[BasisFunction],
    coeffs: BasisExpansionCurveCoeffs,
) -> Array:
    """
    Evaluate a 2D parametric curve at parameter values t.

    Returns:
        points: shape (N, 2)
    """
    t = np.asarray(t, dtype=np.float64)
    if t.ndim != 1:
        raise ValueError("t must be a 1D array.")

    num_basis = len(basis_functions)
    if coeffs.x_coeffs.shape != (num_basis,) or coeffs.y_coeffs.shape != (num_basis,):
        raise ValueError("Coefficient shapes must match the number of basis functions.")

    basis_matrix = np.stack([phi(t) for phi in basis_functions], axis=1)  # (N, M)
    x = basis_matrix @ coeffs.x_coeffs
    y = basis_matrix @ coeffs.y_coeffs
    return np.stack([x, y], axis=1)


def make_fourier_basis_functions(max_freq: int) -> list[BasisFunction]:
    """
    Create Fourier basis functions:
        cos(t), sin(t), cos(2t), sin(2t), ..., cos(Kt), sin(Kt)
    """
    if max_freq < 1:
        raise ValueError("max_freq must be at least 1.")

    basis_functions: list[BasisFunction] = []
    for k in range(1, max_freq + 1):
        basis_functions.append(lambda t, k=k: np.cos(k * t))
        basis_functions.append(lambda t, k=k: np.sin(k * t))
    return basis_functions


def make_fourier_coeff_std(
    max_freq: int,
    scale: float = 1.0,
    decay_power: float = 2.0,
) -> Array:
    """
    For each frequency k, both cos(k t) and sin(k t) get:
        std = scale / k^decay_power
    """
    if max_freq < 1:
        raise ValueError("max_freq must be at least 1.")
    if scale <= 0:
        raise ValueError("scale must be positive.")
    if decay_power <= 0:
        raise ValueError("decay_power must be positive.")

    stds = []
    for k in range(1, max_freq + 1):
        s = scale / (k ** decay_power)
        stds.extend([s, s])

    return np.asarray(stds, dtype=np.float64)


def generate_random_fourier_curve(
    t: Array,
    max_freq: int = 5,
    scale: float = 1.0,
    decay_power: float = 2.0,
    rng: np.random.Generator | None = None,
) -> tuple[Array, BasisExpansionCurveCoeffs]:
    """
    Generate a random smooth Fourier curve evaluated at parameter values t.
    """
    basis_functions = make_fourier_basis_functions(max_freq)
    coeff_std = make_fourier_coeff_std(
        max_freq=max_freq,
        scale=scale,
        decay_power=decay_power,
    )

    coeffs = generate_random_basis_expansion_coeffs(
        num_basis_functions=len(basis_functions),
        coeff_std=coeff_std,
        rng=rng,
    )

    points = evaluate_basis_expansion_curve(t, basis_functions, coeffs)
    return points, coeffs


def _orientation(a: Array, b: Array, c: Array) -> float:
    """
    Signed area / orientation test for three 2D points.
    Positive  -> counterclockwise
    Negative  -> clockwise
    Zero      -> collinear
    """
    a = np.asarray(a, dtype=np.float64).reshape(2)
    b = np.asarray(b, dtype=np.float64).reshape(2)
    c = np.asarray(c, dtype=np.float64).reshape(2)

    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def _on_segment(a: Array, b: Array, c: Array, eps: float = 1e-12) -> bool:
    """
    Check whether point c lies on segment [a,b], assuming near-collinearity.
    """
    a = np.asarray(a, dtype=np.float64).reshape(2)
    b = np.asarray(b, dtype=np.float64).reshape(2)
    c = np.asarray(c, dtype=np.float64).reshape(2)

    return (
        min(a[0], b[0]) - eps <= c[0] <= max(a[0], b[0]) + eps
        and min(a[1], b[1]) - eps <= c[1] <= max(a[1], b[1]) + eps
    )


def _segments_intersect(
    p1: Array,
    p2: Array,
    q1: Array,
    q2: Array,
    eps: float = 1e-12,
) -> bool:
    """
    Check whether closed segments [p1,p2] and [q1,q2] intersect.
    """
    p1 = np.asarray(p1, dtype=np.float64).reshape(2)
    p2 = np.asarray(p2, dtype=np.float64).reshape(2)
    q1 = np.asarray(q1, dtype=np.float64).reshape(2)
    q2 = np.asarray(q2, dtype=np.float64).reshape(2)

    o1 = _orientation(p1, p2, q1)
    o2 = _orientation(p1, p2, q2)
    o3 = _orientation(q1, q2, p1)
    o4 = _orientation(q1, q2, p2)

    # Proper intersection
    if ((o1 > eps and o2 < -eps) or (o1 < -eps and o2 > eps)) and \
       ((o3 > eps and o4 < -eps) or (o3 < -eps and o4 > eps)):
        return True

    # Collinear / endpoint cases
    if abs(o1) <= eps and _on_segment(p1, p2, q1, eps):
        return True
    if abs(o2) <= eps and _on_segment(p1, p2, q2, eps):
        return True
    if abs(o3) <= eps and _on_segment(q1, q2, p1, eps):
        return True
    if abs(o4) <= eps and _on_segment(q1, q2, p2, eps):
        return True

    return False


def curve_has_self_intersections(
    points: Array,
    closed: bool = True,
) -> bool:
    """
    Check whether a sampled polyline has self-intersections.
    """
    points = np.asarray(points, dtype=np.float64)

    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError(f"points must have shape (N, 2), got {points.shape}")

    n = len(points)
    if n < 4:
        return False

    if closed:
        num_segments = n
        seg_start = points
        seg_end = np.roll(points, -1, axis=0)
    else:
        num_segments = n - 1
        seg_start = points[:-1]
        seg_end = points[1:]

    for i in range(num_segments):
        for j in range(i + 1, num_segments):
            # adjacent segments are allowed to meet at endpoints
            if j == i + 1:
                continue

            # first and last are adjacent in a closed curve
            if closed and i == 0 and j == num_segments - 1:
                continue

            if _segments_intersect(seg_start[i], seg_end[i], seg_start[j], seg_end[j]):
                return True

    return False


def center_curve(points: Array) -> Array:
    """
    Subtract centroid.
    """
    points = np.asarray(points, dtype=np.float64)
    return points - points.mean(axis=0, keepdims=True)


def get_max_abs_extent(points: Array) -> float:
    """
    Returns max absolute coordinate value over x and y.
    """
    points = np.asarray(points, dtype=np.float64)
    return float(np.max(np.abs(points)))


def fit_curve_to_canvas_with_random_size(
    points: Array,
    rng: np.random.Generator | None = None,
    min_size: float = 0.25,
    max_size: float = 0.95,
) -> Array:
    """
    Scale the curve so that it fits inside [-s, s]^2 where
    s is sampled uniformly from [min_size, max_size].

    This preserves size variability across samples.
    """
    points = np.asarray(points, dtype=np.float64)

    if rng is None:
        rng = np.random.default_rng()

    if not (0.0 < min_size <= max_size):
        raise ValueError("Require 0 < min_size <= max_size.")

    current_extent = get_max_abs_extent(points)
    if current_extent <= 1e-14:
        raise ValueError("Curve has near-zero extent; cannot scale.")

    target_extent = rng.uniform(min_size, max_size)
    scale_factor = target_extent / current_extent
    return points * scale_factor

def chaikin_subdivide(points: Array, num_iters: int = 2, closed: bool = True) -> Array:
    points = np.asarray(points, dtype=np.float64)
    pts = points.copy()

    for _ in range(num_iters):
        new_pts = []
        n = len(pts)
        last = n if closed else n - 1

        for i in range(last):
            p = pts[i]
            q = pts[(i + 1) % n]
            new_pts.append(0.75 * p + 0.25 * q)
            new_pts.append(0.25 * p + 0.75 * q)

        if not closed:
            new_pts = [pts[0]] + new_pts + [pts[-1]]

        pts = np.asarray(new_pts, dtype=np.float64)

    return pts


def resample_polyline_uniform(points: Array, num_points: int, closed: bool = True) -> Array:
    points = np.asarray(points, dtype=np.float64)

    if closed:
        pts = np.vstack([points, points[:1]])
    else:
        pts = points

    seg = pts[1:] - pts[:-1]
    seg_len = np.linalg.norm(seg, axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg_len)])
    total = cum[-1]

    if total <= 1e-12:
        raise ValueError("Degenerate polyline.")

    if closed:
        targets = np.linspace(0.0, total, num_points, endpoint=False)
    else:
        targets = np.linspace(0.0, total, num_points, endpoint=True)

    out = []
    j = 0
    for t in targets:
        while j + 1 < len(cum) and cum[j + 1] < t:
            j += 1
        j = min(j, len(seg_len) - 1)
        local_len = seg_len[j]
        alpha = 0.0 if local_len <= 1e-12 else (t - cum[j]) / local_len
        p = (1.0 - alpha) * pts[j] + alpha * pts[j + 1]
        out.append(p)

    return np.asarray(out, dtype=np.float64)


def generate_random_piecewise_curve(
    num_points: int,
    rng: np.random.Generator,
    num_vertices_range: tuple[int, int] = (8, 14),
    radius_range: tuple[float, float] = (0.4, 1.0),
    angle_jitter: float = 0.35,
    smooth_iters_range: tuple[int, int] = (1, 3),
    closed: bool = True,
) -> Array:
    m = int(rng.integers(num_vertices_range[0], num_vertices_range[1] + 1))

    base_angles = np.linspace(0.0, 2.0 * np.pi, m, endpoint=False)
    angle_noise = rng.uniform(-angle_jitter, angle_jitter, size=m)
    angles = np.sort(base_angles + angle_noise)

    radii = rng.uniform(radius_range[0], radius_range[1], size=m)
    verts = np.stack([radii * np.cos(angles), radii * np.sin(angles)], axis=1)

    smooth_iters = int(rng.integers(smooth_iters_range[0], smooth_iters_range[1] + 1))
    pts = chaikin_subdivide(verts, num_iters=smooth_iters, closed=closed)
    pts = resample_polyline_uniform(pts, num_points=num_points, closed=closed)
    pts = center_curve(pts)

    return pts


def warp_curve_sampling(
    points: Array,
    rng: np.random.Generator,
    strength: float = 0.18,
    closed: bool = True,
) -> Array:
    """
    Reparameterization-like warp that makes point spacing non-uniform,
    while preserving the underlying curve geometry.
    """
    points = np.asarray(points, dtype=np.float64)
    n = len(points)

    if closed:
        pts = np.vstack([points, points[:1]])
    else:
        pts = points

    seg = pts[1:] - pts[:-1]
    seg_len = np.linalg.norm(seg, axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg_len)])
    total = cum[-1]

    if total <= 1e-12:
        raise ValueError("Degenerate curve in warp_curve_sampling.")

    s = cum / total

    phases = rng.uniform(0.0, 2.0 * np.pi, size=2)
    warped = s + strength * (
        0.6 * np.sin(2.0 * np.pi * s + phases[0]) +
        0.4 * np.sin(4.0 * np.pi * s + phases[1])
    )

    warped = warped - warped.min()
    warped = warped / max(warped.max(), 1e-12)
    warped = np.maximum.accumulate(warped)

    if closed:
        targets = np.linspace(0.0, 1.0, n, endpoint=False)
    else:
        targets = np.linspace(0.0, 1.0, n, endpoint=True)

    sampled = []
    j = 0
    for t in targets:
        while j + 1 < len(warped) and warped[j + 1] < t:
            j += 1
        j = min(j, len(seg_len) - 1)

        denom = warped[j + 1] - warped[j] if j + 1 < len(warped) else 0.0
        alpha = 0.0 if denom <= 1e-12 else (t - warped[j]) / denom
        p = (1.0 - alpha) * pts[j] + alpha * pts[j + 1]
        sampled.append(p)

    return np.asarray(sampled, dtype=np.float64)
def generate_random_simple_fourier_curve(
    t: Array,
    max_freq: int = 5,
    scale: float = 1.0,
    decay_power: float = 2.0,
    rng: np.random.Generator | None = None,
    max_tries: int = 300,
    center: bool = True,
    fit_to_canvas: bool = True,
    min_size: float = 0.25,
    max_size: float = 0.95,
    enforce_simple: bool = False,
) -> tuple[Array, BasisExpansionCurveCoeffs]:
    """
    Generate a random non-self-intersecting Fourier curve.

    Pipeline:
    1. Sample expressive Fourier curve
    2. Center it if requested
    3. Reject if self-intersecting
    4. Fit into a bounded canvas with random retained size

    Returns:
        points: shape (N, 2)
        coeffs: sampled coefficients
    """
    t = np.asarray(t, dtype=np.float64)
    if t.ndim != 1:
        raise ValueError("t must be a 1D array.")

    if rng is None:
        rng = np.random.default_rng()

    if not enforce_simple:
        points, coeffs = generate_random_fourier_curve(
            t=t,
            max_freq=max_freq,
            scale=scale,
            decay_power=decay_power,
            rng=rng,
        )

        if center:
            points = center_curve(points)

        if fit_to_canvas:
            points = fit_curve_to_canvas_with_random_size(
                points,
                rng=rng,
                min_size=min_size,
                max_size=max_size,
            )

        return points, coeffs

    for _ in range(max_tries):
        points, coeffs = generate_random_fourier_curve(
            t=t,
            max_freq=max_freq,
            scale=scale,
            decay_power=decay_power,
            rng=rng,
        )

        if center:
            points = center_curve(points)

        if curve_has_self_intersections(points, closed=True):
            continue

        if fit_to_canvas:
            points = fit_curve_to_canvas_with_random_size(
                points,
                rng=rng,
                min_size=min_size,
                max_size=max_size,
            )

        return points, coeffs

    raise RuntimeError(
        f"Failed to generate a non-self-intersecting Fourier curve after {max_tries} attempts."
    )