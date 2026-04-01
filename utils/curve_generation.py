from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import math
import numpy as np

Array = np.ndarray
BasisFunction = Callable[[Array], Array]


@dataclass
class BasisExpansionCurveCoeffs:
    x_coeffs: Array
    y_coeffs: Array


def generate_random_basis_expansion_coeffs(
    num_basis_functions: int,
    scale: float = 1.0,
    coeff_std: Array | None = None,
    rng: np.random.Generator | None = None,
) -> BasisExpansionCurveCoeffs:
    if num_basis_functions < 1:
        raise ValueError("num_basis_functions must be at least 1")
    if scale <= 0:
        raise ValueError("scale must be positive")
    if rng is None:
        rng = np.random.default_rng()

    if coeff_std is None:
        coeff_std = np.full(num_basis_functions, scale, dtype=np.float64)
    else:
        coeff_std = np.asarray(coeff_std, dtype=np.float64)
        if coeff_std.shape != (num_basis_functions,):
            raise ValueError("coeff_std must have shape (num_basis_functions,)")
        if np.any(coeff_std <= 0):
            raise ValueError("coeff_std must be positive")

    x_coeffs = rng.normal(0.0, coeff_std, size=num_basis_functions)
    y_coeffs = rng.normal(0.0, coeff_std, size=num_basis_functions)
    return BasisExpansionCurveCoeffs(x_coeffs=x_coeffs, y_coeffs=y_coeffs)


def make_fourier_basis_functions(max_freq: int) -> list[BasisFunction]:
    if max_freq < 1:
        raise ValueError("max_freq must be at least 1")
    basis_functions: list[BasisFunction] = []
    for k in range(1, max_freq + 1):
        basis_functions.append(lambda t, k=k: np.cos(k * t))
        basis_functions.append(lambda t, k=k: np.sin(k * t))
    return basis_functions


def make_fourier_coeff_std(
    max_freq: int,
    scale: float = 1.0,
    decay_power: float = 1.65,
) -> Array:
    if max_freq < 1:
        raise ValueError("max_freq must be at least 1")
    if scale <= 0:
        raise ValueError("scale must be positive")
    if decay_power <= 0:
        raise ValueError("decay_power must be positive")

    stds: list[float] = []
    for k in range(1, max_freq + 1):
        s = scale / (k ** decay_power)
        stds.extend([s, s])
    return np.asarray(stds, dtype=np.float64)


def evaluate_basis_expansion_curve(
    t: Array,
    basis_functions: Sequence[BasisFunction],
    coeffs: BasisExpansionCurveCoeffs,
) -> Array:
    t = np.asarray(t, dtype=np.float64)
    if t.ndim != 1:
        raise ValueError("t must be 1D")
    basis_matrix = np.stack([phi(t) for phi in basis_functions], axis=1)
    x = basis_matrix @ coeffs.x_coeffs
    y = basis_matrix @ coeffs.y_coeffs
    return np.stack([x, y], axis=1)


def generate_random_fourier_curve(
    t: Array,
    max_freq: int = 7,
    scale: float = 1.0,
    decay_power: float = 1.65,
    rng: np.random.Generator | None = None,
) -> tuple[Array, BasisExpansionCurveCoeffs]:
    if rng is None:
        rng = np.random.default_rng()
    basis_functions = make_fourier_basis_functions(max_freq)
    coeff_std = make_fourier_coeff_std(max_freq=max_freq, scale=scale, decay_power=decay_power)
    coeffs = generate_random_basis_expansion_coeffs(
        num_basis_functions=len(basis_functions),
        coeff_std=coeff_std,
        rng=rng,
    )
    points = evaluate_basis_expansion_curve(t, basis_functions, coeffs)
    return points, coeffs


# ---------- geometry helpers ----------

def center_curve(points: Array) -> Array:
    points = np.asarray(points, dtype=np.float64)
    return points - points.mean(axis=0, keepdims=True)


def get_max_abs_extent(points: Array) -> float:
    points = np.asarray(points, dtype=np.float64)
    return float(np.max(np.abs(points)))


def fit_curve_to_canvas_with_random_size(
    points: Array,
    rng: np.random.Generator | None = None,
    min_size: float = 0.30,
    max_size: float = 0.90,
) -> Array:
    points = np.asarray(points, dtype=np.float64)
    if rng is None:
        rng = np.random.default_rng()
    current_extent = get_max_abs_extent(points)
    if current_extent <= 1e-14:
        raise ValueError("curve has near-zero extent")
    target_extent = float(rng.uniform(min_size, max_size))
    return points * (target_extent / current_extent)


def warp_curve_sampling(
    points: Array,
    rng: np.random.Generator,
    strength: float = 0.18,
    closed: bool = True,
) -> Array:
    points = np.asarray(points, dtype=np.float64)
    n = len(points)
    pts = np.vstack([points, points[:1]]) if closed else points
    seg = pts[1:] - pts[:-1]
    seg_len = np.linalg.norm(seg, axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg_len)])
    total = cum[-1]
    if total <= 1e-12:
        raise ValueError("degenerate curve in warp_curve_sampling")

    s = cum / total
    phases = rng.uniform(0.0, 2.0 * np.pi, size=2)
    warped = s + strength * (
        0.6 * np.sin(2.0 * np.pi * s + phases[0])
        + 0.4 * np.sin(4.0 * np.pi * s + phases[1])
    )
    warped = warped - warped.min()
    warped = warped / max(warped.max(), 1e-12)
    warped = np.maximum.accumulate(warped)

    targets = np.linspace(0.0, 1.0, n, endpoint=not closed)
    sampled = []
    j = 0
    for tval in targets:
        while j + 1 < len(warped) and warped[j + 1] < tval:
            j += 1
        j = min(j, len(seg_len) - 1)
        denom = warped[j + 1] - warped[j] if j + 1 < len(warped) else 0.0
        alpha = 0.0 if denom <= 1e-12 else (tval - warped[j]) / denom
        sampled.append((1.0 - alpha) * pts[j] + alpha * pts[j + 1])
    return np.asarray(sampled, dtype=np.float64)


def resample_polyline_uniform(points: Array, num_points: int, closed: bool = True) -> Array:
    points = np.asarray(points, dtype=np.float64)
    pts = np.vstack([points, points[:1]]) if closed else points
    seg = pts[1:] - pts[:-1]
    seg_len = np.linalg.norm(seg, axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg_len)])
    total = cum[-1]
    if total <= 1e-12:
        raise ValueError("degenerate polyline")
    targets = np.linspace(0.0, total, num_points, endpoint=not closed)
    out = []
    j = 0
    for tval in targets:
        while j + 1 < len(cum) and cum[j + 1] < tval:
            j += 1
        j = min(j, len(seg_len) - 1)
        local_len = seg_len[j]
        alpha = 0.0 if local_len <= 1e-12 else (tval - cum[j]) / local_len
        out.append((1.0 - alpha) * pts[j] + alpha * pts[j + 1])
    return np.asarray(out, dtype=np.float64)


def _segment_intersection_mask(points: Array, closed: bool = True) -> Array:
    pts = np.asarray(points, dtype=np.float64)
    if closed:
        p1 = pts
        p2 = np.roll(pts, -1, axis=0)
    else:
        p1 = pts[:-1]
        p2 = pts[1:]
    nseg = len(p1)

    minx = np.minimum(p1[:, 0], p2[:, 0])
    maxx = np.maximum(p1[:, 0], p2[:, 0])
    miny = np.minimum(p1[:, 1], p2[:, 1])
    maxy = np.maximum(p1[:, 1], p2[:, 1])
    bbox = (
        (minx[:, None] <= maxx[None, :])
        & (maxx[:, None] >= minx[None, :])
        & (miny[:, None] <= maxy[None, :])
        & (maxy[:, None] >= miny[None, :])
    )

    idx = np.arange(nseg)
    adjacent = np.abs(idx[:, None] - idx[None, :]) <= 1
    if closed:
        adjacent |= ((idx[:, None] == 0) & (idx[None, :] == nseg - 1))
        adjacent |= ((idx[:, None] == nseg - 1) & (idx[None, :] == 0))
    mask = np.triu(bbox & ~adjacent, k=1)
    return mask


def curve_has_self_intersections(points: Array, closed: bool = True, eps: float = 1e-12) -> bool:
    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError("points must have shape (N,2)")
    if len(pts) < 4:
        return False

    if closed:
        p1 = pts
        p2 = np.roll(pts, -1, axis=0)
    else:
        p1 = pts[:-1]
        p2 = pts[1:]

    cand = _segment_intersection_mask(pts, closed=closed)
    ii, jj = np.where(cand)
    if len(ii) == 0:
        return False

    a = p1[ii]
    b = p2[ii]
    c = p1[jj]
    d = p2[jj]

    def orient(u, v, w):
        return (v[:, 0] - u[:, 0]) * (w[:, 1] - u[:, 1]) - (v[:, 1] - u[:, 1]) * (w[:, 0] - u[:, 0])

    o1 = orient(a, b, c)
    o2 = orient(a, b, d)
    o3 = orient(c, d, a)
    o4 = orient(c, d, b)

    proper = (((o1 > eps) & (o2 < -eps)) | ((o1 < -eps) & (o2 > eps))) & (
        ((o3 > eps) & (o4 < -eps)) | ((o3 < -eps) & (o4 > eps))
    )
    if np.any(proper):
        return True

    # Conservative collinear check only on the few nearly-degenerate candidates.
    near = np.where((np.abs(o1) <= eps) | (np.abs(o2) <= eps) | (np.abs(o3) <= eps) | (np.abs(o4) <= eps))[0]
    for k in near.tolist():
        aa, bb, cc, dd = a[k], b[k], c[k], d[k]

        def on_segment(u, v, w):
            return (
                min(u[0], v[0]) - eps <= w[0] <= max(u[0], v[0]) + eps
                and min(u[1], v[1]) - eps <= w[1] <= max(u[1], v[1]) + eps
            )

        oo1 = (bb[0] - aa[0]) * (cc[1] - aa[1]) - (bb[1] - aa[1]) * (cc[0] - aa[0])
        oo2 = (bb[0] - aa[0]) * (dd[1] - aa[1]) - (bb[1] - aa[1]) * (dd[0] - aa[0])
        oo3 = (dd[0] - cc[0]) * (aa[1] - cc[1]) - (dd[1] - cc[1]) * (aa[0] - cc[0])
        oo4 = (dd[0] - cc[0]) * (bb[1] - cc[1]) - (dd[1] - cc[1]) * (bb[0] - cc[0])
        if abs(oo1) <= eps and on_segment(aa, bb, cc):
            return True
        if abs(oo2) <= eps and on_segment(aa, bb, dd):
            return True
        if abs(oo3) <= eps and on_segment(cc, dd, aa):
            return True
        if abs(oo4) <= eps and on_segment(cc, dd, bb):
            return True
    return False


def _simple_closed_screen(points: Array, intersection_check_points: int) -> bool:
    pts = np.asarray(points, dtype=np.float64)
    # Only reject clearly unusable curves beyond self-intersection.
    extent = get_max_abs_extent(pts)
    if extent <= 1e-8:
        return False
    seg = np.roll(pts, -1, axis=0) - pts
    seg_len = np.linalg.norm(seg, axis=1)
    if np.any(seg_len <= 1e-8):
        return False
    test_pts = resample_polyline_uniform(pts, num_points=intersection_check_points, closed=True)
    return not curve_has_self_intersections(test_pts, closed=True)


def generate_random_simple_fourier_curve(
    t: Array,
    max_freq: int = 7,
    scale: float = 1.0,
    decay_power: float = 1.65,
    rng: np.random.Generator | None = None,
    max_tries: int = 200,
    center: bool = True,
    fit_to_canvas: bool = True,
    min_size: float = 0.30,
    max_size: float = 0.90,
    enforce_simple: bool = True,
    intersection_check_points: int = 320,
) -> tuple[Array, BasisExpansionCurveCoeffs]:
    t = np.asarray(t, dtype=np.float64)
    if t.ndim != 1:
        raise ValueError("t must be 1D")
    if rng is None:
        rng = np.random.default_rng()

    for _ in range(max_tries if enforce_simple else 1):
        points, coeffs = generate_random_fourier_curve(
            t=t,
            max_freq=max_freq,
            scale=scale,
            decay_power=decay_power,
            rng=rng,
        )
        if center:
            points = center_curve(points)
        if enforce_simple and not _simple_closed_screen(points, intersection_check_points=intersection_check_points):
            continue
        if fit_to_canvas:
            points = fit_curve_to_canvas_with_random_size(points, rng=rng, min_size=min_size, max_size=max_size)
        return points, coeffs

    raise RuntimeError("Failed to generate a simple non-self-intersecting Fourier curve")


def sample_smooth_monotone_periodic_reparameterization(
    num_points: int,
    rng: np.random.Generator,
    strength: float = 0.35,
    num_harmonics: int = 3,
    min_density: float = 0.35,
    max_density: float = 3.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns:
        u_grid: uniform target parameter in [0, 2pi), shape (N,)
        t_warped: warped source parameter values in [0, 2pi), shape (N,)
    Interpretation:
        We sample uniformly in u, and evaluate the original curve at t = phi^{-1}(u).
    """
    if num_points < 8:
        raise ValueError("num_points must be at least 8")

    u_grid = np.linspace(0.0, 2.0 * np.pi, num_points, endpoint=False, dtype=np.float64)

    # positive periodic density g(t)
    g = np.ones_like(u_grid)
    for k in range(1, num_harmonics + 1):
        a = rng.normal(0.0, strength / (k ** 1.5))
        b = rng.normal(0.0, strength / (k ** 1.5))
        g += a * np.sin(k * u_grid) + b * np.cos(k * u_grid)

    # enforce positivity and avoid extreme distortions
    g = np.clip(g, min_density, max_density)

    # integrate to build monotone map phi
    dt = 2.0 * np.pi / num_points
    phi = np.cumsum(g) * dt
    phi = phi - phi[0]
    phi = phi / phi[-1]
    phi = phi * (2.0 * np.pi)

    # invert phi numerically: for uniform u_grid, find t_warped with phi(t)=u
    # since phi is monotone, np.interp is enough
    t_base = u_grid
    t_warped = np.interp(u_grid, phi, t_base)

    return u_grid, t_warped

def sample_bounded_stride_indices(
    n_in: int,
    n_out: int,
    rng: np.random.Generator,
    jitter: float = 0.2,
) -> np.ndarray:
    if n_out <= 0:
        raise ValueError("n_out must be positive")
    if n_in <= 0:
        raise ValueError("n_in must be positive")
    if n_out >= n_in:
        return np.arange(n_in, dtype=np.int64)

    stride = n_in / n_out
    min_step = max(1, int(np.floor(stride * (1.0 - jitter))))
    max_step = max(min_step, int(np.ceil(stride * (1.0 + jitter))))

    idxs = [0]
    cur = 0

    while True:
        remaining_slots = n_out - len(idxs)
        if remaining_slots == 0:
            break

        remaining_points = (n_in - 1) - cur

        # latest step allowed so we can still finish
        latest = remaining_points - max(0, remaining_slots - 1) * min_step
        # earliest step allowed so we do not finish too early
        earliest = max(min_step, remaining_points - max(0, remaining_slots - 1) * max_step)

        lo = max(min_step, earliest)
        hi = min(max_step, latest)

        if lo > hi:
            # deterministic fallback
            step = max(1, remaining_points // remaining_slots)
        else:
            step = int(rng.integers(lo, hi + 1))

        cur += step
        idxs.append(cur)

    idxs = np.asarray(idxs, dtype=np.int64)

    if len(idxs) != n_out:
        raise RuntimeError(f"Expected {n_out} indices, got {len(idxs)}")

    if np.any(np.diff(idxs) <= 0):
        raise RuntimeError("Downsample indices are not strictly increasing")

    if idxs[0] < 0 or idxs[-1] >= n_in:
        raise RuntimeError("Downsample indices out of bounds")

    return idxs


def downsample_with_bounded_stride(
    points: Array,
    num_out: int,
    rng: np.random.Generator,
    jitter: float = 0.2,
) -> tuple[Array, np.ndarray]:
    idxs = sample_bounded_stride_indices(len(points), num_out, rng=rng, jitter=jitter)
    return points[idxs], idxs

def generate_random_reparameterized_fourier_curve(
    num_points: int,
    max_freq: int = 7,
    scale: float = 1.0,
    decay_power: float = 1.65,
    rng: np.random.Generator | None = None,
    center: bool = True,
    fit_to_canvas: bool = True,
    min_size: float = 0.30,
    max_size: float = 0.90,
    reparam_strength: float = 0.35,
    reparam_num_harmonics: int = 3,
    reparam_min_density: float = 0.35,
    reparam_max_density: float = 3.0,
    max_tries: int = 1000,
    enforce_simple: bool = True,
    intersection_check_points: int = 320,
    downsample_to_points: int | None = None,
    downsample_jitter: float = 0.2,
) -> tuple[Array, BasisExpansionCurveCoeffs, Array, Array]:
    """
    Returns:
        curve_points: sampled points x(t_warped)
        coeffs
        u_grid: uniform target parameter
        t_warped: actual source parameter values used for evaluation
    """
    if rng is None:
        rng = np.random.default_rng()

    basis_functions = make_fourier_basis_functions(max_freq)
    coeff_std = make_fourier_coeff_std(max_freq=max_freq, scale=scale, decay_power=decay_power)

    num_attempts = max_tries if enforce_simple else 1

    for _ in range(num_attempts):
        coeffs = generate_random_basis_expansion_coeffs(
            num_basis_functions=len(basis_functions),
            coeff_std=coeff_std,
            rng=rng,
        )

        # First check the underlying geometric curve on a dense uniform grid
        t_dense = np.linspace(0.0, 2.0 * np.pi, max(num_points, intersection_check_points), endpoint=False)
        dense_points = evaluate_basis_expansion_curve(t_dense, basis_functions, coeffs)

        if center:
            dense_points = center_curve(dense_points)

        if enforce_simple and not _simple_closed_screen(
            dense_points,
            intersection_check_points=intersection_check_points,
        ):
            continue

        # Now apply monotone reparameterization
        u_grid, t_warped = sample_smooth_monotone_periodic_reparameterization(
            num_points=num_points,
            rng=rng,
            strength=reparam_strength,
            num_harmonics=reparam_num_harmonics,
            min_density=reparam_min_density,
            max_density=reparam_max_density,
        )

        points = evaluate_basis_expansion_curve(t_warped, basis_functions, coeffs)
        if downsample_to_points is not None and downsample_to_points < len(points):
            idxs = sample_bounded_stride_indices(
                len(points),
                downsample_to_points,
                rng=rng,
                jitter=downsample_jitter,
            )
            points = points[idxs]
            u_grid = u_grid[idxs]
            t_warped = t_warped[idxs]

        if center:
            points = center_curve(points)

        # Optional safety check on the sampled polygon after reparameterization
        if enforce_simple and not _simple_closed_screen(
            points,
            intersection_check_points=min(intersection_check_points, num_points),
        ):
            continue

        if fit_to_canvas:
            points = fit_curve_to_canvas_with_random_size(
                points,
                rng=rng,
                min_size=min_size,
                max_size=max_size,
            )

        return points, coeffs, u_grid, t_warped

    raise RuntimeError("Failed to generate a simple non-self-intersecting reparameterized Fourier curve")

