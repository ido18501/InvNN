from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from utils.patch_sampling import sample_patch_around_index
from utils.transformations import sample_transformation, apply_transformation

Array = np.ndarray

# Anchor patch: sample from the original curve using the requested patch mode
# -----------------------------
# Robust GT derivative helper
# -----------------------------
def _resample_closed_curve_uniform_arc_length(points: Array, num_points: int) -> Array:
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("points must have shape (N, 2)")
    if len(points) < 4:
        raise ValueError("Need at least 4 points")
    if num_points < 8:
        raise ValueError("num_points must be at least 8")

    extended = np.vstack([points, points[:1]])
    seg = np.linalg.norm(np.diff(extended, axis=0), axis=1)
    total = float(seg.sum())
    if total <= 1e-12:
        raise ValueError("Degenerate curve")

    cum = np.concatenate([[0.0], np.cumsum(seg)])
    targets = np.linspace(0.0, total, num_points, endpoint=False)

    out = np.empty((num_points, 2), dtype=np.float64)
    j = 0
    for i, s in enumerate(targets):
        while j + 1 < len(cum) and cum[j + 1] <= s:
            j += 1
        if j >= len(seg):
            j = len(seg) - 1
        local_len = seg[j]
        if local_len <= 1e-12:
            out[i] = extended[j]
        else:
            alpha = (s - cum[j]) / local_len
            out[i] = (1.0 - alpha) * extended[j] + alpha * extended[j + 1]
    return out


def _nearest_index(points: Array, query: Array) -> int:
    d2 = ((points - query.reshape(1, 2)) ** 2).sum(axis=1)
    return int(np.argmin(d2))


def _compute_gt_arc_length_derivatives(
    curve_points: Array,
    anchor_index: int,
    *,
    dense_num_points: int = 4096,
) -> tuple[Array, Array, Array]:
    """
    Try project helper first if it exists. Otherwise fall back to a dense
    deterministic arc-length resampling estimate.
    """
    try:
        from utils.derivatives import compute_euclidean_arc_length_derivatives  # type: ignore
        return compute_euclidean_arc_length_derivatives(
            curve_points=curve_points,
            anchor_index=anchor_index,
            dense_num_points=dense_num_points,
        )
    except Exception:
        pass



    curve_points = np.asarray(curve_points, dtype=np.float64)
    dense = _resample_closed_curve_uniform_arc_length(curve_points, num_points=dense_num_points)
    q = curve_points[anchor_index]
    k = _nearest_index(dense, q)

    extended = np.vstack([curve_points, curve_points[:1]])
    total_length = float(np.linalg.norm(np.diff(extended, axis=0), axis=1).sum())
    ds = total_length / dense_num_points

    prev_pt = dense[(k - 1) % dense_num_points]
    curr_pt = dense[k]
    next_pt = dense[(k + 1) % dense_num_points]

    first = (next_pt - prev_pt) / (2.0 * ds)
    first_norm = np.linalg.norm(first)
    if first_norm > 1e-12:
        first = first / first_norm

    second = (next_pt - 2.0 * curr_pt + prev_pt) / (ds ** 2)
    return first.astype(np.float64), second.astype(np.float64), curr_pt.astype(np.float64)


# -----------------------------
# Patch helpers
# -----------------------------
def _extract_patch_points(patch_sample) -> Array:
    """
    sample_patch_around_index returns CurvePatchSample in your project.
    This helper keeps us robust if it ever returns a raw ndarray.
    """
    if hasattr(patch_sample, "points"):
        return np.asarray(patch_sample.points, dtype=np.float64)
    return np.asarray(patch_sample, dtype=np.float64)


def _sample_patch_points(
    curve_points: Array,
    center_index: int,
    patch_size: int,
    half_width: int,
    mode: str,
    closed: bool,
    rng: np.random.Generator,
    jitter_fraction: float,
) -> Array:
    if mode == "intrinsic_ordered_stencil":
        sample = sample_patch_around_index(
            curve_points=curve_points,
            center_index=center_index,
            patch_size=patch_size,
            half_width=0,  # ignored
            mode=mode,
            closed=closed,
            rng=rng,
            jitter_fraction=0.0,  # ignored
        )
    else:
        sample = sample_patch_around_index(
            curve_points=curve_points,
            center_index=center_index,
            patch_size=patch_size,
            half_width=half_width,
            mode=mode,
            closed=closed,
            rng=rng,
            jitter_fraction=jitter_fraction,
        )
    return _extract_patch_points(sample)


# -----------------------------
# Geometry helpers for negatives
# -----------------------------
def _normalize(v: Array, eps: float = 1e-8) -> Array:
    n = np.linalg.norm(v)
    if n < eps:
        return np.zeros_like(v)
    return v / n


def _estimate_patch_geometry_signature_from_points(patch_points: Array) -> tuple[Array, float]:
    pts = np.asarray(patch_points, dtype=np.float64)
    k = len(pts)
    c = k // 2

    left_vec = pts[c] - pts[max(c - 1, 0)]
    right_vec = pts[min(c + 1, k - 1)] - pts[c]

    tangent = _normalize(left_vec + right_vec)

    l = _normalize(left_vec)
    r = _normalize(right_vec)
    curvature_proxy = float(np.linalg.norm(r - l))

    return tangent, curvature_proxy


def _distance_on_curve(i: int, j: int, num_points: int, closed: bool) -> int:
    d = abs(i - j)
    if closed:
        d = min(d, num_points - d)
    return int(d)


@dataclass
class TangentTrainingTuple:
    family: str
    anchor_patch: Array
    positive_patch: Array
    negative_patches: Array
    transform_matrix: Array
    anchor_center_index: int
    negative_center_indices: Array
    gt_first_anchor: Array
    gt_second_anchor: Array
    has_analytic_derivatives: bool = False


def _collect_band_candidates(
    curve_points: Array,
    anchor_center_index: int,
    closed: bool,
    patch_size: int,
    half_width: int,
    patch_mode: str,
    jitter_fraction: float,
    rng: np.random.Generator,
    band_min_offset: int,
    band_max_offset: int,
    min_curvature_gap: float,
    max_tangent_abs_dot: float,
    prefer_close_power: float,
) -> list[tuple[int, int, float]]:
    num_points = len(curve_points)

    anchor_patch = _sample_patch_points(
        curve_points=curve_points,
        center_index=anchor_center_index,
        patch_size=patch_size,
        half_width=half_width,
        mode=patch_mode,
        closed=closed,
        rng=rng,
        jitter_fraction=jitter_fraction,
    )
    anchor_tangent, anchor_curv = _estimate_patch_geometry_signature_from_points(anchor_patch)

    candidates: list[tuple[int, int, float]] = []

    for j in range(num_points):
        if j == anchor_center_index:
            continue

        d = _distance_on_curve(anchor_center_index, j, num_points, closed)
        if not (band_min_offset <= d <= band_max_offset):
            continue

        neg_patch = _sample_patch_points(
            curve_points=curve_points,
            center_index=j,
            patch_size=patch_size,
            half_width=half_width,
            mode=patch_mode,
            closed=closed,
            rng=rng,
            jitter_fraction=jitter_fraction,
        )
        neg_tangent, neg_curv = _estimate_patch_geometry_signature_from_points(neg_patch)

        tangent_abs_dot = float(abs(np.dot(anchor_tangent, neg_tangent)))
        curv_gap = float(abs(anchor_curv - neg_curv))

        # Gentle rejection: only reject if almost identical
        if tangent_abs_dot > max_tangent_abs_dot and curv_gap < min_curvature_gap:
            continue

        weight = 1.0 / (float(d) ** prefer_close_power)
        candidates.append((j, d, weight))

    return candidates


def _sample_from_candidates(
    candidates: list[tuple[int, int, float]],
    num_to_take: int,
    rng: np.random.Generator,
) -> list[int]:
    if num_to_take <= 0 or len(candidates) == 0:
        return []

    idxs = np.asarray([x[0] for x in candidates], dtype=np.int64)
    probs = np.asarray([x[2] for x in candidates], dtype=np.float64)
    probs = probs / probs.sum()

    k = min(num_to_take, len(idxs))
    chosen = rng.choice(idxs, size=k, replace=False, p=probs)
    return [int(x) for x in chosen]


def _sample_negative_center_indices(
    curve_points: Array,
    anchor_center_index: int,
    num_negatives: int,
    min_offset: int,
    max_offset: int,
    closed: bool,
    patch_size: int,
    half_width: int,
    patch_mode: str,
    jitter_fraction: float,
    rng: np.random.Generator,
    min_curvature_gap: float = 0.1,
    max_tangent_abs_dot: float = 0.9,
    prefer_close_power=0.6,
    close_band_min_offset = 20,
    close_band_max_offset = 75,
    mid_band_min_offset = 35,
    mid_band_max_offset = 90,
) -> Array:
    """
    Policy:
    - 3 close + 1 mid for 4 negatives
    - 2 close + 1 mid for 3 negatives
    - 1 close + 1 mid for 2 negatives
    - fallback stays strictly local in [4, 22]
    """
    if num_negatives <= 0:
        return np.zeros((0,), dtype=np.int64)

    close_lo = max(min_offset, close_band_min_offset)
    close_hi = min(max_offset, close_band_max_offset)

    mid_lo = max(min_offset, mid_band_min_offset)
    mid_hi = min(max_offset, mid_band_max_offset)

    if num_negatives >= 4:
        num_close = 3
        num_mid = num_negatives - num_close
    elif num_negatives == 3:
        num_close = 2
        num_mid = 1
    elif num_negatives == 2:
        num_close = 1
        num_mid = 1
    else:
        num_close = 1
        num_mid = 0

    close_candidates: list[tuple[int, int, float]] = []
    mid_candidates: list[tuple[int, int, float]] = []

    if close_lo <= close_hi and num_close > 0:
        close_candidates = _collect_band_candidates(
            curve_points=curve_points,
            anchor_center_index=anchor_center_index,
            closed=closed,
            patch_size=patch_size,
            half_width=half_width,
            patch_mode=patch_mode,
            jitter_fraction=jitter_fraction,
            rng=rng,
            band_min_offset=close_lo,
            band_max_offset=close_hi,
            min_curvature_gap=min_curvature_gap,
            max_tangent_abs_dot=max_tangent_abs_dot,
            prefer_close_power=prefer_close_power,
        )

    if mid_lo <= mid_hi and num_mid > 0:
        mid_candidates = _collect_band_candidates(
            curve_points=curve_points,
            anchor_center_index=anchor_center_index,
            closed=closed,
            patch_size=patch_size,
            half_width=half_width,
            patch_mode=patch_mode,
            jitter_fraction=jitter_fraction,
            rng=rng,
            band_min_offset=mid_lo,
            band_max_offset=mid_hi,
            min_curvature_gap=min_curvature_gap,
            max_tangent_abs_dot=max_tangent_abs_dot,
            prefer_close_power=prefer_close_power,
        )

    chosen: list[int] = []

    chosen.extend(_sample_from_candidates(close_candidates, num_close, rng))
    remaining = num_negatives - len(chosen)

    if remaining > 0 and len(mid_candidates) > 0:
        mid_candidates = [c for c in mid_candidates if c[0] not in chosen]
        chosen.extend(_sample_from_candidates(mid_candidates, min(num_mid, remaining), rng))
        remaining = num_negatives - len(chosen)

    if remaining > 0:
        leftovers = []
        leftovers.extend([c for c in close_candidates if c[0] not in chosen])
        leftovers.extend([c for c in mid_candidates if c[0] not in chosen])

        if len(leftovers) > 0:
            chosen.extend(_sample_from_candidates(leftovers, remaining, rng))
            remaining = num_negatives - len(chosen)

    # Final fallback: stay strictly local inside union [4, 22]
    if remaining > 0:
        relaxed_min = max(min_offset, close_band_min_offset)
        relaxed_max = min(max_offset, mid_band_max_offset)

        relaxed: list[tuple[int, int, float]] = []
        num_points = len(curve_points)

        for j in range(num_points):
            if j == anchor_center_index or j in chosen:
                continue
            d = _distance_on_curve(anchor_center_index, j, num_points, closed)
            if relaxed_min <= d <= relaxed_max:
                weight = 1.0 / (float(d) ** prefer_close_power)
                relaxed.append((j, d, weight))

        if len(relaxed) < remaining:
            raise ValueError("Not enough valid local negative center indices.")

        chosen.extend(_sample_from_candidates(relaxed, remaining, rng))

    return np.asarray(chosen[:num_negatives], dtype=np.int64)


def _compute_anchor_derivatives_with_optional_analytic(
    curve_points,
    anchor_index,
    *,
    family: str,
    coeffs=None,
    t_grid=None,
    dense_num_points=4096,
) -> tuple[Array, Array, bool]:
    """
    If Fourier coeffs + anchor parameter value are available, use exact analytic
    family-aware arc-length derivatives when supported.

    Otherwise fall back to a dense geometric estimate for debugging only.
    In fallback mode, has_analytic_derivatives is False.
    """
    if coeffs is not None and t_grid is not None:
        try:
            from utils.derivatives import compute_single_anchor_fourier_arc_length_derivatives

            t_grid = np.asarray(t_grid, dtype=np.float64)
            if t_grid.ndim == 1 and 0 <= anchor_index < len(t_grid):
                _, first, second = compute_single_anchor_fourier_arc_length_derivatives(
                    t_value=float(t_grid[anchor_index]),
                    coeffs=coeffs,
                    family=family,
                )
                return (
                    np.asarray(first, dtype=np.float64),
                    np.asarray(second, dtype=np.float64),
                    True,
                )
        except Exception:
            pass

    first, second, _ = _compute_gt_arc_length_derivatives(
        curve_points=curve_points,
        anchor_index=anchor_index,
        dense_num_points=dense_num_points,
    )
    return first, second, False

def build_tangent_training_tuple(
    curve_points: Array,
    transform_family: str,
    anchor_center_index: int,
    patch_size: int,
    half_width: int,
    num_negatives: int,
    negative_min_offset: int,
    negative_max_offset: int,
    closed: bool,
    patch_mode: str,
    jitter_fraction: float,
    rng: np.random.Generator,
    transform_kwargs: dict | None = None,
    external_negative_curves: list[Array] | None = None,
    num_cross_curve_negatives: int = 0,
    gt_dense_num_points: int = 4096,
    coeffs=None,
    t_grid: Array | None = None,
) -> TangentTrainingTuple:
    if transform_kwargs is None:
        transform_kwargs = {}

    # Anchor patch
    anchor_patch = _sample_patch_points(
        curve_points=curve_points,
        center_index=anchor_center_index,
        patch_size=patch_size,
        half_width=half_width,
        mode=patch_mode,
        closed=closed,
        rng=rng,
        jitter_fraction=jitter_fraction,
    )

    # Positive patch: transform whole curve, then resample patch
    transform = sample_transformation(
        family=transform_family,
        rng=rng,
        **transform_kwargs,
    )
    transformed_curve = apply_transformation(curve_points, transform)

    positive_patch = _sample_patch_points(
        curve_points=transformed_curve,
        center_index=anchor_center_index,
        patch_size=patch_size,
        half_width=half_width,
        mode=patch_mode,
        closed=closed,
        rng=rng,
        jitter_fraction=jitter_fraction,
    )

    neg_idx = _sample_negative_center_indices(
        curve_points=curve_points,
        anchor_center_index=anchor_center_index,
        num_negatives=num_negatives,
        min_offset=negative_min_offset,
        max_offset=negative_max_offset,
        closed=closed,
        patch_size=patch_size,
        half_width=half_width,
        patch_mode=patch_mode,
        jitter_fraction=jitter_fraction,
        rng=rng,
    )

    negative_patches = []
    num_cross_curve_negatives = int(num_cross_curve_negatives)
    num_in_curve = max(0, num_negatives - num_cross_curve_negatives)

    # Metadata: -1 means "this negative came from another curve"
    negative_center_indices = np.full((num_negatives,), -1, dtype=np.int64)

    for slot, j in enumerate(neg_idx[:num_in_curve]):
        neg_patch = _sample_patch_points(
            curve_points=curve_points,
            center_index=int(j),
            patch_size=patch_size,
            half_width=half_width,
            mode=patch_mode,
            closed=closed,
            rng=rng,
            jitter_fraction=jitter_fraction,
        )
        negative_patches.append(neg_patch)
        negative_center_indices[slot] = int(j)

    if num_cross_curve_negatives > 0:
        if external_negative_curves is None or len(external_negative_curves) < num_cross_curve_negatives:
            raise ValueError("external_negative_curves missing for requested cross-curve negatives.")

        for ext_curve in external_negative_curves[:num_cross_curve_negatives]:
            ext_center = int(rng.integers(0, len(ext_curve)))
            neg_patch = _sample_patch_points(
                curve_points=ext_curve,
                center_index=ext_center,
                patch_size=patch_size,
                half_width=half_width,
                mode=patch_mode,
                closed=closed,
                rng=rng,
                jitter_fraction=jitter_fraction,
            )
            negative_patches.append(neg_patch)

    if len(negative_patches) != num_negatives:
        raise ValueError(
            f"Expected {num_negatives} negative patches, got {len(negative_patches)}."
        )

    negative_patches = np.stack(negative_patches, axis=0)

    gt_first, gt_second, has_analytic_derivatives = _compute_anchor_derivatives_with_optional_analytic(
        curve_points=curve_points,
        anchor_index=anchor_center_index,
        family=transform_family,
        coeffs=coeffs,
        t_grid=t_grid,
        dense_num_points=gt_dense_num_points,
    )

    return TangentTrainingTuple(
        family=transform_family,
        anchor_patch=anchor_patch.astype(np.float32),
        positive_patch=positive_patch.astype(np.float32),
        negative_patches=negative_patches.astype(np.float32),
        transform_matrix=np.asarray(transform.A, dtype=np.float32),
        anchor_center_index=int(anchor_center_index),
        negative_center_indices=negative_center_indices,
        gt_first_anchor=gt_first.astype(np.float32),
        gt_second_anchor=gt_second.astype(np.float32),
        has_analytic_derivatives=has_analytic_derivatives,
    )

def build_random_tangent_training_tuple(
    curve_points: Array,
    transform_family: str,
    patch_size: int,
    half_width: int,
    num_negatives: int,
    negative_min_offset: int,
    negative_max_offset: int,
    closed: bool,
    patch_mode: str,
    jitter_fraction: float,
    rng: np.random.Generator,
    transform_kwargs: dict | None = None,
    external_negative_curves: list[Array] | None = None,
    num_cross_curve_negatives: int = 0,
    gt_dense_num_points: int = 4096,
    coeffs=None,
    t_grid: Array | None = None,
) -> TangentTrainingTuple:
    num_points = len(curve_points)
    if patch_mode == "intrinsic_ordered_stencil":
        margin = patch_size // 2
    else:
        margin = half_width

    valid_center_margin = margin if not closed else 0

    if closed:
        anchor_center_index = int(rng.integers(0, num_points))
    else:
        left = valid_center_margin
        right = num_points - valid_center_margin
        if left >= right:
            raise ValueError("No valid center indices remain for the requested margin.")
        anchor_center_index = int(rng.integers(left, right))

    return build_tangent_training_tuple(
        curve_points=curve_points,
        transform_family=transform_family,
        anchor_center_index=anchor_center_index,
        patch_size=patch_size,
        half_width=half_width,
        num_negatives=num_negatives,
        negative_min_offset=negative_min_offset,
        negative_max_offset=negative_max_offset,
        closed=closed,
        patch_mode=patch_mode,
        jitter_fraction=jitter_fraction,
        rng=rng,
        transform_kwargs=transform_kwargs,
        external_negative_curves=external_negative_curves,
        num_cross_curve_negatives=num_cross_curve_negatives,
        gt_dense_num_points=gt_dense_num_points,
        coeffs=coeffs,
        t_grid=t_grid,
    )

def build_random_invariant_training_tuple(
    curve_points: Array,
    coeffs=None,
    t_grid: Array | None = None,
    transform_family: str = "euclidean",
    patch_size: int = 9,
    half_width: int = 12,
    num_negatives: int = 8,
    negative_min_offset: int = 5,
    negative_max_offset: int = 25,
    closed: bool = True,
    patch_mode: str = "intrinsic_ordered_stencil",
    jitter_fraction: float = 0.25,
    rng: np.random.Generator | None = None,
    transform_kwargs: dict | None = None,
    external_negative_curves: list[Array] | None = None,
    num_cross_curve_negatives: int = 0,
    gt_dense_num_points: int = 4096,
) -> TangentTrainingTuple:
    """
    Backward-compatible entry point expected by datasets/tangent_dataset.py.
    Keeps the new tuple generation logic intact and only adapts the interface.
    """
    if rng is None:
        rng = np.random.default_rng()

    return build_random_tangent_training_tuple(
        curve_points=curve_points,
        transform_family=transform_family,
        patch_size=patch_size,
        half_width=half_width,
        num_negatives=num_negatives,
        negative_min_offset=negative_min_offset,
        negative_max_offset=negative_max_offset,
        closed=closed,
        patch_mode=patch_mode,
        jitter_fraction=jitter_fraction,
        rng=rng,
        transform_kwargs=transform_kwargs,
        external_negative_curves=external_negative_curves,
        num_cross_curve_negatives=num_cross_curve_negatives,
        gt_dense_num_points=gt_dense_num_points,
        coeffs=coeffs,
        t_grid=t_grid,
    )