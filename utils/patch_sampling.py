from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np


Array = np.ndarray
PatchSamplingMode = Literal[
    "uniform_symmetric",
    "random_warp_symmetric",
    "intrinsic_ordered_stencil",
]

def _sample_random_monotone_profile(
    x: Array,
    rng: np.random.Generator,
    warp_strength: float = 0.35,
    num_harmonics: int = 3,
) -> Array:
    """
    Sample a random monotone map on [-1, 1] by building a positive derivative
    profile and integrating it.

    Input:
        x: sorted coordinates in [-1, 1]

    Output:
        warped_x: sorted coordinates in [-1, 1], strictly increasing
    """
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 1:
        raise ValueError("x must be 1D.")
    if len(x) < 2:
        return x.copy()

    # Build positive derivative profile g(x) > 0
    g = np.ones_like(x, dtype=np.float64)

    for k in range(1, num_harmonics + 1):
        a = rng.normal(0.0, warp_strength / (k ** 1.5))
        b = rng.normal(0.0, warp_strength / (k ** 1.5))
        g += a * np.sin(k * np.pi * x) + b * np.cos(k * np.pi * x)

    # Make derivative strictly positive
    g = np.maximum(g, 1e-3)

    # Integrate numerically
    dx = np.diff(x)
    avg_g = 0.5 * (g[:-1] + g[1:])
    y = np.concatenate([[0.0], np.cumsum(avg_g * dx)])

    # Normalize back to [-1, 1]
    y = y - y[0]
    total = y[-1]
    if total <= 1e-12:
        return x.copy()

    y = y / total
    y = 2.0 * y - 1.0
    return y

@dataclass
class CurvePatchSample:
    """
    A local ordered patch around a fixed center point on a curve.

    Attributes:
        points:
            Sampled patch points in global coordinates, shape (patch_size, 2).
        centered_points:
            Same patch, centered at the center point, shape (patch_size, 2).
        center_point:
            The fixed center point p, shape (2,).
        center_index:
            Index of the center point in the dense sampled curve.
        sample_indices:
            Indices in the dense curve used to build the patch, shape (patch_size,).
        relative_offsets:
            Integer offsets relative to center_index, shape (patch_size,).
            For example [-4, -2, 0, 3, 5].
        mode:
            Sampling mode used.
    """
    points: Array
    centered_points: Array
    center_point: Array
    center_index: int
    sample_indices: Array
    relative_offsets: Array
    mode: str


def _ensure_rng(rng: np.random.Generator | None) -> np.random.Generator:
    if rng is None:
        return np.random.default_rng()
    return rng


def _validate_curve_points(curve_points: Array) -> Array:
    curve_points = np.asarray(curve_points, dtype=np.float64)
    if curve_points.ndim != 2 or curve_points.shape[1] != 2:
        raise ValueError("curve_points must have shape (N, 2).")
    if len(curve_points) < 3:
        raise ValueError("curve_points must contain at least 3 points.")
    return curve_points


def _validate_patch_size(patch_size: int) -> int:
    if patch_size < 3:
        raise ValueError("patch_size must be at least 3.")
    if patch_size % 2 == 0:
        raise ValueError("patch_size must be odd so the center point is included.")
    return patch_size


def _compute_max_half_width(
    num_points: int,
    center_index: int,
    closed: bool,
) -> int:
    if closed:
        return (num_points - 1) // 2
    return min(center_index, num_points - 1 - center_index)


def _wrap_or_clip_indices(
    indices: Array,
    num_points: int,
    closed: bool,
) -> Array:
    if closed:
        return np.mod(indices, num_points).astype(np.int64)
    return np.clip(indices, 0, num_points - 1).astype(np.int64)


def _make_uniform_symmetric_offsets(
    half_width: int,
    patch_size: int,
) -> Array:
    """
    Make symmetric integer offsets including 0.
    Example:
        half_width=4, patch_size=5 -> [-4, -2, 0, 2, 4]
    """
    return np.rint(
        np.linspace(-half_width, half_width, patch_size, endpoint=True)
    ).astype(np.int64)

def _make_intrinsic_ordered_offsets(
    patch_size: int,
) -> Array:
    """
    Deterministic intrinsic stencil:
        [-r, ..., -1, 0, 1, ..., r]
    where r = (patch_size - 1) // 2
    """
    if patch_size < 3 or patch_size % 2 == 0:
        raise ValueError("patch_size must be odd and >= 3.")

    r = patch_size // 2
    return np.arange(-r, r + 1, dtype=np.int64)
def _make_random_warp_symmetric_offsets(
    half_width: int,
    patch_size: int,
    rng: np.random.Generator,
    warp_strength: float = 0.35,
    num_harmonics: int = 3,
    max_tries: int = 20,
) -> Array:
    """
    Sample symmetric-looking local offsets by drawing a fresh random monotone
    warp each time.

    The center point is always 0. Left/right spacing is allowed to vary
    irregularly, but offsets remain ordered, unique, local, and include 0.
    """
    if patch_size < 3 or patch_size % 2 == 0:
        raise ValueError("patch_size must be odd and >= 3.")

    if half_width < patch_size // 2:
        # not enough room for distinct ordered offsets
        return _make_uniform_symmetric_offsets(
            half_width=half_width,
            patch_size=patch_size,
        )

    base = np.linspace(-1.0, 1.0, patch_size, endpoint=True)
    center_pos = patch_size // 2

    for _ in range(max_tries):
        warped = _sample_random_monotone_profile(
            base,
            rng=rng,
            warp_strength=warp_strength,
            num_harmonics=num_harmonics,
        )

        # Force exact center to map to 0, preserve monotonicity by shifting sides
        warped[center_pos] = 0.0

        # Rescale each side independently to fill [-1,0] and [0,1]
        left = warped[:center_pos]
        right = warped[center_pos + 1:]

        if len(left) > 0:
            left_min = left.min()
            if abs(left_min) < 1e-12:
                continue
            left = left / abs(left_min)
            left = np.clip(left, -1.0, -1e-6)

        if len(right) > 0:
            right_max = right.max()
            if abs(right_max) < 1e-12:
                continue
            right = right / abs(right_max)
            right = np.clip(right, 1e-6, 1.0)

        warped = np.concatenate([left, np.array([0.0]), right])

        offsets = warped * float(half_width)
        int_offsets = np.rint(offsets).astype(np.int64)
        int_offsets[center_pos] = 0

        # Enforce strict order and sign
        for i in range(center_pos - 1, -1, -1):
            max_allowed = -1 if i == center_pos - 1 else int_offsets[i + 1] - 1
            int_offsets[i] = min(int_offsets[i], max_allowed)

        for i in range(center_pos + 1, patch_size):
            min_allowed = 1 if i == center_pos + 1 else int_offsets[i - 1] + 1
            int_offsets[i] = max(int_offsets[i], min_allowed)

        int_offsets = np.clip(int_offsets, -half_width, half_width)

        if len(np.unique(int_offsets)) == patch_size and 0 in int_offsets:
            return int_offsets

    return _make_uniform_symmetric_offsets(
        half_width=half_width,
        patch_size=patch_size,
    )


def sample_patch_around_index(
    curve_points: Array,
    center_index: int,
    patch_size: int,
    half_width: int,
    mode: PatchSamplingMode = "random_warp_symmetric",
    closed: bool = True,
    rng: np.random.Generator | None = None,
    jitter_fraction: float = 0.25,
) -> CurvePatchSample:
    """
    Sample an ordered local patch around a fixed center point.

    Args:
        curve_points:
            Dense sampled curve points, shape (N, 2).
        center_index:
            Index of the fixed center point p.
        patch_size:
            Number of sampled patch points. Must be odd and >= 3.
        half_width:
            Patch radius in index units: sampled neighbors come from within
            [center_index - half_width, center_index + half_width] along the curve.
        mode:
            "uniform_symmetric", "random_warp_symmetric", or "intrinsic_ordered_stencil"
        closed:
            Whether the curve should wrap around cyclically.
        rng:
            Optional NumPy random generator.
        jitter_fraction:
            Warp/jitter strength used by random_warp_symmetric mode.

    Returns:
        CurvePatchSample
    """
    curve_points = _validate_curve_points(curve_points)
    patch_size = _validate_patch_size(patch_size)
    rng = _ensure_rng(rng)

    num_points = len(curve_points)
    if not (0 <= center_index < num_points):
        raise ValueError("center_index is out of range.")
    if mode != "intrinsic_ordered_stencil":
        if half_width < 1:
            raise ValueError("half_width must be at least 1.")

        max_half_width = _compute_max_half_width(
            num_points=num_points,
            center_index=center_index,
            closed=closed,
        )
        if half_width > max_half_width:
            raise ValueError(
                f"half_width={half_width} is too large for this curve / center_index. "
                f"Maximum allowed is {max_half_width}."
            )

        if patch_size > 2 * half_width + 1:
            raise ValueError(
                "patch_size is too large for the requested half_width. "
                "Need patch_size <= 2 * half_width + 1 so distinct ordered samples are possible."
            )




    if mode == "intrinsic_ordered_stencil":
        relative_offsets = _make_intrinsic_ordered_offsets(
            patch_size=patch_size,
        )

    elif mode == "uniform_symmetric":
        relative_offsets = _make_uniform_symmetric_offsets(
            half_width=half_width,
            patch_size=patch_size,
        )

    elif mode == "random_warp_symmetric":
        relative_offsets = _make_random_warp_symmetric_offsets(
            half_width=half_width,
            patch_size=patch_size,
            rng=rng,
            warp_strength=jitter_fraction,
        )

    else:
        raise ValueError(f"Unsupported mode: {mode}")

    sample_indices = center_index + relative_offsets
    sample_indices = _wrap_or_clip_indices(
        sample_indices,
        num_points=num_points,
        closed=closed,
    )

    points = curve_points[sample_indices]
    center_point = curve_points[center_index]
    centered_points = points - center_point.reshape(1, 2)

    return CurvePatchSample(
        points=points,
        centered_points=centered_points,
        center_point=center_point,
        center_index=int(center_index),
        sample_indices=sample_indices.astype(np.int64),
        relative_offsets=relative_offsets.astype(np.int64),
        mode=mode,
    )


def sample_random_patch(
    curve_points: Array,
    patch_size: int,
    half_width: int,
    mode: PatchSamplingMode = "random_warp_symmetric",
    closed: bool = True,
    rng: np.random.Generator | None = None,
    jitter_fraction: float = 0.25,
    valid_center_margin: int | None = None,
) -> CurvePatchSample:
    """
    Sample a random patch by first choosing a center index, then sampling around it.

    This is useful later for dataset generation.

    Args:
        curve_points:
            Dense sampled curve points, shape (N, 2).
        patch_size:
            Odd patch size >= 3.
        half_width:
            Neighborhood radius in index units.
        mode:
            Sampling mode.
        closed:
            Whether the curve is closed.
        rng:
            Optional NumPy random generator.
        jitter_fraction:
            Jitter strength for jittered mode.
        valid_center_margin:
            For open curves, optionally keep random centers away from boundaries.

    Returns:
        CurvePatchSample
    """
    curve_points = _validate_curve_points(curve_points)
    patch_size = _validate_patch_size(patch_size)
    rng = _ensure_rng(rng)

    num_points = len(curve_points)

    if closed:
        center_index = int(rng.integers(0, num_points))
    else:
        if valid_center_margin is None:
            valid_center_margin = half_width
        left = valid_center_margin
        right = num_points - valid_center_margin
        if left >= right:
            raise ValueError("No valid center indices remain for the requested margin.")
        center_index = int(rng.integers(left, right))

    return sample_patch_around_index(
        curve_points=curve_points,
        center_index=center_index,
        patch_size=patch_size,
        half_width=half_width,
        mode=mode,
        closed=closed,
        rng=rng,
        jitter_fraction=jitter_fraction,
    )
