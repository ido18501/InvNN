from __future__ import annotations

import numpy as np

from utils.curve_generation import BasisExpansionCurveCoeffs

Array = np.ndarray


def _validate_t(t: Array) -> Array:
    t = np.asarray(t, dtype=np.float64)
    if t.ndim != 1:
        raise ValueError("t must be a 1D array.")
    return t


def _det2(a: Array, b: Array) -> Array:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.shape != b.shape or a.shape[-1] != 2:
        raise ValueError("a and b must have matching shape (..., 2).")
    return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]


def evaluate_fourier_curve_and_parameter_derivatives(
    t: Array,
    coeffs: BasisExpansionCurveCoeffs,
) -> tuple[Array, Array, Array, Array]:
    """
    Exact evaluation for the Fourier basis used in curve_generation.py.

    x(t), y(t) are represented as:
        sum_k a_k cos(kt) + b_k sin(kt)

    Returns:
        points:      (N, 2)
        first_dt:    (N, 2)
        second_dt:   (N, 2)
        third_dt:    (N, 2)
    """
    t = _validate_t(t)

    x_coeffs = np.asarray(coeffs.x_coeffs, dtype=np.float64)
    y_coeffs = np.asarray(coeffs.y_coeffs, dtype=np.float64)
    if x_coeffs.shape != y_coeffs.shape:
        raise ValueError("x_coeffs and y_coeffs must have identical shape.")
    if x_coeffs.ndim != 1 or len(x_coeffs) % 2 != 0:
        raise ValueError("Fourier coefficient arrays must be 1D with even length.")

    max_freq = len(x_coeffs) // 2
    points = np.zeros((len(t), 2), dtype=np.float64)
    first_dt = np.zeros((len(t), 2), dtype=np.float64)
    second_dt = np.zeros((len(t), 2), dtype=np.float64)
    third_dt = np.zeros((len(t), 2), dtype=np.float64)

    for k in range(1, max_freq + 1):
        xc = x_coeffs[2 * (k - 1)]
        xs = x_coeffs[2 * (k - 1) + 1]
        yc = y_coeffs[2 * (k - 1)]
        ys = y_coeffs[2 * (k - 1) + 1]

        ck = np.cos(k * t)
        sk = np.sin(k * t)

        points[:, 0] += xc * ck + xs * sk
        points[:, 1] += yc * ck + ys * sk

        first_dt[:, 0] += -k * xc * sk + k * xs * ck
        first_dt[:, 1] += -k * yc * sk + k * ys * ck

        second_dt[:, 0] += -(k ** 2) * xc * ck - (k ** 2) * xs * sk
        second_dt[:, 1] += -(k ** 2) * yc * ck - (k ** 2) * ys * sk

        third_dt[:, 0] += (k ** 3) * xc * sk - (k ** 3) * xs * ck
        third_dt[:, 1] += (k ** 3) * yc * sk - (k ** 3) * ys * ck

    return points, first_dt, second_dt, third_dt


def _compute_sigma_and_sigma_prime(
    family: str,
    first_dt: Array,
    second_dt: Array,
    third_dt: Array,
    eps: float = 1e-12,
) -> tuple[Array, Array]:
    """
    Compute geometric arc-length density sigma = ds/dt and its derivative sigma_t
    for the requested geometry family.

    Supported families here:
      - euclidean
      - similarity
      - equi_affine

    Notes:
    - We use absolute values to keep sigma >= 0 for similarity/equi-affine.
    - For equi-affine, the curve must be nondegenerate: det(gamma_t, gamma_tt) != 0.
    """
    first_dt = np.asarray(first_dt, dtype=np.float64)
    second_dt = np.asarray(second_dt, dtype=np.float64)
    third_dt = np.asarray(third_dt, dtype=np.float64)

    if first_dt.shape != second_dt.shape or first_dt.shape != third_dt.shape or first_dt.shape[-1] != 2:
        raise ValueError("All derivative arrays must have matching shape (..., 2).")

    family = str(family).lower()

    speed = np.linalg.norm(first_dt, axis=-1)
    speed_safe = np.clip(speed, eps, None)

    dot12 = np.sum(first_dt * second_dt, axis=-1)
    speed_t = dot12 / speed_safe

    det12 = _det2(first_dt, second_dt)
    det13 = _det2(first_dt, third_dt)

    if family == "euclidean":
        sigma = speed_safe
        sigma_t = speed_t
        return sigma, sigma_t

    if family == "similarity":
        abs_det12 = np.abs(det12)
        sign_det12 = np.sign(det12)

        sigma = abs_det12 / (speed_safe ** 2)
        sigma = np.clip(sigma, eps, None)

        sigma_t = (
            sign_det12 * det13 / (speed_safe ** 2)
            - 2.0 * abs_det12 * speed_t / (speed_safe ** 3)
        )
        return sigma, sigma_t

    if family == "equi_affine":
        abs_det12 = np.abs(det12)
        sign_det12 = np.sign(det12)

        abs_det12_safe = np.clip(abs_det12, eps, None)
        sigma = abs_det12_safe ** (1.0 / 3.0)

        sigma_t = sign_det12 * det13 / (3.0 * (abs_det12_safe ** (2.0 / 3.0)))
        return sigma, sigma_t

    if family == "affine":
        raise NotImplementedError(
            "Full planar affine arc-length is not implemented here. "
            "Use 'equi_affine', 'euclidean', or 'similarity'."
        )

    raise ValueError(f"Unsupported family: {family}")


def compute_arc_length_derivatives_from_parameter_derivatives(
    family: str,
    first_dt: Array,
    second_dt: Array,
    third_dt: Array,
    eps: float = 1e-12,
) -> tuple[Array, Array]:
    """
    Convert exact parameter derivatives to geometric arc-length derivatives.

    For ds = sigma(t) dt:

        d gamma / ds = gamma_t / sigma
        d^2 gamma / ds^2 = gamma_tt / sigma^2 - gamma_t * sigma_t / sigma^3
    """
    sigma, sigma_t = _compute_sigma_and_sigma_prime(
        family=family,
        first_dt=first_dt,
        second_dt=second_dt,
        third_dt=third_dt,
        eps=eps,
    )

    sigma = np.asarray(sigma, dtype=np.float64)[..., None]
    sigma_t = np.asarray(sigma_t, dtype=np.float64)[..., None]

    first_ds = first_dt / sigma
    second_ds = second_dt / (sigma ** 2) - first_dt * sigma_t / (sigma ** 3)
    return first_ds, second_ds


def compute_fourier_arc_length_derivatives(
    t: Array,
    coeffs: BasisExpansionCurveCoeffs,
    family: str = "euclidean",
) -> tuple[Array, Array, Array]:
    points, first_dt, second_dt, third_dt = evaluate_fourier_curve_and_parameter_derivatives(t, coeffs)
    first_ds, second_ds = compute_arc_length_derivatives_from_parameter_derivatives(
        family=family,
        first_dt=first_dt,
        second_dt=second_dt,
        third_dt=third_dt,
    )
    return points, first_ds, second_ds


def compute_single_anchor_fourier_arc_length_derivatives(
    t_value: float,
    coeffs: BasisExpansionCurveCoeffs,
    family: str = "euclidean",
) -> tuple[Array, Array, Array]:
    t = np.asarray([t_value], dtype=np.float64)
    points, first_ds, second_ds = compute_fourier_arc_length_derivatives(
        t=t,
        coeffs=coeffs,
        family=family,
    )
    return points[0], first_ds[0], second_ds[0]


# -----------------------------
# Backward-compatible euclidean wrappers
# -----------------------------
def compute_euclidean_arc_length_derivatives_from_parameter_derivatives(
    first_dt: Array,
    second_dt: Array,
    third_dt: Array | None = None,
    eps: float = 1e-12,
) -> tuple[Array, Array]:
    """
    Backward-compatible Euclidean wrapper.

    third_dt is accepted for API consistency but not needed explicitly beyond sigma_t logic;
    if omitted, we still use the direct Euclidean formulas.
    """
    first_dt = np.asarray(first_dt, dtype=np.float64)
    second_dt = np.asarray(second_dt, dtype=np.float64)

    if third_dt is None:
        speed = np.linalg.norm(first_dt, axis=-1, keepdims=True)
        speed_safe = np.clip(speed, eps, None)
        first_ds = first_dt / speed_safe

        dot = np.sum(first_dt * second_dt, axis=-1, keepdims=True)
        second_ds = second_dt / (speed_safe ** 2) - first_dt * dot / (speed_safe ** 4)
        return first_ds, second_ds

    return compute_arc_length_derivatives_from_parameter_derivatives(
        family="euclidean",
        first_dt=first_dt,
        second_dt=second_dt,
        third_dt=third_dt,
        eps=eps,
    )


def compute_fourier_euclidean_arc_length_derivatives(
    t: Array,
    coeffs: BasisExpansionCurveCoeffs,
) -> tuple[Array, Array, Array]:
    return compute_fourier_arc_length_derivatives(t=t, coeffs=coeffs, family="euclidean")