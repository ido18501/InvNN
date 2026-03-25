from __future__ import annotations

import numpy as np

from utils.curve_generation import BasisExpansionCurveCoeffs

Array = np.ndarray


def _validate_t(t: Array) -> Array:
    t = np.asarray(t, dtype=np.float64)
    if t.ndim != 1:
        raise ValueError("t must be a 1D array.")
    return t


def evaluate_fourier_curve_and_parameter_derivatives(
    t: Array,
    coeffs: BasisExpansionCurveCoeffs,
) -> tuple[Array, Array, Array]:
    """
    Exact evaluation for the Fourier basis used in curve_generation.py.

    x(t), y(t) are represented as:
        sum_k a_k cos(kt) + b_k sin(kt)

    Returns:
        points:      (N, 2)
        first_dt:    (N, 2)
        second_dt:   (N, 2)
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

    return points, first_dt, second_dt


def compute_euclidean_arc_length_derivatives_from_parameter_derivatives(
    first_dt: Array,
    second_dt: Array,
    eps: float = 1e-12,
) -> tuple[Array, Array]:
    """
    Convert exact parameter derivatives to Euclidean arc-length derivatives.

    For gamma(t), with v = ||gamma'(t)||,

        d gamma / ds = gamma'(t) / v
        d^2 gamma / ds^2 = gamma''(t) / v^2 - gamma'(t)<gamma'(t),gamma''(t)>/v^4
    """
    first_dt = np.asarray(first_dt, dtype=np.float64)
    second_dt = np.asarray(second_dt, dtype=np.float64)
    if first_dt.shape != second_dt.shape or first_dt.shape[-1] != 2:
        raise ValueError("first_dt and second_dt must have matching shape (..., 2).")

    speed = np.linalg.norm(first_dt, axis=-1, keepdims=True)
    speed_safe = np.clip(speed, eps, None)
    first_ds = first_dt / speed_safe

    dot = np.sum(first_dt * second_dt, axis=-1, keepdims=True)
    second_ds = second_dt / (speed_safe ** 2) - first_dt * dot / (speed_safe ** 4)
    return first_ds, second_ds


def compute_fourier_euclidean_arc_length_derivatives(
    t: Array,
    coeffs: BasisExpansionCurveCoeffs,
) -> tuple[Array, Array, Array]:
    points, first_dt, second_dt = evaluate_fourier_curve_and_parameter_derivatives(t, coeffs)
    first_ds, second_ds = compute_euclidean_arc_length_derivatives_from_parameter_derivatives(
        first_dt=first_dt,
        second_dt=second_dt,
    )
    return points, first_ds, second_ds


def compute_single_anchor_fourier_arc_length_derivatives(
    t_value: float,
    coeffs: BasisExpansionCurveCoeffs,
) -> tuple[Array, Array, Array]:
    t = np.asarray([t_value], dtype=np.float64)
    points, first_ds, second_ds = compute_fourier_euclidean_arc_length_derivatives(t=t, coeffs=coeffs)
    return points[0], first_ds[0], second_ds[0]
