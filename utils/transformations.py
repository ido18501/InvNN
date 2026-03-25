from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np


Array = np.ndarray
TransformFamily = Literal["euclidean", "similarity", "equi_affine", "affine"]


@dataclass
class Transformation2D:
    """
    Represents a 2D transformation of the form

        x -> A x + b

    for the currently supported families:
        - euclidean
        - similarity
        - equi_affine
        - affine
    """
    family: TransformFamily
    A: Array            # shape (2, 2)
    b: Array            # shape (2,)
    params: dict


def _ensure_rng(rng: np.random.Generator | None) -> np.random.Generator:
    if rng is None:
        return np.random.default_rng()
    return rng


def _rotation_matrix(theta: float) -> Array:
    c = np.cos(theta)
    s = np.sin(theta)
    return np.array([[c, -s], [s, c]], dtype=np.float64)


def _reflection_matrix() -> Array:
    """
    Reflection across the x-axis.
    Combined with rotations, this can produce any reflection.
    """
    return np.array([[1.0, 0.0], [0.0, -1.0]], dtype=np.float64)


def _sample_rotation(
    rng: np.random.Generator,
    angle_range: tuple[float, float] = (-np.pi, np.pi),
) -> Array:
    theta = rng.uniform(angle_range[0], angle_range[1])
    return _rotation_matrix(theta)


def _sample_translation(
    rng: np.random.Generator,
    translation_range: tuple[float, float] = (-0.25, 0.25),
) -> Array:
    low, high = translation_range
    return rng.uniform(low, high, size=2).astype(np.float64)


def _sample_log_uniform(
    rng: np.random.Generator,
    value_range: tuple[float, float],
) -> float:
    low, high = value_range
    if low <= 0 or high <= 0:
        raise ValueError("Log-uniform sampling requires positive bounds.")
    log_low = np.log(low)
    log_high = np.log(high)
    return float(np.exp(rng.uniform(log_low, log_high)))


def _sample_signed_shear(
    rng: np.random.Generator,
    shear_range: tuple[float, float],
) -> float:
    low, high = shear_range
    return float(rng.uniform(low, high))


def _sample_euclidean(
    rng: np.random.Generator,
    angle_range: tuple[float, float],
    allow_reflection: bool,
    translation_range: tuple[float, float],
) -> Transformation2D:
    R = _sample_rotation(rng, angle_range=angle_range)

    reflected = False
    if allow_reflection and rng.random() < 0.5:
        R = R @ _reflection_matrix()
        reflected = True

    b = _sample_translation(rng, translation_range=translation_range)

    return Transformation2D(
        family="euclidean",
        A=R,
        b=b,
        params={
            "reflected": reflected,
        },
    )


def _sample_similarity(
    rng: np.random.Generator,
    angle_range: tuple[float, float],
    allow_reflection: bool,
    translation_range: tuple[float, float],
    scale_range: tuple[float, float],
) -> Transformation2D:
    base = _sample_euclidean(
        rng=rng,
        angle_range=angle_range,
        allow_reflection=allow_reflection,
        translation_range=translation_range,
    )

    scale = _sample_log_uniform(rng, scale_range)
    A = scale * base.A

    return Transformation2D(
        family="similarity",
        A=A,
        b=base.b,
        params={
            **base.params,
            "scale": scale,
        },
    )


def _sample_affine(
    rng: np.random.Generator,
    angle_range: tuple[float, float],
    allow_reflection: bool,
    translation_range: tuple[float, float],
    scale_x_range: tuple[float, float],
    scale_y_range: tuple[float, float],
    shear_range: tuple[float, float],
) -> Transformation2D:
    """
    Structured affine sampling:
        A = R1 @ Sh @ D @ R2
    where:
        R1, R2 rotations
        D diagonal scaling
        Sh shear
    """
    R1 = _sample_rotation(rng, angle_range=angle_range)
    R2 = _sample_rotation(rng, angle_range=angle_range)

    sx = _sample_log_uniform(rng, scale_x_range)
    sy = _sample_log_uniform(rng, scale_y_range)

    D = np.array([[sx, 0.0], [0.0, sy]], dtype=np.float64)

    sh = _sample_signed_shear(rng, shear_range)
    Sh = np.array([[1.0, sh], [0.0, 1.0]], dtype=np.float64)

    A = R1 @ Sh @ D @ R2

    reflected = False
    if allow_reflection and rng.random() < 0.5:
        A = A @ _reflection_matrix()
        reflected = True

    b = _sample_translation(rng, translation_range=translation_range)

    return Transformation2D(
        family="affine",
        A=A,
        b=b,
        params={
            "scale_x": sx,
            "scale_y": sy,
            "shear": sh,
            "reflected": reflected,
            "det": float(np.linalg.det(A)),
        },
    )


def _sample_equi_affine(
    rng: np.random.Generator,
    angle_range: tuple[float, float],
    allow_reflection: bool,
    translation_range: tuple[float, float],
    anisotropy_range: tuple[float, float],
    shear_range: tuple[float, float],
) -> Transformation2D:
    """
    Structured equi-affine sampling:
        A = R1 @ Sh @ D @ R2
    with det(D) = 1, det(Sh)=1, det(R1)=det(R2)=1.
    Optionally multiply by a reflection, giving determinant -1.

    D is parameterized as diag(a, 1/a), with a > 0.
    """
    R1 = _sample_rotation(rng, angle_range=angle_range)
    R2 = _sample_rotation(rng, angle_range=angle_range)

    a = _sample_log_uniform(rng, anisotropy_range)
    D = np.array([[a, 0.0], [0.0, 1.0 / a]], dtype=np.float64)

    sh = _sample_signed_shear(rng, shear_range)
    Sh = np.array([[1.0, sh], [0.0, 1.0]], dtype=np.float64)

    A = R1 @ Sh @ D @ R2

    reflected = False
    if allow_reflection and rng.random() < 0.5:
        A = A @ _reflection_matrix()
        reflected = True

    b = _sample_translation(rng, translation_range=translation_range)

    return Transformation2D(
        family="equi_affine",
        A=A,
        b=b,
        params={
            "anisotropy": a,
            "shear": sh,
            "reflected": reflected,
            "det": float(np.linalg.det(A)),
        },
    )


def sample_transformation(
    family: TransformFamily,
    rng: np.random.Generator | None = None,
    angle_range: tuple[float, float] = (-np.pi, np.pi),
    allow_reflection: bool = True,
    translation_range: tuple[float, float] = (-0.25, 0.25),
    scale_range: tuple[float, float] = (0.7, 1.4),
    scale_x_range: tuple[float, float] = (0.7, 1.4),
    scale_y_range: tuple[float, float] = (0.7, 1.4),
    anisotropy_range: tuple[float, float] = (0.7, 1.4),
    shear_range: tuple[float, float] = (-0.35, 0.35),
) -> Transformation2D:
    """
    Sample a random 2D transformation from a supported family.

    Args:
        family:
            One of:
                "euclidean"
                "similarity"
                "equi_affine"
                "affine"
        rng:
            Optional NumPy random generator.
        angle_range:
            Range for sampled rotation angles.
        allow_reflection:
            Whether reflections are allowed.
        translation_range:
            Uniform range for each translation coordinate.
        scale_range:
            For similarity transforms.
        scale_x_range, scale_y_range:
            For full affine transforms.
        anisotropy_range:
            For equi-affine diagonal term diag(a, 1/a).
        shear_range:
            Shear strength range.

    Returns:
        Transformation2D
    """
    rng = _ensure_rng(rng)

    if family == "euclidean":
        return _sample_euclidean(
            rng=rng,
            angle_range=angle_range,
            allow_reflection=allow_reflection,
            translation_range=translation_range,
        )

    if family == "similarity":
        return _sample_similarity(
            rng=rng,
            angle_range=angle_range,
            allow_reflection=allow_reflection,
            translation_range=translation_range,
            scale_range=scale_range,
        )

    if family == "equi_affine":
        return _sample_equi_affine(
            rng=rng,
            angle_range=angle_range,
            allow_reflection=allow_reflection,
            translation_range=translation_range,
            anisotropy_range=anisotropy_range,
            shear_range=shear_range,
        )

    if family == "affine":
        return _sample_affine(
            rng=rng,
            angle_range=angle_range,
            allow_reflection=allow_reflection,
            translation_range=translation_range,
            scale_x_range=scale_x_range,
            scale_y_range=scale_y_range,
            shear_range=shear_range,
        )

    raise ValueError(f"Unsupported transformation family: {family}")


def apply_transformation(
    points: Array,
    transform: Transformation2D,
) -> Array:
    """
    Apply x -> A x + b to points.

    Args:
        points:
            Array of shape (N, 2)
        transform:
            Transformation2D

    Returns:
        transformed_points:
            Array of shape (N, 2)
    """
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("points must have shape (N, 2).")

    return points @ transform.A.T + transform.b.reshape(1, 2)


def apply_linear_part(
    vectors: Array,
    transform: Transformation2D,
    normalize: bool = False,
    eps: float = 1e-12,
) -> Array:
    """
    Apply only the linear part A to vectors.

    This is what you will later want for tangent transport
    in Euclidean / similarity / affine / equi-affine families.

    Args:
        vectors:
            Array of shape (..., 2)
        transform:
            Transformation2D
        normalize:
            Whether to renormalize the output vectors.
        eps:
            Numerical stability constant for normalization.

    Returns:
        transformed vectors of same shape
    """
    vectors = np.asarray(vectors, dtype=np.float64)
    if vectors.shape[-1] != 2:
        raise ValueError("vectors must have last dimension 2.")

    out = vectors @ transform.A.T

    if normalize:
        norms = np.linalg.norm(out, axis=-1, keepdims=True)
        out = out / np.clip(norms, eps, None)

    return out


def transform_tangent_vectors(
    tangents: Array,
    transform: Transformation2D,
    eps: float = 1e-12,
) -> Array:
    """
    Transform tangent directions under the currently supported families.

    Since tangents are directions, we apply the linear part A and renormalize.

    Args:
        tangents:
            Array of shape (..., 2)
        transform:
            Transformation2D

    Returns:
        transformed tangent directions of same shape
    """
    return apply_linear_part(
        tangents,
        transform=transform,
        normalize=True,
        eps=eps,
    )
