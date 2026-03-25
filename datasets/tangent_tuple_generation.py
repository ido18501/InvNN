from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from utils.patch_sampling import sample_patch_around_index
from utils.transformations import sample_transformation, apply_transformation
from utils.derivatives import compute_euclidean_arc_length_derivatives

Array = np.ndarray


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


def _sample_negative_center_indices(
    num_points: int,
    anchor_center_index: int,
    num_negatives: int,
    min_offset: int,
    max_offset: int,
    closed: bool,
    rng: np.random.Generator,
    distance_weight_power: float = 2.5,
) -> Array:
    candidates: list[int] = []
    distances: list[int] = []
    for j in range(num_points):
        if j == anchor_center_index:
            continue
        d = abs(j - anchor_center_index)
        if closed:
            d = min(d, num_points - d)
        if min_offset <= d <= max_offset:
            candidates.append(j)
            distances.append(d)
    if len(candidates) < num_negatives:
        raise ValueError("Not enough valid negative center indices")

    candidates_arr = np.asarray(candidates, dtype=np.int64)
    distances_arr = np.asarray(distances, dtype=np.float64)
    # Hard negatives: prefer closer valid offsets, but still allow the whole band.
    weights = 1.0 / np.power(np.maximum(distances_arr, 1.0), distance_weight_power)
    weights = weights / weights.sum()
    chosen = rng.choice(candidates_arr, size=num_negatives, replace=False, p=weights)
    return np.asarray(chosen, dtype=np.int64)


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
    negative_distance_weight_power: float = 2.5,
) -> TangentTrainingTuple:
    if transform_kwargs is None:
        transform_kwargs = {}

    anchor_patch = sample_patch_around_index(
        curve_points=curve_points,
        center_index=anchor_center_index,
        patch_size=patch_size,
        half_width=half_width,
        mode=patch_mode,
        closed=closed,
        rng=rng,
        jitter_fraction=jitter_fraction,
    )

    transform = sample_transformation(family=transform_family, rng=rng, **transform_kwargs)
    positive_patch = apply_transformation(anchor_patch, transform)

    neg_idx = _sample_negative_center_indices(
        num_points=len(curve_points),
        anchor_center_index=anchor_center_index,
        num_negatives=num_negatives,
        min_offset=negative_min_offset,
        max_offset=negative_max_offset,
        closed=closed,
        rng=rng,
        distance_weight_power=negative_distance_weight_power,
    )

    negative_patches = []
    num_in_curve = num_negatives - int(num_cross_curve_negatives)
    for j in neg_idx[:num_in_curve]:
        neg_patch = sample_patch_around_index(
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

    if num_cross_curve_negatives > 0:
        if external_negative_curves is None or len(external_negative_curves) < num_cross_curve_negatives:
            raise ValueError("external_negative_curves missing for requested cross-curve negatives")
        for ext_curve in external_negative_curves[:num_cross_curve_negatives]:
            ext_center = int(rng.integers(0, len(ext_curve)))
            neg_patch = sample_patch_around_index(
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

    negative_patches = np.stack(negative_patches, axis=0)

    gt_first, gt_second, _ = compute_euclidean_arc_length_derivatives(
        curve_points=curve_points,
        anchor_index=anchor_center_index,
        dense_num_points=gt_dense_num_points,
    )

    return TangentTrainingTuple(
        family=transform_family,
        anchor_patch=anchor_patch.astype(np.float32),
        positive_patch=positive_patch.astype(np.float32),
        negative_patches=negative_patches.astype(np.float32),
        transform_matrix=np.asarray(transform.A, dtype=np.float32),
        anchor_center_index=int(anchor_center_index),
        negative_center_indices=neg_idx.astype(np.int64),
        gt_first_anchor=gt_first.astype(np.float32),
        gt_second_anchor=gt_second.astype(np.float32),
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
    negative_distance_weight_power: float = 2.5,
) -> TangentTrainingTuple:
    num_points = len(curve_points)
    valid_center_margin = half_width if not closed else 0
    if closed:
        anchor_center_index = int(rng.integers(0, num_points))
    else:
        left = valid_center_margin
        right = num_points - valid_center_margin
        if left >= right:
            raise ValueError("No valid center indices remain for the requested margin")
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
        negative_distance_weight_power=negative_distance_weight_power,
    )
