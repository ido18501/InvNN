from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from datasets.tangent_tuple_generation import build_random_invariant_training_tuple
from utils.curve_generation import (
    BasisExpansionCurveCoeffs,
    fit_curve_to_canvas_with_random_size,
    generate_random_simple_fourier_curve,
    warp_curve_sampling,
)

Array = np.ndarray


@dataclass
class InvariantSampleTensors:
    anchor: torch.Tensor
    positive: torch.Tensor
    negatives: torch.Tensor
    transform_matrix: torch.Tensor
    family: str
    anchor_center_index: int
    negative_center_indices: torch.Tensor
    gt_first_anchor: torch.Tensor
    gt_second_anchor: torch.Tensor
    has_analytic_derivatives: torch.Tensor


class PregeneratedCurveBank:
    def __init__(self, bank_path: str | Path) -> None:
        bank_path = Path(bank_path)
        data = np.load(bank_path, allow_pickle=False)
        self.curve_points = np.asarray(data['curve_points'], dtype=np.float32)
        self.x_coeffs = np.asarray(data['x_coeffs'], dtype=np.float64) if 'x_coeffs' in data.files else None
        self.y_coeffs = np.asarray(data['y_coeffs'], dtype=np.float64) if 'y_coeffs' in data.files else None
        if 't_grid' in data.files:
            tg = np.asarray(data['t_grid'], dtype=np.float64)
            self.t_grid = tg
        else:
            self.t_grid = None

        if self.curve_points.ndim != 3 or self.curve_points.shape[-1] != 2:
            raise ValueError('curve_points must have shape (K, N, 2).')
        self.num_curves = self.curve_points.shape[0]

        if self.x_coeffs is not None and self.y_coeffs is not None:
            if self.x_coeffs.shape[0] != self.num_curves or self.y_coeffs.shape[0] != self.num_curves:
                raise ValueError('Coefficient banks must align with number of curves.')
            self.has_coeffs = True
        else:
            self.has_coeffs = False

    def __len__(self) -> int:
        return self.num_curves

    def get(self, idx: int) -> tuple[Array, BasisExpansionCurveCoeffs | None, Array | None]:
        curve = self.curve_points[idx].astype(np.float32)
        coeffs = None
        if self.has_coeffs:
            coeffs = BasisExpansionCurveCoeffs(
                x_coeffs=self.x_coeffs[idx].copy(),
                y_coeffs=self.y_coeffs[idx].copy(),
            )
        t_grid = None
        if self.t_grid is not None:
            if self.t_grid.ndim == 1:
                t_grid = self.t_grid.copy()
            else:
                t_grid = self.t_grid[idx].copy()
        return curve, coeffs, t_grid


class TangentDataset(Dataset):
    """
    Invariant-only dataset with two modes:

    - source='generated'    : on-the-fly Fourier curves
    - source='pregenerated' : load curves from a bank on disk

    The optimization signal never uses derivatives.
    Exact analytic Euclidean arc-length derivatives are returned only as diagnostics.
    """

    def __init__(
        self,
        *,
        length: int,
        family: str,
        source: str = 'generated',
        bank_path: str | None = None,
        num_curve_points: int = 300,
        fourier_max_freq: int = 5,
        fourier_scale: float = 0.9,
        fourier_decay_power: float = 2.0,
        curve_max_tries: int = 300,
        curve_min_size: float = 0.45,
        curve_max_size: float = 0.75,
        patch_size: int = 9,
        half_width: int = 12,
        half_width_range: tuple[int, int] | None = None,
        num_negatives: int = 8,
        negative_min_offset: int = 5,
        negative_max_offset: int = 25,
        negative_other_curve_fraction: float = 0.5,
        patch_mode: str = 'random_warp_symmetric',
        jitter_fraction: float = 0.25,
        closed: bool = True,
        transform_kwargs: dict[str, Any] | None = None,
        return_centered: bool = True,
        point_noise_std: float = 0.0,
        warp_sampling_prob: float = 0.7,
        warp_sampling_strength: float = 0.18,
        dtype: torch.dtype = torch.float32,
        seed: int | None = None,
    ) -> None:
        self.length = int(length)
        self.family = family
        self.source = source
        self.bank = PregeneratedCurveBank(bank_path) if source == 'pregenerated' else None
        if source == 'pregenerated' and self.bank is None:
            raise ValueError('bank_path is required when source="pregenerated".')

        self.num_curve_points = num_curve_points
        self.fourier_max_freq = fourier_max_freq
        self.fourier_scale = fourier_scale
        self.fourier_decay_power = fourier_decay_power
        self.curve_max_tries = curve_max_tries
        self.curve_min_size = curve_min_size
        self.curve_max_size = curve_max_size
        self.patch_size = patch_size
        self.half_width = half_width
        self.half_width_range = half_width_range
        self.num_negatives = num_negatives
        self.negative_min_offset = negative_min_offset
        self.negative_max_offset = negative_max_offset
        self.negative_other_curve_fraction = negative_other_curve_fraction
        self.patch_mode = patch_mode
        self.jitter_fraction = jitter_fraction
        self.closed = closed
        self.transform_kwargs = {} if transform_kwargs is None else dict(transform_kwargs)
        self.return_centered = return_centered
        self.point_noise_std = point_noise_std
        self.warp_sampling_prob = warp_sampling_prob
        self.warp_sampling_strength = warp_sampling_strength
        self.dtype = dtype
        self._base_seed = seed

    def __len__(self) -> int:
        return self.length

    def _make_rng(self, index: int) -> np.random.Generator:
        if self._base_seed is None:
            return np.random.default_rng()
        return np.random.default_rng(self._base_seed + index)

    def _sample_half_width(self, rng: np.random.Generator) -> int:
        if self.half_width_range is None:
            return self.half_width
        low, high = self.half_width_range
        return int(rng.integers(low, high + 1))

    def _generate_curve(self, rng: np.random.Generator) -> tuple[Array, BasisExpansionCurveCoeffs, Array]:
        t = np.linspace(0.0, 2.0 * np.pi, self.num_curve_points, endpoint=False)
        curve_points, coeffs = generate_random_simple_fourier_curve(
            t=t,
            max_freq=self.fourier_max_freq,
            scale=self.fourier_scale,
            decay_power=self.fourier_decay_power,
            rng=rng,
            max_tries=self.curve_max_tries,
            center=True,
            fit_to_canvas=True,
            min_size=self.curve_min_size,
            max_size=self.curve_max_size,
            enforce_simple=False,
        )
        if rng.random() < self.warp_sampling_prob:
            curve_points = warp_curve_sampling(
                curve_points,
                rng=rng,
                strength=self.warp_sampling_strength,
                closed=self.closed,
            )
        if self.point_noise_std > 0.0:
            curve_points = curve_points + rng.normal(0.0, self.point_noise_std, size=curve_points.shape)
            curve_points = fit_curve_to_canvas_with_random_size(
                curve_points,
                rng=rng,
                min_size=self.curve_min_size,
                max_size=self.curve_max_size,
            )
        return curve_points, coeffs, t

    def _load_curve(self, rng: np.random.Generator, idx: int) -> tuple[Array, BasisExpansionCurveCoeffs | None, Array | None]:
        assert self.bank is not None
        bank_idx = idx % len(self.bank)
        curve_points, coeffs, t_grid = self.bank.get(bank_idx)
        if rng.random() < self.warp_sampling_prob:
            curve_points = warp_curve_sampling(
                curve_points,
                rng=rng,
                strength=self.warp_sampling_strength,
                closed=self.closed,
            )
        return curve_points, coeffs, t_grid

    def _get_curve(self, rng: np.random.Generator, idx: int) -> tuple[Array, BasisExpansionCurveCoeffs | None, Array | None]:
        if self.source == 'generated':
            return self._generate_curve(rng)
        if self.source == 'pregenerated':
            return self._load_curve(rng, idx)
        raise ValueError(f'Unsupported source: {self.source}')

    def __getitem__(self, index: int) -> InvariantSampleTensors:
        rng = self._make_rng(index)
        curve_points, coeffs, t_grid = self._get_curve(rng, index)

        num_cross_curve_negatives = int(round(self.num_negatives * self.negative_other_curve_fraction))
        external_negative_curves: list[Array] = []
        for i in range(num_cross_curve_negatives):
            ext_curve, _, _ = self._get_curve(rng, index + 100000 + i)
            external_negative_curves.append(ext_curve)

        half_width = self._sample_half_width(rng)
        tuple_sample = build_random_invariant_training_tuple(
            curve_points=curve_points,
            coeffs=coeffs,
            t_grid=t_grid,
            transform_family=self.family,
            patch_size=self.patch_size,
            half_width=half_width,
            num_negatives=self.num_negatives,
            negative_min_offset=self.negative_min_offset,
            negative_max_offset=self.negative_max_offset,
            closed=self.closed,
            patch_mode=self.patch_mode,
            jitter_fraction=self.jitter_fraction,
            rng=rng,
            transform_kwargs=self.transform_kwargs,
            external_negative_curves=external_negative_curves if num_cross_curve_negatives > 0 else None,
            num_cross_curve_negatives=num_cross_curve_negatives,
        )

        anchor = torch.as_tensor(tuple_sample.anchor_patch, dtype=self.dtype)
        positive = torch.as_tensor(tuple_sample.positive_patch, dtype=self.dtype)
        negatives = torch.as_tensor(tuple_sample.negative_patches, dtype=self.dtype)
        transform_matrix = torch.as_tensor(tuple_sample.transform_matrix, dtype=self.dtype)
        gt_first_anchor = torch.as_tensor(tuple_sample.gt_first_anchor, dtype=self.dtype)
        gt_second_anchor = torch.as_tensor(tuple_sample.gt_second_anchor, dtype=self.dtype)
        has_analytic_derivatives = torch.tensor(tuple_sample.has_analytic_derivatives, dtype=torch.bool)

        if self.return_centered:
            anchor_center = anchor[self.patch_size // 2].clone()
            positive_center = positive[self.patch_size // 2].clone()
            anchor = anchor - anchor_center.unsqueeze(0)
            positive = positive - positive_center.unsqueeze(0)
            negatives = negatives - negatives[:, self.patch_size // 2, :].unsqueeze(1)

        return InvariantSampleTensors(
            anchor=anchor,
            positive=positive,
            negatives=negatives,
            transform_matrix=transform_matrix,
            family=tuple_sample.family,
            anchor_center_index=tuple_sample.anchor_center_index,
            negative_center_indices=torch.as_tensor(tuple_sample.negative_center_indices, dtype=torch.long),
            gt_first_anchor=gt_first_anchor,
            gt_second_anchor=gt_second_anchor,
            has_analytic_derivatives=has_analytic_derivatives,
        )
