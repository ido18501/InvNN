from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from datasets.tangent_dataset import PregeneratedCurveBank
from utils.derivatives import compute_single_anchor_fourier_arc_length_derivatives
from utils.patch_sampling import sample_patch_around_index


@dataclass
class SecondOrderSample:
    patch: torch.Tensor                  # [K,2], centered
    target_second: torch.Tensor         # [2]
    center_index: int
    relative_offsets: torch.Tensor      # [K]
    curve_idx: int


class SecondOrderCurveDataset(Dataset):
    """
    Supervised dataset for learning a local stencil w(X) such that:

        sum_j w_j * X_j  ~=  d²x/ds²  at the patch center

    Important design choices:
    - Uses ONLY PregeneratedCurveBank for data access.
    - Requires analytic Fourier coeffs + t_grid in the bank, because the target
      second derivative is part of the actual training signal now.
    - Samples local centered intrinsic patches from closed curves.
    """

    def __init__(
        self,
        bank_path: str | Path,
        *,
        length: int,
        family: str = "euclidean",
        patch_size: int = 9,
        patch_mode: str = "intrinsic_ordered_stencil",
        half_width: int = 0,
        closed: bool = True,
        return_centered: bool = True,
        seed: int | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        self.bank = PregeneratedCurveBank(bank_path)
        self.length = int(length)
        self.family = str(family)
        self.patch_size = int(patch_size)
        self.patch_mode = str(patch_mode)
        self.half_width = int(half_width)
        self.closed = bool(closed)
        self.return_centered = bool(return_centered)
        self.seed = seed
        self.dtype = dtype

        if self.patch_size < 3 or self.patch_size % 2 == 0:
            raise ValueError("patch_size must be odd and >= 3.")

        if self.patch_mode == "intrinsic_ordered_stencil":
            self.half_width = 0

        if not self.bank.has_coeffs:
            raise ValueError(
                "This experiment requires analytic targets, but the bank does not "
                "contain Fourier coefficients."
            )
        if self.bank.t_grid is None:
            raise ValueError(
                "This experiment requires t_grid in the bank in order to evaluate "
                "analytic arc-length second derivatives."
            )

    def __len__(self) -> int:
        return self.length

    def _make_rng(self, idx: int) -> np.random.Generator:
        if self.seed is None:
            return np.random.default_rng()
        return np.random.default_rng(self.seed + idx)

    def __getitem__(self, idx: int) -> SecondOrderSample:
        rng = self._make_rng(idx)
        curve_idx = idx % len(self.bank)

        curve_points, coeffs, t_grid = self.bank.get(curve_idx)
        if coeffs is None or t_grid is None:
            raise RuntimeError("Missing coeffs or t_grid for analytic target computation.")

        n = len(curve_points)
        center_index = int(rng.integers(0, n))

        patch_sample = sample_patch_around_index(
            curve_points=curve_points,
            center_index=center_index,
            patch_size=self.patch_size,
            half_width=self.half_width,
            mode=self.patch_mode,
            closed=self.closed,
            rng=rng,
            jitter_fraction=0.0,
        )

        patch = np.asarray(patch_sample.points, dtype=np.float32)
        if self.return_centered:
            patch = patch - patch[self.patch_size // 2 : self.patch_size // 2 + 1]

        _, _, second = compute_single_anchor_fourier_arc_length_derivatives(
            t_value=float(t_grid[center_index]),
            coeffs=coeffs,
            family=self.family,
        )

        return SecondOrderSample(
            patch=torch.as_tensor(patch, dtype=self.dtype),
            target_second=torch.as_tensor(second.astype(np.float32), dtype=self.dtype),
            center_index=center_index,
            relative_offsets=torch.as_tensor(
                np.asarray(patch_sample.relative_offsets, dtype=np.int64),
                dtype=torch.long,
            ),
            curve_idx=curve_idx,
        )


@dataclass
class SecondOrderBatch:
    patch: torch.Tensor
    target_second: torch.Tensor
    center_index: torch.Tensor
    relative_offsets: torch.Tensor
    curve_idx: torch.Tensor


def second_order_collate_fn(batch: list[SecondOrderSample]) -> SecondOrderBatch:
    if len(batch) == 0:
        raise ValueError("Cannot collate an empty batch.")

    return SecondOrderBatch(
        patch=torch.stack([x.patch for x in batch], dim=0),
        target_second=torch.stack([x.target_second for x in batch], dim=0),
        center_index=torch.tensor([x.center_index for x in batch], dtype=torch.long),
        relative_offsets=torch.stack([x.relative_offsets for x in batch], dim=0),
        curve_idx=torch.tensor([x.curve_idx for x in batch], dtype=torch.long),
    )
