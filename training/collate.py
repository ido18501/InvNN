from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch

from datasets.tangent_dataset import InvariantSampleTensors


@dataclass
class TangentBatch:
    anchor: torch.Tensor
    positive: torch.Tensor
    negatives: torch.Tensor
    transform_matrix: torch.Tensor
    family: list[str]
    anchor_center_index: torch.Tensor
    negative_center_indices: torch.Tensor
    gt_first_anchor: torch.Tensor
    gt_second_anchor: torch.Tensor
    has_analytic_derivatives: torch.Tensor



def tangent_collate_fn(batch: Sequence[InvariantSampleTensors]) -> TangentBatch:
    if len(batch) == 0:
        raise ValueError('Cannot collate an empty batch.')

    return TangentBatch(
        anchor=torch.stack([s.anchor for s in batch], dim=0),
        positive=torch.stack([s.positive for s in batch], dim=0),
        negatives=torch.stack([s.negatives for s in batch], dim=0),
        transform_matrix=torch.stack([s.transform_matrix for s in batch], dim=0),
        family=[s.family for s in batch],
        anchor_center_index=torch.tensor([s.anchor_center_index for s in batch], dtype=torch.long),
        negative_center_indices=torch.stack([s.negative_center_indices for s in batch], dim=0),
        gt_first_anchor=torch.stack([s.gt_first_anchor for s in batch], dim=0),
        gt_second_anchor=torch.stack([s.gt_second_anchor for s in batch], dim=0),
        has_analytic_derivatives=torch.stack([s.has_analytic_derivatives for s in batch], dim=0),
    )
