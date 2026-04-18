from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from datasets.tangent_dataset import PregeneratedCurveBank
from models.tangent_model import TangentOperatorModel
from utils.derivatives import compute_fourier_arc_length_derivatives


@dataclass
class CurveBatch:
    curve_indices: list[int]
    curves: list[torch.Tensor]
    coeffs: list[Any | None]
    t_grids: list[np.ndarray | None]


@dataclass
class SampledOperatorTargets:
    anchor_indices: torch.Tensor               # [A]
    patch_indices_anchor: torch.Tensor         # [A,K]
    abs_patches_anchor: torch.Tensor           # [A,K,2]
    centered_patches_anchor: torch.Tensor      # [A,K,2]
    gt_first_anchor: torch.Tensor              # [A,2]
    gt_second_anchor: torch.Tensor             # [A,2]
    closure_indices: torch.Tensor              # [U]
    patch_indices_closure: torch.Tensor        # [U,K]
    abs_patches_closure: torch.Tensor          # [U,K,2]
    centered_patches_closure: torch.Tensor     # [U,K,2]
    closure_positions_for_anchor_support: torch.Tensor  # [A,K]


@dataclass
class FullCurveOperatorState:
    patch_indices: torch.Tensor        # [N,K]
    abs_patches: torch.Tensor          # [N,K,2]
    centered_patches: torch.Tensor     # [N,K,2]
    weights: torch.Tensor              # [N,K]
    direct1: torch.Tensor              # [N,2]
    global1: torch.Tensor              # [N,2]
    global2: torch.Tensor              # [N,2]
    global2_alt: torch.Tensor          # [N,2]
    row_sum: torch.Tensor              # [N]


REQUIRED_MODEL_CONFIG_FIELDS = [
    'patch_size',
    'operator_hidden_dims',
    'signature_hidden_dims',
    'signature_out_dim',
    'signature_center_radius',
    'head_dropout',
    'disable_normalize_projector',
    'operator_init_scale',
    'learn_output_scale',
    'disable_centered_input_for_operator',
]


def load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text())


def load_pretrained_run(run_dir: str | Path) -> tuple[Path, dict[str, Any], Path]:
    run_dir = Path(run_dir)
    done_path = run_dir / 'DONE'
    config_path = run_dir / 'config.json'
    best_model_path = run_dir / 'best_model.pt'

    if not done_path.exists():
        raise FileNotFoundError(f'Missing DONE marker: {done_path}')
    if not config_path.exists():
        raise FileNotFoundError(f'Missing config.json: {config_path}')
    if not best_model_path.exists():
        raise FileNotFoundError(f'Missing best_model.pt: {best_model_path}')

    cfg = load_json(config_path)
    return best_model_path, cfg, config_path


def _parse_int_list(value: Any, name: str) -> list[int]:
    if value is None:
        return []

    if isinstance(value, bool):
        raise ValueError(f'{name} is bool, expected int list: {value}')

    if isinstance(value, int):
        return [int(value)]

    if isinstance(value, (list, tuple)):
        out: list[int] = []
        for x in value:
            if isinstance(x, bool):
                raise ValueError(f'{name} contains bool, expected ints: {value}')
            out.append(int(x))
        return out

    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        return [int(x.strip()) for x in text.split(',') if x.strip()]

    raise ValueError(f'Unsupported type for {name}: type={type(value)} value={value}')


def _parse_bool(value: Any, name: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {'true', '1', 'yes', 'y'}:
            return True
        if text in {'false', '0', 'no', 'n'}:
            return False
        raise ValueError(f'Could not parse bool field {name} from string: {value}')
    if isinstance(value, (int, float)):
        return bool(value)
    raise ValueError(f'Unsupported type for bool field {name}: type={type(value)} value={value}')


def instantiate_model_from_pretrained_config(cfg: dict[str, Any]) -> TangentOperatorModel:
    missing = [k for k in REQUIRED_MODEL_CONFIG_FIELDS if k not in cfg]
    if missing:
        raise KeyError(f'Missing required model config fields: {missing}')

    model = TangentOperatorModel(
        patch_size=int(cfg['patch_size']),
        operator_hidden_dims=_parse_int_list(cfg['operator_hidden_dims'], 'operator_hidden_dims'),
        signature_hidden_dims=_parse_int_list(cfg['signature_hidden_dims'], 'signature_hidden_dims'),
        signature_out_dim=int(cfg['signature_out_dim']),
        signature_center_radius=int(cfg['signature_center_radius']),
        head_dropout=float(cfg['head_dropout']),
        normalize_projector=not _parse_bool(cfg['disable_normalize_projector'], 'disable_normalize_projector'),
        init_scale=float(cfg['operator_init_scale']),
        learn_scale=_parse_bool(cfg['learn_output_scale'], 'learn_output_scale'),
        centered_input_for_operator=not _parse_bool(
            cfg['disable_centered_input_for_operator'],
            'disable_centered_input_for_operator',
        ),
    )
    return model


def freeze_signature_head(model: TangentOperatorModel) -> None:
    for p in model.signature_head.parameters():
        p.requires_grad = False


def cyclic_patch_indices(num_points: int, patch_size: int, centers: torch.Tensor) -> torch.Tensor:
    if patch_size % 2 == 0:
        raise ValueError(f'patch_size must be odd, got {patch_size}')
    radius = patch_size // 2
    offsets = torch.arange(-radius, radius + 1, device=centers.device, dtype=torch.long)
    return (centers[:, None] + offsets[None, :]) % int(num_points)


def gather_patches(curve_xy: torch.Tensor, patch_indices: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    abs_patches = curve_xy[patch_indices]
    center_pos = patch_indices.shape[1] // 2
    centers = abs_patches[:, center_pos:center_pos + 1, :]
    centered = abs_patches - centers
    return abs_patches, centered


@torch.no_grad()
def analytic_arc_length_derivatives_for_curve(
    curve_xy: torch.Tensor,
    coeffs: Any | None,
    t_grid: np.ndarray | None,
    family: str,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    if coeffs is None or t_grid is None:
        raise ValueError('Analytic derivatives requested but coeffs/t_grid are missing.')

    _, first, second = compute_fourier_arc_length_derivatives(
        t=np.asarray(t_grid, dtype=np.float64),
        coeffs=coeffs,
        family=family,
    )
    gt1 = torch.as_tensor(first, dtype=curve_xy.dtype, device=device)
    gt2 = torch.as_tensor(second, dtype=curve_xy.dtype, device=device)
    if gt1.shape != curve_xy.shape or gt2.shape != curve_xy.shape:
        raise RuntimeError(
            f'Analytic derivative shapes do not match curve shape. '
            f'curve={tuple(curve_xy.shape)} gt1={tuple(gt1.shape)} gt2={tuple(gt2.shape)}'
        )
    return gt1, gt2


@torch.no_grad()
def sample_anchor_indices(
    num_points: int,
    num_anchors: int,
    device: torch.device,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    num_anchors = min(int(num_anchors), int(num_points))
    perm = torch.randperm(num_points, generator=generator, device=device)
    chosen = perm[:num_anchors]
    chosen, _ = torch.sort(chosen)
    return chosen


@torch.no_grad()
def build_sampled_operator_targets(
    curve_xy: torch.Tensor,
    gt_first: torch.Tensor,
    gt_second: torch.Tensor,
    patch_size: int,
    num_anchors: int,
    generator: torch.Generator | None = None,
) -> SampledOperatorTargets:
    device = curve_xy.device
    num_points = int(curve_xy.shape[0])

    anchor_indices = sample_anchor_indices(
        num_points=num_points,
        num_anchors=num_anchors,
        device=device,
        generator=generator,
    )
    patch_indices_anchor = cyclic_patch_indices(num_points, patch_size, anchor_indices)
    abs_patches_anchor, centered_patches_anchor = gather_patches(curve_xy, patch_indices_anchor)

    closure_indices = torch.unique(patch_indices_anchor.reshape(-1), sorted=True)
    patch_indices_closure = cyclic_patch_indices(num_points, patch_size, closure_indices)
    abs_patches_closure, centered_patches_closure = gather_patches(curve_xy, patch_indices_closure)

    lookup = torch.full((num_points,), -1, dtype=torch.long, device=device)
    lookup[closure_indices] = torch.arange(closure_indices.numel(), device=device)
    closure_positions_for_anchor_support = lookup[patch_indices_anchor]

    return SampledOperatorTargets(
        anchor_indices=anchor_indices,
        patch_indices_anchor=patch_indices_anchor,
        abs_patches_anchor=abs_patches_anchor,
        centered_patches_anchor=centered_patches_anchor,
        gt_first_anchor=gt_first[anchor_indices],
        gt_second_anchor=gt_second[anchor_indices],
        closure_indices=closure_indices,
        patch_indices_closure=patch_indices_closure,
        abs_patches_closure=abs_patches_closure,
        centered_patches_closure=centered_patches_closure,
        closure_positions_for_anchor_support=closure_positions_for_anchor_support,
    )


def get_weights_and_direct(model: TangentOperatorModel, centered_patches: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    weights = model.get_weights(centered_patches)
    direct = model.apply_weights(weights, centered_patches)
    return weights, direct


def apply_operator_rowwise(weights: torch.Tensor, support_values: torch.Tensor) -> torch.Tensor:
    return torch.einsum('ak,akd->ad', weights, support_values)


def apply_operator_field(weights: torch.Tensor, support_field: torch.Tensor) -> torch.Tensor:
    return torch.einsum('nk,nkd->nd', weights, support_field)


def compute_sampled_operator_predictions(
    model: TangentOperatorModel,
    sampled: SampledOperatorTargets,
) -> dict[str, torch.Tensor]:
    weights_anchor, direct1_anchor = get_weights_and_direct(model, sampled.centered_patches_anchor)
    weights_closure, direct1_closure = get_weights_and_direct(model, sampled.centered_patches_closure)

    global1_anchor = apply_operator_rowwise(weights_anchor, sampled.abs_patches_anchor)
    global1_closure = apply_operator_rowwise(weights_closure, sampled.abs_patches_closure)

    closure_support_global1 = global1_closure[sampled.closure_positions_for_anchor_support]
    closure_support_direct1 = direct1_closure[sampled.closure_positions_for_anchor_support]

    global2_anchor = apply_operator_rowwise(weights_anchor, closure_support_global1)
    global2_alt_anchor = apply_operator_rowwise(weights_anchor, closure_support_direct1)

    return {
        'weights_anchor': weights_anchor,
        'weights_closure': weights_closure,
        'direct1_anchor': direct1_anchor,
        'global1_anchor': global1_anchor,
        'global2_anchor': global2_anchor,
        'global2_alt_anchor': global2_alt_anchor,
        'row_sum_anchor': weights_anchor.sum(dim=-1),
        'row_sum_closure': weights_closure.sum(dim=-1),
    }


@torch.no_grad()
def compute_full_curve_operator_state(
    model: TangentOperatorModel,
    curve_xy: torch.Tensor,
    patch_size: int,
) -> FullCurveOperatorState:
    device = curve_xy.device
    num_points = int(curve_xy.shape[0])
    all_indices = torch.arange(num_points, device=device)
    patch_indices = cyclic_patch_indices(num_points, patch_size, all_indices)
    abs_patches, centered_patches = gather_patches(curve_xy, patch_indices)

    weights, direct1 = get_weights_and_direct(model, centered_patches)
    global1 = apply_operator_rowwise(weights, abs_patches)

    global1_support = global1[patch_indices]
    direct1_support = direct1[patch_indices]

    global2 = apply_operator_field(weights, global1_support)
    global2_alt = apply_operator_field(weights, direct1_support)

    return FullCurveOperatorState(
        patch_indices=patch_indices,
        abs_patches=abs_patches,
        centered_patches=centered_patches,
        weights=weights,
        direct1=direct1,
        global1=global1,
        global2=global2,
        global2_alt=global2_alt,
        row_sum=weights.sum(dim=-1),
    )


@torch.no_grad()
def maybe_flip_operator_sign(
    model: TangentOperatorModel,
    bank: PregeneratedCurveBank,
    family: str,
    patch_size: int,
    device: torch.device,
    num_curves: int = 8,
    num_anchors: int = 64,
    negative_threshold: float = -0.25,
) -> dict[str, float]:
    cos_vals: list[float] = []

    for curve_idx in range(min(num_curves, len(bank))):
        curve_np, coeffs, t_grid = bank.get(curve_idx)
        if coeffs is None or t_grid is None:
            continue
        curve_xy = torch.as_tensor(curve_np, dtype=torch.float32, device=device)
        gt1, gt2 = analytic_arc_length_derivatives_for_curve(curve_xy, coeffs, t_grid, family, device)
        sampled = build_sampled_operator_targets(
            curve_xy=curve_xy,
            gt_first=gt1,
            gt_second=gt2,
            patch_size=patch_size,
            num_anchors=num_anchors,
        )
        preds = compute_sampled_operator_predictions(model, sampled)
        pred1 = preds['global1_anchor']
        gt = sampled.gt_first_anchor
        pred_n = pred1 / (pred1.norm(dim=-1, keepdim=True) + 1e-8)
        gt_n = gt / (gt.norm(dim=-1, keepdim=True) + 1e-8)
        cos = (pred_n * gt_n).sum(dim=-1)
        cos_vals.append(float(cos.mean().item()))

    mean_signed_cos = float(np.mean(cos_vals)) if cos_vals else float('nan')
    flipped = False

    if np.isfinite(mean_signed_cos) and mean_signed_cos < negative_threshold:
        last_linear = None
        for module in reversed(list(model.operator_head.net)):
            if isinstance(module, torch.nn.Linear):
                last_linear = module
                break
        if last_linear is None:
            raise RuntimeError('Could not find final linear layer in operator_head for sign flip.')
        with torch.no_grad():
            last_linear.weight.mul_(-1.0)
            if last_linear.bias is not None:
                last_linear.bias.mul_(-1.0)
        flipped = True

    return {
        'mean_signed_cos_before_optional_flip': mean_signed_cos,
        'flipped': float(flipped),
    }


class CurveBankIndexSampler:
    def __init__(self, num_items: int, batch_size: int, seed: int) -> None:
        self.num_items = int(num_items)
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.epoch = 0

    def iter_batches(self, shuffle: bool) -> list[list[int]]:
        rng = np.random.default_rng(self.seed + self.epoch)
        order = np.arange(self.num_items, dtype=np.int64)
        if shuffle:
            rng.shuffle(order)
        batches = [order[i:i + self.batch_size].tolist() for i in range(0, len(order), self.batch_size)]
        self.epoch += 1
        return batches
