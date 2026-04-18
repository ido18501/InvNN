from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict
import math

import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm.auto import tqdm

try:
    from datasets.tangent_dataset import PregeneratedCurveBank
except Exception:
    from tangent_dataset import PregeneratedCurveBank


Array = np.ndarray


def _det2(a: Array, b: Array) -> Array:
    return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]


def _evaluate_fourier_curve_and_parameter_derivatives(
    t: Array,
    x_coeffs: Array,
    y_coeffs: Array,
) -> tuple[Array, Array, Array, Array]:
    t = np.asarray(t, dtype=np.float64)
    x_coeffs = np.asarray(x_coeffs, dtype=np.float64)
    y_coeffs = np.asarray(y_coeffs, dtype=np.float64)
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
    speed = np.linalg.norm(first_dt, axis=-1)
    speed_safe = np.clip(speed, eps, None)
    dot12 = np.sum(first_dt * second_dt, axis=-1)
    speed_t = dot12 / speed_safe

    det12 = _det2(first_dt, second_dt)
    det13 = _det2(first_dt, third_dt)

    family = family.lower()
    if family == 'euclidean':
        return speed_safe, speed_t
    if family == 'similarity':
        abs_det12 = np.abs(det12)
        sign_det12 = np.sign(det12)
        sigma = np.clip(abs_det12 / (speed_safe ** 2), eps, None)
        sigma_t = sign_det12 * det13 / (speed_safe ** 2) - 2.0 * abs_det12 * speed_t / (speed_safe ** 3)
        return sigma, sigma_t
    if family == 'equi_affine':
        abs_det12 = np.abs(det12)
        sign_det12 = np.sign(det12)
        abs_det12_safe = np.clip(abs_det12, eps, None)
        sigma = abs_det12_safe ** (1.0 / 3.0)
        sigma_t = sign_det12 * det13 / (3.0 * (abs_det12_safe ** (2.0 / 3.0)))
        return sigma, sigma_t
    raise NotImplementedError(f'Family {family!r} is not supported by this fine-tune trainer.')


def _compute_arc_length_derivatives(
    family: str,
    first_dt: Array,
    second_dt: Array,
    third_dt: Array,
) -> tuple[Array, Array]:
    sigma, sigma_t = _compute_sigma_and_sigma_prime(family, first_dt, second_dt, third_dt)
    sigma = sigma[..., None]
    sigma_t = sigma_t[..., None]
    first_ds = first_dt / sigma
    second_ds = second_dt / (sigma ** 2) - first_dt * sigma_t / (sigma ** 3)
    return first_ds, second_ds


def _compute_single_anchor_gt(
    *,
    t_value: float,
    x_coeffs: Array,
    y_coeffs: Array,
    family: str,
) -> tuple[Array, Array]:
    t = np.asarray([t_value], dtype=np.float64)
    _, first_dt, second_dt, third_dt = _evaluate_fourier_curve_and_parameter_derivatives(t, x_coeffs, y_coeffs)
    first_ds, second_ds = _compute_arc_length_derivatives(family, first_dt, second_dt, third_dt)
    return first_ds[0].astype(np.float32), second_ds[0].astype(np.float32)


def _fallback_gt_from_sampled_curve(curve_points: Array, anchor_index: int) -> tuple[Array, Array]:
    n = len(curve_points)
    prev_pt = curve_points[(anchor_index - 1) % n]
    curr_pt = curve_points[anchor_index]
    next_pt = curve_points[(anchor_index + 1) % n]

    first = (next_pt - prev_pt) * 0.5
    first_norm = float(np.linalg.norm(first))
    if first_norm > 1e-12:
        first = first / first_norm
    second = next_pt - 2.0 * curr_pt + prev_pt
    return first.astype(np.float32), second.astype(np.float32)


def _extract_intrinsic_patch(
    curve: torch.Tensor,
    centers: torch.Tensor,
    patch_size: int,
    return_centered: bool = True,
) -> torch.Tensor:
    if patch_size % 2 == 0:
        raise ValueError('patch_size must be odd.')
    if curve.ndim != 3 or curve.shape[-1] != 2:
        raise ValueError(f'curve must have shape [B,N,2], got {tuple(curve.shape)}')
    if centers.ndim != 1 or centers.shape[0] != curve.shape[0]:
        raise ValueError('centers must have shape [B].')

    bsz, n, _ = curve.shape
    radius = patch_size // 2
    offsets = torch.arange(-radius, radius + 1, device=curve.device, dtype=torch.long)
    idx = (centers[:, None] + offsets[None, :]) % n
    patch = curve.gather(1, idx[..., None].expand(bsz, patch_size, 2))
    if return_centered:
        patch = patch - patch[:, radius:radius + 1, :]
    return patch


@dataclass
class FineTuneCurveSample:
    curve_points: torch.Tensor
    gt_first_all: torch.Tensor
    gt_second_all: torch.Tensor


class FineTuneCurveDataset(Dataset):
    """
    Full-curve dataset for fine-tuning.

    IMPORTANT:
    Uses only PregeneratedCurveBank for curve access.
    """

    def __init__(
        self,
        *,
        bank_path: str,
        family: str,
        length: int | None = None,
        seed: int = 123,
    ) -> None:
        self.family = str(family)
        self.bank = PregeneratedCurveBank(bank_path)
        self.length = len(self.bank) if length is None else int(length)
        self.seed = int(seed)

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> FineTuneCurveSample:
        bank_idx = index % len(self.bank)
        curve_points, coeffs, t_grid = self.bank.get(bank_idx)

        if coeffs is not None and t_grid is not None:
            gt_first_all = np.empty_like(curve_points, dtype=np.float32)
            gt_second_all = np.empty_like(curve_points, dtype=np.float32)
            for i, t_value in enumerate(np.asarray(t_grid, dtype=np.float64)):
                first, second = _compute_single_anchor_gt(
                    t_value=float(t_value),
                    x_coeffs=coeffs.x_coeffs,
                    y_coeffs=coeffs.y_coeffs,
                    family=self.family,
                )
                gt_first_all[i] = first
                gt_second_all[i] = second
        else:
            gt_first = []
            gt_second = []
            for i in range(len(curve_points)):
                f1, f2 = _fallback_gt_from_sampled_curve(curve_points, i)
                gt_first.append(f1)
                gt_second.append(f2)
            gt_first_all = np.asarray(gt_first, dtype=np.float32)
            gt_second_all = np.asarray(gt_second, dtype=np.float32)

        return FineTuneCurveSample(
            curve_points=torch.as_tensor(curve_points, dtype=torch.float32),
            gt_first_all=torch.as_tensor(gt_first_all, dtype=torch.float32),
            gt_second_all=torch.as_tensor(gt_second_all, dtype=torch.float32),
        )


def finetune_curve_collate(batch: list[FineTuneCurveSample]) -> FineTuneCurveSample:
    return FineTuneCurveSample(
        curve_points=torch.stack([b.curve_points for b in batch], dim=0),
        gt_first_all=torch.stack([b.gt_first_all for b in batch], dim=0),
        gt_second_all=torch.stack([b.gt_second_all for b in batch], dim=0),
    )


@dataclass
class TrainOutput:
    loss: float
    stats: Dict[str, float]


class SecondOrderFineTuneTrainer:
    def __init__(
        self,
        *,
        model,
        optimizer,
        scheduler,
        loss_fn,
        device,
        checkpoint_dir: str | Path,
        patch_size: int,
        grad_clip_norm: float | None = None,
        anchors_per_curve: int = 8,
        seed: int = 123,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss_fn = loss_fn
        self.device = torch.device(device)
        self.patch_size = int(patch_size)
        self.grad_clip_norm = grad_clip_norm
        self.anchors_per_curve = int(anchors_per_curve)
        self.seed = int(seed)

        self.model.to(self.device)
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.reference_params = [p.detach().clone().to(self.device) for p in self.model.parameters() if p.requires_grad]

    def _move_batch(self, batch: FineTuneCurveSample) -> FineTuneCurveSample:
        batch.curve_points = batch.curve_points.to(self.device)
        batch.gt_first_all = batch.gt_first_all.to(self.device)
        batch.gt_second_all = batch.gt_second_all.to(self.device)
        return batch

    def _sample_anchor_indices(self, curve_points: torch.Tensor, epoch: int, batch_idx: int) -> torch.Tensor:
        bsz, n, _ = curve_points.shape
        g = torch.Generator(device='cpu')
        g.manual_seed(self.seed + 100000 * epoch + 1000 * batch_idx)
        return torch.randint(low=0, high=n, size=(bsz, self.anchors_per_curve), generator=g)

    def _build_supervised_predictions(
        self,
        curve_points: torch.Tensor,
        gt_first_all: torch.Tensor,
        gt_second_all: torch.Tensor,
        anchor_idx: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        bsz, n, _ = curve_points.shape
        a = self.anchors_per_curve
        k = self.patch_size
        radius = k // 2
        offsets = torch.arange(-radius, radius + 1, device=curve_points.device, dtype=torch.long)

        flat_anchor_idx = anchor_idx.reshape(-1)
        repeat_curve = curve_points[:, None, :, :].expand(bsz, a, n, 2).reshape(bsz * a, n, 2)
        repeat_gt1 = gt_first_all[:, None, :, :].expand(bsz, a, n, 2).reshape(bsz * a, n, 2)
        repeat_gt2 = gt_second_all[:, None, :, :].expand(bsz, a, n, 2).reshape(bsz * a, n, 2)

        anchor_patch = _extract_intrinsic_patch(repeat_curve, flat_anchor_idx, self.patch_size, return_centered=True)
        anchor_out = self.model(anchor_patch)
        pred1 = anchor_out['pred']
        weights = anchor_out['weights']

        neighbor_centers = (flat_anchor_idx[:, None] + offsets[None, :]) % n
        flat_neighbor_centers = neighbor_centers.reshape(-1)

        neighbor_curve = repeat_curve[:, None, :, :].expand(bsz * a, k, n, 2).reshape(bsz * a * k, n, 2)
        neighbor_patches = _extract_intrinsic_patch(
            neighbor_curve,
            flat_neighbor_centers,
            self.patch_size,
            return_centered=True,
        )
        neighbor_out = self.model(neighbor_patches)
        neighbor_pred1 = neighbor_out['pred'].reshape(bsz * a, k, 2)

        pred2 = torch.einsum('bk,bkd->bd', weights, neighbor_pred1)

        gt1 = repeat_gt1.gather(1, flat_anchor_idx[:, None, None].expand(bsz * a, 1, 2)).squeeze(1)
        gt2 = repeat_gt2.gather(1, flat_anchor_idx[:, None, None].expand(bsz * a, 1, 2)).squeeze(1)

        return {
            'pred1': pred1,
            'pred2': pred2,
            'gt1': gt1,
            'gt2': gt2,
            'weights': weights,
            'row_sum': weights.sum(dim=-1),
        }

    def train_step(self, batch: FineTuneCurveSample, *, epoch: int, batch_idx: int) -> TrainOutput:
        self.model.train()
        batch = self._move_batch(batch)
        self.optimizer.zero_grad(set_to_none=True)

        anchor_idx = self._sample_anchor_indices(batch.curve_points, epoch=epoch, batch_idx=batch_idx).to(self.device)
        out = self._build_supervised_predictions(
            curve_points=batch.curve_points,
            gt_first_all=batch.gt_first_all,
            gt_second_all=batch.gt_second_all,
            anchor_idx=anchor_idx,
        )

        current_params = [p for p in self.model.parameters() if p.requires_grad]
        loss, stats = self.loss_fn(
            pred1=out['pred1'],
            gt1=out['gt1'],
            pred2=out['pred2'],
            gt2=out['gt2'],
            row_sum=out['row_sum'],
            current_params=current_params,
            reference_params=self.reference_params,
            return_stats=True,
        )
        loss.backward()

        if self.grad_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)
        self.optimizer.step()
        return TrainOutput(loss=float(loss.item()), stats=stats)

    @torch.no_grad()
    def eval_step(self, batch: FineTuneCurveSample, *, epoch: int, batch_idx: int) -> TrainOutput:
        self.model.eval()
        batch = self._move_batch(batch)
        anchor_idx = self._sample_anchor_indices(batch.curve_points, epoch=epoch, batch_idx=batch_idx).to(self.device)
        out = self._build_supervised_predictions(
            curve_points=batch.curve_points,
            gt_first_all=batch.gt_first_all,
            gt_second_all=batch.gt_second_all,
            anchor_idx=anchor_idx,
        )
        current_params = [p for p in self.model.parameters() if p.requires_grad]
        loss, stats = self.loss_fn(
            pred1=out['pred1'],
            gt1=out['gt1'],
            pred2=out['pred2'],
            gt2=out['gt2'],
            row_sum=out['row_sum'],
            current_params=current_params,
            reference_params=self.reference_params,
            return_stats=True,
        )
        return TrainOutput(loss=float(loss.item()), stats=stats)

    def _run_loader(self, loader, *, train: bool, epoch: int, desc: str) -> Dict[str, float]:
        metrics: Dict[str, float] = {}
        n = 0
        iterator = tqdm(loader, desc=desc, leave=False, dynamic_ncols=True)
        for batch_idx, batch in enumerate(iterator):
            out = self.train_step(batch, epoch=epoch, batch_idx=batch_idx) if train else self.eval_step(batch, epoch=epoch, batch_idx=batch_idx)
            for k, v in out.stats.items():
                metrics[k] = metrics.get(k, 0.0) + float(v)
            n += 1
            iterator.set_postfix(
                loss=f"{out.stats.get('loss', float('nan')):.4f}",
                mse1=f"{out.stats.get('mse1', float('nan')):.4f}",
                mse2=f"{out.stats.get('mse2', float('nan')):.4f}",
                cos1=f"{out.stats.get('cos1', float('nan')):.3f}",
                cos2=f"{out.stats.get('cos2', float('nan')):.3f}",
            )
        for k in list(metrics.keys()):
            metrics[k] /= max(n, 1)
        return metrics

    @staticmethod
    def _print_epoch_summary(epoch: int, train_metrics: Dict[str, float], val_metrics: Dict[str, float]) -> None:
        print(f"\nEpoch {epoch}", flush=True)
        print(
            'train | '
            f"loss={train_metrics.get('loss', float('nan')):.4f} "
            f"mse1={train_metrics.get('mse1', float('nan')):.6f} "
            f"mse2={train_metrics.get('mse2', float('nan')):.6f} "
            f"cos1={train_metrics.get('cos1', float('nan')):.4f} "
            f"cos2={train_metrics.get('cos2', float('nan')):.4f} "
            f"mag1={train_metrics.get('mag1_ratio', float('nan')):.4f} "
            f"mag2={train_metrics.get('mag2_ratio', float('nan')):.4f} "
            f"rowsum={train_metrics.get('row_sum_abs_mean', float('nan')):.6f}",
            flush=True,
        )
        print(
            'val   | '
            f"loss={val_metrics.get('loss', float('nan')):.4f} "
            f"mse1={val_metrics.get('mse1', float('nan')):.6f} "
            f"mse2={val_metrics.get('mse2', float('nan')):.6f} "
            f"cos1={val_metrics.get('cos1', float('nan')):.4f} "
            f"cos2={val_metrics.get('cos2', float('nan')):.4f} "
            f"mag1={val_metrics.get('mag1_ratio', float('nan')):.4f} "
            f"mag2={val_metrics.get('mag2_ratio', float('nan')):.4f} "
            f"rowsum={val_metrics.get('row_sum_abs_mean', float('nan')):.6f}",
            flush=True,
        )

    def fit(self, train_loader, val_loader, *, num_epochs: int, early_stopping_patience: int = 10):
        best_val = float('inf')
        best_epoch = 0
        patience = 0
        best_model_path = self.checkpoint_dir / 'best_model.pt'
        torch.save(self.model.state_dict(), self.checkpoint_dir / 'init_model.pt')

        for epoch in range(1, num_epochs + 1):
            train_metrics = self._run_loader(train_loader, train=True, epoch=epoch, desc=f'train {epoch}/{num_epochs}')
            val_metrics = self._run_loader(val_loader, train=False, epoch=epoch, desc=f'val   {epoch}/{num_epochs}')
            self._print_epoch_summary(epoch, train_metrics, val_metrics)

            val_loss = val_metrics.get('loss', float('inf'))
            if self.scheduler is not None:
                self.scheduler.step(val_loss)
                print(f"lr={self.optimizer.param_groups[0]['lr']:.6g}", flush=True)

            if val_loss < best_val:
                best_val = val_loss
                best_epoch = epoch
                patience = 0
                torch.save(self.model.state_dict(), best_model_path)
                print('✓ saved new best model', flush=True)
            else:
                patience += 1
                print(f'no improvement ({patience}/{early_stopping_patience})', flush=True)

            if patience >= early_stopping_patience:
                print('Early stopping triggered', flush=True)
                break

        print(f'\nBest validation epoch: {best_epoch}', flush=True)
        self.model.load_state_dict(torch.load(best_model_path, map_location=self.device))
        return best_model_path

    def evaluate(self, loader, *, split_name: str = 'test') -> Dict[str, float]:
        metrics = self._run_loader(loader, train=False, epoch=999999, desc=split_name)
        print(f'\n{split_name.capitalize()} metrics', flush=True)
        print(metrics, flush=True)
        return metrics
