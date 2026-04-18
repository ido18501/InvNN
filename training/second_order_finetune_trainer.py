from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import math
from typing import Iterable

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm

from datasets.tangent_dataset import PregeneratedCurveBank
from utils.derivatives import compute_fourier_arc_length_derivatives


class FullCurveBankDataset(Dataset):
    """
    Thin dataset wrapper over PregeneratedCurveBank.
    Returns full curves plus full first/second arc-length derivatives.
    """

    def __init__(self, bank_path: str | Path, family: str = 'euclidean', dtype: torch.dtype = torch.float32) -> None:
        self.bank = PregeneratedCurveBank(bank_path)
        self.family = str(family)
        self.dtype = dtype

    def __len__(self) -> int:
        return len(self.bank)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | int | bool]:
        curve_points, coeffs, t_grid = self.bank.get(idx)
        if coeffs is None or t_grid is None:
            raise RuntimeError(
                'Second-order fine-tuning requires pregenerated banks with Fourier coefficients and t_grid.'
            )

        _, gt_first, gt_second = compute_fourier_arc_length_derivatives(
            t=np.asarray(t_grid, dtype=np.float64),
            coeffs=coeffs,
            family=self.family,
        )
        return {
            'curve_points': torch.as_tensor(curve_points, dtype=self.dtype),
            'gt_first': torch.as_tensor(gt_first, dtype=self.dtype),
            'gt_second': torch.as_tensor(gt_second, dtype=self.dtype),
            'curve_index': int(idx),
        }


@dataclass
class CurveBatch:
    curve_points: torch.Tensor
    gt_first: torch.Tensor
    gt_second: torch.Tensor
    curve_index: torch.Tensor


@dataclass
class OperatorTrainingOutputs:
    pred1: torch.Tensor
    pred2: torch.Tensor
    effective_anchor_weights: torch.Tensor
    anchor_indices: torch.Tensor


@dataclass
class EvalSummary:
    metrics: dict[str, float]
    raw: dict[str, np.ndarray]



def full_curve_collate(batch: list[dict]) -> CurveBatch:
    return CurveBatch(
        curve_points=torch.stack([x['curve_points'] for x in batch], dim=0),
        gt_first=torch.stack([x['gt_first'] for x in batch], dim=0),
        gt_second=torch.stack([x['gt_second'] for x in batch], dim=0),
        curve_index=torch.tensor([x['curve_index'] for x in batch], dtype=torch.long),
    )


class SecondOrderFineTuneTrainer:
    def __init__(
        self,
        *,
        model,
        optimizer,
        scheduler,
        loss_fn,
        device: str | torch.device,
        checkpoint_dir: str | Path,
        patch_size: int,
        sign_check_batches: int = 4,
        sign_check_threshold: float = -0.25,
        grad_clip_norm: float | None = 1.0,
        train_num_anchors: int = 128,
        eval_batch_size_curves: int = 4,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss_fn = loss_fn
        self.device = torch.device(device)
        self.patch_size = int(patch_size)
        self.radius = self.patch_size // 2
        self.sign_check_batches = int(sign_check_batches)
        self.sign_check_threshold = float(sign_check_threshold)
        self.grad_clip_norm = grad_clip_norm
        self.train_num_anchors = int(train_num_anchors)
        self.eval_batch_size_curves = int(eval_batch_size_curves)

        self.model.to(self.device)
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def _move_batch(self, batch: CurveBatch) -> CurveBatch:
        batch.curve_points = batch.curve_points.to(self.device)
        batch.gt_first = batch.gt_first.to(self.device)
        batch.gt_second = batch.gt_second.to(self.device)
        batch.curve_index = batch.curve_index.to(self.device)
        return batch

    @staticmethod
    def _sample_anchor_indices(num_points: int, num_anchors: int, device: torch.device, generator: torch.Generator) -> torch.Tensor:
        if num_anchors >= num_points:
            return torch.arange(num_points, device=device, dtype=torch.long)
        perm = torch.randperm(num_points, generator=generator, device=device)
        return perm[:num_anchors]

    def _cyclic_gather_patches(self, curve_points: torch.Tensor, center_indices: torch.Tensor) -> torch.Tensor:
        """
        curve_points: [N,2]
        center_indices: [M]
        returns patches: [M,K,2]
        """
        offsets = torch.arange(-self.radius, self.radius + 1, device=curve_points.device, dtype=torch.long)
        idx = (center_indices[:, None] + offsets[None, :]) % curve_points.shape[0]
        return curve_points[idx]

    def _effective_weights(self, model_out: dict[str, torch.Tensor]) -> torch.Tensor:
        weights = model_out['weights']
        scale = getattr(self.model, 'output_scale', None)
        if scale is None:
            return weights
        return scale * weights

    def _predict_on_indices(self, curve_points: torch.Tensor, center_indices: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        patches = self._cyclic_gather_patches(curve_points, center_indices)
        out = self.model(patches)
        effective_weights = self._effective_weights(out)
        pred = torch.einsum('mk,mkd->md', effective_weights, patches)
        return pred, effective_weights, patches

    def _operator_forward_sampled(self, curve_points: torch.Tensor, anchor_indices: torch.Tensor) -> OperatorTrainingOutputs:
        """
        Two-hop operator-aware approximation.

        For pred1 at anchors, we use the effective operator row on X.
        For pred2 at anchors, we first evaluate pred1 on every neighbor appearing in the anchor stencils,
        then apply each anchor row to those first-order neighbor values.
        """
        pred1_anchor, effective_anchor_weights, _ = self._predict_on_indices(curve_points, anchor_indices)

        offsets = torch.arange(-self.radius, self.radius + 1, device=curve_points.device, dtype=torch.long)
        neighbor_idx_matrix = (anchor_indices[:, None] + offsets[None, :]) % curve_points.shape[0]  # [A,K]
        unique_neighbors, inverse = torch.unique(neighbor_idx_matrix.reshape(-1), sorted=True, return_inverse=True)

        pred1_neighbors, _, _ = self._predict_on_indices(curve_points, unique_neighbors)
        pred1_neighbor_matrix = pred1_neighbors[inverse.view(anchor_indices.shape[0], self.patch_size)]

        pred2_anchor = torch.einsum('ak,akd->ad', effective_anchor_weights, pred1_neighbor_matrix)

        return OperatorTrainingOutputs(
            pred1=pred1_anchor,
            pred2=pred2_anchor,
            effective_anchor_weights=effective_anchor_weights,
            anchor_indices=anchor_indices,
        )

    def _full_curve_operator_eval(self, curve_points: torch.Tensor) -> dict[str, torch.Tensor]:
        n = curve_points.shape[0]
        all_idx = torch.arange(n, device=curve_points.device, dtype=torch.long)
        patches = self._cyclic_gather_patches(curve_points, all_idx)
        out = self.model(patches)
        effective_weights = self._effective_weights(out)  # [N,K]
        direct1 = torch.einsum('nk,nkd->nd', effective_weights, patches)

        offsets = torch.arange(-self.radius, self.radius + 1, device=curve_points.device, dtype=torch.long)
        neighbor_idx = (all_idx[:, None] + offsets[None, :]) % n

        global1 = direct1
        global2 = torch.einsum('nk,nkd->nd', effective_weights, global1[neighbor_idx])

        direct1_field_centered = torch.zeros_like(patches)
        direct1_field_centered[:, self.radius, :] = direct1
        global2_alt = torch.einsum('nk,nkd->nd', effective_weights, direct1_field_centered)

        return {
            'effective_weights': effective_weights,
            'direct1': direct1,
            'global1': global1,
            'global2': global2,
            'global2_alt': global2_alt,
        }

    @staticmethod
    def _cosine_and_angle(pred: np.ndarray, gt: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        pred_n = pred / np.clip(np.linalg.norm(pred, axis=-1, keepdims=True), 1e-8, None)
        gt_n = gt / np.clip(np.linalg.norm(gt, axis=-1, keepdims=True), 1e-8, None)
        cos = np.sum(pred_n * gt_n, axis=-1)
        cos = np.clip(cos, -1.0, 1.0)
        angle = np.degrees(np.arccos(cos))
        return cos, angle

    @staticmethod
    def _pearson(x: np.ndarray, y: np.ndarray) -> float:
        x = np.asarray(x, dtype=np.float64).reshape(-1)
        y = np.asarray(y, dtype=np.float64).reshape(-1)
        if x.size < 2 or y.size < 2:
            return float('nan')
        x = x - x.mean()
        y = y - y.mean()
        denom = np.linalg.norm(x) * np.linalg.norm(y)
        if denom <= 1e-12:
            return float('nan')
        return float(np.dot(x, y) / denom)

    @staticmethod
    def _rankdata_average(x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        order = np.argsort(x, kind='mergesort')
        ranks = np.empty_like(order, dtype=np.float64)
        xs = x[order]
        start = 0
        while start < len(x):
            end = start + 1
            while end < len(x) and xs[end] == xs[start]:
                end += 1
            avg_rank = 0.5 * (start + end - 1) + 1.0
            ranks[order[start:end]] = avg_rank
            start = end
        return ranks

    def _spearman(self, x: np.ndarray, y: np.ndarray) -> float:
        rx = self._rankdata_average(x)
        ry = self._rankdata_average(y)
        return self._pearson(rx, ry)

    def _vector_metrics(self, pred: np.ndarray, gt: np.ndarray, prefix: str) -> dict[str, float]:
        cos, angle = self._cosine_and_angle(pred, gt)
        pred_norm = np.linalg.norm(pred, axis=-1)
        gt_norm = np.linalg.norm(gt, axis=-1)
        norm_err = np.abs(pred_norm - gt_norm)

        out = {
            f'{prefix}_cosine_mean': float(np.mean(cos)),
            f'{prefix}_abs_cosine_mean': float(np.mean(np.abs(cos))),
            f'{prefix}_angle_mean': float(np.mean(angle)),
            f'{prefix}_mse': float(np.mean((pred - gt) ** 2)),
            f'{prefix}_pred_norm_mean': float(np.mean(pred_norm)),
            f'{prefix}_pred_norm_median': float(np.median(pred_norm)),
            f'{prefix}_norm_error_mean': float(np.mean(norm_err)),
            f'{prefix}_norm_error_median': float(np.median(norm_err)),
        }

        if prefix.startswith('global2'):
            pred_n = pred_norm.reshape(-1)
            gt_n = gt_norm.reshape(-1)
            out[f'{prefix}_norm_spearman'] = self._spearman(pred_n, gt_n)
            out[f'{prefix}_norm_pearson'] = self._pearson(pred_n, gt_n)
            out[f'{prefix}_log1p_norm_pearson'] = self._pearson(np.log1p(pred_n), np.log1p(gt_n))
            if pred_n.size >= 2 and np.std(gt_n) > 1e-12:
                slope, intercept = np.polyfit(gt_n, pred_n, deg=1)
                out[f'{prefix}_norm_fit_slope'] = float(slope)
                out[f'{prefix}_norm_fit_intercept'] = float(intercept)
            else:
                out[f'{prefix}_norm_fit_slope'] = float('nan')
                out[f'{prefix}_norm_fit_intercept'] = float('nan')
        return out

    def _full_evaluate_loader(self, loader: DataLoader, split_name: str, tag: str) -> EvalSummary:
        self.model.eval()
        local1_pred, global1_pred, global2_pred, global2_alt_pred = [], [], [], []
        gt1_all, gt2_all = [], []

        with torch.no_grad():
            for batch in tqdm(loader, desc=f'{split_name} {tag}', leave=False, dynamic_ncols=True):
                batch = self._move_batch(batch)
                bsz = batch.curve_points.shape[0]
                for b in range(bsz):
                    outputs = self._full_curve_operator_eval(batch.curve_points[b])
                    local1_pred.append(outputs['direct1'].detach().cpu().numpy())
                    global1_pred.append(outputs['global1'].detach().cpu().numpy())
                    global2_pred.append(outputs['global2'].detach().cpu().numpy())
                    global2_alt_pred.append(outputs['global2_alt'].detach().cpu().numpy())
                    gt1_all.append(batch.gt_first[b].detach().cpu().numpy())
                    gt2_all.append(batch.gt_second[b].detach().cpu().numpy())

        local1_pred = np.concatenate(local1_pred, axis=0)
        global1_pred = np.concatenate(global1_pred, axis=0)
        global2_pred = np.concatenate(global2_pred, axis=0)
        global2_alt_pred = np.concatenate(global2_alt_pred, axis=0)
        gt1_all = np.concatenate(gt1_all, axis=0)
        gt2_all = np.concatenate(gt2_all, axis=0)

        metrics = {}
        metrics.update(self._vector_metrics(local1_pred, gt1_all, 'local1'))
        metrics.update(self._vector_metrics(global1_pred, gt1_all, 'global1'))
        metrics.update(self._vector_metrics(global2_pred, gt2_all, 'global2'))
        metrics.update(self._vector_metrics(global2_alt_pred, gt2_all, 'global2_alt'))

        return EvalSummary(
            metrics=metrics,
            raw={
                'local1_pred': local1_pred,
                'global1_pred': global1_pred,
                'global2_pred': global2_pred,
                'global2_alt_pred': global2_alt_pred,
                'gt1': gt1_all,
                'gt2': gt2_all,
            },
        )

    def _mean_signed_cosine_first(self, loader: DataLoader) -> float:
        self.model.eval()
        vals: list[float] = []
        with torch.no_grad():
            for i, batch in enumerate(loader):
                if i >= self.sign_check_batches:
                    break
                batch = self._move_batch(batch)
                bsz, n, _ = batch.curve_points.shape
                for b in range(bsz):
                    outputs = self._full_curve_operator_eval(batch.curve_points[b])
                    pred = outputs['global1'].detach().cpu().numpy()
                    gt = batch.gt_first[b].detach().cpu().numpy()
                    cos, _ = self._cosine_and_angle(pred, gt)
                    vals.append(float(np.mean(cos)))
        if not vals:
            return float('nan')
        return float(np.mean(vals))

    def _flip_operator_sign_once(self) -> None:
        scale = getattr(self.model, 'output_scale', None)
        if scale is not None:
            with torch.no_grad():
                scale.mul_(-1.0)
            return

        last_linear = None
        for module in reversed(list(self.model.operator_head.net)):
            if isinstance(module, torch.nn.Linear):
                last_linear = module
                break
        if last_linear is None:
            raise RuntimeError('Could not find final linear layer in operator_head for sign flip.')
        with torch.no_grad():
            last_linear.weight.mul_(-1.0)
            if last_linear.bias is not None:
                last_linear.bias.mul_(-1.0)

    def maybe_apply_sign_flip(self, val_loader: DataLoader) -> dict[str, float]:
        before = self._mean_signed_cosine_first(val_loader)
        flipped = False
        after = before
        if math.isfinite(before) and before < self.sign_check_threshold:
            self._flip_operator_sign_once()
            after = self._mean_signed_cosine_first(val_loader)
            flipped = True
        stats = {
            'init_signed_cosine_before_flip': float(before),
            'init_signed_cosine_after_flip': float(after),
            'sign_flip_applied': float(flipped),
        }
        print(
            f"[sign check] before={before:.6f} after={after:.6f} flipped={'yes' if flipped else 'no'}",
            flush=True,
        )
        return stats

    def train_one_epoch(self, loader: DataLoader, epoch: int, num_epochs: int, seed: int) -> dict[str, float]:
        self.model.train()
        metrics: dict[str, float] = {}
        count = 0
        gen = torch.Generator(device=self.device.type if self.device.type != 'mps' else 'cpu')
        gen.manual_seed(seed + epoch)

        iterator = tqdm(loader, desc=f'train {epoch}/{num_epochs}', leave=False, dynamic_ncols=True)
        for batch in iterator:
            batch = self._move_batch(batch)
            self.optimizer.zero_grad(set_to_none=True)

            total_loss = batch.curve_points.new_zeros(())
            batch_stats_accum: dict[str, float] = {}
            bsz = batch.curve_points.shape[0]

            for b in range(bsz):
                n = batch.curve_points.shape[1]
                anchor_idx = self._sample_anchor_indices(n, self.train_num_anchors, self.device, gen)
                out = self._operator_forward_sampled(batch.curve_points[b], anchor_idx)
                gt1 = batch.gt_first[b, anchor_idx]
                gt2 = batch.gt_second[b, anchor_idx]

                loss, stats = self.loss_fn(
                    pred1=out.pred1,
                    gt1=gt1,
                    pred2=out.pred2,
                    gt2=gt2,
                    effective_weights=out.effective_anchor_weights,
                    return_stats=True,
                )
                total_loss = total_loss + loss / bsz
                for k, v in stats.items():
                    batch_stats_accum[k] = batch_stats_accum.get(k, 0.0) + float(v) / bsz

            total_loss.backward()
            if self.grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)
            self.optimizer.step()

            batch_stats_accum['loss'] = float(total_loss.item())
            for k, v in batch_stats_accum.items():
                metrics[k] = metrics.get(k, 0.0) + float(v)
            count += 1
            iterator.set_postfix(loss=f"{batch_stats_accum['loss']:.4f}", vec1=f"{batch_stats_accum.get('vec1_loss', float('nan')):.4f}", vec2=f"{batch_stats_accum.get('vec2_loss', float('nan')):.4f}")

        for k in list(metrics.keys()):
            metrics[k] /= max(count, 1)
        return metrics

    @torch.no_grad()
    def validate_one_epoch(self, loader: DataLoader, epoch: int, num_epochs: int) -> dict[str, float]:
        self.model.eval()
        metrics: dict[str, float] = {}
        count = 0
        iterator = tqdm(loader, desc=f'val   {epoch}/{num_epochs}', leave=False, dynamic_ncols=True)
        for batch in iterator:
            batch = self._move_batch(batch)
            batch_stats_accum: dict[str, float] = {}
            bsz = batch.curve_points.shape[0]
            for b in range(bsz):
                n = batch.curve_points.shape[1]
                anchor_idx = torch.arange(n, device=self.device, dtype=torch.long)
                out = self._operator_forward_sampled(batch.curve_points[b], anchor_idx)
                gt1 = batch.gt_first[b, anchor_idx]
                gt2 = batch.gt_second[b, anchor_idx]
                _, stats = self.loss_fn(
                    pred1=out.pred1,
                    gt1=gt1,
                    pred2=out.pred2,
                    gt2=gt2,
                    effective_weights=out.effective_anchor_weights,
                    return_stats=True,
                )
                for k, v in stats.items():
                    batch_stats_accum[k] = batch_stats_accum.get(k, 0.0) + float(v) / bsz
            for k, v in batch_stats_accum.items():
                metrics[k] = metrics.get(k, 0.0) + float(v)
            count += 1
        for k in list(metrics.keys()):
            metrics[k] /= max(count, 1)
        return metrics

    @staticmethod
    def _print_epoch_summary(epoch: int, train_metrics: dict[str, float], val_metrics: dict[str, float]) -> None:
        print(f"\nEpoch {epoch}", flush=True)
        print(
            "train | "
            f"loss={train_metrics.get('loss', float('nan')):.4f} "
            f"vec1={train_metrics.get('vec1_loss', float('nan')):.4f} "
            f"vec2={train_metrics.get('vec2_loss', float('nan')):.4f} "
            f"cos1={train_metrics.get('cos1_loss', float('nan')):.4f} "
            f"cos2={train_metrics.get('cos2_loss', float('nan')):.4f} "
            f"log2={train_metrics.get('log2_loss', float('nan')):.4f} "
            f"rowsum={train_metrics.get('rowsum_loss', float('nan')):.4f}",
            flush=True,
        )
        print(
            "val   | "
            f"loss={val_metrics.get('loss', float('nan')):.4f} "
            f"vec1={val_metrics.get('vec1_loss', float('nan')):.4f} "
            f"vec2={val_metrics.get('vec2_loss', float('nan')):.4f} "
            f"cos1={val_metrics.get('cos1_loss', float('nan')):.4f} "
            f"cos2={val_metrics.get('cos2_loss', float('nan')):.4f} "
            f"log2={val_metrics.get('log2_loss', float('nan')):.4f} "
            f"rowsum={val_metrics.get('rowsum_loss', float('nan')):.4f}",
            flush=True,
        )

    def fit(
        self,
        *,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: DataLoader,
        num_epochs: int,
        early_stopping_patience: int,
        seed: int,
    ) -> dict[str, object]:
        sign_stats = self.maybe_apply_sign_flip(val_loader)

        before_val = self._full_evaluate_loader(val_loader, split_name='val', tag='before')
        before_test = self._full_evaluate_loader(test_loader, split_name='test', tag='before')

        best_val = float('inf')
        best_epoch = 0
        patience = 0
        best_model_path = self.checkpoint_dir / 'best_model.pt'
        torch.save(self.model.state_dict(), self.checkpoint_dir / 'init_model.pt')

        history: list[dict[str, object]] = []
        for epoch in range(1, num_epochs + 1):
            train_metrics = self.train_one_epoch(train_loader, epoch=epoch, num_epochs=num_epochs, seed=seed)
            val_metrics = self.validate_one_epoch(val_loader, epoch=epoch, num_epochs=num_epochs)
            self._print_epoch_summary(epoch, train_metrics, val_metrics)

            val_loss = float(val_metrics.get('loss', float('inf')))
            if self.scheduler is not None:
                self.scheduler.step(val_loss)
                current_lr = self.optimizer.param_groups[0]['lr']
                print(f'lr={current_lr:.6g}', flush=True)

            if val_loss < best_val:
                best_val = val_loss
                best_epoch = epoch
                patience = 0
                torch.save(self.model.state_dict(), best_model_path)
                print('✓ saved new best model', flush=True)
            else:
                patience += 1
                print(f'no improvement ({patience}/{early_stopping_patience})', flush=True)

            history.append({'epoch': epoch, 'train': train_metrics, 'val': val_metrics})
            if patience >= early_stopping_patience:
                print('Early stopping triggered', flush=True)
                break

        self.model.load_state_dict(torch.load(best_model_path, map_location=self.device))

        after_val = self._full_evaluate_loader(val_loader, split_name='val', tag='after')
        after_test = self._full_evaluate_loader(test_loader, split_name='test', tag='after')

        summary = {
            'sign_check': sign_stats,
            'best_epoch': best_epoch,
            'best_val_loss': best_val,
            'history': history,
            'before': {
                'val': before_val.metrics,
                'test': before_test.metrics,
            },
            'after': {
                'val': after_val.metrics,
                'test': after_test.metrics,
            },
        }
        with open(self.checkpoint_dir / 'summary.json', 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2)

        return summary
