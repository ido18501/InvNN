from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable

import numpy as np
import torch
from tqdm.auto import tqdm

try:
    from training.collate import TangentBatch
except Exception:
    from collate import TangentBatch

try:
    from datasets.tangent_dataset import PregeneratedCurveBank
except Exception:
    from tangent_dataset import PregeneratedCurveBank

try:
    from utils.derivatives import compute_fourier_arc_length_derivatives
except Exception:
    from derivatives import compute_fourier_arc_length_derivatives


@dataclass
class TrainOutput:
    loss: float
    stats: Dict[str, float]


class FirstDerivativeFineTuneTrainer:
    def __init__(
        self,
        model,
        optimizer,
        scheduler,
        loss_fn,
        device,
        grad_clip_norm=None,
        checkpoint_dir='checkpoints_first_derivative',
    ):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss_fn = loss_fn
        self.device = torch.device(device)
        self.grad_clip_norm = grad_clip_norm
        self.model.to(self.device)
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True, parents=True)

    def _move_batch(self, batch: TangentBatch) -> TangentBatch:
        batch.anchor = batch.anchor.to(self.device)
        batch.gt_first_anchor = batch.gt_first_anchor.to(self.device)
        batch.has_analytic_derivatives = batch.has_analytic_derivatives.to(self.device)
        return batch

    @staticmethod
    def _cosine_and_angle(pred: torch.Tensor, gt: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        pred_n = pred / (pred.norm(dim=-1, keepdim=True) + 1e-8)
        gt_n = gt / (gt.norm(dim=-1, keepdim=True) + 1e-8)
        cos = (pred_n * gt_n).sum(dim=-1).clamp(-1.0, 1.0)
        angle = torch.rad2deg(torch.acos(cos))
        return cos, angle

    def _valid_mask(self, batch: TangentBatch) -> torch.Tensor:
        return batch.has_analytic_derivatives.bool() & torch.isfinite(batch.gt_first_anchor).all(dim=-1)

    def train_step(self, batch: TangentBatch) -> TrainOutput:
        self.model.train()
        batch = self._move_batch(batch)
        valid = self._valid_mask(batch)
        if valid.sum().item() == 0:
            return TrainOutput(loss=float('nan'), stats={})

        self.optimizer.zero_grad(set_to_none=True)
        pred = self.model(batch.anchor)['pred'][valid]
        gt = batch.gt_first_anchor[valid]
        loss, stats = self.loss_fn(pred=pred, gt=gt, return_stats=True)
        loss.backward()

        if self.grad_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)
        self.optimizer.step()
        return TrainOutput(loss=float(loss.item()), stats=stats)

    @torch.no_grad()
    def eval_step(self, batch: TangentBatch) -> TrainOutput:
        self.model.eval()
        batch = self._move_batch(batch)
        valid = self._valid_mask(batch)
        if valid.sum().item() == 0:
            return TrainOutput(loss=float('nan'), stats={})

        pred = self.model(batch.anchor)['pred'][valid]
        gt = batch.gt_first_anchor[valid]
        loss, stats = self.loss_fn(pred=pred, gt=gt, return_stats=True)
        return TrainOutput(loss=float(loss.item()), stats=stats)

    def _run_loader(self, loader, train: bool, desc: str) -> Dict[str, float]:
        metrics: Dict[str, float] = {}
        n = 0
        iterator = tqdm(loader, desc=desc, leave=False, dynamic_ncols=True)
        for batch in iterator:
            out = self.train_step(batch) if train else self.eval_step(batch)
            if len(out.stats) == 0:
                continue
            for k, v in out.stats.items():
                if isinstance(v, float) and (v != v):
                    continue
                metrics[k] = metrics.get(k, 0.0) + float(v)
            n += 1
            iterator.set_postfix(
                loss=f"{out.stats.get('loss', float('nan')):.5f}",
                cos=f"{out.stats.get('cosine_mean', float('nan')):.4f}",
                pn=f"{out.stats.get('pred_norm_mean', float('nan')):.4f}",
            )
        for k in list(metrics.keys()):
            metrics[k] /= max(n, 1)
        return metrics

    def _print_epoch_summary(self, epoch: int, train_metrics: Dict[str, float], val_metrics: Dict[str, float]) -> None:
        print(f"\nEpoch {epoch}", flush=True)
        print(
            "train | "
            f"loss={train_metrics.get('loss', float('nan')):.6f} "
            f"mse={train_metrics.get('mse_loss', float('nan')):.6f} "
            f"norm={train_metrics.get('norm_loss', float('nan')):.6f} "
            f"cos={train_metrics.get('cosine_mean', float('nan')):.4f} "
            f"pred_norm_mean={train_metrics.get('pred_norm_mean', float('nan')):.4f} "
            f"pred_norm_median={train_metrics.get('pred_norm_median', float('nan')):.4f} "
            f"norm_err={train_metrics.get('norm_error_mean', float('nan')):.4f}",
            flush=True,
        )
        print(
            "val   | "
            f"loss={val_metrics.get('loss', float('nan')):.6f} "
            f"mse={val_metrics.get('mse_loss', float('nan')):.6f} "
            f"norm={val_metrics.get('norm_loss', float('nan')):.6f} "
            f"cos={val_metrics.get('cosine_mean', float('nan')):.4f} "
            f"pred_norm_mean={val_metrics.get('pred_norm_mean', float('nan')):.4f} "
            f"pred_norm_median={val_metrics.get('pred_norm_median', float('nan')):.4f} "
            f"norm_err={val_metrics.get('norm_error_mean', float('nan')):.4f}",
            flush=True,
        )

    def fit(self, train_loader, val_loader, num_epochs, early_stopping_patience=10):
        best_val = float('inf')
        best_epoch = 0
        patience = 0
        best_model_path = self.checkpoint_dir / 'best_model.pt'
        torch.save(self.model.state_dict(), self.checkpoint_dir / 'init_model.pt')

        history: list[dict] = []
        for epoch in range(1, num_epochs + 1):
            train_metrics = self._run_loader(train_loader, train=True, desc=f'train {epoch}/{num_epochs}')
            val_metrics = self._run_loader(val_loader, train=False, desc=f'val   {epoch}/{num_epochs}')
            val_loss = val_metrics.get('loss', float('inf'))
            self._print_epoch_summary(epoch, train_metrics, val_metrics)

            if self.scheduler is not None:
                self.scheduler.step(val_loss)
                current_lr = self.optimizer.param_groups[0]['lr']
                print(f'lr={current_lr:.6g}', flush=True)

            history.append({'epoch': epoch, 'train': train_metrics, 'val': val_metrics})
            (self.checkpoint_dir / 'history.json').write_text(json.dumps(history, indent=2))

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

    def evaluate(self, loader, split_name='test'):
        metrics = self._run_loader(loader, train=False, desc=f'{split_name}')
        print(f'\n{split_name.capitalize()} metrics', flush=True)
        print(metrics, flush=True)
        return metrics


# -----------------------
# Global post-training evaluation
# -----------------------

def _uniform_offsets(patch_size: int, half_width: int) -> np.ndarray:
    return np.rint(np.linspace(-half_width, half_width, patch_size, endpoint=True)).astype(np.int64)


def _patch_offsets(patch_size: int, patch_mode: str, half_width: int) -> np.ndarray:
    if patch_mode == 'intrinsic_ordered_stencil':
        r = patch_size // 2
        return np.arange(-r, r + 1, dtype=np.int64)
    if patch_mode == 'uniform_symmetric':
        return _uniform_offsets(patch_size, half_width)
    raise ValueError(
        'Global operator assembly is currently only deterministic for '
        'patch_mode in {intrinsic_ordered_stencil, uniform_symmetric}. '
        f'Got {patch_mode!r}.'
    )


def _make_centered_cyclic_patches(curve_points: np.ndarray, offsets: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    curve_points = np.asarray(curve_points, dtype=np.float32)
    n = len(curve_points)
    row_indices = []
    patches = []
    for i in range(n):
        idx = (i + offsets) % n
        pts = curve_points[idx]
        patches.append(pts - curve_points[i:i+1])
        row_indices.append(idx)
    return np.stack(patches, axis=0), np.stack(row_indices, axis=0)


def _numeric_first_derivative_from_curve(curve_points: np.ndarray) -> np.ndarray:
    pts = np.asarray(curve_points, dtype=np.float64)
    prev_pt = np.roll(pts, 1, axis=0)
    next_pt = np.roll(pts, -1, axis=0)
    raw = next_pt - prev_pt
    nrm = np.linalg.norm(raw, axis=-1, keepdims=True)
    return (raw / np.clip(nrm, 1e-12, None)).astype(np.float32)


@torch.no_grad()
def evaluate_global_first_derivative(
    *,
    model,
    bank_path: str,
    device: str,
    patch_size: int,
    patch_mode: str,
    half_width: int,
    max_curves: int | None = None,
) -> dict:
    device_t = torch.device(device)
    model = model.to(device_t)
    model.eval()

    bank = PregeneratedCurveBank(bank_path)
    offsets = _patch_offsets(patch_size=patch_size, patch_mode=patch_mode, half_width=half_width)

    per_curve = []
    Ws = []
    W2_norms = []

    num_curves = len(bank) if max_curves is None else min(len(bank), int(max_curves))
    for i in range(num_curves):
        curve_points, coeffs, t_grid = bank.get(i)
        curve_points = np.asarray(curve_points, dtype=np.float32)
        n = len(curve_points)

        patches, row_indices = _make_centered_cyclic_patches(curve_points, offsets)
        patch_t = torch.from_numpy(patches).to(device_t)
        weights = model(patch_t)['weights'].detach().cpu().numpy().astype(np.float64)

        W = np.zeros((n, n), dtype=np.float64)
        rows = np.arange(n)[:, None]
        W[rows, row_indices] = weights
        global1 = W @ curve_points.astype(np.float64)

        if coeffs is not None and t_grid is not None:
            _, gt_first, _ = compute_fourier_arc_length_derivatives(
                t=np.asarray(t_grid, dtype=np.float64),
                coeffs=coeffs,
                family='euclidean',
            )
            gt_first = gt_first.astype(np.float64)
        else:
            gt_first = _numeric_first_derivative_from_curve(curve_points).astype(np.float64)

        g1_norm = np.linalg.norm(global1, axis=-1)
        gt_norm = np.linalg.norm(gt_first, axis=-1)
        global1_u = global1 / np.clip(g1_norm[:, None], 1e-12, None)
        gt_u = gt_first / np.clip(gt_norm[:, None], 1e-12, None)
        cos = np.sum(global1_u * gt_u, axis=-1)
        mse = np.mean((global1 - gt_first) ** 2)
        abs_norm_error = np.abs(g1_norm - 1.0)

        per_curve.append({
            'curve_index': i,
            'global1_cosine_mean': float(np.mean(cos)),
            'global1_mse': float(mse),
            'global1_norm_mean': float(np.mean(g1_norm)),
            'global1_norm_median': float(np.median(g1_norm)),
            'global1_norm_error_mean': float(np.mean(abs_norm_error)),
            'global1_norm_error_median': float(np.median(abs_norm_error)),
        })
        Ws.append(np.linalg.norm(W, ord='fro'))
        W2_norms.append(np.linalg.norm(W @ W, ord='fro'))

    summary = {
        'num_curves': int(num_curves),
        'global1_cosine_mean': float(np.mean([x['global1_cosine_mean'] for x in per_curve])),
        'global1_mse_mean': float(np.mean([x['global1_mse'] for x in per_curve])),
        'global1_norm_mean': float(np.mean([x['global1_norm_mean'] for x in per_curve])),
        'global1_norm_median_mean': float(np.mean([x['global1_norm_median'] for x in per_curve])),
        'global1_norm_error_mean': float(np.mean([x['global1_norm_error_mean'] for x in per_curve])),
        'global1_norm_error_median_mean': float(np.mean([x['global1_norm_error_median'] for x in per_curve])),
        'W_fro_mean': float(np.mean(Ws)) if Ws else float('nan'),
        'W2_fro_mean': float(np.mean(W2_norms)) if W2_norms else float('nan'),
        'per_curve': per_curve,
    }
    return summary
