from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tqdm.auto import tqdm

from datasets.tangent_dataset import PregeneratedCurveBank
from models.tangent_model import TangentOperatorModel
from training.operator_finetune_utils import (
    CurveBankIndexSampler,
    analytic_arc_length_derivatives_for_curve,
    build_sampled_operator_targets,
    compute_full_curve_operator_state,
    compute_sampled_operator_predictions,
)
from training.second_order_finetune_losses import RobustOperatorSupervisionLoss


@dataclass
class EvalAccumulator:
    cosine_sum: float = 0.0
    abs_cosine_sum: float = 0.0
    angle_sum: float = 0.0
    mse_sum: float = 0.0
    pred_norm_sum: float = 0.0
    norm_error_sum: float = 0.0
    count: int = 0
    pred_norm_values: list[float] | None = None
    norm_error_values: list[float] | None = None
    gt_norm_values: list[float] | None = None

    def __post_init__(self):
        if self.pred_norm_values is None:
            self.pred_norm_values = []
        if self.norm_error_values is None:
            self.norm_error_values = []
        if self.gt_norm_values is None:
            self.gt_norm_values = []


@dataclass
class CurveEvalResult:
    metrics: dict[str, float]
    series: dict[str, np.ndarray]


class SecondOrderFineTuneTrainer:
    def __init__(
        self,
        *,
        model: TangentOperatorModel,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.ReduceLROnPlateau | None,
        loss_fn: RobustOperatorSupervisionLoss,
        train_bank: PregeneratedCurveBank,
        val_bank: PregeneratedCurveBank,
        test_bank: PregeneratedCurveBank,
        family: str,
        patch_size: int,
        device: str | torch.device,
        checkpoint_dir: str | Path,
        grad_clip_norm: float | None = 1.0,
        num_anchors_train: int = 128,
        num_anchors_eval_sampled: int = 256,
        batch_size_curves: int = 4,
        seed: int = 123,
        max_eval_curves: int | None = None,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss_fn = loss_fn
        self.train_bank = train_bank
        self.val_bank = val_bank
        self.test_bank = test_bank
        self.family = family
        self.patch_size = int(patch_size)
        self.device = torch.device(device)
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.grad_clip_norm = grad_clip_norm
        self.num_anchors_train = int(num_anchors_train)
        self.num_anchors_eval_sampled = int(num_anchors_eval_sampled)
        self.batch_size_curves = int(batch_size_curves)
        self.seed = int(seed)
        self.max_eval_curves = max_eval_curves

        self.model.to(self.device)
        self._reference_params = [p.detach().clone().to(self.device) for p in self.model.operator_head.parameters() if p.requires_grad]
        self._train_sampler = CurveBankIndexSampler(len(self.train_bank), self.batch_size_curves, self.seed)
        self._val_sampler = CurveBankIndexSampler(len(self.val_bank), self.batch_size_curves, self.seed + 999)

    def _operator_params(self) -> list[torch.Tensor]:
        return [p for p in self.model.operator_head.parameters() if p.requires_grad]

    def _load_curve(self, bank: PregeneratedCurveBank, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        curve_np, coeffs, t_grid = bank.get(idx)
        if coeffs is None or t_grid is None:
            raise RuntimeError(
                f'Pregenerated bank item {idx} does not contain coeffs/t_grid, '
                'but stage-2 fine-tune requires analytic derivatives.'
            )
        curve_xy = torch.as_tensor(curve_np, dtype=torch.float32, device=self.device)
        gt1, gt2 = analytic_arc_length_derivatives_for_curve(
            curve_xy=curve_xy,
            coeffs=coeffs,
            t_grid=t_grid,
            family=self.family,
            device=self.device,
        )
        return curve_xy, gt1, gt2

    def _rng_for(self, epoch: int, batch_idx: int, curve_idx: int) -> torch.Generator:
        g = torch.Generator(device=self.device.type if self.device.type == 'cuda' else 'cpu')
        g.manual_seed(self.seed + 100000 * epoch + 1000 * batch_idx + curve_idx)
        return g

    def _run_curve_sampled_step(
        self,
        *,
        curve_xy: torch.Tensor,
        gt1: torch.Tensor,
        gt2: torch.Tensor,
        num_anchors: int,
        generator: torch.Generator,
        train: bool,
    ) -> dict[str, float] | tuple[torch.Tensor, dict[str, float]]:
        sampled = build_sampled_operator_targets(
            curve_xy=curve_xy,
            gt_first=gt1,
            gt_second=gt2,
            patch_size=self.patch_size,
            num_anchors=num_anchors,
            generator=generator,
        )
        preds = compute_sampled_operator_predictions(self.model, sampled)

        out = self.loss_fn(
            pred1=preds['global1_anchor'],
            gt1=sampled.gt_first_anchor,
            pred2=preds['global2_anchor'],
            gt2=sampled.gt_second_anchor,
            row_sum=preds['row_sum_anchor'],
            current_params=self._operator_params(),
            reference_params=self._reference_params,
            return_stats=True,
        )
        loss, stats = out

        with torch.no_grad():
            alt_cos = torch.nn.functional.cosine_similarity(preds['global2_alt_anchor'], sampled.gt_second_anchor, dim=-1)
            stats['second_alt_cos_mean'] = float(alt_cos.mean().item())
            stats['num_anchors'] = float(sampled.anchor_indices.numel())

        if train:
            return loss, stats
        return stats

    def _run_epoch(self, *, epoch: int, train: bool) -> dict[str, float]:
        self.model.train(train)
        batches = self._train_sampler.iter_batches(shuffle=True) if train else self._val_sampler.iter_batches(shuffle=False)
        bank = self.train_bank if train else self.val_bank

        totals: dict[str, float] = {}
        denom = 0
        desc = f"{'train' if train else 'val'} {epoch}"
        iterator = tqdm(enumerate(batches), total=len(batches), leave=False, dynamic_ncols=True, desc=desc)

        for batch_idx, curve_indices in iterator:
            if train:
                self.optimizer.zero_grad(set_to_none=True)
                batch_loss = None
                batch_stats_accum: dict[str, float] = {}

                for local_pos, curve_idx in enumerate(curve_indices):
                    curve_xy, gt1, gt2 = self._load_curve(bank, curve_idx)
                    generator = self._rng_for(epoch, batch_idx, curve_idx)
                    loss, stats = self._run_curve_sampled_step(
                        curve_xy=curve_xy,
                        gt1=gt1,
                        gt2=gt2,
                        num_anchors=self.num_anchors_train,
                        generator=generator,
                        train=True,
                    )
                    batch_loss = loss if batch_loss is None else batch_loss + loss
                    for k, v in stats.items():
                        batch_stats_accum[k] = batch_stats_accum.get(k, 0.0) + float(v)

                if batch_loss is None:
                    continue
                batch_loss = batch_loss / max(len(curve_indices), 1)
                batch_loss.backward()
                if self.grad_clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)
                self.optimizer.step()

                batch_stats_accum['loss'] = float(batch_loss.item())
                for k in list(batch_stats_accum.keys()):
                    batch_stats_accum[k] /= max(len(curve_indices), 1)
                batch_stats = batch_stats_accum
            else:
                with torch.no_grad():
                    batch_stats_accum: dict[str, float] = {}
                    for curve_idx in curve_indices:
                        curve_xy, gt1, gt2 = self._load_curve(bank, curve_idx)
                        generator = self._rng_for(epoch, batch_idx, curve_idx)
                        stats = self._run_curve_sampled_step(
                            curve_xy=curve_xy,
                            gt1=gt1,
                            gt2=gt2,
                            num_anchors=self.num_anchors_eval_sampled,
                            generator=generator,
                            train=False,
                        )
                        for k, v in stats.items():
                            batch_stats_accum[k] = batch_stats_accum.get(k, 0.0) + float(v)
                    for k in list(batch_stats_accum.keys()):
                        batch_stats_accum[k] /= max(len(curve_indices), 1)
                    batch_stats = batch_stats_accum

            denom += 1
            for k, v in batch_stats.items():
                totals[k] = totals.get(k, 0.0) + float(v)

            iterator.set_postfix(
                loss=f"{batch_stats.get('loss', float('nan')):.4f}",
                cos1=f"{batch_stats.get('first_cos_mean', float('nan')):.3f}",
                cos2=f"{batch_stats.get('second_cos_mean', float('nan')):.3f}",
            )

        if denom == 0:
            return {}
        for k in list(totals.keys()):
            totals[k] /= denom
        return totals

    @staticmethod
    def _cosine_and_angle(pred: torch.Tensor, gt: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        pred_n = pred / (pred.norm(dim=-1, keepdim=True) + 1e-8)
        gt_n = gt / (gt.norm(dim=-1, keepdim=True) + 1e-8)
        cos = (pred_n * gt_n).sum(dim=-1).clamp(-1.0, 1.0)
        angle = torch.rad2deg(torch.acos(cos))
        return cos, angle

    @staticmethod
    def _rankdata(values: np.ndarray) -> np.ndarray:
        order = np.argsort(values)
        ranks = np.empty_like(order, dtype=np.float64)
        ranks[order] = np.arange(len(values), dtype=np.float64)
        return ranks

    @classmethod
    def _pearson(cls, x: np.ndarray, y: np.ndarray) -> float:
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        if len(x) < 2:
            return float('nan')
        x = x - x.mean()
        y = y - y.mean()
        denom = np.sqrt(np.sum(x * x) * np.sum(y * y))
        if denom <= 1e-12:
            return float('nan')
        return float(np.sum(x * y) / denom)

    @classmethod
    def _spearman(cls, x: np.ndarray, y: np.ndarray) -> float:
        return cls._pearson(cls._rankdata(x), cls._rankdata(y))

    @classmethod
    def _linear_fit(cls, x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
        if len(x) < 2:
            return float('nan'), float('nan')
        slope, intercept = np.polyfit(x, y, deg=1)
        return float(slope), float(intercept)

    def _update_accumulator(self, acc: EvalAccumulator, pred: torch.Tensor, gt: torch.Tensor) -> None:
        cos, angle = self._cosine_and_angle(pred, gt)
        pred_norm = pred.norm(dim=-1)
        gt_norm = gt.norm(dim=-1)
        norm_err = (pred_norm - gt_norm).abs()
        mse = ((pred - gt) ** 2).mean(dim=-1)

        acc.cosine_sum += float(cos.sum().item())
        acc.abs_cosine_sum += float(cos.abs().sum().item())
        acc.angle_sum += float(angle.sum().item())
        acc.mse_sum += float(mse.sum().item())
        acc.pred_norm_sum += float(pred_norm.sum().item())
        acc.norm_error_sum += float(norm_err.sum().item())
        acc.count += int(pred.shape[0])
        acc.pred_norm_values.extend(pred_norm.detach().cpu().tolist())
        acc.norm_error_values.extend(norm_err.detach().cpu().tolist())
        acc.gt_norm_values.extend(gt_norm.detach().cpu().tolist())

    def _finalize_accumulator(self, prefix: str, acc: EvalAccumulator) -> dict[str, float]:
        denom = max(acc.count, 1)
        out = {
            f'{prefix}_cosine_mean': acc.cosine_sum / denom,
            f'{prefix}_abs_cosine_mean': acc.abs_cosine_sum / denom,
            f'{prefix}_angle_mean': acc.angle_sum / denom,
            f'{prefix}_mse': acc.mse_sum / denom,
            f'{prefix}_pred_norm_mean': acc.pred_norm_sum / denom,
            f'{prefix}_pred_norm_median': float(np.median(acc.pred_norm_values)) if acc.pred_norm_values else float('nan'),
            f'{prefix}_norm_error_mean': acc.norm_error_sum / denom,
            f'{prefix}_norm_error_median': float(np.median(acc.norm_error_values)) if acc.norm_error_values else float('nan'),
        }
        return out

    @torch.no_grad()
    def evaluate_full_bank(self, bank: PregeneratedCurveBank, split_name: str, save_path: str | Path | None = None) -> dict[str, Any]:
        self.model.eval()

        local1_acc = EvalAccumulator()
        global1_acc = EvalAccumulator()
        global2_acc = EvalAccumulator()
        global2_alt_acc = EvalAccumulator()

        global2_pred_norms: list[float] = []
        global2_gt_norms: list[float] = []
        global2_alt_pred_norms: list[float] = []
        row_sum_abs_means: list[float] = []

        num_curves = len(bank) if self.max_eval_curves is None else min(len(bank), int(self.max_eval_curves))

        for curve_idx in tqdm(range(num_curves), desc=f'full eval {split_name}', leave=False, dynamic_ncols=True):
            curve_xy, gt1, gt2 = self._load_curve(bank, curve_idx)
            state = compute_full_curve_operator_state(self.model, curve_xy, self.patch_size)

            self._update_accumulator(local1_acc, state.direct1, gt1)
            self._update_accumulator(global1_acc, state.global1, gt1)
            self._update_accumulator(global2_acc, state.global2, gt2)
            self._update_accumulator(global2_alt_acc, state.global2_alt, gt2)

            global2_pred_norms.extend(state.global2.norm(dim=-1).detach().cpu().tolist())
            global2_gt_norms.extend(gt2.norm(dim=-1).detach().cpu().tolist())
            global2_alt_pred_norms.extend(state.global2_alt.norm(dim=-1).detach().cpu().tolist())
            row_sum_abs_means.append(float(state.row_sum.abs().mean().item()))

        summary: dict[str, Any] = {
            'split': split_name,
            'num_curves': num_curves,
            **self._finalize_accumulator('local1', local1_acc),
            **self._finalize_accumulator('global1', global1_acc),
            **self._finalize_accumulator('global2', global2_acc),
            **self._finalize_accumulator('global2_alt', global2_alt_acc),
            'row_sum_abs_mean': float(np.mean(row_sum_abs_means)) if row_sum_abs_means else float('nan'),
        }

        pred2 = np.asarray(global2_pred_norms, dtype=np.float64)
        gt2n = np.asarray(global2_gt_norms, dtype=np.float64)
        pred2_alt = np.asarray(global2_alt_pred_norms, dtype=np.float64)

        summary.update({
            'global2_norm_spearman': self._spearman(pred2, gt2n),
            'global2_norm_pearson': self._pearson(pred2, gt2n),
            'global2_log1p_norm_pearson': self._pearson(np.log1p(pred2), np.log1p(gt2n)),
            'global2_alt_norm_spearman': self._spearman(pred2_alt, gt2n),
            'global2_alt_norm_pearson': self._pearson(pred2_alt, gt2n),
            'global2_alt_log1p_norm_pearson': self._pearson(np.log1p(pred2_alt), np.log1p(gt2n)),
        })
        slope, intercept = self._linear_fit(gt2n, pred2)
        alt_slope, alt_intercept = self._linear_fit(gt2n, pred2_alt)
        summary.update({
            'global2_norm_fit_slope': slope,
            'global2_norm_fit_intercept': intercept,
            'global2_alt_norm_fit_slope': alt_slope,
            'global2_alt_norm_fit_intercept': alt_intercept,
        })

        if save_path is not None:
            Path(save_path).write_text(json.dumps(summary, indent=2))
        return summary

    def _print_epoch_summary(self, epoch: int, train_stats: dict[str, float], val_stats: dict[str, float]) -> None:
        print(f'\nEpoch {epoch}')
        print(
            'train | '
            f"loss={train_stats.get('loss', float('nan')):.4f} "
            f"cos1={train_stats.get('first_cos_mean', float('nan')):.4f} "
            f"cos2={train_stats.get('second_cos_mean', float('nan')):.4f} "
            f"mag1={train_stats.get('first_mag_loss', float('nan')):.4f} "
            f"mag2={train_stats.get('second_mag_loss', float('nan')):.4f} "
            f"log2={train_stats.get('second_log_mag_loss', float('nan')):.4f} "
            f"rowsum={train_stats.get('row_sum_abs_mean', float('nan')):.5f}"
        )
        print(
            'val   | '
            f"loss={val_stats.get('loss', float('nan')):.4f} "
            f"cos1={val_stats.get('first_cos_mean', float('nan')):.4f} "
            f"cos2={val_stats.get('second_cos_mean', float('nan')):.4f} "
            f"alt2={val_stats.get('second_alt_cos_mean', float('nan')):.4f} "
            f"mag1={val_stats.get('first_mag_loss', float('nan')):.4f} "
            f"mag2={val_stats.get('second_mag_loss', float('nan')):.4f} "
            f"log2={val_stats.get('second_log_mag_loss', float('nan')):.4f}"
        )
        if self.scheduler is not None:
            current_lr = self.optimizer.param_groups[0]['lr']
            print(f'lr={current_lr:.6g}')

    def fit(self, *, num_epochs: int, early_stopping_patience: int = 10) -> Path:
        best_val = float('inf')
        patience = 0
        best_epoch = 0
        best_model_path = self.checkpoint_dir / 'best_model.pt'
        init_model_path = self.checkpoint_dir / 'init_model.pt'
        torch.save(self.model.state_dict(), init_model_path)

        history: list[dict[str, Any]] = []

        for epoch in range(1, num_epochs + 1):
            train_stats = self._run_epoch(epoch=epoch, train=True)
            val_stats = self._run_epoch(epoch=epoch, train=False)
            self._print_epoch_summary(epoch, train_stats, val_stats)

            val_loss = float(val_stats.get('loss', float('inf')))
            if self.scheduler is not None and math.isfinite(val_loss):
                self.scheduler.step(val_loss)

            history.append({'epoch': epoch, 'train': train_stats, 'val': val_stats})
            (self.checkpoint_dir / 'history.json').write_text(json.dumps(history, indent=2))

            if val_loss < best_val:
                best_val = val_loss
                best_epoch = epoch
                patience = 0
                torch.save(self.model.state_dict(), best_model_path)
                print('✓ saved new best model')
            else:
                patience += 1
                print(f'no improvement ({patience}/{early_stopping_patience})')

            if patience >= early_stopping_patience:
                print('Early stopping triggered')
                break

        print(f'Best validation epoch: {best_epoch}')
        self.model.load_state_dict(torch.load(best_model_path, map_location=self.device))
        return best_model_path
