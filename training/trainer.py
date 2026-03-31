from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import torch
from tqdm.auto import tqdm

from training.collate import TangentBatch


@dataclass
class TrainOutput:
    loss: float
    stats: Dict[str, float]


class TangentTrainer:
    def __init__(self, model, optimizer, loss_fn, device, grad_clip_norm=None, checkpoint_dir='checkpoints'):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = torch.device(device)
        self.grad_clip_norm = grad_clip_norm
        self.model.to(self.device)
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True, parents=True)

    def _move_batch(self, batch: TangentBatch) -> TangentBatch:
        batch.anchor = batch.anchor.to(self.device)
        batch.positive = batch.positive.to(self.device)
        batch.negatives = batch.negatives.to(self.device)
        batch.transform_matrix = batch.transform_matrix.to(self.device)
        batch.gt_first_anchor = batch.gt_first_anchor.to(self.device)
        batch.gt_second_anchor = batch.gt_second_anchor.to(self.device)
        batch.has_analytic_derivatives = batch.has_analytic_derivatives.to(self.device)
        return batch

    def _forward_triplet(self, batch: TangentBatch):
        anchor_out = self.model(batch.anchor)
        positive_out = self.model(batch.positive)

        B, M, P, C = batch.negatives.shape
        flat_neg = batch.negatives.view(B * M, P, C)
        neg_out = self.model(flat_neg)
        return anchor_out, positive_out, neg_out

    @staticmethod
    def _cosine_and_angle(pred: torch.Tensor, gt: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        pred_n = pred / (pred.norm(dim=-1, keepdim=True) + 1e-8)
        gt_n = gt / (gt.norm(dim=-1, keepdim=True) + 1e-8)
        cos = (pred_n * gt_n).sum(dim=-1).clamp(-1.0, 1.0)
        angle = torch.rad2deg(torch.acos(cos))
        return cos, angle

    def _derivative_metrics(
        self,
        *,
        pred: torch.Tensor,
        gt_first: torch.Tensor,
        has_analytic: torch.Tensor,
    ) -> Dict[str, float]:
        valid = has_analytic.bool() & torch.isfinite(gt_first).all(dim=-1)
        if valid.sum().item() == 0:
            return {
                'analytic_fraction': 0.0,
                'first_cosine_mean': float('nan'),
                'first_angle_deg_mean': float('nan'),
                'first_mse': float('nan'),
                'pred_first_norm_mean': float('nan'),
                'gt_first_norm_mean': float('nan'),
            }

        pred = pred[valid]
        gt_first = gt_first[valid]

        first_cos, first_angle = self._cosine_and_angle(pred, gt_first)

        return {
            'analytic_fraction': float(valid.float().mean().item()),
            'first_cosine_mean': float(first_cos.mean().item()),
            'first_angle_deg_mean': float(first_angle.mean().item()),
            'first_mse': float(torch.mean((pred - gt_first) ** 2).item()),
            'pred_first_norm_mean': float(pred.norm(dim=-1).mean().item()),
            'gt_first_norm_mean': float(gt_first.norm(dim=-1).mean().item()),
        }

    def evaluate_once(self, loader, split_name="init"):
        self.model.eval()

        total = {}
        count = 0

        with torch.no_grad():
            for batch in loader:
                batch = self._move_batch(batch)

                anchor_out, positive_out, neg_out = self._forward_triplet(batch)
                loss_inputs = self._build_loss_inputs(batch, anchor_out, positive_out, neg_out)
                _, stats = self.loss_fn(return_stats=True, **loss_inputs)

                stats.update(
                    self._derivative_metrics(
                        pred=anchor_out['pred'],
                        gt_first=batch.gt_first_anchor,
                        has_analytic=batch.has_analytic_derivatives,
                    )
                )

                for k, v in stats.items():
                    if isinstance(v, float) and (v != v):
                        continue
                    total[k] = total.get(k, 0.0) + float(v)
                count += 1

        print(f"\n[{split_name} evaluation BEFORE training]")
        for k in sorted(total.keys()):
            print(f"{k}: {total[k] / max(count, 1):.6f}")
    def _build_loss_inputs(self, batch: TangentBatch, anchor_out: dict, positive_out: dict, neg_out: dict) -> dict:
        B, M = batch.negatives.shape[:2]

        pred_anchor_equivariant = torch.einsum('bd,bed->be', anchor_out['pred'], batch.transform_matrix)
        pred_positive = positive_out['pred']
        pred_negatives = neg_out['pred'].view(B, M, -1)

        return {
            'pred_anchor_equivariant': pred_anchor_equivariant,
            'pred_positive': pred_positive,
            'pred_negatives': pred_negatives,
            'weights': anchor_out['weights'],
        }

    def train_step(self, batch: TangentBatch) -> TrainOutput:
        self.model.train()
        batch = self._move_batch(batch)
        self.optimizer.zero_grad(set_to_none=True)

        anchor_out, positive_out, neg_out = self._forward_triplet(batch)
        loss_inputs = self._build_loss_inputs(batch, anchor_out, positive_out, neg_out)
        loss, stats = self.loss_fn(return_stats=True, **loss_inputs)
        loss.backward()

        if self.grad_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)
        self.optimizer.step()

        with torch.no_grad():
            stats.update(
                self._derivative_metrics(
                    pred=anchor_out['pred'],
                    gt_first=batch.gt_first_anchor,
                    has_analytic=batch.has_analytic_derivatives,
                )
            )
        return TrainOutput(loss=float(loss.item()), stats=stats)

    @torch.no_grad()
    def eval_step(self, batch: TangentBatch) -> TrainOutput:
        self.model.eval()
        batch = self._move_batch(batch)
        anchor_out, positive_out, neg_out = self._forward_triplet(batch)
        loss_inputs = self._build_loss_inputs(batch, anchor_out, positive_out, neg_out)
        loss, stats = self.loss_fn(return_stats=True, **loss_inputs)
        stats.update(
            self._derivative_metrics(
                pred=anchor_out['pred'],
                gt_first=batch.gt_first_anchor,
                has_analytic=batch.has_analytic_derivatives,
            )
        )
        return TrainOutput(loss=float(loss.item()), stats=stats)

    def _run_loader(self, loader, train: bool, desc: str):
        metrics = {}
        n = 0
        iterator = tqdm(loader, desc=desc, leave=False, dynamic_ncols=True)
        for batch in iterator:
            out = self.train_step(batch) if train else self.eval_step(batch)
            for k, v in out.stats.items():
                if isinstance(v, float) and (v != v):
                    continue
                metrics[k] = metrics.get(k, 0.0) + float(v)
            n += 1
            if 'loss' in out.stats:
                iterator.set_postfix(
                    loss=f"{out.stats['loss']:.4f}",
                    eqmse=f"{out.stats.get('eq_raw_loss', float('nan')):.4f}",
                    eqcos=f"{out.stats.get('eq_cos_mean', float('nan')):.3f}",
                )
        for k in list(metrics.keys()):
            metrics[k] /= max(n, 1)
        return metrics

    def _print_epoch_summary(self, epoch: int, train_metrics: Dict[str, float], val_metrics: Dict[str, float]) -> None:
        print(f"\nEpoch {epoch}", flush=True)
        print(
            "train | "
            f"loss={train_metrics.get('loss', float('nan')):.4f} "
            f"nce={train_metrics.get('nce_loss', float('nan')):.4f} "
            f"eqmse={train_metrics.get('eq_raw_loss', float('nan')):.6f} "
            f"eqnorm={train_metrics.get('eq_norm_mse', float('nan')):.6f} "
            f"eqcos={train_metrics.get('eq_cos_mean', float('nan')):.4f} "
            f"wnorm={train_metrics.get('weight_l2_mean', float('nan')):.4f}",
            flush=True,
        )
        print(
            "train analytic  | "
            f"cos1={train_metrics.get('first_cosine_mean', float('nan')):.4f} "
            f"ang1={train_metrics.get('first_angle_deg_mean', float('nan')):.2f}° "
            f"mse1={train_metrics.get('first_mse', float('nan')):.6f}",
            flush=True,
        )
        print(
            "val   | "
            f"loss={val_metrics.get('loss', float('nan')):.4f} "
            f"nce={val_metrics.get('nce_loss', float('nan')):.4f} "
            f"eqmse={val_metrics.get('eq_raw_loss', float('nan')):.6f} "
            f"eqnorm={val_metrics.get('eq_norm_mse', float('nan')):.6f} "
            f"eqcos={val_metrics.get('eq_cos_mean', float('nan')):.4f} "
            f"analytic={val_metrics.get('analytic_fraction', float('nan')):.2f}",
            flush=True,
        )
        print(
            "val   analytic  | "
            f"cos1={val_metrics.get('first_cosine_mean', float('nan')):.4f} "
            f"ang1={val_metrics.get('first_angle_deg_mean', float('nan')):.2f}° "
            f"mse1={val_metrics.get('first_mse', float('nan')):.6f}",
            flush=True,
        )

    def fit(self, train_loader, val_loader, num_epochs, early_stopping_patience=10):
        best_val = float('inf')
        best_epoch = 0
        patience = 0
        best_model_path = self.checkpoint_dir / 'best_model.pt'
        self.evaluate_once(train_loader, split_name="train_init")
        self.evaluate_once(val_loader, split_name="val_init")
        for epoch in range(1, num_epochs + 1):
            train_metrics = self._run_loader(train_loader, train=True, desc=f'train {epoch}/{num_epochs}')
            val_metrics = self._run_loader(val_loader, train=False, desc=f'val   {epoch}/{num_epochs}')
            val_loss = val_metrics.get('loss', float('inf'))
            self._print_epoch_summary(epoch, train_metrics, val_metrics)

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
        print(
            f"{split_name} analytic | "
            f"cos1={metrics.get('first_cosine_mean', float('nan')):.4f} "
            f"ang1={metrics.get('first_angle_deg_mean', float('nan')):.2f}° "
            f"mse1={metrics.get('first_mse', float('nan')):.6f}",
            flush=True,
        )
        return metrics