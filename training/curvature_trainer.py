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


class CurvatureTrainer:
    def __init__(
        self,
        model,
        optimizer,
        scheduler,
        loss_fn,
        device,
        grad_clip_norm=None,
        checkpoint_dir="checkpoints",
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
        batch.gt_second_anchor = batch.gt_second_anchor.to(self.device)
        return batch

    def train_step(self, batch: TangentBatch) -> TrainOutput:
        self.model.train()
        batch = self._move_batch(batch)

        self.optimizer.zero_grad(set_to_none=True)

        out = self.model(batch.anchor)
        pred = out["pred"]
        weights = out["weights"]

        loss, stats = self.loss_fn(
            pred=pred,
            gt_second=batch.gt_second_anchor,
            weights=weights,
            return_stats=True,
        )

        loss.backward()

        if self.grad_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)

        self.optimizer.step()

        return TrainOutput(loss=float(loss.item()), stats=stats)

    @torch.no_grad()
    def eval_step(self, batch: TangentBatch) -> TrainOutput:
        self.model.eval()
        batch = self._move_batch(batch)

        out = self.model(batch.anchor)
        pred = out["pred"]
        weights = out["weights"]

        loss, stats = self.loss_fn(
            pred=pred,
            gt_second=batch.gt_second_anchor,
            weights=weights,
            return_stats=True,
        )

        return TrainOutput(loss=float(loss.item()), stats=stats)

    def _run_loader(self, loader, train: bool, desc: str):
        metrics = {}
        n = 0

        iterator = tqdm(loader, desc=desc, leave=False, dynamic_ncols=True)

        for batch in iterator:
            out = self.train_step(batch) if train else self.eval_step(batch)

            for k, v in out.stats.items():
                metrics[k] = metrics.get(k, 0.0) + float(v)

            n += 1

            iterator.set_postfix(
                loss=f"{out.stats['loss']:.4f}",
                mse=f"{out.stats.get('mse', float('nan')):.4f}",
                cos=f"{out.stats.get('cosine_mean', float('nan')):.3f}",
            )

        for k in metrics:
            metrics[k] /= max(n, 1)

        return metrics

    def fit(self, train_loader, val_loader, num_epochs, early_stopping_patience=10):
        best_val = float("inf")
        patience = 0

        best_model_path = self.checkpoint_dir / "best_model.pt"

        for epoch in range(1, num_epochs + 1):
            train_metrics = self._run_loader(train_loader, True, f"train {epoch}")
            val_metrics = self._run_loader(val_loader, False, f"val {epoch}")

            print(f"\nEpoch {epoch}")
            print(
                f"train | loss={train_metrics['loss']:.4f} "
                f"mse={train_metrics['mse']:.6f} "
                f"cos={train_metrics['cosine_mean']:.4f}"
            )
            print(
                f"val   | loss={val_metrics['loss']:.4f} "
                f"mse={val_metrics['mse']:.6f} "
                f"cos={val_metrics['cosine_mean']:.4f}"
            )

            if self.scheduler is not None:
                self.scheduler.step(val_metrics["loss"])

            if val_metrics["loss"] < best_val:
                best_val = val_metrics["loss"]
                patience = 0
                torch.save(self.model.state_dict(), best_model_path)
                print("✓ saved best")
            else:
                patience += 1

            if patience >= early_stopping_patience:
                print("Early stopping")
                break

        self.model.load_state_dict(torch.load(best_model_path))
        return best_model_path

    def evaluate(self, loader):
        metrics = self._run_loader(loader, False, "test")

        print("\nTest metrics:")
        print(metrics)

        return metrics