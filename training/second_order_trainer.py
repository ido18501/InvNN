from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import torch
from tqdm.auto import tqdm

from second_order_dataset import SecondOrderBatch


@dataclass
class StepOutput:
    loss: float
    stats: Dict[str, float]


class SecondOrderTrainer:
    def __init__(
        self,
        *,
        model,
        optimizer,
        scheduler,
        loss_fn,
        device,
        grad_clip_norm: float | None = None,
        checkpoint_dir: str | Path = "checkpoints_second_order_only",
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss_fn = loss_fn
        self.device = torch.device(device)
        self.grad_clip_norm = grad_clip_norm

        self.model.to(self.device)
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def _move_batch(self, batch: SecondOrderBatch) -> SecondOrderBatch:
        batch.patch = batch.patch.to(self.device)
        batch.target_second = batch.target_second.to(self.device)
        batch.center_index = batch.center_index.to(self.device)
        batch.relative_offsets = batch.relative_offsets.to(self.device)
        batch.curve_idx = batch.curve_idx.to(self.device)
        return batch

    def train_step(self, batch: SecondOrderBatch) -> StepOutput:
        self.model.train()
        batch = self._move_batch(batch)

        self.optimizer.zero_grad(set_to_none=True)

        out = self.model(batch.patch)
        loss, stats = self.loss_fn(
            pred_second=out["pred_second"],
            target_second=batch.target_second,
            weights=out["weights"],
            return_stats=True,
        )
        loss.backward()

        if self.grad_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)

        self.optimizer.step()
        return StepOutput(loss=float(loss.item()), stats=stats)

    @torch.no_grad()
    def eval_step(self, batch: SecondOrderBatch) -> StepOutput:
        self.model.eval()
        batch = self._move_batch(batch)

        out = self.model(batch.patch)
        loss, stats = self.loss_fn(
            pred_second=out["pred_second"],
            target_second=batch.target_second,
            weights=out["weights"],
            return_stats=True,
        )
        return StepOutput(loss=float(loss.item()), stats=stats)

    def _run_loader(self, loader, *, train: bool, desc: str) -> Dict[str, float]:
        sums: Dict[str, float] = {}
        n = 0

        iterator = tqdm(loader, desc=desc, leave=False, dynamic_ncols=True)
        for batch in iterator:
            out = self.train_step(batch) if train else self.eval_step(batch)
            for k, v in out.stats.items():
                sums[k] = sums.get(k, 0.0) + float(v)
            n += 1

            iterator.set_postfix(
                loss=f"{out.stats.get('loss', float('nan')):.4f}",
                cos2=f"{out.stats.get('cos2_mean', float('nan')):.3f}",
                mse2=f"{out.stats.get('mse2', float('nan')):.4f}",
            )

        if n == 0:
            return sums

        for k in list(sums.keys()):
            sums[k] /= n
        return sums

    def _print_summary(self, epoch: int, train_metrics: Dict[str, float], val_metrics: Dict[str, float]) -> None:
        print(f"\nEpoch {epoch}", flush=True)
        print(
            "train | "
            f"loss={train_metrics.get('loss', float('nan')):.4f} "
            f"mse2={train_metrics.get('mse2', float('nan')):.6f} "
            f"cos2={train_metrics.get('cos2_mean', float('nan')):.4f} "
            f"angle2={train_metrics.get('angle2_deg_mean', float('nan')):.2f}° "
            f"lognorm2={train_metrics.get('log_norm_mse2', float('nan')):.6f} "
            f"rowsum={train_metrics.get('rowsum_penalty', float('nan')):.6f}",
            flush=True,
        )
        print(
            "val   | "
            f"loss={val_metrics.get('loss', float('nan')):.4f} "
            f"mse2={val_metrics.get('mse2', float('nan')):.6f} "
            f"cos2={val_metrics.get('cos2_mean', float('nan')):.4f} "
            f"angle2={val_metrics.get('angle2_deg_mean', float('nan')):.2f}° "
            f"lognorm2={val_metrics.get('log_norm_mse2', float('nan')):.6f} "
            f"rowsum={val_metrics.get('rowsum_penalty', float('nan')):.6f} "
            f"ratio={val_metrics.get('norm_ratio_mean', float('nan')):.4f}",
            flush=True,
        )

    def fit(self, train_loader, val_loader, *, num_epochs: int, early_stopping_patience: int = 10) -> Path:
        best_val = float("inf")
        best_epoch = 0
        patience = 0

        best_model_path = self.checkpoint_dir / "best_model.pt"
        history_path = self.checkpoint_dir / "history.jsonl"

        torch.save(self.model.state_dict(), self.checkpoint_dir / "init_model.pt")

        with history_path.open("w", encoding="utf-8") as history_f:
            for epoch in range(1, num_epochs + 1):
                train_metrics = self._run_loader(train_loader, train=True, desc=f"train {epoch}/{num_epochs}")
                val_metrics = self._run_loader(val_loader, train=False, desc=f"val {epoch}/{num_epochs}")

                self._print_summary(epoch, train_metrics, val_metrics)

                record = {
                    "epoch": epoch,
                    "train": train_metrics,
                    "val": val_metrics,
                }
                history_f.write(json.dumps(record) + "\n")
                history_f.flush()

                val_loss = val_metrics.get("loss", float("inf"))

                if self.scheduler is not None:
                    if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                        self.scheduler.step(val_loss)
                    else:
                        self.scheduler.step()
                    print(f"lr={self.optimizer.param_groups[0]['lr']:.6g}", flush=True)

                if val_loss < best_val:
                    best_val = val_loss
                    best_epoch = epoch
                    patience = 0
                    torch.save(self.model.state_dict(), best_model_path)
                    print("✓ saved new best model", flush=True)
                else:
                    patience += 1
                    print(f"no improvement ({patience}/{early_stopping_patience})", flush=True)

                if patience >= early_stopping_patience:
                    print("Early stopping triggered", flush=True)
                    break

        print(f"\nBest validation epoch: {best_epoch}", flush=True)
        self.model.load_state_dict(torch.load(best_model_path, map_location=self.device))
        (self.checkpoint_dir / "DONE").write_text("", encoding="utf-8")
        return best_model_path

    def evaluate(self, loader, *, split_name: str = "test") -> Dict[str, float]:
        metrics = self._run_loader(loader, train=False, desc=split_name)
        print(f"\n{split_name.capitalize()} metrics", flush=True)
        print(json.dumps(metrics, indent=2), flush=True)
        return metrics
