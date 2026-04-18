from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import torch
from tqdm.auto import tqdm

from operator_finetune_utils import (
    FullCurveBankDataset,
    apply_local_stencils_to_curve,
    cosine_mean,
    extract_cyclic_patches,
    load_model_from_config,
    maybe_fix_global_sign,
    operator_losses,
    project_first_derivative_stencils,
    second_order_confidence,
)



def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Operator-level second-order fine-tuning for TangentOperatorModel.")
    p.add_argument("--family", type=str, default="euclidean")
    p.add_argument("--pretrained-checkpoint", type=str, required=True)
    p.add_argument("--config", type=str, required=True)
    p.add_argument("--checkpoint-dir", type=str, required=True)
    p.add_argument("--train-bank", type=str, required=True)
    p.add_argument("--val-bank", type=str, required=True)
    p.add_argument("--test-bank", type=str, required=True)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--num-epochs", type=int, default=20)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--grad-clip-norm", type=float, default=1.0)
    p.add_argument("--early-stopping-patience", type=int, default=6)

    p.add_argument("--alpha-cos1", type=float, default=0.25)
    p.add_argument("--alpha-vec1", type=float, default=1.0)
    p.add_argument("--beta-cos2", type=float, default=1.0)
    p.add_argument("--beta-log2", type=float, default=1.0)
    p.add_argument("--beta-lin2", type=float, default=0.15)
    p.add_argument("--lambda-stay-close", type=float, default=0.05)
    p.add_argument("--ridge", type=float, default=1e-6)
    p.add_argument("--valid-quantile-cap", type=float, default=0.99)
    p.add_argument("--sign-fix", action="store_true")
    return p.parse_args()


class OperatorFinetuner:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.device = torch.device(args.device)
        self.checkpoint_dir = Path(args.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        (self.checkpoint_dir / "config.json").write_text(json.dumps(vars(args), indent=2))

        self.model = load_model_from_config(args.pretrained_checkpoint, args.config, self.device)
        self.base_model = load_model_from_config(args.pretrained_checkpoint, args.config, self.device)
        self.base_model.eval()

        self.train_ds = FullCurveBankDataset(args.train_bank, family=args.family)
        self.val_ds = FullCurveBankDataset(args.val_bank, family=args.family)
        self.test_ds = FullCurveBankDataset(args.test_bank, family=args.family)

        self.sign = 1
        if args.sign_fix:
            self.sign = maybe_fix_global_sign(self.model, self.val_ds, device=self.device)

        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    def _forward_curve(self, curve_points: torch.Tensor, model: torch.nn.Module) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        patches = extract_cyclic_patches(curve_points, model.patch_size)
        raw_weights = model.get_weights(patches)
        proj_weights, ds = project_first_derivative_stencils(raw_weights, patches, ridge=self.args.ridge)
        global1 = apply_local_stencils_to_curve(proj_weights, patches, sign=self.sign)
        global2 = apply_local_stencils_to_curve(proj_weights, extract_cyclic_patches(global1, model.patch_size), sign=self.sign)
        return raw_weights, proj_weights, global1, global2

    def _step(self, sample, train: bool) -> dict[str, float]:
        curve = sample.curve_points.to(self.device)
        gt_first = sample.gt_first.to(self.device)
        gt_second = sample.gt_second.to(self.device)

        if train:
            self.model.train()
            self.optimizer.zero_grad(set_to_none=True)
        else:
            self.model.eval()

        with torch.set_grad_enabled(train):
            _, proj_weights, global1, global2 = self._forward_curve(curve, self.model)
            with torch.no_grad():
                _, base_proj_weights, base_global1, _ = self._forward_curve(curve, self.base_model)

            conf2, valid2, conf_stats = second_order_confidence(
                gt_second=gt_second,
                curve_points=curve,
                quantile_cap=self.args.valid_quantile_cap,
            )
            loss_main, stats = operator_losses(
                global1=global1,
                global2=global2,
                gt_first=gt_first,
                gt_second=gt_second,
                conf2=conf2,
                valid2=valid2,
                alpha_cos1=self.args.alpha_cos1,
                alpha_vec1=self.args.alpha_vec1,
                beta_cos2=self.args.beta_cos2,
                beta_log2=self.args.beta_log2,
                beta_lin2=self.args.beta_lin2,
            )
            stay_close = torch.nn.functional.mse_loss(global1, base_global1) + 0.1 * torch.nn.functional.mse_loss(proj_weights, base_proj_weights)
            loss = loss_main + self.args.lambda_stay_close * stay_close

            if train:
                loss.backward()
                if self.args.grad_clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.grad_clip_norm)
                self.optimizer.step()

        stats["loss"] = float(loss.item())
        stats["stay_close"] = float(stay_close.item())
        stats.update(conf_stats)
        return stats

    @staticmethod
    def _aggregate(stats_list: list[dict[str, float]]) -> dict[str, float]:
        out = defaultdict(float)
        count = defaultdict(int)
        for stats in stats_list:
            for k, v in stats.items():
                if isinstance(v, float) and (v != v):
                    continue
                out[k] += float(v)
                count[k] += 1
        return {k: out[k] / max(count[k], 1) for k in out}

    def _run_split(self, dataset: FullCurveBankDataset, train: bool, desc: str) -> dict[str, float]:
        stats_list = []
        iterator = tqdm(range(len(dataset)), desc=desc, leave=False, dynamic_ncols=True)
        for idx in iterator:
            stats = self._step(dataset[idx], train=train)
            stats_list.append(stats)
            iterator.set_postfix(loss=f"{stats['loss']:.4f}", cos1=f"{stats['cos1']:.4f}", cos2=f"{stats['cos2_full']:.4f}", pear2=f"{stats.get('pearson2_valid', float('nan')):.4f}")
        return self._aggregate(stats_list)

    @staticmethod
    def _print_epoch(epoch: int, train_stats: dict[str, float], val_stats: dict[str, float]) -> None:
        print(f"\nEpoch {epoch}")
        print(
            "train | "
            f"loss={train_stats.get('loss', float('nan')):.4f} "
            f"cos1={train_stats.get('cos1', float('nan')):.4f} "
            f"cos2={train_stats.get('cos2_full', float('nan')):.4f} "
            f"log2={train_stats.get('log2_huber', float('nan')):.4f} "
            f"lin2={train_stats.get('lin2_huber', float('nan')):.4f} "
            f"pear2_valid={train_stats.get('pearson2_valid', float('nan')):.4f} "
            f"slope2_valid={train_stats.get('slope2_valid', float('nan')):.4f}"
        )
        print(
            "val   | "
            f"loss={val_stats.get('loss', float('nan')):.4f} "
            f"cos1={val_stats.get('cos1', float('nan')):.4f} "
            f"cos2={val_stats.get('cos2_full', float('nan')):.4f} "
            f"log2={val_stats.get('log2_huber', float('nan')):.4f} "
            f"lin2={val_stats.get('lin2_huber', float('nan')):.4f} "
            f"pear2_valid={val_stats.get('pearson2_valid', float('nan')):.4f} "
            f"slope2_valid={val_stats.get('slope2_valid', float('nan')):.4f} "
            f"valid2={val_stats.get('valid2_fraction', float('nan')):.3f}"
        )

    def fit(self) -> Path:
        best_score = float("-inf")
        patience = 0
        best_path = self.checkpoint_dir / "best_model.pt"
        torch.save(self.model.state_dict(), self.checkpoint_dir / "init_model.pt")

        for epoch in range(1, self.args.num_epochs + 1):
            train_stats = self._run_split(self.train_ds, train=True, desc=f"train {epoch}/{self.args.num_epochs}")
            val_stats = self._run_split(self.val_ds, train=False, desc=f"val {epoch}/{self.args.num_epochs}")
            self._print_epoch(epoch, train_stats, val_stats)

            score = val_stats.get("pearson2_valid", float("-inf")) - abs(val_stats.get("slope2_valid", 1.0) - 1.0)
            if score > best_score:
                best_score = score
                patience = 0
                torch.save(self.model.state_dict(), best_path)
                print("✓ saved new best model")
            else:
                patience += 1
                print(f"no improvement ({patience}/{self.args.early_stopping_patience})")
            if patience >= self.args.early_stopping_patience:
                print("Early stopping triggered")
                break

        self.model.load_state_dict(torch.load(best_path, map_location=self.device))
        return best_path

    def evaluate(self, dataset: FullCurveBankDataset, split_name: str) -> dict[str, float]:
        stats = self._run_split(dataset, train=False, desc=split_name)
        print(f"\n{split_name} metrics")
        print(json.dumps(stats, indent=2))
        return stats



def main() -> None:
    args = parse_args()
    trainer = OperatorFinetuner(args)
    best_path = trainer.fit()
    print(f"Best model saved at: {best_path}")
    trainer.evaluate(trainer.test_ds, split_name="test")


if __name__ == "__main__":
    main()
