from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from second_order_dataset import SecondOrderCurveDataset, second_order_collate_fn
from second_order_losses import SecondOrderOperatorLoss
from second_order_operator_model import SecondOrderOperatorModel
from second_order_trainer import SecondOrderTrainer


def parse_int_list(text: str) -> list[int]:
    text = text.strip()
    if not text:
        return []
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--family", type=str, default="euclidean", choices=["euclidean", "similarity", "equi_affine"])

    p.add_argument("--train-bank", type=str, required=True)
    p.add_argument("--val-bank", type=str, required=True)
    p.add_argument("--test-bank", type=str, required=True)
    p.add_argument("--checkpoint-dir", type=str, required=True)

    p.add_argument("--train-length", type=int, default=4096)
    p.add_argument("--val-length", type=int, default=1024)
    p.add_argument("--test-length", type=int, default=1024)

    p.add_argument("--patch-size", type=int, default=9)
    p.add_argument("--patch-mode", type=str, default="intrinsic_ordered_stencil")
    p.add_argument("--half-width", type=int, default=0)

    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--num-epochs", type=int, default=60)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--grad-clip-norm", type=float, default=1.0)
    p.add_argument("--early-stopping-patience", type=int, default=10)

    p.add_argument("--hidden-dims", type=str, default="256,256")
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--disable-zero-sum-enforcement", action="store_true")
    p.add_argument("--learn-output-scale", action="store_true")

    p.add_argument("--lambda-mse", type=float, default=1.0)
    p.add_argument("--lambda-cos", type=float, default=0.1)
    p.add_argument("--lambda-log-norm", type=float, default=0.1)
    p.add_argument("--lambda-rowsum", type=float, default=1.0)

    p.add_argument("--lr-scheduler", type=str, default="plateau", choices=["none", "plateau"])
    p.add_argument("--lr-patience", type=int, default=4)
    p.add_argument("--lr-factor", type=float, default=0.3)
    p.add_argument("--lr-min", type=float, default=1e-5)

    return p.parse_args()


def make_dataset(args, split: str) -> SecondOrderCurveDataset:
    bank = getattr(args, f"{split}_bank")
    length = getattr(args, f"{split}_length")
    return SecondOrderCurveDataset(
        bank_path=bank,
        length=length,
        family=args.family,
        patch_size=args.patch_size,
        patch_mode=args.patch_mode,
        half_width=args.half_width,
        closed=True,
        return_centered=True,
        seed=args.seed + {"train": 0, "val": 10000, "test": 20000}[split],
    )


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    (checkpoint_dir / "config.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")

    train_dataset = make_dataset(args, "train")
    val_dataset = make_dataset(args, "val")
    test_dataset = make_dataset(args, "test")

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=second_order_collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=second_order_collate_fn,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=second_order_collate_fn,
    )

    model = SecondOrderOperatorModel(
        patch_size=args.patch_size,
        hidden_dims=parse_int_list(args.hidden_dims),
        dropout=args.dropout,
        enforce_zero_sum=not args.disable_zero_sum_enforcement,
        learn_output_scale=args.learn_output_scale,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    scheduler = None
    if args.lr_scheduler == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=args.lr_factor,
            patience=args.lr_patience,
            min_lr=args.lr_min,
        )

    loss_fn = SecondOrderOperatorLoss(
        lambda_mse=args.lambda_mse,
        lambda_cos=args.lambda_cos,
        lambda_log_norm=args.lambda_log_norm,
        lambda_rowsum=args.lambda_rowsum,
    )

    trainer = SecondOrderTrainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        loss_fn=loss_fn,
        device=args.device,
        grad_clip_norm=args.grad_clip_norm,
        checkpoint_dir=checkpoint_dir,
    )

    best_model_path = trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=args.num_epochs,
        early_stopping_patience=args.early_stopping_patience,
    )
    print(f"Best model saved at: {best_model_path}")

    trainer.evaluate(test_loader, split_name="test")


if __name__ == "__main__":
    main()
