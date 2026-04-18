from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from datasets.tangent_dataset import TangentDataset
from models.tangent_model import TangentOperatorModel
from training.collate import tangent_collate_fn
from training.curvature_losses import CurvatureVectorLoss
from training.curvature_trainer import CurvatureTrainer


def parse_int_list(text: str):
    return [int(x) for x in text.split(",") if x]


def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument("--family", type=str, default="euclidean")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--checkpoint-dir", type=str, required=True)

    p.add_argument("--train-bank", type=str, required=True)
    p.add_argument("--val-bank", type=str, required=True)
    p.add_argument("--test-bank", type=str, required=True)

    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--num-epochs", type=int, default=40)
    p.add_argument("--lr", type=float, default=1e-3)

    p.add_argument("--patch-size", type=int, default=9)

    p.add_argument("--operator-hidden-dims", type=str, default="256,256")

    # loss
    p.add_argument("--lambda-rel", type=float, default=1.0)
    p.add_argument("--lambda-abs", type=float, default=0.05)
    p.add_argument("--lambda-cos", type=float, default=0.0)
    p.add_argument("--lambda-reg", type=float, default=1e-2)
    p.add_argument("--lambda-weight-sum", type=float, default=1e-2)
    p.add_argument("--tau-scale", type=float, default=1.0)
    p.add_argument("--tau-min", type=float, default=1e-3)

    return p.parse_args()


def make_dataset(bank_path, args):
    return TangentDataset(
        length=4096,
        family=args.family,
        source="pregenerated",
        bank_path=bank_path,
        patch_size=args.patch_size,
        patch_mode="intrinsic_ordered_stencil",
    )


def main():
    args = parse_args()

    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    (checkpoint_dir / "config.json").write_text(json.dumps(vars(args), indent=2))

    train_dataset = make_dataset(args.train_bank, args)
    val_dataset = make_dataset(args.val_bank, args)
    test_dataset = make_dataset(args.test_bank, args)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=tangent_collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=tangent_collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=tangent_collate_fn)

    model = TangentOperatorModel(
        patch_size=args.patch_size,
        operator_hidden_dims=parse_int_list(args.operator_hidden_dims),
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    loss_fn = CurvatureVectorLoss(
        lambda_rel=args.lambda_rel,
        lambda_abs=args.lambda_abs,
        lambda_cos=args.lambda_cos,
        lambda_reg=args.lambda_reg,
        lambda_weight_sum=args.lambda_weight_sum,
        tau_scale=args.tau_scale,
        tau_min=args.tau_min,
    )

    trainer = CurvatureTrainer(
        model=model,
        optimizer=optimizer,
        scheduler=None,
        loss_fn=loss_fn,
        device=args.device,
        checkpoint_dir=checkpoint_dir,
    )

    trainer.fit(train_loader, val_loader, num_epochs=args.num_epochs)
    trainer.evaluate(test_loader)


if __name__ == "__main__":
    main()