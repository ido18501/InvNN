from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from models.tangent_model import TangentOperatorModel
from training.second_derivative_operator_losses import RobustSecondDerivativeLoss
from training.second_derivative_operator_trainer import (
    FullCurveSecondDerivativeDataset,
    SecondDerivativeOperatorTrainer,
    full_curve_collate,
)


def parse_int_list(text: str | list[int]) -> list[int]:
    if isinstance(text, list):
        return [int(x) for x in text]
    text = text.strip()
    if not text:
        return []
    return [int(x.strip()) for x in text.split(',') if x.strip()]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--output-dir', type=str, required=True)
    p.add_argument('--family', type=str, default='euclidean', choices=['euclidean', 'similarity', 'equi_affine'])
    p.add_argument('--train-bank', type=str, required=True)
    p.add_argument('--val-bank', type=str, required=True)
    p.add_argument('--test-bank', type=str, required=True)

    p.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--seed', type=int, default=123)

    p.add_argument('--curve-batch-size', type=int, default=4)
    p.add_argument('--num-workers', type=int, default=2)
    p.add_argument('--num-epochs', type=int, default=20)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--weight-decay', type=float, default=1e-5)
    p.add_argument('--grad-clip-norm', type=float, default=1.0)
    p.add_argument('--early-stopping-patience', type=int, default=6)
    p.add_argument('--train-num-anchors', type=int, default=128)

    p.add_argument('--patch-size', type=int, default=21)
    p.add_argument('--operator-hidden-dims', type=str, default='256,256')
    p.add_argument('--signature-hidden-dims', type=str, default='128,64')
    p.add_argument('--signature-out-dim', type=int, default=64)
    p.add_argument('--signature-center-radius', type=int, default=0)
    p.add_argument('--head-dropout', type=float, default=0.0)
    p.add_argument('--disable-normalize-projector', action='store_true')
    p.add_argument('--disable-centered-input-for-operator', action='store_true')
    p.add_argument('--operator-init-scale', type=float, default=0.05)
    p.add_argument('--learn-output-scale', action='store_true')

    p.add_argument('--lambda-vec', type=float, default=1.0)
    p.add_argument('--lambda-cos', type=float, default=0.02)
    p.add_argument('--lambda-log', type=float, default=0.01)
    p.add_argument('--lambda-rowsum', type=float, default=0.0)
    p.add_argument('--lambda-weight-l2', type=float, default=1e-5)
    p.add_argument('--huber-delta', type=float, default=0.1)

    p.add_argument('--lr-scheduler', type=str, default='plateau', choices=['none', 'plateau'])
    p.add_argument('--lr-patience', type=int, default=2)
    p.add_argument('--lr-factor', type=float, default=0.5)
    p.add_argument('--lr-min', type=float, default=1e-6)
    return p.parse_args()


def build_loader(bank_path: str, family: str, batch_size: int, num_workers: int, shuffle: bool) -> DataLoader:
    dataset = FullCurveSecondDerivativeDataset(bank_path=bank_path, family=family)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=full_curve_collate,
        drop_last=False,
    )


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / 'config.json', 'w', encoding='utf-8') as f:
        json.dump(vars(args), f, indent=2)

    model = TangentOperatorModel(
        patch_size=args.patch_size,
        operator_hidden_dims=parse_int_list(args.operator_hidden_dims),
        signature_hidden_dims=parse_int_list(args.signature_hidden_dims),
        signature_out_dim=args.signature_out_dim,
        signature_center_radius=args.signature_center_radius,
        head_dropout=args.head_dropout,
        normalize_projector=not args.disable_normalize_projector,
        init_scale=args.operator_init_scale,
        learn_scale=args.learn_output_scale,
        centered_input_for_operator=not args.disable_centered_input_for_operator,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = None
    if args.lr_scheduler == 'plateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=args.lr_factor,
            patience=args.lr_patience,
            min_lr=args.lr_min,
        )

    loss_fn = RobustSecondDerivativeLoss(
        lambda_vec=args.lambda_vec,
        lambda_cos=args.lambda_cos,
        lambda_log=args.lambda_log,
        lambda_rowsum=args.lambda_rowsum,
        lambda_weight_l2=args.lambda_weight_l2,
        huber_delta=args.huber_delta,
    )

    train_loader = build_loader(args.train_bank, family=args.family, batch_size=args.curve_batch_size, num_workers=args.num_workers, shuffle=True)
    val_loader = build_loader(args.val_bank, family=args.family, batch_size=args.curve_batch_size, num_workers=args.num_workers, shuffle=False)
    test_loader = build_loader(args.test_bank, family=args.family, batch_size=args.curve_batch_size, num_workers=args.num_workers, shuffle=False)

    trainer = SecondDerivativeOperatorTrainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        loss_fn=loss_fn,
        device=args.device,
        checkpoint_dir=output_dir,
        patch_size=args.patch_size,
        grad_clip_norm=args.grad_clip_norm,
        train_num_anchors=args.train_num_anchors,
    )

    summary = trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        num_epochs=args.num_epochs,
        early_stopping_patience=args.early_stopping_patience,
        seed=args.seed,
    )

    print(f"Direct second-derivative training complete. Output directory: {output_dir}")
    print(json.dumps(summary['after']['test'], indent=2))


if __name__ == '__main__':
    main()
