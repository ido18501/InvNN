from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

try:
    from models.tangent_model import TangentOperatorModel
except Exception:
    from tangent_model import TangentOperatorModel

from robust_finetune_loss import RobustDerivativeLoss
from second_order_finetune_trainer import (
    FineTuneCurveDataset,
    SecondOrderFineTuneTrainer,
    finetune_curve_collate,
)


def parse_int_list(text: str) -> list[int]:
    text = text.strip()
    if not text:
        return []
    return [int(x.strip()) for x in text.split(',') if x.strip()]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--family', type=str, default='euclidean', choices=['euclidean', 'similarity', 'equi_affine'])
    p.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--seed', type=int, default=123)

    p.add_argument('--pretrained-checkpoint', type=str, required=True)
    p.add_argument('--checkpoint-dir', type=str, required=True)

    p.add_argument('--train-bank', type=str, required=True)
    p.add_argument('--val-bank', type=str, required=True)
    p.add_argument('--test-bank', type=str, default=None)
    p.add_argument('--train-length', type=int, default=None)
    p.add_argument('--val-length', type=int, default=None)
    p.add_argument('--test-length', type=int, default=None)

    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--num-workers', type=int, default=4)
    p.add_argument('--anchors-per-curve', type=int, default=8)

    p.add_argument('--num-epochs', type=int, default=20)
    p.add_argument('--lr', type=float, default=2e-5)
    p.add_argument('--weight-decay', type=float, default=1e-4)
    p.add_argument('--grad-clip-norm', type=float, default=1.0)
    p.add_argument('--early-stopping-patience', type=int, default=6)

    p.add_argument('--lr-scheduler', type=str, default='plateau', choices=['none', 'plateau'])
    p.add_argument('--lr-patience', type=int, default=2)
    p.add_argument('--lr-factor', type=float, default=0.5)
    p.add_argument('--lr-min', type=float, default=1e-6)

    p.add_argument('--patch-size', type=int, required=True)
    p.add_argument('--operator-hidden-dims', type=str, default='256,256')
    p.add_argument('--signature-hidden-dims', type=str, default='128,64')
    p.add_argument('--signature-out-dim', type=int, default=64)
    p.add_argument('--signature-center-radius', type=int, default=0)
    p.add_argument('--head-dropout', type=float, default=0.0)
    p.add_argument('--disable-normalize-projector', action='store_true')
    p.add_argument('--disable-centered-input-for-operator', action='store_true')
    p.add_argument('--operator-init-scale', type=float, default=0.05)
    p.add_argument('--learn-output-scale', action='store_true')

    p.add_argument('--lambda-first', type=float, default=1.0)
    p.add_argument('--lambda-second', type=float, default=0.25)
    p.add_argument('--delta-first', type=float, default=0.05)
    p.add_argument('--delta-second', type=float, default=0.25)
    p.add_argument('--lambda-row-sum', type=float, default=0.0)
    p.add_argument('--lambda-prox', type=float, default=1e-6)
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    (checkpoint_dir / 'config.json').write_text(json.dumps(vars(args), indent=2))

    train_dataset = FineTuneCurveDataset(
        bank_path=args.train_bank,
        family=args.family,
        length=args.train_length,
        seed=args.seed,
    )
    val_dataset = FineTuneCurveDataset(
        bank_path=args.val_bank,
        family=args.family,
        length=args.val_length,
        seed=args.seed + 10000,
    )
    test_dataset = None
    if args.test_bank is not None:
        test_dataset = FineTuneCurveDataset(
            bank_path=args.test_bank,
            family=args.family,
            length=args.test_length,
            seed=args.seed + 20000,
        )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=finetune_curve_collate,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=finetune_curve_collate,
        drop_last=False,
    )
    test_loader = None
    if test_dataset is not None:
        test_loader = DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
            collate_fn=finetune_curve_collate,
            drop_last=False,
        )

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

    state = torch.load(args.pretrained_checkpoint, map_location='cpu')
    model.load_state_dict(state)

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

    loss_fn = RobustDerivativeLoss(
        lambda_first=args.lambda_first,
        lambda_second=args.lambda_second,
        delta_first=args.delta_first,
        delta_second=args.delta_second,
        lambda_row_sum=args.lambda_row_sum,
        lambda_prox=args.lambda_prox,
    )

    trainer = SecondOrderFineTuneTrainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        loss_fn=loss_fn,
        device=args.device,
        checkpoint_dir=checkpoint_dir,
        patch_size=args.patch_size,
        grad_clip_norm=args.grad_clip_norm,
        anchors_per_curve=args.anchors_per_curve,
        seed=args.seed,
    )

    # preserve the starting pretrained state in the new fine-tune directory
    torch.save(model.state_dict(), checkpoint_dir / 'pretrained_init_model.pt')

    best_model_path = trainer.fit(
        train_loader,
        val_loader,
        num_epochs=args.num_epochs,
        early_stopping_patience=args.early_stopping_patience,
    )
    print(f'Best model saved at: {best_model_path}')

    if test_loader is not None:
        trainer.evaluate(test_loader, split_name='test')

    (checkpoint_dir / 'DONE').write_text('ok\n')


if __name__ == '__main__':
    main()
