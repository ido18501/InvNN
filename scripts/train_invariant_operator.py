from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from datasets.tangent_dataset import TangentDataset
from models.tangent_model import TangentOperatorModel
from training.collate import tangent_collate_fn
from training.losses import EquivariantVectorLoss
from training.trainer import TangentTrainer


def parse_int_list(text: str) -> list[int]:
    text = text.strip()
    if not text:
        return []
    return [int(x.strip()) for x in text.split(',') if x.strip()]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--family', type=str, default='affine', choices=['euclidean', 'similarity', 'equi_affine', 'affine'])
    p.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--seed', type=int, default=123)
    p.add_argument('--checkpoint-dir', type=str, required=True)

    p.add_argument('--train-source', type=str, default='generated', choices=['generated', 'pregenerated'])
    p.add_argument('--val-source', type=str, default='generated', choices=['generated', 'pregenerated'])
    p.add_argument('--test-source', type=str, default='generated', choices=['generated', 'pregenerated'])
    p.add_argument('--train-bank', type=str, default=None)
    p.add_argument('--val-bank', type=str, default=None)
    p.add_argument('--test-bank', type=str, default=None)
    p.add_argument('--train-length', type=int, default=4096)
    p.add_argument('--val-length', type=int, default=1024)
    p.add_argument('--test-length', type=int, default=1024)

    p.add_argument('--batch-size', type=int, default=128)
    p.add_argument('--num-workers', type=int, default=4)
    p.add_argument('--num-epochs', type=int, default=40)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--weight-decay', type=float, default=1e-4)
    p.add_argument('--grad-clip-norm', type=float, default=1.0)
    p.add_argument('--early-stopping-patience', type=int, default=10)

    p.add_argument('--patch-size', type=int, default=9)
    p.add_argument('--half-width', type=int, default=12)
    p.add_argument('--num-negatives', type=int, default=8)
    p.add_argument('--negative-min-offset', type=int, default=5)
    p.add_argument('--negative-max-offset', type=int, default=25)
    p.add_argument('--negative-other-curve-fraction', type=float, default=0.5)
    p.add_argument('--patch-mode', type=str, default='intrinsic_ordered_stencil')
    p.add_argument('--jitter-fraction', type=float, default=0.0)
    p.add_argument('--num-curve-points', type=int, default=4000)

    p.add_argument('--reparam-strength', type=float, default=0.15)
    p.add_argument('--reparam-num-harmonics', type=int, default=2)
    p.add_argument('--reparam-min-density', type=float, default=0.7)
    p.add_argument('--reparam-max-density', type=float, default=1.5)
    p.add_argument('--fourier-max-freq', type=int, default=5)
    p.add_argument('--fourier-scale', type=float, default=0.9)
    p.add_argument('--fourier-decay-power', type=float, default=2.0)

    p.add_argument('--operator-hidden-dims', type=str, default='256,256')
    p.add_argument('--signature-hidden-dims', type=str, default='128,64')
    p.add_argument('--signature-out-dim', type=int, default=64)
    p.add_argument('--signature-center-radius', type=int, default=0)
    p.add_argument('--head-dropout', type=float, default=0.0)
    p.add_argument('--disable-normalize-projector', action='store_true')
    p.add_argument('--disable-centered-input-for-operator', action='store_true')
    p.add_argument('--operator-init-scale', type=float, default=0.05)
    p.add_argument('--learn-output-scale', action='store_true')

    p.add_argument('--temperature', type=float, default=0.1)
    p.add_argument('--lambda-nce', type=float, default=1.0)
    p.add_argument('--lambda-eq', type=float, default=1.0)
    p.add_argument('--lambda-reg', type=float, default=1e-4)


    p.add_argument('--reparametrize-prob', type=float, default=0.7)
    return p.parse_args()


def make_dataset(args, split: str) -> TangentDataset:
    source = getattr(args, f'{split}_source')
    bank = getattr(args, f'{split}_bank')
    length = getattr(args, f'{split}_length')
    return TangentDataset(
        length=length,
        family=args.family,
        source=source,
        bank_path=bank,
        num_curve_points=args.num_curve_points,
        fourier_max_freq=args.fourier_max_freq,
        fourier_scale=args.fourier_scale,
        fourier_decay_power=args.fourier_decay_power,
        patch_size=args.patch_size,
        half_width=args.half_width,
        num_negatives=args.num_negatives,
        negative_min_offset=args.negative_min_offset,
        negative_max_offset=args.negative_max_offset,
        negative_other_curve_fraction=args.negative_other_curve_fraction,
        patch_mode=args.patch_mode,
        jitter_fraction=args.jitter_fraction,
        seed=args.seed + {'train': 0, 'val': 10000, 'test': 20000}[split],
        reparametrize_prob=args.reparametrize_prob,
        reparam_strength=args.reparam_strength,
        reparam_num_harmonics=args.reparam_num_harmonics,
        reparam_min_density=args.reparam_min_density,
        reparam_max_density=args.reparam_max_density,
    )


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    (checkpoint_dir / 'config.json').write_text(json.dumps(vars(args), indent=2))

    train_dataset = make_dataset(args, 'train')
    val_dataset = make_dataset(args, 'val')
    test_dataset = make_dataset(args, 'test')

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, collate_fn=tangent_collate_fn, drop_last=False)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True, collate_fn=tangent_collate_fn, drop_last=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True, collate_fn=tangent_collate_fn, drop_last=False)

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
    loss_fn = EquivariantVectorLoss(
        temperature=args.temperature,
        lambda_nce=args.lambda_nce,
        lambda_eq=args.lambda_eq,
        lambda_reg=args.lambda_reg,
    )

    trainer = TangentTrainer(
        model=model,
        optimizer=optimizer,
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
    print(f'Best model saved at: {best_model_path}')
    trainer.evaluate(test_loader, split_name='test')


if __name__ == '__main__':
    main()