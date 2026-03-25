from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from datasets.tangent_dataset import TangentDataset
from models.tangent_model import TangentOperatorModel
from training.collate import tangent_collate_fn
from training.losses import EquivariantMatrixOperatorLoss
from training.trainer import TangentTrainer
from scripts.train_invariant_operator import build_locality_matrix, parse_int_list


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', type=str, required=True)
    p.add_argument('--family', type=str, default='affine', choices=['euclidean', 'similarity', 'equi_affine', 'affine'])
    p.add_argument('--source', type=str, default='generated', choices=['generated', 'pregenerated'])
    p.add_argument('--bank', type=str, default=None)
    p.add_argument('--length', type=int, default=2048)
    p.add_argument('--batch-size', type=int, default=128)
    p.add_argument('--num-workers', type=int, default=4)
    p.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--patch-size', type=int, default=9)
    p.add_argument('--half-width', type=int, default=12)
    p.add_argument('--num-negatives', type=int, default=8)
    p.add_argument('--operator-hidden-dims', type=str, default='256,256')
    p.add_argument('--signature-hidden-dims', type=str, default='128,64')
    p.add_argument('--signature-out-dim', type=int, default=64)
    p.add_argument('--signature-center-radius', type=int, default=0)
    p.add_argument('--operator-bandwidth', type=int, default=None)
    p.add_argument('--operator-init-scale', type=float, default=0.05)
    p.add_argument('--disable-normalize-projector', action='store_true')
    p.add_argument('--disable-center-operator', action='store_true')
    p.add_argument('--disable-centered-input-for-operator', action='store_true')
    p.add_argument('--temperature', type=float, default=0.1)
    p.add_argument('--lambda-nce', type=float, default=1.0)
    p.add_argument('--lambda-eq', type=float, default=1.0)
    p.add_argument('--lambda-sum', type=float, default=0.1)
    p.add_argument('--lambda-reg', type=float, default=1e-4)
    p.add_argument('--lambda-loc', type=float, default=0.0)
    return p.parse_args()


def main():
    args = parse_args()
    dataset = TangentDataset(length=args.length, family=args.family, source=args.source, bank_path=args.bank, patch_size=args.patch_size, half_width=args.half_width, num_negatives=args.num_negatives, seed=12345)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True, collate_fn=tangent_collate_fn)

    model = TangentOperatorModel(
        patch_size=args.patch_size,
        operator_hidden_dims=parse_int_list(args.operator_hidden_dims),
        signature_hidden_dims=parse_int_list(args.signature_hidden_dims),
        signature_out_dim=args.signature_out_dim,
        signature_center_radius=args.signature_center_radius,
        normalize_projector=not args.disable_normalize_projector,
        center_operator=not args.disable_center_operator,
        operator_bandwidth=args.operator_bandwidth,
        init_scale=args.operator_init_scale,
        centered_input_for_operator=not args.disable_centered_input_for_operator,
    )
    ckpt = torch.load(Path(args.checkpoint), map_location=args.device)
    model.load_state_dict(ckpt)

    trainer = TangentTrainer(
        model=model,
        optimizer=torch.optim.AdamW(model.parameters(), lr=1e-3),
        loss_fn=EquivariantMatrixOperatorLoss(
            temperature=args.temperature,
            lambda_nce=args.lambda_nce,
            lambda_eq=args.lambda_eq,
            lambda_sum=args.lambda_sum,
            lambda_reg=args.lambda_reg,
            lambda_loc=args.lambda_loc,
            locality_matrix=build_locality_matrix(args.patch_size),
        ),
        device=args.device,
        checkpoint_dir=Path(args.checkpoint).parent,
    )
    trainer.evaluate(loader, split_name='eval')


if __name__ == '__main__':
    main()
