from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from datasets.tangent_dataset import TangentDataset
from models.tangent_model import TangentOperatorModel
from training.collate import tangent_collate_fn
from training.losses import InvariantOperatorLoss
from training.trainer import TangentTrainer



def parse_int_list(s: str) -> list[int]:
    return [int(x) for x in s.split(',') if x.strip()]



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
    p.add_argument('--point-mlp-dims', type=str, default='64,64,128')
    p.add_argument('--projector-dims', type=str, default='128')
    p.add_argument('--head-dims', type=str, default='128,64')
    p.add_argument('--invariant-dim', type=int, default=64)
    return p.parse_args()



def main():
    args = parse_args()
    dataset = TangentDataset(
        length=args.length,
        family=args.family,
        source=args.source,
        bank_path=args.bank,
        patch_size=args.patch_size,
        half_width=args.half_width,
        num_negatives=args.num_negatives,
        seed=12345,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=tangent_collate_fn,
    )

    model = TangentOperatorModel(
        patch_size=args.patch_size,
        point_mlp_dims=parse_int_list(args.point_mlp_dims),
        projector_dims=parse_int_list(args.projector_dims),
        head_dims=parse_int_list(args.head_dims),
        invariant_dim=args.invariant_dim,
    )
    ckpt = torch.load(Path(args.checkpoint), map_location=args.device)
    model.load_state_dict(ckpt)

    trainer = TangentTrainer(
        model=model,
        optimizer=torch.optim.AdamW(model.parameters(), lr=1e-3),
        loss_fn=InvariantOperatorLoss(),
        device=args.device,
        checkpoint_dir=Path(args.checkpoint).parent,
    )
    trainer.evaluate(loader, split_name='eval')


if __name__ == '__main__':
    main()
