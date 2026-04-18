from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from models.tangent_model import TangentOperatorModel
from training.second_order_finetune_losses import RobustOperatorVectorLoss
from training.projected_second_order_finetune_trainer import (
    FullCurveBankDataset,
    ProjectedSecondOrderFineTuneTrainer,
    full_curve_collate,
)

REQUIRED_MODEL_CONFIG_FIELDS = [
    'patch_size',
    'operator_hidden_dims',
    'signature_hidden_dims',
    'signature_out_dim',
    'signature_center_radius',
    'head_dropout',
    'disable_normalize_projector',
    'operator_init_scale',
    'learn_output_scale',
    'disable_centered_input_for_operator',
]


def parse_int_list_maybe(value):
    if isinstance(value, list):
        return [int(x) for x in value]
    if isinstance(value, str):
        value = value.strip()
        return [int(x.strip()) for x in value.split(',') if x.strip()] if value else []
    raise TypeError(f'Cannot parse integer list from type {type(value)!r}')


def load_pretrained_config(pretrained_model_path: str | Path) -> tuple[dict, Path]:
    model_path = Path(pretrained_model_path)
    if model_path.name != 'best_model.pt':
        raise ValueError('pretrained-model-path must point to best_model.pt')
    run_dir = model_path.parent
    config_path = run_dir / 'config.json'
    done_path = run_dir / 'DONE'
    if not config_path.exists():
        raise FileNotFoundError(f'Missing config.json next to checkpoint: {config_path}')
    if not done_path.exists():
        raise FileNotFoundError(f'Missing DONE marker for pretrained run: {done_path}')
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    missing = [k for k in REQUIRED_MODEL_CONFIG_FIELDS if k not in config]
    if missing:
        raise KeyError(f'Pretrained config missing required fields: {missing}')
    return config, run_dir


def instantiate_model_from_config(config: dict) -> TangentOperatorModel:
    return TangentOperatorModel(
        patch_size=int(config['patch_size']),
        operator_hidden_dims=parse_int_list_maybe(config['operator_hidden_dims']),
        signature_hidden_dims=parse_int_list_maybe(config['signature_hidden_dims']),
        signature_out_dim=int(config['signature_out_dim']),
        signature_center_radius=int(config['signature_center_radius']),
        head_dropout=float(config['head_dropout']),
        normalize_projector=not bool(config['disable_normalize_projector']),
        init_scale=float(config['operator_init_scale']),
        learn_scale=bool(config['learn_output_scale']),
        centered_input_for_operator=not bool(config['disable_centered_input_for_operator']),
    )


def build_loader(bank_path: str, family: str, batch_size: int, num_workers: int, shuffle: bool) -> DataLoader:
    ds = FullCurveBankDataset(bank_path=bank_path, family=family)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True, collate_fn=full_curve_collate, drop_last=False)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--pretrained-model-path', type=str, required=True)
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
    p.add_argument('--lr', type=float, default=1e-5)
    p.add_argument('--weight-decay', type=float, default=1e-5)
    p.add_argument('--grad-clip-norm', type=float, default=1.0)
    p.add_argument('--early-stopping-patience', type=int, default=6)
    p.add_argument('--train-num-anchors', type=int, default=128)
    p.add_argument('--moment-order', type=int, default=3)
    p.add_argument('--projection-ridge', type=float, default=1e-6)
    p.add_argument('--lambda1-vec', type=float, default=1.0)
    p.add_argument('--lambda2-vec', type=float, default=0.1)
    p.add_argument('--lambda1-cos', type=float, default=0.02)
    p.add_argument('--lambda2-cos', type=float, default=0.02)
    p.add_argument('--lambda2-log', type=float, default=0.01)
    p.add_argument('--lambda-rowsum', type=float, default=0.0)
    p.add_argument('--lambda-weight-l2', type=float, default=1e-5)
    p.add_argument('--huber-delta1', type=float, default=0.05)
    p.add_argument('--huber-delta2', type=float, default=0.05)
    p.add_argument('--lr-scheduler', type=str, default='plateau', choices=['none', 'plateau'])
    p.add_argument('--lr-patience', type=int, default=2)
    p.add_argument('--lr-factor', type=float, default=0.5)
    p.add_argument('--lr-min', type=float, default=1e-6)
    p.add_argument('--sign-check-batches', type=int, default=4)
    p.add_argument('--sign-check-threshold', type=float, default=-0.25)
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    pretrained_config, pretrained_run_dir = load_pretrained_config(args.pretrained_model_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / 'finetune_config.json', 'w', encoding='utf-8') as f:
        json.dump(vars(args), f, indent=2)
    with open(output_dir / 'pretrained_config_snapshot.json', 'w', encoding='utf-8') as f:
        json.dump(pretrained_config, f, indent=2)

    model = instantiate_model_from_config(pretrained_config)
    state_dict = torch.load(args.pretrained_model_path, map_location='cpu')
    model.load_state_dict(state_dict)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = None
    if args.lr_scheduler == 'plateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=args.lr_factor, patience=args.lr_patience, min_lr=args.lr_min,
        )

    loss_fn = RobustOperatorVectorLoss(
        lambda1_vec=args.lambda1_vec,
        lambda2_vec=args.lambda2_vec,
        lambda1_cos=args.lambda1_cos,
        lambda2_cos=args.lambda2_cos,
        lambda2_log=args.lambda2_log,
        lambda_rowsum=args.lambda_rowsum,
        lambda_weight_l2=args.lambda_weight_l2,
        huber_delta1=args.huber_delta1,
        huber_delta2=args.huber_delta2,
    )

    train_loader = build_loader(args.train_bank, args.family, args.curve_batch_size, args.num_workers, True)
    val_loader = build_loader(args.val_bank, args.family, args.curve_batch_size, args.num_workers, False)
    test_loader = build_loader(args.test_bank, args.family, args.curve_batch_size, args.num_workers, False)

    trainer = ProjectedSecondOrderFineTuneTrainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        loss_fn=loss_fn,
        device=args.device,
        checkpoint_dir=output_dir,
        patch_size=int(pretrained_config['patch_size']),
        moment_order=args.moment_order,
        projection_ridge=args.projection_ridge,
        sign_check_batches=args.sign_check_batches,
        sign_check_threshold=args.sign_check_threshold,
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
    print(f'Fine-tune complete. Output directory: {output_dir}')
    print(f'Loaded pretrained run from: {pretrained_run_dir}')
    print(json.dumps(summary['after']['test'], indent=2))


if __name__ == '__main__':
    main()
