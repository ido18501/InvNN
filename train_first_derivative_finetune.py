from __future__ import annotations

import argparse
import json
import torch
import torch.nn as nn
from pathlib import Path
from torch.utils.data import DataLoader

try:
    from datasets.tangent_dataset import TangentDataset
except Exception:
    from tangent_dataset import TangentDataset

try:
    from models.tangent_model import TangentOperatorModel
except Exception:
    from tangent_model import TangentOperatorModel

try:
    from training.collate import tangent_collate_fn
except Exception:
    from collate import tangent_collate_fn


def flip_operator_sign(model):
    last_linear = None
    for module in reversed(model.operator_head.net):
        if isinstance(module, nn.Linear):
            last_linear = module
            break

    if last_linear is None:
        raise RuntimeError("Could not find final Linear layer in model.operator_head.net")

    with torch.no_grad():
        last_linear.weight.mul_(-1.0)
        if last_linear.bias is not None:
            last_linear.bias.mul_(-1.0)

@torch.no_grad()
def estimate_signed_cosine(model, loader, device, max_batches=5):
    model.eval()
    vals = []

    for batch_idx, batch in enumerate(loader):
        if batch_idx >= max_batches:
            break

        anchor = batch.anchor.to(device)
        gt = batch.gt_first_anchor.to(device)
        has_analytic = batch.has_analytic_derivatives.to(device)

        out = model(anchor)
        pred = out["pred"]

        valid = has_analytic.bool() & torch.isfinite(gt).all(dim=-1)
        if valid.sum().item() == 0:
            continue

        pred = pred[valid]
        gt = gt[valid]

        pred_n = pred / (pred.norm(dim=-1, keepdim=True) + 1e-8)
        gt_n = gt / (gt.norm(dim=-1, keepdim=True) + 1e-8)
        cos = (pred_n * gt_n).sum(dim=-1)

        vals.append(cos.mean().item())

    if not vals:
        return float("nan")
    return sum(vals) / len(vals)

def parse_int_list(text: str) -> list[int]:
    text = text.strip()
    if not text:
        return []
    return [int(x.strip()) for x in text.split(',') if x.strip()]

try:
    from training.first_derivative_losses import FirstDerivativeLoss
except Exception:
    from first_derivative_losses import FirstDerivativeLoss

try:
    from training.first_derivative_trainer import FirstDerivativeFineTuneTrainer, evaluate_global_first_derivative
except Exception:
    from first_derivative_trainer import FirstDerivativeFineTuneTrainer, evaluate_global_first_derivative


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--seed', type=int, default=123)
    p.add_argument('--pretrained-checkpoint', type=str, required=True)
    p.add_argument('--checkpoint-dir', type=str, required=True)

    p.add_argument('--train-source', type=str, default='pregenerated', choices=['generated', 'pregenerated'])
    p.add_argument('--val-source', type=str, default='pregenerated', choices=['generated', 'pregenerated'])
    p.add_argument('--test-source', type=str, default='pregenerated', choices=['generated', 'pregenerated'])
    p.add_argument('--train-bank', type=str, default=None)
    p.add_argument('--val-bank', type=str, default=None)
    p.add_argument('--test-bank', type=str, default=None)
    p.add_argument('--train-length', type=int, default=4096)
    p.add_argument('--val-length', type=int, default=1024)
    p.add_argument('--test-length', type=int, default=1024)

    p.add_argument('--family', type=str, default='euclidean', choices=['euclidean', 'similarity', 'equi_affine', 'affine'])
    p.add_argument('--batch-size', type=int, default=128)
    p.add_argument('--num-workers', type=int, default=4)
    p.add_argument('--num-epochs', type=int, default=30)
    p.add_argument('--lr', type=float, default=1e-4)
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

    p.add_argument('--alpha-norm', type=float, default=1.0)
    p.add_argument('--target-norm', type=float, default=1.0)

    p.add_argument('--downsample-to-points', type=int, default=None)
    p.add_argument('--downsample-jitter', type=float, default=0.2)
    p.add_argument('--reparametrize-prob', type=float, default=0.7)
    p.add_argument('--disable-return-centered', action='store_true')

    p.add_argument('--lr-scheduler', type=str, default='plateau', choices=['none', 'plateau'])
    p.add_argument('--lr-patience', type=int, default=4)
    p.add_argument('--lr-factor', type=float, default=0.3)
    p.add_argument('--lr-min', type=float, default=1e-6)

    p.add_argument('--global-eval-bank', type=str, default=None)
    p.add_argument('--global-eval-max-curves', type=int, default=64)
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
        downsample_to_points=args.downsample_to_points,
        return_centered=not args.disable_return_centered,
        downsample_jitter=args.downsample_jitter,
    )


def build_model(args) -> TangentOperatorModel:
    return TangentOperatorModel(
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

    model = build_model(args)
    state = torch.load(args.pretrained_checkpoint, map_location='cpu')
    model.load_state_dict(state, strict=True)

    model.to(device)

    # 🔍 Check sign
    init_cos = quick_signed_cosine_check(model, val_loader, device)
    print(f"[pre-ft sign check] mean cosine = {init_cos:.6f}")

    if init_cos < 0:
        print("[pre-ft] Detected flipped operator → applying sign flip")
        flip_operator_sign(model)

        flipped_cos = quick_signed_cosine_check(model, val_loader, device)
        print(f"[pre-ft] cosine after flip = {flipped_cos:.6f}")

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
    loss_fn = FirstDerivativeLoss(alpha_norm=args.alpha_norm, target_norm=args.target_norm)

    trainer = FirstDerivativeFineTuneTrainer(
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
    print(f'Best fine-tuned model saved at: {best_model_path}')

    test_metrics = trainer.evaluate(test_loader, split_name='test')
    (checkpoint_dir / 'test_metrics.json').write_text(json.dumps(test_metrics, indent=2))

    global_bank = args.global_eval_bank or args.test_bank
    if global_bank is not None:
        global_metrics = evaluate_global_first_derivative(
            model=trainer.model,
            bank_path=global_bank,
            device=args.device,
            patch_size=args.patch_size,
            patch_mode=args.patch_mode,
            half_width=args.half_width,
            max_curves=args.global_eval_max_curves,
        )
        print('\nGlobal first-derivative evaluation')
        print(json.dumps(global_metrics, indent=2))
        (checkpoint_dir / 'global_eval.json').write_text(json.dumps(global_metrics, indent=2))


if __name__ == '__main__':
    main()
