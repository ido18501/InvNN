from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from datasets.tangent_dataset import PregeneratedCurveBank
from training.operator_finetune_utils import (
    freeze_signature_head,
    instantiate_model_from_pretrained_config,
    load_pretrained_run,
    maybe_flip_operator_sign,
)
from training.second_order_finetune_losses import RobustOperatorSupervisionLoss
from training.second_order_finetune_trainer import SecondOrderFineTuneTrainer


DATASET_CHOICES = [
    'data_complex_f20_250to180',
    'data_complex_f20_500to300',
    'data_complex_f20_1000to500',
    'data_complex_f20_2000to1000',
    'data_complex_f20_3000to1500',
    'data_complex_f20_4000to2000',
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument('--pretrained-run-dir', type=str, required=True)
    p.add_argument('--train-bank', type=str, required=True)
    p.add_argument('--val-bank', type=str, required=True)
    p.add_argument('--test-bank', type=str, required=True)
    p.add_argument('--dataset-name', type=str, required=True, choices=DATASET_CHOICES)
    p.add_argument('--family', type=str, default='euclidean', choices=['euclidean', 'similarity', 'equi_affine'])
    p.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--seed', type=int, default=123)
    p.add_argument('--output-dir', type=str, required=True)

    p.add_argument('--batch-size-curves', type=int, default=4)
    p.add_argument('--num-anchors-train', type=int, default=128)
    p.add_argument('--num-anchors-eval-sampled', type=int, default=256)
    p.add_argument('--num-epochs', type=int, default=20)
    p.add_argument('--early-stopping-patience', type=int, default=6)
    p.add_argument('--lr', type=float, default=2e-5)
    p.add_argument('--weight-decay', type=float, default=1e-5)
    p.add_argument('--grad-clip-norm', type=float, default=1.0)

    p.add_argument('--lr-scheduler', type=str, default='plateau', choices=['none', 'plateau'])
    p.add_argument('--lr-patience', type=int, default=2)
    p.add_argument('--lr-factor', type=float, default=0.5)
    p.add_argument('--lr-min', type=float, default=1e-6)

    p.add_argument('--lambda-first-dir', type=float, default=1.0)
    p.add_argument('--lambda-first-mag', type=float, default=0.25)
    p.add_argument('--lambda-second-dir', type=float, default=0.75)
    p.add_argument('--lambda-second-mag', type=float, default=0.15)
    p.add_argument('--lambda-second-log-mag', type=float, default=0.20)
    p.add_argument('--lambda-zero-sum', type=float, default=0.05)
    p.add_argument('--lambda-weight-drift', type=float, default=1e-5)

    p.add_argument('--huber-delta-first-mag', type=float, default=0.05)
    p.add_argument('--huber-delta-second-mag', type=float, default=0.05)
    p.add_argument('--huber-delta-zero-sum', type=float, default=0.02)

    p.add_argument('--sign-check-curves', type=int, default=8)
    p.add_argument('--sign-check-anchors', type=int, default=64)
    p.add_argument('--sign-flip-negative-threshold', type=float, default=-0.25)
    p.add_argument('--disable-sign-check', action='store_true')
    p.add_argument('--max-eval-curves', type=int, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    best_model_path, pretrained_cfg, pretrained_config_path = load_pretrained_run(args.pretrained_run_dir)

    print('patch_size raw:', pretrained_cfg.get('patch_size'), type(pretrained_cfg.get('patch_size')))
    print('operator_hidden_dims raw:', pretrained_cfg.get('operator_hidden_dims'),
          type(pretrained_cfg.get('operator_hidden_dims')))
    print('signature_hidden_dims raw:', pretrained_cfg.get('signature_hidden_dims'),
          type(pretrained_cfg.get('signature_hidden_dims')))
    print('disable_normalize_projector raw:', pretrained_cfg.get('disable_normalize_projector'),
          type(pretrained_cfg.get('disable_normalize_projector')))
    print('learn_output_scale raw:', pretrained_cfg.get('learn_output_scale'),
          type(pretrained_cfg.get('learn_output_scale')))
    print('disable_centered_input_for_operator raw:', pretrained_cfg.get('disable_centered_input_for_operator'),
          type(pretrained_cfg.get('disable_centered_input_for_operator')))

    model = instantiate_model_from_pretrained_config(pretrained_cfg)
    state = torch.load(best_model_path, map_location='cpu')
    model.load_state_dict(state)
    freeze_signature_head(model)

    train_bank = PregeneratedCurveBank(args.train_bank)
    val_bank = PregeneratedCurveBank(args.val_bank)
    test_bank = PregeneratedCurveBank(args.test_bank)

    sign_info = {'sign_check_disabled': float(args.disable_sign_check)}
    if not args.disable_sign_check:
        sign_info = maybe_flip_operator_sign(
            model=model,
            bank=val_bank,
            family=args.family,
            patch_size=int(pretrained_cfg['patch_size']),
            device=torch.device(args.device),
            num_curves=args.sign_check_curves,
            num_anchors=args.sign_check_anchors,
            negative_threshold=args.sign_flip_negative_threshold,
        )

    loss_fn = RobustOperatorSupervisionLoss(
        lambda_first_dir=args.lambda_first_dir,
        lambda_first_mag=args.lambda_first_mag,
        lambda_second_dir=args.lambda_second_dir,
        lambda_second_mag=args.lambda_second_mag,
        lambda_second_log_mag=args.lambda_second_log_mag,
        lambda_zero_sum=args.lambda_zero_sum,
        lambda_weight_drift=args.lambda_weight_drift,
        huber_delta_first_mag=args.huber_delta_first_mag,
        huber_delta_second_mag=args.huber_delta_second_mag,
        huber_delta_zero_sum=args.huber_delta_zero_sum,
    )

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = None
    if args.lr_scheduler == 'plateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=args.lr_factor,
            patience=args.lr_patience,
            min_lr=args.lr_min,
        )

    trainer = SecondOrderFineTuneTrainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        loss_fn=loss_fn,
        train_bank=train_bank,
        val_bank=val_bank,
        test_bank=test_bank,
        family=args.family,
        patch_size=int(pretrained_cfg['patch_size']),
        device=args.device,
        checkpoint_dir=output_dir,
        grad_clip_norm=args.grad_clip_norm,
        num_anchors_train=args.num_anchors_train,
        num_anchors_eval_sampled=args.num_anchors_eval_sampled,
        batch_size_curves=args.batch_size_curves,
        seed=args.seed,
        max_eval_curves=args.max_eval_curves,
    )

    run_config = {
        'args': vars(args),
        'pretrained_config_path': str(pretrained_config_path),
        'pretrained_best_model_path': str(best_model_path),
        'pretrained_config': pretrained_cfg,
        'sign_info': sign_info,
    }
    (output_dir / 'finetune_config.json').write_text(json.dumps(run_config, indent=2))

    before_val = trainer.evaluate_full_bank(val_bank, 'val_before', save_path=output_dir / 'val_before.json')
    before_test = trainer.evaluate_full_bank(test_bank, 'test_before', save_path=output_dir / 'test_before.json')

    best_path = trainer.fit(
        num_epochs=args.num_epochs,
        early_stopping_patience=args.early_stopping_patience,
    )

    after_val = trainer.evaluate_full_bank(val_bank, 'val_after', save_path=output_dir / 'val_after.json')
    after_test = trainer.evaluate_full_bank(test_bank, 'test_after', save_path=output_dir / 'test_after.json')

    summary = {
        'sign_info': sign_info,
        'best_model_path': str(best_path),
        'val_before': before_val,
        'val_after': after_val,
        'test_before': before_test,
        'test_after': after_test,
    }
    (output_dir / 'summary.json').write_text(json.dumps(summary, indent=2))
    (output_dir / 'DONE').write_text('stage2 fine-tune complete\n')

    print('\nSaved summary to:', output_dir / 'summary.json')
    print('Saved best model to:', best_path)


if __name__ == '__main__':
    main()
