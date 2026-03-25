# Invariant Operator Project

This is a standalone rewrite of the tangent-operator project around an **invariant-only training objective**.

## Goal

Train a model on local curve patches using only invariance over a transformation family.
The model never sees derivative labels during optimization.

The model learns:
- an invariant embedding `z(patch)`
- a single scalar stencil `W` predicted from that invariant code

Then for diagnostics only:
- `W(x)` is compared to the exact analytic Euclidean arc-length first derivative
- `W(W(x))` is compared to the exact analytic Euclidean arc-length second derivative

Those derivative comparisons are **never used in the loss**.

## Directory layout

Create a fresh project directory and place the files exactly like this:

```text
invariant_operator_project/
├── README.md
├── requirements.txt
├── datasets/
│   ├── __init__.py
│   ├── tangent_dataset.py
│   └── tangent_tuple_generation.py
├── models/
│   ├── __init__.py
│   └── tangent_model.py
├── training/
│   ├── __init__.py
│   ├── collate.py
│   ├── losses.py
│   └── trainer.py
├── utils/
│   ├── __init__.py
│   ├── curve_generation.py
│   ├── derivatives.py
│   ├── patch_sampling.py
│   └── transformations.py
└── scripts/
    ├── generate_curve_bank.py
    ├── train_invariant_operator.py
    └── evaluate_checkpoint.py
```

## Two dataset modes

### 1. On-the-fly generation
Random Fourier curves generated each sample.

### 2. Pregenerated bank
Use a precomputed `.npz` file with:
- `curve_points` shape `(K, N, 2)`
- `x_coeffs` shape `(K, M)`
- `y_coeffs` shape `(K, M)`
- `t_grid` shape `(N,)` or `(K, N)`

This preserves exact analytic derivative diagnostics.

## Important behavior preserved

- random Fourier curves
- optional global warp of the full curve sampling
- random local patch warp
- positive made from the same local support under a sampled transformation
- in-curve and cross-curve negatives
- no derivative labels in the objective

## Typical workflow

### Generate a pregenerated bank

```bash
python scripts/generate_curve_bank.py \
  --output data/curve_bank_train.npz \
  --num-curves 5000 \
  --num-points 1000 \
  --max-freq 5
```

### Train

```bash
python scripts/train_invariant_operator.py \
  --family affine \
  --train-source pregenerated \
  --train-bank data/curve_bank_train.npz \
  --val-source pregenerated \
  --val-bank data/curve_bank_val.npz \
  --test-source pregenerated \
  --test-bank data/curve_bank_test.npz \
  --patch-size 9 \
  --half-width 12 \
  --batch-size 128 \
  --num-epochs 40 \
  --checkpoint-dir checkpoints/invariant_affine
```

### Evaluate a checkpoint

```bash
python scripts/evaluate_checkpoint.py \
  --checkpoint checkpoints/invariant_affine/best_model.pt \
  --family affine \
  --source pregenerated \
  --bank data/curve_bank_test.npz
```

## Notes

- Progress bars are shown live with `tqdm` during train / val / test.
- Epoch summaries print both the training loss terms and the derivative-emergence metrics.
- If derivative labels are unavailable, the trainer will skip those diagnostics safely.
