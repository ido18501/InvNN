from __future__ import annotations

import argparse
import numpy as np

from datasets.tangent_tuple_generation import build_random_invariant_training_tuple
from utils.curve_generation import (
    generate_random_simple_fourier_curve,
    generate_random_reparameterized_fourier_curve,
)


def _pairwise_adjacent_distances(patch: np.ndarray) -> np.ndarray:
    return np.linalg.norm(patch[1:] - patch[:-1], axis=1)


def _ratio_stats(x: np.ndarray, eps: float = 1e-12) -> tuple[float, float]:
    x = np.asarray(x, dtype=np.float64)
    if len(x) == 0:
        return 0.0, 0.0
    return (
        float(np.max(x) / max(np.min(x), eps)),
        float(np.percentile(x, 90) / max(np.percentile(x, 10), eps)),
    )


def _patch_metrics(patch: np.ndarray) -> dict[str, float]:
    patch = np.asarray(patch, dtype=np.float64)
    k = len(patch)
    c = k // 2

    adj = _pairwise_adjacent_distances(patch)
    adj_ratio, adj_p90_p10 = _ratio_stats(adj)

    left_span = float(np.sum(adj[:c])) if c > 0 else 0.0
    right_span = float(np.sum(adj[c:])) if c > 0 else 0.0
    lr_balance = max(left_span, right_span) / max(min(left_span, right_span), 1e-12) if c > 0 else 1.0

    center_left = float(np.linalg.norm(patch[c] - patch[c - 1])) if c > 0 else 0.0
    center_right = float(np.linalg.norm(patch[c + 1] - patch[c])) if c > 0 else 0.0
    center_balance = (
        max(center_left, center_right) / max(min(center_left, center_right), 1e-12)
        if c > 0 else 1.0
    )

    return {
        "adj_min": float(np.min(adj)) if len(adj) else 0.0,
        "adj_max": float(np.max(adj)) if len(adj) else 0.0,
        "adj_ratio": adj_ratio,
        "adj_p90_p10": adj_p90_p10,
        "left_span": left_span,
        "right_span": right_span,
        "lr_balance": lr_balance,
        "center_left": center_left,
        "center_right": center_right,
        "center_balance": center_balance,
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--num-samples", type=int, default=200)
    p.add_argument("--patch-size", type=int, default=9)
    p.add_argument("--num-curve-points", type=int, default=4000)
    p.add_argument("--family", type=str, default="euclidean")
    p.add_argument("--num-negatives", type=int, default=4)
    p.add_argument("--negative-min-offset", type=int, default=5)
    p.add_argument("--negative-max-offset", type=int, default=25)
    p.add_argument("--fourier-max-freq", type=int, default=7)
    p.add_argument("--fourier-scale", type=float, default=1.0)
    p.add_argument("--fourier-decay-power", type=float, default=1.65)
    p.add_argument("--curve-max-tries", type=int, default=300)
    p.add_argument("--curve-min-size", type=float, default=0.45)
    p.add_argument("--curve-max-size", type=float, default=0.75)
    p.add_argument("--reparametrize-prob", type=float, default=0.7)
    p.add_argument("--reparam-strength", type=float, default=0.15)
    p.add_argument("--reparam-num-harmonics", type=int, default=2)
    p.add_argument("--reparam-min-density", type=float, default=0.7)
    p.add_argument("--reparam-max-density", type=float, default=1.5)
    p.add_argument("--seed", type=int, default=123)
    return p.parse_args()


def _generate_curve(args: argparse.Namespace, rng: np.random.Generator):
    use_reparam = rng.random() < args.reparametrize_prob

    if use_reparam:
        curve_points, coeffs, _, t_grid = generate_random_reparameterized_fourier_curve(
            num_points=args.num_curve_points,
            max_freq=args.fourier_max_freq,
            scale=args.fourier_scale,
            decay_power=args.fourier_decay_power,
            rng=rng,
            center=True,
            fit_to_canvas=True,
            min_size=args.curve_min_size,
            max_size=args.curve_max_size,
            reparam_strength=args.reparam_strength,
            reparam_num_harmonics=args.reparam_num_harmonics,
            reparam_min_density=args.reparam_min_density,
            reparam_max_density=args.reparam_max_density,
        )
    else:
        t_grid = np.linspace(0.0, 2.0 * np.pi, args.num_curve_points, endpoint=False)
        curve_points, coeffs = generate_random_simple_fourier_curve(
            t=t_grid,
            max_freq=args.fourier_max_freq,
            scale=args.fourier_scale,
            decay_power=args.fourier_decay_power,
            rng=rng,
            max_tries=args.curve_max_tries,
            center=True,
            fit_to_canvas=True,
            min_size=args.curve_min_size,
            max_size=args.curve_max_size,
            enforce_simple=False,
        )

    return curve_points, coeffs, t_grid


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    anchor_adj_ratios = []
    anchor_adj_p90p10 = []
    anchor_lr_balances = []
    anchor_center_balances = []

    pos_adj_ratios = []
    neg_adj_ratios = []

    bad_adj = 0
    bad_balance = 0

    for i in range(args.num_samples):
        curve_points, coeffs, t_grid = _generate_curve(args, rng)

        tuple_sample = build_random_invariant_training_tuple(
            curve_points=curve_points,
            coeffs=coeffs,
            t_grid=t_grid,
            transform_family=args.family,
            patch_size=args.patch_size,
            half_width=0,
            num_negatives=args.num_negatives,
            negative_min_offset=args.negative_min_offset,
            negative_max_offset=args.negative_max_offset,
            closed=True,
            patch_mode="intrinsic_ordered_stencil",
            jitter_fraction=0.0,
            rng=rng,
            transform_kwargs=None,
            external_negative_curves=None,
            num_cross_curve_negatives=0,
        )

        anchor = np.asarray(tuple_sample.anchor_patch, dtype=np.float64)
        positive = np.asarray(tuple_sample.positive_patch, dtype=np.float64)
        negatives = np.asarray(tuple_sample.negative_patches, dtype=np.float64)

        am = _patch_metrics(anchor)
        pm = _patch_metrics(positive)

        anchor_adj_ratios.append(am["adj_ratio"])
        anchor_adj_p90p10.append(am["adj_p90_p10"])
        anchor_lr_balances.append(am["lr_balance"])
        anchor_center_balances.append(am["center_balance"])
        pos_adj_ratios.append(pm["adj_ratio"])

        if am["adj_p90_p10"] > 4.0:
            bad_adj += 1
        if am["lr_balance"] > 3.0 or am["center_balance"] > 3.0:
            bad_balance += 1

        for neg in negatives:
            nm = _patch_metrics(neg)
            neg_adj_ratios.append(nm["adj_ratio"])

        if i < 10:
            print(
                f"[sample {i:03d}] "
                f"anchor adj_ratio={am['adj_ratio']:.3f} | "
                f"anchor p90/p10={am['adj_p90_p10']:.3f} | "
                f"lr_balance={am['lr_balance']:.3f} | "
                f"center_balance={am['center_balance']:.3f}"
            )

    print("\n=== Summary ===")
    print(f"anchor mean adj_ratio:         {np.mean(anchor_adj_ratios):.3f}")
    print(f"anchor mean adj_p90/p10:       {np.mean(anchor_adj_p90p10):.3f}")
    print(f"anchor mean lr_balance:        {np.mean(anchor_lr_balances):.3f}")
    print(f"anchor mean center_balance:    {np.mean(anchor_center_balances):.3f}")
    print(f"positive mean adj_ratio:       {np.mean(pos_adj_ratios):.3f}")
    print(f"negative mean adj_ratio:       {np.mean(neg_adj_ratios):.3f}")
    print(f"fraction bad adj patches:      {bad_adj / max(args.num_samples, 1):.3f}")
    print(f"fraction bad balance patches:  {bad_balance / max(args.num_samples, 1):.3f}")


if __name__ == "__main__":
    main()
