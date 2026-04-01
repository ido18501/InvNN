from __future__ import annotations

import argparse
import numpy as np
import matplotlib.pyplot as plt

from datasets.tangent_tuple_generation import build_random_invariant_training_tuple
from utils.curve_generation import (
    generate_random_simple_fourier_curve,
    curve_has_self_intersections,
)
from utils.derivatives import evaluate_fourier_curve_and_parameter_derivatives
from utils.transformations import apply_transformation, Transformation2D


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--num-examples", type=int, default=6)
    p.add_argument("--family", type=str, default="euclidean",
                   choices=["euclidean", "similarity", "equi_affine", "affine"])
    p.add_argument("--patch-size", type=int, default=9)
    p.add_argument("--context-size", type=int, default=121)   # visualization only
    p.add_argument("--num-negatives", type=int, default=3)
    p.add_argument("--negative-min-offset", type=int, default=5)
    p.add_argument("--negative-max-offset", type=int, default=25)
    p.add_argument("--num-curve-points", type=int, default=4000)
    p.add_argument("--fourier-max-freq", type=int, default=7)
    p.add_argument("--fourier-scale", type=float, default=1.0)
    p.add_argument("--fourier-decay-power", type=float, default=1.65)
    p.add_argument("--curve-max-tries", type=int, default=10000)
    p.add_argument("--curve-min-size", type=float, default=0.45)
    p.add_argument("--curve-max-size", type=float, default=0.75)
    p.add_argument("--reparametrize-prob", type=float, default=0.7)
    p.add_argument("--reparam-strength", type=float, default=0.15)
    p.add_argument("--reparam-num-harmonics", type=int, default=2)
    p.add_argument("--reparam-min-density", type=float, default=0.7)
    p.add_argument("--reparam-max-density", type=float, default=1.5)
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--downsample-to-points", type=int, default=None)
    p.add_argument("--downsample-jitter", type=float, default=0.2)
    return p.parse_args()


def sample_smooth_monotone_periodic_reparameterization(
    num_points: int,
    rng: np.random.Generator,
    strength: float = 0.15,
    num_harmonics: int = 2,
    min_density: float = 0.7,
    max_density: float = 1.5,
) -> tuple[np.ndarray, np.ndarray]:
    u_grid = np.linspace(0.0, 2.0 * np.pi, num_points, endpoint=False, dtype=np.float64)

    g = np.ones_like(u_grid)
    for k in range(1, num_harmonics + 1):
        a = rng.normal(0.0, strength / (k ** 1.5))
        b = rng.normal(0.0, strength / (k ** 1.5))
        g += a * np.sin(k * u_grid) + b * np.cos(k * u_grid)

    g = g / np.mean(g)
    g = np.clip(g, min_density, max_density)

    dt = 2.0 * np.pi / num_points
    phi = np.cumsum(g) * dt
    phi = phi - phi[0]
    phi = phi / phi[-1]
    phi = phi * (2.0 * np.pi)

    t_base = u_grid
    t_warped = np.interp(u_grid, phi, t_base)
    return u_grid, t_warped


def generate_simple_reparameterized_curve(args: argparse.Namespace, rng: np.random.Generator):
    """
    Always start from a simple base Fourier curve, then reparameterize it smoothly.
    This preserves non-self-intersection.
    """
    t_uniform = np.linspace(0.0, 2.0 * np.pi, args.num_curve_points, endpoint=False)

    base_curve, coeffs = generate_random_simple_fourier_curve(
        t=t_uniform,
        max_freq=args.fourier_max_freq,
        scale=args.fourier_scale,
        decay_power=args.fourier_decay_power,
        rng=rng,
        max_tries=args.curve_max_tries,
        center=True,
        fit_to_canvas=True,
        min_size=args.curve_min_size,
        max_size=args.curve_max_size,
        enforce_simple=True,
    )

    use_reparam = rng.random() < args.reparametrize_prob
    if not use_reparam:
        return base_curve, coeffs, t_uniform, False

    _, _, _, _ = evaluate_fourier_curve_and_parameter_derivatives(t_uniform, coeffs)
    _, t_warped = sample_smooth_monotone_periodic_reparameterization(
        num_points=args.num_curve_points,
        rng=rng,
        strength=args.reparam_strength,
        num_harmonics=args.reparam_num_harmonics,
        min_density=args.reparam_min_density,
        max_density=args.reparam_max_density,
    )
    curve_points, _, _, _ = evaluate_fourier_curve_and_parameter_derivatives(t_warped, coeffs)

    # Re-center and rescale exactly like the base generator logic would have done globally
    curve_points = curve_points - curve_points.mean(axis=0, keepdims=True)
    extent = np.max(np.abs(curve_points))
    target_extent = rng.uniform(args.curve_min_size, args.curve_max_size)
    curve_points = curve_points * (target_extent / max(extent, 1e-12))

    if curve_has_self_intersections(curve_points, closed=True):
        # very conservative fallback: keep simple uniform sample
        return base_curve, coeffs, t_uniform, False

    return curve_points.astype(np.float64), coeffs, t_warped.astype(np.float64), True


def intrinsic_context_indices(center: int, context_size: int, n: int) -> np.ndarray:
    r = context_size // 2
    offs = np.arange(-r, r + 1, dtype=np.int64)
    return np.mod(center + offs, n)


def center_points(points: np.ndarray) -> np.ndarray:
    c = points[len(points) // 2]
    return points - c.reshape(1, 2)


def plot_full_curve(ax, curve: np.ndarray, center_idx: int, neg_idxs: np.ndarray, title: str) -> None:
    ax.plot(curve[:, 0], curve[:, 1], linewidth=1.0)
    ax.scatter([curve[center_idx, 0]], [curve[center_idx, 1]], s=50, marker="x")
    if len(neg_idxs) > 0:
        ax.scatter(curve[neg_idxs, 0], curve[neg_idxs, 1], s=18)
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=9)
    ax.axis("off")


def plot_context_and_stencil(ax, context: np.ndarray, patch: np.ndarray, title: str) -> None:
    # show context faintly
    ax.plot(context[:, 0], context[:, 1], linewidth=1.0, alpha=0.35)

    # show stencil clearly
    ax.plot(patch[:, 0], patch[:, 1], color="orange", marker="o", markersize=5, linewidth=2.5)
    c = len(patch) // 2
    ax.scatter([patch[c, 0]], [patch[c, 1]], color="red", s=55, marker="x")

    # zoom around stencil, not full context
    xmin, xmax = patch[:, 0].min(), patch[:, 0].max()
    ymin, ymax = patch[:, 1].min(), patch[:, 1].max()

    dx = xmax - xmin
    dy = ymax - ymin
    span = max(dx, dy, 1e-6)

    cx = 0.5 * (xmin + xmax)
    cy = 0.5 * (ymin + ymax)

    zoom = 0.8   # smaller => tighter zoom, larger => more context around patch
    half = 0.5 * zoom * span

    ax.set_xlim(cx - half, cx + half)
    ax.set_ylim(cy - half, cy + half)

    ax.set_aspect("equal")
    ax.set_title(title, fontsize=9)
    ax.axis("off")


def main() -> None:
    args = parse_args()
    if args.context_size % 2 == 0:
        raise ValueError("--context-size must be odd")
    if args.context_size < args.patch_size:
        raise ValueError("--context-size must be >= --patch-size")

    rng = np.random.default_rng(args.seed)

    cols = 1 + 1 + 1 + args.num_negatives
    rows = args.num_examples
    fig, axes = plt.subplots(rows, cols, figsize=(3.2 * cols, 3.0 * rows))
    if rows == 1:
        axes = np.expand_dims(axes, axis=0)

    for i in range(args.num_examples):
        curve_points, coeffs, t_grid, used_reparam = generate_simple_reparameterized_curve(args, rng)

        tup = build_random_invariant_training_tuple(
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

        row = axes[i]

        plot_full_curve(
            row[0],
            curve_points,
            tup.anchor_center_index,
            tup.negative_center_indices,
            title=f"curve {i} | {'reparam' if used_reparam else 'uniform'}",
        )

        anchor_ctx_idx = intrinsic_context_indices(
            tup.anchor_center_index, args.context_size, len(curve_points)
        )
        anchor_context = center_points(curve_points[anchor_ctx_idx])
        anchor_patch = center_points(np.asarray(tup.anchor_patch, dtype=np.float64))
        plot_context_and_stencil(row[1], anchor_context, anchor_patch, "anchor context + stencil")

        # For positive, use the same local context transformed by A (translation irrelevant after centering)
        A = np.asarray(tup.transform_matrix, dtype=np.float64)
        positive_context = anchor_context @ A.T
        positive_patch = center_points(np.asarray(tup.positive_patch, dtype=np.float64))
        plot_context_and_stencil(row[2], positive_context, positive_patch, f"positive ({args.family})")

        for j in range(args.num_negatives):
            neg_center = int(tup.negative_center_indices[j])
            neg_ctx_idx = intrinsic_context_indices(neg_center, args.context_size, len(curve_points))
            neg_context = center_points(curve_points[neg_ctx_idx])
            neg_patch = center_points(np.asarray(tup.negative_patches[j], dtype=np.float64))
            plot_context_and_stencil(row[3 + j], neg_context, neg_patch, f"neg {j}")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
