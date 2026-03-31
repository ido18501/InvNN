from __future__ import annotations

import argparse
import numpy as np
import matplotlib.pyplot as plt

from utils.curve_generation import (
    generate_random_basis_expansion_coeffs,
    make_fourier_basis_functions,
    make_fourier_coeff_std,
    evaluate_basis_expansion_curve,
    center_curve,
    fit_curve_to_canvas_with_random_size,
    sample_smooth_monotone_periodic_reparameterization,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--num-points", type=int, default=4000)
    p.add_argument("--smooth-points", type=int, default=40000)
    p.add_argument("--max-freq", type=int, default=7)
    p.add_argument("--scale", type=float, default=1.0)
    p.add_argument("--decay-power", type=float, default=1.65)
    p.add_argument("--strength", type=float, default=0.35)
    p.add_argument("--num-harmonics", type=int, default=3)
    p.add_argument("--min-density", type=float, default=0.35)
    p.add_argument("--max-density", type=float, default=3.0)
    p.add_argument("--seed", type=int, default=123)
    return p.parse_args()


def make_example(rng, args):
    basis_functions = make_fourier_basis_functions(args.max_freq)
    coeff_std = make_fourier_coeff_std(args.max_freq, args.scale, args.decay_power)
    coeffs = generate_random_basis_expansion_coeffs(len(basis_functions), coeff_std=coeff_std, rng=rng)

    t_smooth = np.linspace(0.0, 2.0 * np.pi, args.smooth_points, endpoint=False)
    smooth = evaluate_basis_expansion_curve(t_smooth, basis_functions, coeffs)
    smooth = center_curve(smooth)

    _, t_warped = sample_smooth_monotone_periodic_reparameterization(
        num_points=args.num_points,
        rng=rng,
        strength=args.strength,
        num_harmonics=args.num_harmonics,
        min_density=args.min_density,
        max_density=args.max_density,
    )
    sampled = evaluate_basis_expansion_curve(t_warped, basis_functions, coeffs)
    sampled = center_curve(sampled)

    target_extent = rng.uniform(0.3, 0.9)
    smooth = smooth * (target_extent / max(np.max(np.abs(smooth)), 1e-12))
    sampled = sampled * (target_extent / max(np.max(np.abs(sampled)), 1e-12))

    seg = np.linalg.norm(np.roll(sampled, -1, axis=0) - sampled, axis=1)
    return smooth, sampled, t_warped, seg


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    ax0, ax1 = axes

    state = {"idx": 0}

    def redraw():
        smooth, sampled, t_warped, seg = make_example(rng, args)
        ax0.clear()
        ax1.clear()

        ax0.plot(smooth[:, 0], smooth[:, 1], lw=1.5, label="smooth curve")
        ax0.scatter(sampled[:, 0], sampled[:, 1], s=8, alpha=0.7, label="sampled points")
        ax0.set_title(f"geometry view | example {state['idx']}")
        ax0.set_aspect("equal")
        ax0.legend()

        ax1.plot(seg, lw=1.0, label="segment lengths")
        ax1.set_title(
            f"spacing view | min={seg.min():.4e}, max={seg.max():.4e}, ratio={seg.max()/max(seg.min(),1e-12):.2f}"
        )
        ax1.legend()

        fig.suptitle("Press right/left or n/p for next example")
        fig.canvas.draw_idle()

    def on_key(event):
        if event.key in ["right", "n", " "]:
            state["idx"] += 1
            redraw()
        elif event.key in ["left", "p"]:
            state["idx"] -= 1
            redraw()

    fig.canvas.mpl_connect("key_press_event", on_key)
    redraw()
    plt.show()


if __name__ == "__main__":
    main()
