from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_curves(npz_path: str) -> np.ndarray:
    data = np.load(npz_path, allow_pickle=True)
    for key in ["curves", "curve_points", "points"]:
        if key in data:
            arr = data[key]
            if arr.ndim == 3 and arr.shape[-1] == 2:
                return arr
    raise RuntimeError(f"Could not find curves array in {npz_path}")


def sample_patch_indices(center: int, patch_size: int, half_width: int, n_points: int) -> np.ndarray:
    idx = np.linspace(center - half_width, center + half_width, patch_size)
    idx = np.round(idx).astype(int)
    idx = np.clip(idx, 0, n_points - 1)
    return idx


def choose_negative_centers(
    center: int,
    n_points: int,
    num_negatives: int,
    min_offset: int,
    max_offset: int,
    rng: np.random.Generator,
) -> list[int]:
    offsets = []
    for d in range(min_offset, max_offset + 1):
        offsets.append(-d)
        offsets.append(d)

    offsets = np.array(offsets, dtype=int)

    valid = []
    weights = []
    for off in offsets:
        c = center + off
        if 0 <= c < n_points:
            valid.append(c)
            # harder negatives = prefer smaller offsets
            weights.append(1.0 / abs(off))

    valid = np.array(valid, dtype=int)
    weights = np.array(weights, dtype=float)
    weights /= weights.sum()

    if len(valid) == 0:
        return []

    k = min(num_negatives, len(valid))
    chosen = rng.choice(valid, size=k, replace=False, p=weights)
    return chosen.tolist()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--bank", type=str, required=True)
    p.add_argument("--output-dir", type=str, required=True)
    p.add_argument("--num-examples", type=int, default=12)
    p.add_argument("--patch-size", type=int, default=9)
    p.add_argument("--half-width", type=int, default=12)
    p.add_argument("--num-negatives", type=int, default=4)
    p.add_argument("--min-negative-offset", type=int, default=20)
    p.add_argument("--max-negative-offset", type=int, default=120)
    p.add_argument("--seed", type=int, default=123)
    args = p.parse_args()

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    curves = load_curves(args.bank)

    for ex in range(args.num_examples):
        curve_idx = int(rng.integers(0, len(curves)))
        curve = np.asarray(curves[curve_idx], dtype=float)
        n = len(curve)

        margin = max(args.half_width, args.max_negative_offset + 2)
        if n <= 2 * margin + 1:
            continue

        anchor_center = int(rng.integers(margin, n - margin))
        positive_center = anchor_center  # same center, transformed later in training

        neg_centers = choose_negative_centers(
            center=anchor_center,
            n_points=n,
            num_negatives=args.num_negatives,
            min_offset=args.min_negative_offset,
            max_offset=args.max_negative_offset,
            rng=rng,
        )

        anchor_idx = sample_patch_indices(anchor_center, args.patch_size, args.half_width, n)
        pos_idx = sample_patch_indices(positive_center, args.patch_size, args.half_width, n)
        neg_idx_list = [
            sample_patch_indices(c, args.patch_size, args.half_width, n)
            for c in neg_centers
        ]

        fig = plt.figure(figsize=(14, 8))

        ax0 = fig.add_subplot(2, 3, 1)
        ax0.plot(curve[:, 0], curve[:, 1], lw=1.5)
        ax0.scatter(curve[anchor_center, 0], curve[anchor_center, 1], s=80, label="anchor")
        ax0.scatter(curve[positive_center, 0], curve[positive_center, 1], s=80, marker="x", label="positive")
        for i, c in enumerate(neg_centers):
            ax0.scatter(curve[c, 0], curve[c, 1], s=60, marker="^", label=f"neg{i}")
        ax0.set_title(f"curve {curve_idx}: centers")
        ax0.axis("equal")
        ax0.legend(fontsize=8)

        ax1 = fig.add_subplot(2, 3, 2)
        anchor_patch = curve[anchor_idx]
        ax1.plot(anchor_patch[:, 0], anchor_patch[:, 1], "-o")
        ax1.set_title("anchor patch")
        ax1.axis("equal")

        ax2 = fig.add_subplot(2, 3, 3)
        pos_patch = curve[pos_idx]
        ax2.plot(pos_patch[:, 0], pos_patch[:, 1], "-o")
        ax2.set_title("positive patch (same center pre-transform)")
        ax2.axis("equal")

        for j in range(min(3, len(neg_idx_list))):
            ax = fig.add_subplot(2, 3, 4 + j)
            neg_patch = curve[neg_idx_list[j]]
            ax.plot(neg_patch[:, 0], neg_patch[:, 1], "-o")
            dist = abs(neg_centers[j] - anchor_center)
            ax.set_title(f"negative {j} | offset={dist}")
            ax.axis("equal")

        fig.suptitle(
            f"example {ex} | curve={curve_idx} | anchor={anchor_center} | negatives={neg_centers}",
            fontsize=12,
        )
        fig.tight_layout()
        fig.savefig(outdir / f"negative_example_{ex:02d}.png", dpi=160)
        plt.close(fig)

    print(f"saved visualizations to {outdir}")


if __name__ == "__main__":
    main()
