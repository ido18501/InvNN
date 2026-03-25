from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from datasets.tangent_tuple_generation import build_random_tangent_training_tuple


def load_curves(npz_path: str) -> np.ndarray:
    data = np.load(npz_path, allow_pickle=True)
    for key in ["curves", "curve_points", "points"]:
        if key in data:
            arr = data[key]
            if arr.ndim == 3 and arr.shape[-1] == 2:
                return arr
    raise RuntimeError(f"Could not find curves array in {npz_path}")


def curve_distance(i: int, j: int, n: int, closed: bool) -> int:
    d = abs(i - j)
    if closed:
        d = min(d, n - d)
    return d


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--bank", type=str, required=True)
    p.add_argument("--output-dir", type=str, required=True)
    p.add_argument("--num-examples", type=int, default=12)
    p.add_argument("--patch-size", type=int, default=9)
    p.add_argument("--half-width", type=int, default=12)
    p.add_argument("--num-negatives", type=int, default=4)
    p.add_argument("--negative-min-offset", type=int, default=4)
    p.add_argument("--negative-max-offset", type=int, default=22)
    p.add_argument("--patch-mode", type=str, default="random_warp_symmetric")
    p.add_argument("--jitter-fraction", type=float, default=0.25)
    p.add_argument("--closed", action="store_true")
    p.add_argument("--seed", type=int, default=123)
    args = p.parse_args()

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    curves = load_curves(args.bank)
    rng = np.random.default_rng(args.seed)

    for ex in range(args.num_examples):
        curve_idx = int(rng.integers(0, len(curves)))
        curve = np.asarray(curves[curve_idx], dtype=np.float32)

        tuple_ = build_random_tangent_training_tuple(
            curve_points=curve,
            transform_family="euclidean",
            patch_size=args.patch_size,
            half_width=args.half_width,
            num_negatives=args.num_negatives,
            negative_min_offset=args.negative_min_offset,
            negative_max_offset=args.negative_max_offset,
            closed=args.closed,
            patch_mode=args.patch_mode,
            jitter_fraction=args.jitter_fraction,
            rng=rng,
            transform_kwargs={},
            gt_dense_num_points=4096,
        )

        anchor_center = int(tuple_.anchor_center_index)
        neg_centers = [int(x) for x in tuple_.negative_center_indices.tolist()]

        fig = plt.figure(figsize=(14, 8))

        ax0 = fig.add_subplot(2, 3, 1)
        ax0.plot(curve[:, 0], curve[:, 1], lw=1.5)
        ax0.scatter(curve[anchor_center, 0], curve[anchor_center, 1], s=80, label="anchor")
        ax0.scatter(curve[anchor_center, 0], curve[anchor_center, 1], s=80, marker="x", label="positive")
        for i, c in enumerate(neg_centers):
            ax0.scatter(curve[c, 0], curve[c, 1], s=60, marker="^", label=f"neg{i}")
        ax0.set_title(f"curve {curve_idx}: centers")
        ax0.axis("equal")
        ax0.legend(fontsize=8)

        ax1 = fig.add_subplot(2, 3, 2)
        ax1.plot(tuple_.anchor_patch[:, 0], tuple_.anchor_patch[:, 1], "-o")
        ax1.set_title("anchor patch")
        ax1.axis("equal")

        ax2 = fig.add_subplot(2, 3, 3)
        ax2.plot(tuple_.positive_patch[:, 0], tuple_.positive_patch[:, 1], "-o")
        ax2.set_title("positive patch (real transformed+resampled)")
        ax2.axis("equal")

        for j in range(min(3, len(tuple_.negative_patches))):
            ax = fig.add_subplot(2, 3, 4 + j)
            neg_patch = tuple_.negative_patches[j]
            offset = curve_distance(anchor_center, neg_centers[j], len(curve), args.closed)
            ax.plot(neg_patch[:, 0], neg_patch[:, 1], "-o")
            ax.set_title(f"negative {j} | offset={offset}")
            ax.axis("equal")

        fig.suptitle(
            f"example {ex} | curve={curve_idx} | anchor={anchor_center} | negatives={neg_centers}",
            fontsize=12,
        )
        fig.tight_layout()
        fig.savefig(outdir / f"tuple_example_{ex:02d}.png", dpi=160)
        plt.close(fig)

    print(f"saved tuple visualizations to {outdir}")


if __name__ == "__main__":
    main()
