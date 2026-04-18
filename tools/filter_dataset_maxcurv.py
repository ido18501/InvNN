from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from datasets.tangent_dataset import PregeneratedCurveBank
from utils.derivatives import compute_fourier_arc_length_derivatives


def compute_max_curvature(coeffs, t, family: str) -> float:
    _, gt1, gt2 = compute_fourier_arc_length_derivatives(
        t=np.asarray(t, dtype=np.float64),
        coeffs=coeffs,
        family=family,
    )

    fn = np.linalg.norm(gt1, axis=1)
    cross = gt1[:, 0] * gt2[:, 1] - gt1[:, 1] * gt2[:, 0]
    kappa = np.abs(cross) / (fn**3 + 1e-12)
    return float(np.max(kappa))


def should_filter_array(arr: np.ndarray, num_curves: int) -> bool:
    # Only filter arrays whose first dimension matches number of curves.
    # Scalars / 0-D arrays and per-point arrays should be left untouched.
    return isinstance(arr, np.ndarray) and arr.ndim >= 1 and arr.shape[0] == num_curves


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--threshold", type=float, default=1000.0)
    parser.add_argument("--family", default="euclidean")
    args = parser.parse_args()

    bank = PregeneratedCurveBank(args.input)
    num_curves = len(bank)

    keep_indices = []
    max_curvs = []

    for i in range(num_curves):
        _, coeffs, t = bank.get(i)
        max_k = compute_max_curvature(coeffs, t, args.family)
        max_curvs.append(max_k)
        if max_k <= args.threshold:
            keep_indices.append(i)

    keep_indices = np.asarray(keep_indices, dtype=np.int64)
    print(f"Keeping {len(keep_indices)} / {num_curves} curves")

    data = np.load(args.input, allow_pickle=True)

    filtered = {}
    for k in data.files:
        arr = data[k]
        if should_filter_array(arr, num_curves):
            filtered[k] = arr[keep_indices]
        else:
            filtered[k] = arr

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **filtered)

    if len(keep_indices) > 0:
        kept_curvs = np.asarray(max_curvs)[keep_indices]
        print(f"Saved to {args.output}")
        print(f"Kept max-curvature stats: mean={kept_curvs.mean():.2f}, max={kept_curvs.max():.2f}")
    else:
        print(f"Saved empty filtered dataset to {args.output}")


if __name__ == "__main__":
    main()
