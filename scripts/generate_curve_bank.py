from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from utils.curve_generation import generate_random_simple_fourier_curve



def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=str, required=True)
    p.add_argument('--num-curves', type=int, default=5000)
    p.add_argument('--num-points', type=int, default=1000)
    p.add_argument('--max-freq', type=int, default=5)
    p.add_argument('--scale', type=float, default=0.9)
    p.add_argument('--decay-power', type=float, default=2.0)
    p.add_argument('--curve-max-tries', type=int, default=300)
    p.add_argument('--min-size', type=float, default=0.45)
    p.add_argument('--max-size', type=float, default=0.75)
    p.add_argument('--seed', type=int, default=123)
    return p.parse_args()



def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    t = np.linspace(0.0, 2.0 * np.pi, args.num_points, endpoint=False)

    curves = []
    x_coeffs = []
    y_coeffs = []

    for _ in range(args.num_curves):
        pts, coeffs = generate_random_simple_fourier_curve(
            t=t,
            max_freq=args.max_freq,
            scale=args.scale,
            decay_power=args.decay_power,
            rng=rng,
            max_tries=args.curve_max_tries,
            center=True,
            fit_to_canvas=True,
            min_size=args.min_size,
            max_size=args.max_size,
            enforce_simple=False,
        )
        curves.append(pts.astype(np.float32))
        x_coeffs.append(coeffs.x_coeffs.astype(np.float64))
        y_coeffs.append(coeffs.y_coeffs.astype(np.float64))

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        curve_points=np.stack(curves, axis=0),
        x_coeffs=np.stack(x_coeffs, axis=0),
        y_coeffs=np.stack(y_coeffs, axis=0),
        t_grid=t.astype(np.float64),
    )
    print(f'Saved bank to {output}')
    print(f'curve_points shape: {np.stack(curves, axis=0).shape}')


if __name__ == '__main__':
    main()
