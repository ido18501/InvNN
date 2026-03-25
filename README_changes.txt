Files in this bundle:
- utils/curve_generation.py
- datasets/tangent_tuple_generation.py
- scripts/generate_curve_bank.py

What changed:
1) Fourier curve generation now defaults to more expressive curves (higher max_freq, slower coefficient decay).
2) Simple-curve enforcement is done by rejecting only self-intersecting / degenerate sampled curves.
3) Bank generation is multiprocessing-enabled and saves .npz files with multiple compatibility keys.
4) Negative sampling is harder: within the valid offset band, closer negatives are sampled with higher probability.

Notes:
- The generation script saves: curves, curve_points, points, x_coeffs, y_coeffs, t_values, t_grid.
- This should maximize compatibility with existing pregenerated loaders.
