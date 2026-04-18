from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from utils.curve_generation import generate_random_reparameterized_fourier_curve
from utils.derivatives import (
    evaluate_fourier_curve_and_parameter_derivatives,
    compute_arc_length_derivatives_from_parameter_derivatives,
)

try:
    from scipy.signal import savgol_filter
    HAS_SCIPY = True
except Exception:
    HAS_SCIPY = False


Array = np.ndarray


@dataclass
class Regime:
    name: str
    max_freq: int
    decay_power: float
    scale: float


def _wrap_idx(i: int, n: int) -> int:
    return i % n


def _norm(x: Array, axis: int = -1, eps: float = 1e-12) -> Array:
    return np.sqrt(np.sum(x * x, axis=axis) + eps)


def _unit(v: Array, eps: float = 1e-12) -> Array:
    return v / np.maximum(_norm(v, axis=-1, eps=eps)[..., None], eps)


def _circumcircle_second(points: Array) -> Array:
    n = len(points)
    out = np.zeros((n, 2), dtype=np.float64)
    for i in range(n):
        p0 = points[_wrap_idx(i - 1, n)]
        p1 = points[i]
        p2 = points[_wrap_idx(i + 1, n)]

        a = p1 - p0
        b = p2 - p1
        c = p2 - p0
        la = np.linalg.norm(a)
        lb = np.linalg.norm(b)
        lc = np.linalg.norm(c)
        area2 = abs(a[0] * c[1] - a[1] * c[0])
        denom = max(la * lb * lc, 1e-12)
        kappa = 2.0 * area2 / denom

        tan = _unit(c)
        normal = np.array([-tan[1], tan[0]], dtype=np.float64)
        turn = a[0] * b[1] - a[1] * b[0]
        if turn < 0:
            normal = -normal
        out[i] = kappa * normal
    return out


def _turning_angle_second(points: Array) -> Array:
    n = len(points)
    out = np.zeros((n, 2), dtype=np.float64)
    for i in range(n):
        pm = points[_wrap_idx(i - 1, n)]
        p = points[i]
        pp = points[_wrap_idx(i + 1, n)]
        vm = p - pm
        vp = pp - p
        hm = np.linalg.norm(vm)
        hp = np.linalg.norm(vp)
        if hm <= 1e-12 or hp <= 1e-12:
            continue
        tm = vm / hm
        tp = vp / hp
        cosv = np.clip(np.dot(tm, tp), -1.0, 1.0)
        ang = np.arccos(cosv)
        ds = 0.5 * (hm + hp)
        kappa = ang / max(ds, 1e-12)
        chord = pp - pm
        tan = _unit(chord)
        normal = np.array([-tan[1], tan[0]], dtype=np.float64)
        if vm[0] * vp[1] - vm[1] * vp[0] < 0:
            normal = -normal
        out[i] = kappa * normal
    return out


def _savgol_second(points: Array, window: int, degree: int) -> Array:
    if not HAS_SCIPY:
        raise RuntimeError('SciPy not installed; cannot run Savitzky-Golay baseline.')
    n = len(points)
    ext = np.concatenate([points[-window:], points, points[:window]], axis=0)
    dx = savgol_filter(ext[:, 0], window_length=window, polyorder=degree, deriv=1, delta=1.0, mode='interp')
    dy = savgol_filter(ext[:, 1], window_length=window, polyorder=degree, deriv=1, delta=1.0, mode='interp')
    ddx = savgol_filter(ext[:, 0], window_length=window, polyorder=degree, deriv=2, delta=1.0, mode='interp')
    ddy = savgol_filter(ext[:, 1], window_length=window, polyorder=degree, deriv=2, delta=1.0, mode='interp')
    dx = dx[window:window + n]
    dy = dy[window:window + n]
    ddx = ddx[window:window + n]
    ddy = ddy[window:window + n]
    first = np.stack([dx, dy], axis=1)
    second = np.stack([ddx, ddy], axis=1)
    speed = _norm(first)
    dot12 = np.sum(first * second, axis=1)
    second_ds = second / np.maximum(speed[:, None] ** 2, 1e-12) - first * dot12[:, None] / np.maximum(speed[:, None] ** 4, 1e-12)
    return second_ds


def _spearman(x: Array, y: Array) -> float:
    rx = pd.Series(x).rank(method='average').to_numpy()
    ry = pd.Series(y).rank(method='average').to_numpy()
    if np.std(rx) < 1e-12 or np.std(ry) < 1e-12:
        return float('nan')
    return float(np.corrcoef(rx, ry)[0, 1])


def _metrics(pred: Array, gt: Array) -> dict[str, float]:
    pred_n = _unit(pred)
    gt_n = _unit(gt)
    cos = np.clip(np.sum(pred_n * gt_n, axis=1), -1.0, 1.0)
    pred_norm = _norm(pred)
    gt_norm = _norm(gt)
    slope, intercept = np.polyfit(gt_norm, pred_norm, 1)
    return {
        'cosine_mean': float(np.mean(cos)),
        'angle_mean_deg': float(np.degrees(np.mean(np.arccos(cos)))),
        'mse': float(np.mean(np.sum((pred - gt) ** 2, axis=1))),
        'norm_spearman': _spearman(pred_norm, gt_norm),
        'norm_pearson': float(np.corrcoef(pred_norm, gt_norm)[0, 1]) if np.std(pred_norm) > 1e-12 and np.std(gt_norm) > 1e-12 else float('nan'),
        'log1p_norm_pearson': float(np.corrcoef(np.log1p(pred_norm), np.log1p(gt_norm))[0, 1]),
        'norm_fit_slope': float(slope),
        'norm_fit_intercept': float(intercept),
        'pred_norm_mean': float(np.mean(pred_norm)),
        'gt_norm_mean': float(np.mean(gt_norm)),
    }


def _tail_stats(gt_norm: Array) -> dict[str, float]:
    q = np.quantile(gt_norm, [0.5, 0.9, 0.99, 0.999, 0.9999])
    return {
        'gt_norm_min': float(np.min(gt_norm)),
        'gt_norm_median': float(q[0]),
        'gt_norm_q90': float(q[1]),
        'gt_norm_q99': float(q[2]),
        'gt_norm_q999': float(q[3]),
        'gt_norm_q9999': float(q[4]),
        'gt_norm_max': float(np.max(gt_norm)),
        'gt_norm_log10_range': float(np.log10(max(np.max(gt_norm), 1e-12)) - np.log10(max(np.min(gt_norm), 1e-12))),
    }


def generate_regime_points(regime: Regime, n_curves: int, n_points: int, rng: np.random.Generator,
                           reparam_strength: float, reparam_num_harmonics: int,
                           reparam_min_density: float, reparam_max_density: float) -> list[tuple[Array, Array]]:
    out = []
    for _ in range(n_curves):
        pts, coeffs, _, t_warped = generate_random_reparameterized_fourier_curve(
            num_points=n_points,
            max_freq=regime.max_freq,
            scale=regime.scale,
            decay_power=regime.decay_power,
            rng=rng,
            center=True,
            fit_to_canvas=True,
            min_size=0.45,
            max_size=0.75,
            reparam_strength=reparam_strength,
            reparam_num_harmonics=reparam_num_harmonics,
            reparam_min_density=reparam_min_density,
            reparam_max_density=reparam_max_density,
            max_tries=1000,
            enforce_simple=True,
            intersection_check_points=max(320, n_points // 2),
            downsample_to_points=None,
        )
        _, first_dt, second_dt, third_dt = evaluate_fourier_curve_and_parameter_derivatives(t_warped, coeffs)
        _, gt_second = compute_arc_length_derivatives_from_parameter_derivatives(
            family='euclidean',
            first_dt=first_dt,
            second_dt=second_dt,
            third_dt=third_dt,
        )
        out.append((pts, gt_second))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--output-dir', type=str, required=True)
    ap.add_argument('--num-curves', type=int, default=200)
    ap.add_argument('--num-points', type=int, default=1000)
    ap.add_argument('--seed', type=int, default=123)
    ap.add_argument('--reparam-strength', type=float, default=0.15)
    ap.add_argument('--reparam-num-harmonics', type=int, default=2)
    ap.add_argument('--reparam-min-density', type=float, default=0.7)
    ap.add_argument('--reparam-max-density', type=float, default=1.5)
    ap.add_argument('--savgol-window', type=int, default=21)
    ap.add_argument('--savgol-degree', type=int, default=5)
    args = ap.parse_args()

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    regimes = [
        Regime('very_simple_f2_dp3.5', max_freq=2, decay_power=3.5, scale=0.9),
        Regime('simple_f3_dp3.0', max_freq=3, decay_power=3.0, scale=0.9),
        Regime('moderate_f5_dp2.5', max_freq=5, decay_power=2.5, scale=0.9),
        Regime('hard_f7_dp2.0', max_freq=7, decay_power=2.0, scale=0.9),
        Regime('very_hard_f9_dp1.65', max_freq=9, decay_power=1.65, scale=0.9),
    ]

    rows = []
    for regime in regimes:
        curves = generate_regime_points(
            regime=regime,
            n_curves=args.num_curves,
            n_points=args.num_points,
            rng=rng,
            reparam_strength=args.reparam_strength,
            reparam_num_harmonics=args.reparam_num_harmonics,
            reparam_min_density=args.reparam_min_density,
            reparam_max_density=args.reparam_max_density,
        )
        gt_second = np.concatenate([gt for _, gt in curves], axis=0)
        gt_norm = _norm(gt_second)
        record = {'regime': regime.name, **_tail_stats(gt_norm)}

        all_turn = []
        all_circ = []
        all_sg = []
        for pts, gt in curves:
            all_turn.append(_turning_angle_second(pts))
            all_circ.append(_circumcircle_second(pts))
            if HAS_SCIPY:
                all_sg.append(_savgol_second(pts, window=args.savgol_window, degree=args.savgol_degree))
        turn = np.concatenate(all_turn, axis=0)
        circ = np.concatenate(all_circ, axis=0)
        record.update({f'turn_{k}': v for k, v in _metrics(turn, gt_second).items()})
        record.update({f'circ_{k}': v for k, v in _metrics(circ, gt_second).items()})
        if HAS_SCIPY:
            sg = np.concatenate(all_sg, axis=0)
            record.update({f'savgol_{k}': v for k, v in _metrics(sg, gt_second).items()})
        rows.append(record)
        print(json.dumps(record, indent=2))

    df = pd.DataFrame(rows)
    df.to_csv(outdir / 'regime_summary.csv', index=False)
    (outdir / 'regime_summary.json').write_text(df.to_json(orient='records', indent=2))

    # Plots
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(df['regime'], df['gt_norm_q99'], marker='o', label='q99')
    ax.plot(df['regime'], df['gt_norm_q999'], marker='o', label='q999')
    ax.plot(df['regime'], df['gt_norm_max'], marker='o', label='max')
    ax.set_yscale('log')
    ax.set_ylabel('true ||x"(s)|| tail')
    ax.set_title('Curvature tail vs curve complexity (with reparameterization)')
    ax.tick_params(axis='x', rotation=25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(outdir / 'tail_vs_complexity.png', dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(df['regime'], df['turn_log1p_norm_pearson'], marker='o', label='turn log-corr')
    ax.plot(df['regime'], df['turn_norm_pearson'], marker='o', label='turn linear corr')
    ax.plot(df['regime'], df['circ_log1p_norm_pearson'], marker='o', label='circ log-corr')
    ax.plot(df['regime'], df['circ_norm_pearson'], marker='o', label='circ linear corr')
    if HAS_SCIPY:
        ax.plot(df['regime'], df['savgol_log1p_norm_pearson'], marker='o', label='sg log-corr')
        ax.plot(df['regime'], df['savgol_norm_pearson'], marker='o', label='sg linear corr')
    ax.set_ylabel('correlation')
    ax.set_ylim(-0.05, 1.05)
    ax.set_title('Log vs linear magnitude calibration across complexity')
    ax.tick_params(axis='x', rotation=25)
    ax.legend(ncol=2)
    fig.tight_layout()
    fig.savefig(outdir / 'calibration_vs_complexity.png', dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(df['regime'], df['turn_mse'], marker='o', label='turn')
    ax.plot(df['regime'], df['circ_mse'], marker='o', label='circ')
    if HAS_SCIPY:
        ax.plot(df['regime'], df['savgol_mse'], marker='o', label='savgol')
    ax.set_yscale('log')
    ax.set_ylabel('MSE')
    ax.set_title('Second-order MSE across complexity')
    ax.tick_params(axis='x', rotation=25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(outdir / 'mse_vs_complexity.png', dpi=180)
    plt.close(fig)


if __name__ == '__main__':
    main()
