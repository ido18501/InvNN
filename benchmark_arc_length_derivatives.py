from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from torch.utils.data import DataLoader

from datasets.tangent_dataset import PregeneratedCurveBank
from utils.derivatives import compute_fourier_arc_length_derivatives
from train_second_order_finetune import load_pretrained_config, instantiate_model_from_config
from train_second_derivative_operator import parse_int_list as parse_int_list_direct
from training.second_order_finetune_trainer import SecondOrderFineTuneTrainer as UnprojectedFirstTrainer
from training.projected_second_order_finetune_trainer import ProjectedSecondOrderFineTuneTrainer as ProjectedFirstTrainer


ModelType = Literal['first_unprojected', 'first_projected', 'direct_second']


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument('--bank', type=str, required=True)
    p.add_argument('--family', type=str, default='euclidean', choices=['euclidean', 'similarity', 'equi_affine'])
    p.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--num-workers', type=int, default=0)
    p.add_argument('--curve-batch-size', type=int, default=4)
    p.add_argument('--output-dir', type=str, required=True)
    p.add_argument('--spectral-resample-factor', type=int, default=1)
    p.add_argument('--fornberg-stencil', type=int, default=21)
    p.add_argument('--savgol-window', type=int, default=21)
    p.add_argument('--savgol-degree', type=int, default=5)
    p.add_argument(
        '--model-spec', action='append', default=[],
        help='Format: name:type:ckpt_path[:pretrained_best_model_path]. type in {first_unprojected,first_projected,direct_second}'
    )
    return p.parse_args()


@dataclass
class CurveRecord:
    curve_points: np.ndarray
    gt1: np.ndarray
    gt2: np.ndarray
    length: float
    s: np.ndarray


class MetricComputer:
    @staticmethod
    def cosine_and_angle(pred: np.ndarray, gt: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        pred_n = pred / np.clip(np.linalg.norm(pred, axis=-1, keepdims=True), 1e-8, None)
        gt_n = gt / np.clip(np.linalg.norm(gt, axis=-1, keepdims=True), 1e-8, None)
        cos = np.sum(pred_n * gt_n, axis=-1)
        cos = np.clip(cos, -1.0, 1.0)
        angle = np.degrees(np.arccos(cos))
        return cos, angle

    @staticmethod
    def pearson(x: np.ndarray, y: np.ndarray) -> float:
        x = np.asarray(x, dtype=np.float64).reshape(-1)
        y = np.asarray(y, dtype=np.float64).reshape(-1)
        if x.size < 2 or y.size < 2:
            return float('nan')
        x = x - x.mean()
        y = y - y.mean()
        denom = np.linalg.norm(x) * np.linalg.norm(y)
        if denom <= 1e-12:
            return float('nan')
        return float(np.dot(x, y) / denom)

    @staticmethod
    def rankdata_average(x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        order = np.argsort(x, kind='mergesort')
        ranks = np.empty_like(order, dtype=np.float64)
        xs = x[order]
        start = 0
        while start < len(x):
            end = start + 1
            while end < len(x) and xs[end] == xs[start]:
                end += 1
            avg_rank = 0.5 * (start + end - 1) + 1.0
            ranks[order[start:end]] = avg_rank
            start = end
        return ranks

    @classmethod
    def spearman(cls, x: np.ndarray, y: np.ndarray) -> float:
        return cls.pearson(cls.rankdata_average(x), cls.rankdata_average(y))

    @classmethod
    def vector_metrics(cls, pred: np.ndarray, gt: np.ndarray, prefix: str) -> dict[str, float]:
        cos, angle = cls.cosine_and_angle(pred, gt)
        pred_norm = np.linalg.norm(pred, axis=-1)
        gt_norm = np.linalg.norm(gt, axis=-1)
        norm_err = np.abs(pred_norm - gt_norm)
        out = {
            f'{prefix}_cosine_mean': float(np.mean(cos)),
            f'{prefix}_abs_cosine_mean': float(np.mean(np.abs(cos))),
            f'{prefix}_angle_mean': float(np.mean(angle)),
            f'{prefix}_mse': float(np.mean((pred - gt) ** 2)),
            f'{prefix}_pred_norm_mean': float(np.mean(pred_norm)),
            f'{prefix}_pred_norm_median': float(np.median(pred_norm)),
            f'{prefix}_norm_error_mean': float(np.mean(norm_err)),
            f'{prefix}_norm_error_median': float(np.median(norm_err)),
        }
        pred_n = pred_norm.reshape(-1)
        gt_n = gt_norm.reshape(-1)
        out[f'{prefix}_norm_spearman'] = cls.spearman(pred_n, gt_n)
        out[f'{prefix}_norm_pearson'] = cls.pearson(pred_n, gt_n)
        out[f'{prefix}_log1p_norm_pearson'] = cls.pearson(np.log1p(pred_n), np.log1p(gt_n))
        if pred_n.size >= 2 and np.std(gt_n) > 1e-12:
            slope, intercept = np.polyfit(gt_n, pred_n, deg=1)
            out[f'{prefix}_norm_fit_slope'] = float(slope)
            out[f'{prefix}_norm_fit_intercept'] = float(intercept)
        else:
            out[f'{prefix}_norm_fit_slope'] = float('nan')
            out[f'{prefix}_norm_fit_intercept'] = float('nan')
        return out


def closed_curve_arc_length(points: np.ndarray) -> tuple[np.ndarray, float]:
    pts = np.asarray(points, dtype=np.float64)
    ext = np.vstack([pts, pts[:1]])
    seg = np.linalg.norm(np.diff(ext, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    total = float(cum[-1])
    s = cum[:-1]
    return s, total


def periodic_interp_1d(s_query: np.ndarray, s_base: np.ndarray, y_base: np.ndarray, period: float) -> np.ndarray:
    s_query = np.asarray(s_query, dtype=np.float64)
    s_base = np.asarray(s_base, dtype=np.float64)
    y_base = np.asarray(y_base, dtype=np.float64)
    s_q = np.mod(s_query, period)
    s_aug = np.concatenate([s_base, [s_base[0] + period]])
    y_aug = np.concatenate([y_base, [y_base[0]]])
    return np.interp(s_q, s_aug, y_aug)


def spectral_derivatives(points: np.ndarray, resample_factor: int = 1) -> tuple[np.ndarray, np.ndarray]:
    s, L = closed_curve_arc_length(points)
    n = len(points)
    m = int(max(n, n * resample_factor))
    su = np.linspace(0.0, L, m, endpoint=False)
    xu = periodic_interp_1d(su, s, points[:, 0], L)
    yu = periodic_interp_1d(su, s, points[:, 1], L)
    ds = L / m
    k = 2.0 * np.pi * np.fft.fftfreq(m, d=ds)

    def diff(arr: np.ndarray, order: int) -> np.ndarray:
        hat = np.fft.fft(arr)
        return np.fft.ifft((1j * k) ** order * hat).real

    dx = diff(xu, 1)
    dy = diff(yu, 1)
    ddx = diff(xu, 2)
    ddy = diff(yu, 2)
    first_u = np.stack([dx, dy], axis=1)
    second_u = np.stack([ddx, ddy], axis=1)
    first = np.stack([
        periodic_interp_1d(s, su, first_u[:, 0], L),
        periodic_interp_1d(s, su, first_u[:, 1], L),
    ], axis=1)
    second = np.stack([
        periodic_interp_1d(s, su, second_u[:, 0], L),
        periodic_interp_1d(s, su, second_u[:, 1], L),
    ], axis=1)
    return first, second


def finite_diff_weights(x: np.ndarray, x0: float, deriv_order: int) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    m = len(x)
    V = np.vstack([(x - x0) ** p for p in range(m)])
    b = np.zeros(m, dtype=np.float64)
    b[deriv_order] = float(np.math.factorial(deriv_order))
    try:
        w = np.linalg.solve(V, b)
    except np.linalg.LinAlgError:
        w = np.linalg.lstsq(V, b, rcond=None)[0]
    return w


def periodic_signed_offsets(s: np.ndarray, i: int, idxs: np.ndarray, L: float) -> np.ndarray:
    d = s[idxs] - s[i]
    d = (d + 0.5 * L) % L - 0.5 * L
    return d


def fornberg_periodic_derivatives(points: np.ndarray, stencil: int) -> tuple[np.ndarray, np.ndarray]:
    n = len(points)
    if stencil % 2 == 0:
        raise ValueError('fornberg stencil must be odd')
    s, L = closed_curve_arc_length(points)
    r = stencil // 2
    first = np.zeros_like(points, dtype=np.float64)
    second = np.zeros_like(points, dtype=np.float64)
    for i in range(n):
        idxs = (np.arange(i - r, i + r + 1) % n).astype(int)
        offs = periodic_signed_offsets(s, i, idxs, L)
        w1 = finite_diff_weights(offs, 0.0, 1)
        w2 = finite_diff_weights(offs, 0.0, 2)
        first[i] = w1 @ points[idxs]
        second[i] = w2 @ points[idxs]
    return first, second


def savgol_periodic_derivatives(points: np.ndarray, window: int, degree: int) -> tuple[np.ndarray, np.ndarray]:
    n = len(points)
    if window % 2 == 0:
        raise ValueError('savgol window must be odd')
    if degree >= window:
        raise ValueError('savgol degree must be < window')
    s, L = closed_curve_arc_length(points)
    r = window // 2
    first = np.zeros_like(points, dtype=np.float64)
    second = np.zeros_like(points, dtype=np.float64)
    for i in range(n):
        idxs = (np.arange(i - r, i + r + 1) % n).astype(int)
        offs = periodic_signed_offsets(s, i, idxs, L)
        A = np.vstack([offs ** p for p in range(degree + 1)]).T  # [W,deg+1]
        pinv = np.linalg.pinv(A)
        coeff_x = pinv @ points[idxs, 0]
        coeff_y = pinv @ points[idxs, 1]
        first[i, 0] = coeff_x[1] if degree >= 1 else 0.0
        first[i, 1] = coeff_y[1] if degree >= 1 else 0.0
        second[i, 0] = 2.0 * coeff_x[2] if degree >= 2 else 0.0
        second[i, 1] = 2.0 * coeff_y[2] if degree >= 2 else 0.0
    return first, second


def load_curves(bank_path: str, family: str) -> list[CurveRecord]:
    bank = PregeneratedCurveBank(bank_path)
    out: list[CurveRecord] = []
    for idx in range(len(bank)):
        curve_points, coeffs, t_grid = bank.get(idx)
        if coeffs is None or t_grid is None:
            raise RuntimeError('Benchmark requires coeffs and t_grid in bank')
        _, gt1, gt2 = compute_fourier_arc_length_derivatives(np.asarray(t_grid, dtype=np.float64), coeffs, family=family)
        s, L = closed_curve_arc_length(curve_points)
        out.append(CurveRecord(curve_points=np.asarray(curve_points, dtype=np.float64), gt1=gt1, gt2=gt2, length=L, s=s))
    return out


def instantiate_first_model_for_ckpt(ckpt_path: str):
    cfg, _ = load_pretrained_config(ckpt_path)
    model = instantiate_model_from_config(cfg)
    state = torch.load(ckpt_path, map_location='cpu')
    model.load_state_dict(state)
    return model, int(cfg['patch_size'])


def instantiate_model_from_finetune_dir(finetune_ckpt: str, pretrained_best: str):
    cfg, _ = load_pretrained_config(pretrained_best)
    model = instantiate_model_from_config(cfg)
    state = torch.load(finetune_ckpt, map_location='cpu')
    model.load_state_dict(state)
    return model, int(cfg['patch_size'])


def instantiate_direct_second_model(ckpt_path: str):
    ckpt = Path(ckpt_path)
    cfg_path = ckpt.parent / 'config.json'
    if not cfg_path.exists():
        raise FileNotFoundError(f'Missing config.json for direct second model: {cfg_path}')
    cfg = json.loads(cfg_path.read_text())
    from models.tangent_model import TangentOperatorModel
    model = TangentOperatorModel(
        patch_size=int(cfg['patch_size']),
        operator_hidden_dims=parse_int_list_direct(cfg['operator_hidden_dims']),
        signature_hidden_dims=parse_int_list_direct(cfg['signature_hidden_dims']),
        signature_out_dim=int(cfg['signature_out_dim']),
        signature_center_radius=int(cfg['signature_center_radius']),
        head_dropout=float(cfg['head_dropout']),
        normalize_projector=not bool(cfg['disable_normalize_projector']),
        init_scale=float(cfg['operator_init_scale']),
        learn_scale=bool(cfg['learn_output_scale']),
        centered_input_for_operator=not bool(cfg['disable_centered_input_for_operator']),
    )
    state = torch.load(ckpt_path, map_location='cpu')
    model.load_state_dict(state)
    return model, int(cfg['patch_size'])


def eval_first_unprojected(model, patch_size: int, curves: list[CurveRecord], device: str) -> dict[str, np.ndarray]:
    trainer = UnprojectedFirstTrainer(model=model, optimizer=None, scheduler=None, loss_fn=None, device=device, checkpoint_dir='/tmp/bench', patch_size=patch_size)
    model.eval()
    g1_list, g2_list = [], []
    with torch.no_grad():
        for rec in curves:
            cp = torch.as_tensor(rec.curve_points, dtype=torch.float32, device=device)
            out = trainer._full_curve_operator_eval(cp)
            g1_list.append(out['global1'].detach().cpu().numpy())
            g2_list.append(out['global2'].detach().cpu().numpy())
    return {'global1_pred': np.concatenate(g1_list, axis=0), 'global2_pred': np.concatenate(g2_list, axis=0)}


def eval_first_projected(model, patch_size: int, curves: list[CurveRecord], device: str, moment_order: int = 3, projection_ridge: float = 1e-6) -> dict[str, np.ndarray]:
    trainer = ProjectedFirstTrainer(model=model, optimizer=None, scheduler=None, loss_fn=None, device=device, checkpoint_dir='/tmp/bench', patch_size=patch_size, moment_order=moment_order, projection_ridge=projection_ridge)
    model.eval()
    g1_list, g2_list = [], []
    with torch.no_grad():
        for rec in curves:
            cp = torch.as_tensor(rec.curve_points, dtype=torch.float32, device=device)
            out = trainer._full_curve_operator_eval(cp)
            g1_list.append(out['global1'].detach().cpu().numpy())
            g2_list.append(out['global2'].detach().cpu().numpy())
    return {'global1_pred': np.concatenate(g1_list, axis=0), 'global2_pred': np.concatenate(g2_list, axis=0)}


def eval_direct_second(model, patch_size: int, curves: list[CurveRecord], device: str) -> dict[str, np.ndarray]:
    radius = patch_size // 2
    offsets = torch.arange(-radius, radius + 1, dtype=torch.long, device=device)
    model = model.to(device).eval()
    preds = []
    with torch.no_grad():
        for rec in curves:
            cp = torch.as_tensor(rec.curve_points, dtype=torch.float32, device=device)
            n = cp.shape[0]
            idx = torch.arange(n, device=device, dtype=torch.long)
            neigh = (idx[:, None] + offsets[None, :]) % n
            patches = cp[neigh]
            out = model(patches)
            weights = out['weights']
            scale = getattr(model, 'output_scale', None)
            if scale is not None:
                weights = scale * weights
            pred = torch.einsum('nk,nkd->nd', weights, patches)
            preds.append(pred.detach().cpu().numpy())
    return {'direct2_pred': np.concatenate(preds, axis=0)}


def parse_model_spec(text: str) -> tuple[str, ModelType, str, str | None]:
    parts = text.split(':')
    if len(parts) not in (3, 4):
        raise ValueError(f'Invalid model spec: {text}')
    name = parts[0]
    mtype = parts[1]
    ckpt = parts[2]
    pretrained = parts[3] if len(parts) == 4 else None
    if mtype not in {'first_unprojected', 'first_projected', 'direct_second'}:
        raise ValueError(f'Unknown model type: {mtype}')
    return name, mtype, ckpt, pretrained


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    curves = load_curves(args.bank, args.family)
    gt1 = np.concatenate([c.gt1 for c in curves], axis=0)
    gt2 = np.concatenate([c.gt2 for c in curves], axis=0)

    methods: dict[str, dict[str, np.ndarray]] = {}

    # Numerical baselines
    spec_first, spec_second = [], []
    forn_first, forn_second = [], []
    sg_first, sg_second = [], []
    for rec in curves:
        a1, a2 = spectral_derivatives(rec.curve_points, resample_factor=args.spectral_resample_factor)
        b1, b2 = fornberg_periodic_derivatives(rec.curve_points, stencil=args.fornberg_stencil)
        c1, c2 = savgol_periodic_derivatives(rec.curve_points, window=args.savgol_window, degree=args.savgol_degree)
        spec_first.append(a1); spec_second.append(a2)
        forn_first.append(b1); forn_second.append(b2)
        sg_first.append(c1); sg_second.append(c2)

    methods['spectral'] = {'global1_pred': np.concatenate(spec_first, axis=0), 'global2_pred': np.concatenate(spec_second, axis=0)}
    methods['fornberg'] = {'global1_pred': np.concatenate(forn_first, axis=0), 'global2_pred': np.concatenate(forn_second, axis=0)}
    methods['savgol'] = {'global1_pred': np.concatenate(sg_first, axis=0), 'global2_pred': np.concatenate(sg_second, axis=0)}

    # Learned models
    for spec in args.model_spec:
        name, mtype, ckpt, pretrained = parse_model_spec(spec)
        if mtype == 'first_unprojected':
            model, ps = instantiate_model_from_finetune_dir(ckpt, pretrained or ckpt)
            methods[name] = eval_first_unprojected(model, ps, curves, args.device)
        elif mtype == 'first_projected':
            if pretrained is None:
                raise ValueError('first_projected requires pretrained best_model path as 4th field')
            model, ps = instantiate_model_from_finetune_dir(ckpt, pretrained)
            methods[name] = eval_first_projected(model, ps, curves, args.device)
        elif mtype == 'direct_second':
            model, ps = instantiate_direct_second_model(ckpt)
            methods[name] = eval_direct_second(model, ps, curves, args.device)

    metric_rows: list[dict[str, object]] = []
    summary: dict[str, dict[str, float]] = {}
    mc = MetricComputer()

    for name, preds in methods.items():
        metrics: dict[str, float] = {}
        if 'global1_pred' in preds:
            metrics.update(mc.vector_metrics(preds['global1_pred'], gt1, 'global1'))
        if 'global2_pred' in preds:
            metrics.update(mc.vector_metrics(preds['global2_pred'], gt2, 'global2'))
        if 'direct2_pred' in preds:
            metrics.update(mc.vector_metrics(preds['direct2_pred'], gt2, 'direct2'))
        summary[name] = metrics
        row = {'method': name}
        row.update(metrics)
        metric_rows.append(row)

    with open(out_dir / 'benchmark_summary.json', 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)

    # wide CSV
    all_keys = sorted({k for row in metric_rows for k in row.keys() if k != 'method'})
    with open(out_dir / 'benchmark_summary.csv', 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['method'] + all_keys)
        writer.writeheader()
        for row in metric_rows:
            writer.writerow(row)

    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
