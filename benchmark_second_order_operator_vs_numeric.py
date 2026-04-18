from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from datasets.tangent_dataset import PregeneratedCurveBank
from models.tangent_model import TangentOperatorModel
from utils.derivatives import compute_single_anchor_fourier_arc_length_derivatives


Array = np.ndarray
Tensor = torch.Tensor


def parse_int_list(x: str) -> list[int]:
    x = x.strip()
    if not x:
        return []
    return [int(v.strip()) for v in x.split(",") if v.strip()]


def load_model(config_path: str, checkpoint_path: str, device: str) -> TangentOperatorModel:
    cfg = json.loads(Path(config_path).read_text())
    model = TangentOperatorModel(
        patch_size=cfg["patch_size"],
        operator_hidden_dims=parse_int_list(cfg.get("operator_hidden_dims", "256,256")),
        signature_hidden_dims=parse_int_list(cfg.get("signature_hidden_dims", "128,64")),
        signature_out_dim=int(cfg.get("signature_out_dim", 64)),
        signature_center_radius=int(cfg.get("signature_center_radius", 0)),
        head_dropout=float(cfg.get("head_dropout", 0.0)),
        normalize_projector=not bool(cfg.get("disable_normalize_projector", False)),
        init_scale=float(cfg.get("operator_init_scale", 0.05)),
        learn_scale=bool(cfg.get("learn_output_scale", False)),
        centered_input_for_operator=not bool(cfg.get("disable_centered_input_for_operator", False)),
    )
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def cyclic_patch(points: Array, center: int, patch_size: int) -> tuple[Array, Array]:
    r = patch_size // 2
    offsets = np.arange(-r, r + 1, dtype=np.int64)
    idx = (center + offsets) % len(points)
    patch = points[idx]
    return patch, idx


def signed_arc_offsets(points: Array, idx: Array, center_pos: int) -> Array:
    patch = points[idx]
    K = len(idx)
    out = np.zeros(K, dtype=np.float64)
    c = center_pos
    for j in range(c + 1, K):
        out[j] = out[j - 1] + np.linalg.norm(patch[j] - patch[j - 1])
    for j in range(c - 1, -1, -1):
        out[j] = out[j + 1] - np.linalg.norm(patch[j + 1] - patch[j])
    return out


def project_first_derivative_row(raw_w: Array, ds: Array, ridge: float = 1e-6) -> Array:
    # min ||w - raw||^2 + ridge ||w||^2 s.t. sum w = 0, sum w ds = 1
    raw_w = np.asarray(raw_w, dtype=np.float64)
    ds = np.asarray(ds, dtype=np.float64)
    K = raw_w.shape[0]
    Hinv = np.eye(K, dtype=np.float64) / (1.0 + ridge)
    A = np.stack([np.ones(K, dtype=np.float64), ds], axis=0)  # [2,K]
    b = np.array([0.0, 1.0], dtype=np.float64)
    AHAt = A @ Hinv @ A.T
    rhs = A @ raw_w - b
    lam = np.linalg.solve(AHAt, rhs)
    w = raw_w - Hinv @ A.T @ lam
    return w


@torch.no_grad()
def model_global_operator_outputs(
    model: TangentOperatorModel,
    curve: Array,
    patch_size: int,
    device: str,
    ridge: float,
    sign: float,
) -> tuple[Array, Array, Array]:
    N = len(curve)
    patches = []
    ds_rows = []
    for i in range(N):
        patch, idx = cyclic_patch(curve, i, patch_size)
        patch_centered = patch - patch[patch_size // 2]
        patches.append(patch_centered)
        ds_rows.append(signed_arc_offsets(curve, idx, patch_size // 2))
    patches_t = torch.as_tensor(np.stack(patches, axis=0), dtype=torch.float32, device=device)
    raw = model(patches_t)
    raw_w = sign * raw["weights"].detach().cpu().numpy()

    proj_w = np.empty_like(raw_w, dtype=np.float64)
    for i in range(N):
        proj_w[i] = project_first_derivative_row(raw_w[i], ds_rows[i], ridge=ridge)

    global1 = np.zeros_like(curve, dtype=np.float64)
    for i in range(N):
        patch, _ = cyclic_patch(curve, i, patch_size)
        global1[i] = proj_w[i] @ patch

    global2 = np.zeros_like(curve, dtype=np.float64)
    for i in range(N):
        patch1, idx = cyclic_patch(global1, i, patch_size)
        global2[i] = proj_w[i] @ patch1

    return global1, global2, proj_w


def local_poly_second(points: Array, window: int) -> Array:
    assert window % 2 == 1 and window >= 5
    N = len(points)
    r = window // 2
    out = np.zeros_like(points, dtype=np.float64)
    for i in range(N):
        idx = (np.arange(i - r, i + r + 1) % N).astype(np.int64)
        patch = points[idx]
        s = np.zeros(window, dtype=np.float64)
        for j in range(r + 1, window):
            s[j] = s[j - 1] + np.linalg.norm(patch[j] - patch[j - 1])
        for j in range(r - 1, -1, -1):
            s[j] = s[j + 1] - np.linalg.norm(patch[j + 1] - patch[j])
        V = np.stack([np.ones_like(s), s, s ** 2], axis=1)
        cx, *_ = np.linalg.lstsq(V, patch[:, 0], rcond=None)
        cy, *_ = np.linalg.lstsq(V, patch[:, 1], rcond=None)
        out[i, 0] = 2.0 * cx[2]
        out[i, 1] = 2.0 * cy[2]
    return out


def resample_uniform_arc(points: Array, num_points: int) -> tuple[Array, Array]:
    pts = np.asarray(points, dtype=np.float64)
    ext = np.vstack([pts, pts[:1]])
    seg = np.linalg.norm(np.diff(ext, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    total = cum[-1]
    targets = np.linspace(0.0, total, num_points, endpoint=False)
    out = np.empty((num_points, 2), dtype=np.float64)
    src_idx = np.empty(num_points, dtype=np.float64)
    j = 0
    for k, t in enumerate(targets):
        while j + 1 < len(cum) and cum[j + 1] <= t:
            j += 1
        j = min(j, len(seg) - 1)
        L = seg[j]
        a = 0.0 if L <= 1e-12 else (t - cum[j]) / L
        out[k] = (1 - a) * ext[j] + a * ext[j + 1]
        src_idx[k] = j + a
    return out, src_idx


def spectral_second(points: Array, resample_points: int | None = None) -> Array:
    N = len(points)
    M = N if resample_points is None else int(resample_points)
    uni, src = resample_uniform_arc(points, M)
    x = uni[:, 0]
    y = uni[:, 1]
    k = np.fft.fftfreq(M, d=1.0 / M) * 2.0 * np.pi
    d2mult = -(k ** 2)
    x2 = np.fft.ifft(d2mult * np.fft.fft(x)).real
    y2 = np.fft.ifft(d2mult * np.fft.fft(y)).real
    # convert from normalized parameter to arc-length on resampled unit-speed grid
    ext = np.vstack([uni, uni[:1]])
    ds = np.linalg.norm(np.diff(ext, axis=0), axis=1).mean()
    sec = np.stack([x2, y2], axis=1) / (((2.0 * np.pi / M) / ds) ** 2)
    # map back by nearest resampled sample
    nearest = np.rint(np.linspace(0, M - 1, N, endpoint=False)).astype(np.int64)
    return sec[nearest]


def gt_arc_derivatives(curve: Array, coeffs: Any, t_grid: Any) -> tuple[Array, Array]:
    if coeffs is None or t_grid is None:
        # fallback: local poly for first, second. Only for benchmarking when analytic missing.
        second = local_poly_second(curve, window= nine_or_less(len(curve)))
        first = np.zeros_like(curve)
        for i in range(len(curve)):
            idx = (np.arange(i - 1, i + 2) % len(curve)).astype(np.int64)
            patch = curve[idx]
            ds_l = np.linalg.norm(patch[1] - patch[0])
            ds_r = np.linalg.norm(patch[2] - patch[1])
            first[i] = (patch[2] - patch[0]) / max(ds_l + ds_r, 1e-8)
        return first, second
    first = np.zeros_like(curve, dtype=np.float64)
    second = np.zeros_like(curve, dtype=np.float64)
    for i in range(len(curve)):
        _, f, s = compute_single_anchor_fourier_arc_length_derivatives(float(t_grid[i]), coeffs, family="euclidean")
        first[i] = f
        second[i] = s
    return first, second


def nine_or_less(n: int) -> int:
    for w in [9, 7, 5]:
        if w < n:
            return w
    return 5


def cosine_mean(pred: Array, gt: Array, mask: Array | None = None) -> float:
    pn = pred / np.clip(np.linalg.norm(pred, axis=-1, keepdims=True), 1e-8, None)
    gn = gt / np.clip(np.linalg.norm(gt, axis=-1, keepdims=True), 1e-8, None)
    c = np.sum(pn * gn, axis=-1)
    if mask is not None:
        c = c[mask]
    return float(np.mean(c))


def pearson(x: Array, y: Array) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.size < 2:
        return float("nan")
    x = x - x.mean()
    y = y - y.mean()
    denom = np.sqrt((x * x).sum() * (y * y).sum())
    return float((x * y).sum() / denom) if denom > 1e-12 else float("nan")


def rankdata(a: Array) -> Array:
    order = np.argsort(a)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(len(a), dtype=np.float64)
    return ranks


def spearman(x: Array, y: Array) -> float:
    return pearson(rankdata(x), rankdata(y))


def fit_slope_intercept(x: Array, y: Array) -> tuple[float, float]:
    if x.size < 2:
        return float("nan"), float("nan")
    A = np.stack([x, np.ones_like(x)], axis=1)
    sol, *_ = np.linalg.lstsq(A, y, rcond=None)
    return float(sol[0]), float(sol[1])


def method_metrics(pred_second: Array, gt_second: Array, valid: Array) -> dict[str, float]:
    pred_norm = np.linalg.norm(pred_second, axis=-1)
    gt_norm = np.linalg.norm(gt_second, axis=-1)
    slope, intercept = fit_slope_intercept(gt_norm[valid], pred_norm[valid])
    return {
        "cos2_full": cosine_mean(pred_second, gt_second),
        "mse2_full": float(np.mean((pred_second - gt_second) ** 2)),
        "mse2_valid": float(np.mean(np.sum((pred_second[valid] - gt_second[valid]) ** 2, axis=-1))),
        "pear2_valid": pearson(gt_norm[valid], pred_norm[valid]),
        "spear2_valid": spearman(gt_norm[valid], pred_norm[valid]),
        "logcorr2_valid": pearson(np.log1p(gt_norm[valid]), np.log1p(pred_norm[valid])),
        "slope2_valid": slope,
        "intercept2_valid": intercept,
        "pred_norm_mean": float(pred_norm.mean()),
        "gt_norm_mean": float(gt_norm.mean()),
        "valid_fraction": float(valid.mean()),
    }


def aggregate_dicts(dicts: list[dict[str, float]]) -> dict[str, float]:
    keys = dicts[0].keys()
    return {k: float(np.nanmean([d[k] for d in dicts])) for k in keys}


def choose_sign(model: TangentOperatorModel, bank: PregeneratedCurveBank, patch_size: int, device: str, ridge: float) -> float:
    curve, coeffs, t_grid = bank.get(0)
    gt_first, _ = gt_arc_derivatives(curve.astype(np.float64), coeffs, t_grid)
    global1, _, _ = model_global_operator_outputs(model, curve.astype(np.float64), patch_size, device, ridge, sign=1.0)
    c = cosine_mean(global1, gt_first)
    return -1.0 if c < -0.25 else 1.0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--bank", type=str, required=True)
    p.add_argument("--config", type=str, required=True)
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--ridge", type=float, default=1e-6)
    p.add_argument("--sign-fix", action="store_true")
    p.add_argument("--valid-quantile-cap", type=float, default=0.90)
    p.add_argument("--poly-window", type=int, default=9)
    p.add_argument("--spectral-resample-points", type=int, default=2048)
    p.add_argument("--limit-curves", type=int, default=0)
    p.add_argument("--output", type=str, default="")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    bank = PregeneratedCurveBank(args.bank)
    cfg = json.loads(Path(args.config).read_text())
    patch_size = int(cfg["patch_size"])
    model = load_model(args.config, args.checkpoint, args.device)
    sign = choose_sign(model, bank, patch_size, args.device, args.ridge) if args.sign_fix else 1.0

    num = len(bank) if args.limit_curves <= 0 else min(len(bank), args.limit_curves)
    op_stats: list[dict[str, float]] = []
    poly_stats: list[dict[str, float]] = []
    spec_stats: list[dict[str, float]] = []

    for i in range(num):
        curve, coeffs, t_grid = bank.get(i)
        curve = curve.astype(np.float64)
        gt_first, gt_second = gt_arc_derivatives(curve, coeffs, t_grid)

        gt_norm = np.linalg.norm(gt_second, axis=-1)
        q = np.quantile(gt_norm, args.valid_quantile_cap)
        valid = gt_norm <= q

        _, op_second, _ = model_global_operator_outputs(model, curve, patch_size, args.device, args.ridge, sign)
        poly_second = local_poly_second(curve, args.poly_window)
        spec_second = spectral_second(curve, args.spectral_resample_points)

        op_stats.append(method_metrics(op_second, gt_second, valid))
        poly_stats.append(method_metrics(poly_second, gt_second, valid))
        spec_stats.append(method_metrics(spec_second, gt_second, valid))

    out = {
        "operator_projected": aggregate_dicts(op_stats),
        "local_poly": aggregate_dicts(poly_stats),
        "spectral": aggregate_dicts(spec_stats),
        "num_curves": num,
        "valid_quantile_cap": args.valid_quantile_cap,
        "poly_window": args.poly_window,
        "spectral_resample_points": args.spectral_resample_points,
        "sign_used": sign,
    }

    text = json.dumps(out, indent=2)
    print(text)
    if args.output:
        Path(args.output).write_text(text)


if __name__ == "__main__":
    main()
