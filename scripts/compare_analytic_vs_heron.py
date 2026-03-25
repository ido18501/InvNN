from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from utils.curve_generation import BasisExpansionCurveCoeffs
from utils.derivatives import compute_fourier_euclidean_arc_length_derivatives


def normalize(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.clip(n, eps, None)


def left_normal(t: np.ndarray) -> np.ndarray:
    return np.stack([-t[:, 1], t[:, 0]], axis=-1)


def heron_tangent_and_curvature(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    3-point tangent + Heron signed curvature-vector estimate on a closed curve.
    Returns:
        tangent: (N,2)
        curvature_vec: (N,2)
    """
    p_prev = np.roll(points, 1, axis=0)
    p = points
    p_next = np.roll(points, -1, axis=0)

    chord = p_next - p_prev
    tangent = normalize(chord)

    a = np.linalg.norm(p - p_prev, axis=1)
    b = np.linalg.norm(p_next - p, axis=1)
    c = np.linalg.norm(p_next - p_prev, axis=1)

    cross_z = (
        (p[:, 0] - p_prev[:, 0]) * (p_next[:, 1] - p_prev[:, 1])
        - (p[:, 1] - p_prev[:, 1]) * (p_next[:, 0] - p_prev[:, 0])
    )

    signed_kappa = 2.0 * cross_z / np.clip(a * b * c, 1e-12, None)
    normal = left_normal(tangent)
    curvature_vec = signed_kappa[:, None] * normal
    return tangent, curvature_vec


def angle_deg(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    u_n = normalize(u)
    v_n = normalize(v)
    dots = np.sum(u_n * v_n, axis=1)
    dots = np.clip(dots, -1.0, 1.0)
    return np.degrees(np.arccos(dots))


def make_curve_overlay_plot(
    ax,
    points: np.ndarray,
    gt_tan: np.ndarray,
    heron_tan: np.ndarray,
    gt_curv: np.ndarray,
    heron_curv: np.ndarray,
    stride: int,
    title: str,
) -> None:
    ax.plot(points[:, 0], points[:, 1], linewidth=1.0)
    idx = np.arange(0, len(points), stride)

    ax.quiver(
        points[idx, 0], points[idx, 1],
        gt_tan[idx, 0], gt_tan[idx, 1],
        angles="xy", scale_units="xy", scale=12, width=0.003
    )
    ax.quiver(
        points[idx, 0], points[idx, 1],
        heron_tan[idx, 0], heron_tan[idx, 1],
        angles="xy", scale_units="xy", scale=12, width=0.003, alpha=0.65
    )

    gt_curv_scaled = gt_curv.copy()
    heron_curv_scaled = heron_curv.copy()

    gt_scale = max(np.percentile(np.linalg.norm(gt_curv, axis=1), 95), 1e-6)
    hr_scale = max(np.percentile(np.linalg.norm(heron_curv, axis=1), 95), 1e-6)

    gt_curv_scaled /= gt_scale
    heron_curv_scaled /= hr_scale

    ax.quiver(
        points[idx, 0], points[idx, 1],
        gt_curv_scaled[idx, 0], gt_curv_scaled[idx, 1],
        angles="xy", scale_units="xy", scale=18, width=0.0025
    )
    ax.quiver(
        points[idx, 0], points[idx, 1],
        heron_curv_scaled[idx, 0], heron_curv_scaled[idx, 1],
        angles="xy", scale_units="xy", scale=18, width=0.0025, alpha=0.65
    )

    ax.set_aspect("equal")
    ax.set_title(title)
    ax.axis("off")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--bank", type=str, required=True)
    p.add_argument("--num-curves", type=int, default=4)
    p.add_argument("--curve-offset", type=int, default=0)
    p.add_argument("--quiver-stride", type=int, default=25)
    p.add_argument("--outdir", type=str, default="debug/heron_compare")
    args = p.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    data = np.load(args.bank, allow_pickle=False)
    curve_points = np.asarray(data["curve_points"], dtype=np.float64)

    if "x_coeffs" not in data.files or "y_coeffs" not in data.files:
        raise ValueError("Bank must contain x_coeffs and y_coeffs for analytic derivatives.")
    if "t_grid" in data.files:
        t_grid = np.asarray(data["t_grid"], dtype=np.float64)
    elif "t_values" in data.files:
        t_grid = np.asarray(data["t_values"], dtype=np.float64)
    else:
        t_grid = np.linspace(0.0, 2.0 * np.pi, curve_points.shape[1], endpoint=False, dtype=np.float64)

    x_coeffs = np.asarray(data["x_coeffs"], dtype=np.float64)
    y_coeffs = np.asarray(data["y_coeffs"], dtype=np.float64)

    tan_angle_errors = []
    curv_angle_errors = []
    curv_mag_abs_errors = []
    curv_mag_rel_errors = []

    start = args.curve_offset
    end = min(start + args.num_curves, len(curve_points))

    for curve_idx in range(start, end):
        pts = curve_points[curve_idx]
        coeffs = BasisExpansionCurveCoeffs(
            x_coeffs=x_coeffs[curve_idx],
            y_coeffs=y_coeffs[curve_idx],
        )

        _, gt_tan, gt_curv = compute_fourier_euclidean_arc_length_derivatives(
            t=t_grid,
            coeffs=coeffs,
        )
        heron_tan, heron_curv = heron_tangent_and_curvature(pts)

        tan_err = angle_deg(gt_tan, heron_tan)
        curv_err = angle_deg(gt_curv, heron_curv)

        gt_mag = np.linalg.norm(gt_curv, axis=1)
        hr_mag = np.linalg.norm(heron_curv, axis=1)
        abs_err = np.abs(gt_mag - hr_mag)
        rel_err = abs_err / np.clip(gt_mag, 1e-8, None)

        tan_angle_errors.append(tan_err)
        curv_angle_errors.append(curv_err)
        curv_mag_abs_errors.append(abs_err)
        curv_mag_rel_errors.append(rel_err)

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        make_curve_overlay_plot(
            axes[0], pts, gt_tan, heron_tan, gt_curv, heron_curv,
            stride=args.quiver_stride,
            title=f"curve {curve_idx}: overlays",
        )

        axes[1].plot(gt_mag, label="analytic |curvature|")
        axes[1].plot(hr_mag, label="Heron |curvature|", alpha=0.8)
        axes[1].set_title("curvature magnitude")
        axes[1].legend()

        axes[2].plot(tan_err, label="tangent angle err (deg)")
        axes[2].plot(curv_err, label="curvature angle err (deg)", alpha=0.85)
        axes[2].set_title("direction errors")
        axes[2].legend()

        fig.tight_layout()
        fig.savefig(outdir / f"curve_{curve_idx:03d}.png", dpi=180)
        plt.close(fig)

    tan_angle_errors = np.concatenate(tan_angle_errors)
    curv_angle_errors = np.concatenate(curv_angle_errors)
    curv_mag_abs_errors = np.concatenate(curv_mag_abs_errors)
    curv_mag_rel_errors = np.concatenate(curv_mag_rel_errors)

    print("\n=== Analytic vs Heron summary ===")
    print(f"curves checked: {end - start}")
    print(f"points checked: {len(tan_angle_errors)}")
    print()
    print(f"tangent angle mean (deg):   {tan_angle_errors.mean():.6f}")
    print(f"tangent angle median (deg): {np.median(tan_angle_errors):.6f}")
    print(f"curvature angle mean (deg): {curv_angle_errors.mean():.6f}")
    print(f"curvature angle median(deg):{np.median(curv_angle_errors):.6f}")
    print(f"curvature |.| abs err mean: {curv_mag_abs_errors.mean():.6f}")
    print(f"curvature |.| abs err med:  {np.median(curv_mag_abs_errors):.6f}")
    print(f"curvature |.| rel err mean: {curv_mag_rel_errors.mean():.6f}")
    print(f"curvature |.| rel err med:  {np.median(curv_mag_rel_errors):.6f}")
    print(f"\nSaved figures to: {outdir}")


if __name__ == "__main__":
    main()
