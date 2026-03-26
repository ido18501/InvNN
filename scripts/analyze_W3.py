from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from datasets.tangent_dataset import TangentDataset
from models.tangent_model import TangentOperatorModel
from training.collate import tangent_collate_fn


def parse_int_list(text: str) -> list[int]:
    text = str(text).strip()
    if not text:
        return []
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def summarize_array(x: np.ndarray) -> dict:
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        return {"count": 0}
    return {
        "count": int(x.size),
        "mean": float(np.mean(x)),
        "std": float(np.std(x)),
        "median": float(np.median(x)),
        "min": float(np.min(x)),
        "max": float(np.max(x)),
        "p05": float(np.percentile(x, 5)),
        "p25": float(np.percentile(x, 25)),
        "p75": float(np.percentile(x, 75)),
        "p95": float(np.percentile(x, 95)),
    }


def circular_summary_deg(angles_deg: np.ndarray) -> dict:
    angles_deg = np.asarray(angles_deg, dtype=np.float64)
    if angles_deg.size == 0:
        return {"count": 0}
    rad = np.deg2rad(angles_deg)
    C = np.mean(np.cos(rad))
    S = np.mean(np.sin(rad))
    mean_rad = np.arctan2(S, C)
    R = np.sqrt(C * C + S * S)
    circ_std = np.sqrt(max(0.0, -2.0 * np.log(max(R, 1e-12))))
    return {
        "circular_mean_deg": float(np.rad2deg(mean_rad)),
        "circular_std_deg": float(np.rad2deg(circ_std)),
        "resultant_length": float(R),
    }


def save_hist(values: np.ndarray, title: str, xlabel: str, path: Path, bins: int = 60):
    values = np.asarray(values)
    if values.size == 0:
        return
    plt.figure(figsize=(6, 4))
    plt.hist(values, bins=bins)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("count")
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def move_batch(batch, device):
    batch.anchor = batch.anchor.to(device)
    batch.positive = batch.positive.to(device)
    batch.negatives = batch.negatives.to(device)
    batch.transform_matrix = batch.transform_matrix.to(device)
    batch.gt_first_anchor = batch.gt_first_anchor.to(device)
    batch.gt_second_anchor = batch.gt_second_anchor.to(device)
    batch.has_analytic_derivatives = batch.has_analytic_derivatives.to(device)
    return batch


def apply_W(W: torch.Tensor, X: torch.Tensor) -> torch.Tensor:
    return torch.einsum("bij,bjd->bid", W, X)


def build_dataset_from_config(cfg: dict, split: str) -> TangentDataset:
    seed_offset = {"train": 0, "val": 10000, "test": 20000}[split]
    return TangentDataset(
        length=int(cfg[f"{split}_length"]),
        family=cfg["family"],
        source=cfg[f"{split}_source"],
        bank_path=cfg[f"{split}_bank"],
        num_curve_points=int(cfg["num_curve_points"]),
        fourier_max_freq=int(cfg["fourier_max_freq"]),
        fourier_scale=float(cfg["fourier_scale"]),
        fourier_decay_power=float(cfg["fourier_decay_power"]),
        patch_size=int(cfg["patch_size"]),
        half_width=int(cfg["half_width"]),
        num_negatives=int(cfg["num_negatives"]),
        negative_min_offset=int(cfg["negative_min_offset"]),
        negative_max_offset=int(cfg["negative_max_offset"]),
        negative_other_curve_fraction=float(cfg["negative_other_curve_fraction"]),
        patch_mode=cfg["patch_mode"],
        jitter_fraction=float(cfg["jitter_fraction"]),
        warp_sampling_prob=float(cfg["warp_sampling_prob"]),
        warp_sampling_strength=float(cfg["warp_sampling_strength"]),
        seed=int(cfg["seed"]) + seed_offset,
    )


def build_model_from_config(cfg: dict, device: torch.device) -> TangentOperatorModel:
    model = TangentOperatorModel(
        patch_size=int(cfg["patch_size"]),
        operator_hidden_dims=parse_int_list(cfg["operator_hidden_dims"]),
        signature_hidden_dims=parse_int_list(cfg["signature_hidden_dims"]),
        signature_out_dim=int(cfg["signature_out_dim"]),
        signature_center_radius=int(cfg["signature_center_radius"]),
        head_dropout=float(cfg["head_dropout"]),
        normalize_projector=not bool(cfg["disable_normalize_projector"]),
        center_operator=not bool(cfg["disable_center_operator"]),
        operator_bandwidth=cfg["operator_bandwidth"],
        init_scale=float(cfg["operator_init_scale"]),
        learn_scale=bool(cfg["learn_output_scale"]),
        centered_input_for_operator=not bool(cfg["disable_centered_input_for_operator"]),
    ).to(device)
    return model


def resample_closed_curve_uniform_arc_length(points: np.ndarray, num_points: int) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    ext = np.vstack([points, points[:1]])
    seg = np.linalg.norm(np.diff(ext, axis=0), axis=1)
    total = float(seg.sum())
    if total <= 1e-12:
        raise ValueError("Degenerate curve")
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    targets = np.linspace(0.0, total, num_points, endpoint=False)

    out = np.empty((num_points, 2), dtype=np.float64)
    j = 0
    for i, s in enumerate(targets):
        while j + 1 < len(cum) and cum[j + 1] <= s:
            j += 1
        if j >= len(seg):
            j = len(seg) - 1
        local_len = seg[j]
        if local_len <= 1e-12:
            out[i] = ext[j]
        else:
            alpha = (s - cum[j]) / local_len
            out[i] = (1.0 - alpha) * ext[j] + alpha * ext[j + 1]
    return out


def nearest_index(points: np.ndarray, query: np.ndarray) -> int:
    d2 = ((points - query.reshape(1, 2)) ** 2).sum(axis=1)
    return int(np.argmin(d2))


def numerical_derivatives_at_anchor(curve_points: np.ndarray, anchor_index: int, dense_num_points: int = 4096):
    dense = resample_closed_curve_uniform_arc_length(curve_points, num_points=dense_num_points)

    ext = np.vstack([curve_points, curve_points[:1]])
    total = float(np.linalg.norm(np.diff(ext, axis=0), axis=1).sum())
    ds = total / dense_num_points

    q = np.asarray(curve_points[anchor_index], dtype=np.float64)
    k = nearest_index(dense, q)

    prev_pt = dense[(k - 1) % dense_num_points]
    curr_pt = dense[k]
    next_pt = dense[(k + 1) % dense_num_points]

    first = (next_pt - prev_pt) / (2.0 * ds)
    nf = np.linalg.norm(first)
    if nf > 1e-12:
        first = first / nf
    else:
        first = np.zeros(2, dtype=np.float64)

    second = (next_pt - 2.0 * curr_pt + prev_pt) / (ds ** 2)
    return first.astype(np.float64), second.astype(np.float64)


def signed_angle_deg(pred: np.ndarray, ref: np.ndarray) -> float:
    eps = 1e-12
    nr = np.linalg.norm(ref)
    npred = np.linalg.norm(pred)
    if nr <= eps or npred <= eps:
        return 0.0
    r = ref / nr
    p = pred / npred
    r_perp = np.array([-r[1], r[0]], dtype=np.float64)
    x = np.dot(p, r)
    y = np.dot(p, r_perp)
    return float(np.rad2deg(np.arctan2(y, x)))


def plot_patch_with_field(ax, pts: np.ndarray, field: np.ndarray, title: str):
    ax.plot(pts[:, 0], pts[:, 1], "-o", ms=3)
    ax.quiver(pts[:, 0], pts[:, 1], field[:, 0], field[:, 1], angles="xy", scale_units="xy", scale=1.0)
    ax.set_title(title)
    ax.axis("equal")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-dir", type=str, required=True)
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--out-dir", type=str, required=True)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--fd-dense-points", type=int, default=4096)
    parser.add_argument("--num-examples", type=int, default=6)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_dir = Path(args.checkpoint_dir)
    cfg = json.loads((checkpoint_dir / "config.json").read_text())
    device = torch.device(args.device)

    dataset = build_dataset_from_config(cfg, args.split)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=tangent_collate_fn,
        drop_last=False,
    )

    model = build_model_from_config(cfg, device)
    state_dict = torch.load(checkpoint_dir / "best_model.pt", map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    cos_W3_first = []
    cos_W3_second = []
    ang_W3_first = []
    ang_W3_second = []

    saved_examples = []
    global_dataset_index = 0

    with torch.no_grad():
        for batch in loader:
            batch = move_batch(batch, device)

            out = model(batch.anchor)
            W = out["operator"]
            F1 = out["field_first"]
            F2 = apply_W(W, F1)
            F3 = apply_W(W, F2)

            center = F1.shape[1] // 2
            F3_center = F3[:, center, :].detach().cpu().numpy()
            anchor_pts = batch.anchor.detach().cpu().numpy()
            center_indices = batch.anchor_center_index.detach().cpu().numpy()

            for b in range(F3.shape[0]):
                ds_idx = global_dataset_index + b

                rng = dataset._make_rng(ds_idx)
                curve_points, _, _ = dataset._get_curve(rng, ds_idx)
                curve_points = np.asarray(curve_points, dtype=np.float64)

                gt_first, gt_second = numerical_derivatives_at_anchor(
                    curve_points=curve_points,
                    anchor_index=int(center_indices[b]),
                    dense_num_points=args.fd_dense_points,
                )

                pred = F3_center[b]

                c1 = 0.0 if np.linalg.norm(pred) < 1e-12 else float(np.dot(pred, gt_first) / (np.linalg.norm(pred) * np.linalg.norm(gt_first) + 1e-12))
                c2 = 0.0 if np.linalg.norm(pred) < 1e-12 or np.linalg.norm(gt_second) < 1e-12 else float(np.dot(pred, gt_second) / (np.linalg.norm(pred) * np.linalg.norm(gt_second) + 1e-12))

                cos_W3_first.append(c1)
                cos_W3_second.append(c2)
                ang_W3_first.append(signed_angle_deg(pred, gt_first))
                ang_W3_second.append(signed_angle_deg(pred, gt_second))

                if len(saved_examples) < args.num_examples:
                    saved_examples.append({
                        "dataset_idx": ds_idx,
                        "pts": anchor_pts[b],
                        "F1": F1[b].detach().cpu().numpy(),
                        "F2": F2[b].detach().cpu().numpy(),
                        "F3": F3[b].detach().cpu().numpy(),
                    })

            global_dataset_index += F3.shape[0]

    summary = {
        "W3_vs_first_cos": summarize_array(np.asarray(cos_W3_first)),
        "W3_vs_second_cos": summarize_array(np.asarray(cos_W3_second)),
        "W3_vs_first_angle_deg": {
            **summarize_array(np.asarray(ang_W3_first)),
            **circular_summary_deg(np.asarray(ang_W3_first)),
        },
        "W3_vs_second_angle_deg": {
            **summarize_array(np.asarray(ang_W3_second)),
            **circular_summary_deg(np.asarray(ang_W3_second)),
        },
    }

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    save_hist(np.asarray(cos_W3_first), "W^3 center vs numerical first derivative", "cos(W^3 center, gt_first)", out_dir / "hist_W3_vs_first_cos.png")
    save_hist(np.asarray(cos_W3_second), "W^3 center vs numerical second derivative", "cos(W^3 center, gt_second)", out_dir / "hist_W3_vs_second_cos.png")
    save_hist(np.asarray(ang_W3_first), "W^3 residual angle vs numerical first derivative", "deg", out_dir / "hist_W3_vs_first_angle.png")
    save_hist(np.asarray(ang_W3_second), "W^3 residual angle vs numerical second derivative", "deg", out_dir / "hist_W3_vs_second_angle.png")

    for i, ex in enumerate(saved_examples):
        fig, axes = plt.subplots(1, 4, figsize=(18, 4.5))
        plot_patch_with_field(axes[0], ex["pts"], np.zeros_like(ex["F1"]), f"Patch #{ex['dataset_idx']}")
        plot_patch_with_field(axes[1], ex["pts"], ex["F1"], "W(X)X")
        plot_patch_with_field(axes[2], ex["pts"], ex["F2"], "W²(X)X")
        plot_patch_with_field(axes[3], ex["pts"], ex["F3"], "W³(X)X")
        plt.tight_layout()
        plt.savefig(out_dir / f"example_{i:02d}.png", dpi=180)
        plt.close()

    with open(out_dir / "quick_report.txt", "w") as f:
        f.write(json.dumps(summary, indent=2))

    print(f"Saved W^3 analysis to: {out_dir}")


if __name__ == "__main__":
    main()
