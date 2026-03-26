from __future__ import annotations

import argparse
import heapq
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
from utils.derivatives import compute_single_anchor_fourier_arc_length_derivatives


# ----------------------------
# helpers
# ----------------------------

def parse_int_list(text: str) -> list[int]:
    text = str(text).strip()
    if not text:
        return []
    return [int(x.strip()) for x in text.split(",") if x.strip()]


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
    # W: [B,K,K], X: [B,K,2] -> [B,K,2]
    return torch.einsum("bij,bjd->bid", W, X)


def flatten_cos(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    a_flat = a.reshape(a.shape[0], -1)
    b_flat = b.reshape(b.shape[0], -1)
    return F.cosine_similarity(a_flat, b_flat, dim=-1)


def summarize_array(x: np.ndarray) -> dict:
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        return {
            "count": 0,
            "mean": None,
            "std": None,
            "median": None,
            "min": None,
            "max": None,
            "p05": None,
            "p25": None,
            "p75": None,
            "p95": None,
        }
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
        return {"count": 0, "circular_mean_deg": None, "circular_std_deg": None, "resultant_length": None}
    rad = np.deg2rad(angles_deg)
    C = np.mean(np.cos(rad))
    S = np.mean(np.sin(rad))
    mean_rad = np.arctan2(S, C)
    R = np.sqrt(C * C + S * S)
    # circular std approximation
    circ_std = np.sqrt(max(0.0, -2.0 * np.log(max(R, 1e-12))))
    return {
        "count": int(angles_deg.size),
        "circular_mean_deg": float(np.rad2deg(mean_rad)),
        "circular_std_deg": float(np.rad2deg(circ_std)),
        "resultant_length": float(R),
    }


def frame_residual_angles_deg(pred: torch.Tensor, gt_dir: torch.Tensor) -> np.ndarray:
    """
    pred, gt_dir: [N,2]
    returns signed residual angles in degrees:
      angle between pred direction and gt direction, measured in gt-local frame.
    """
    pred_u = pred / (pred.norm(dim=-1, keepdim=True) + 1e-12)
    gt_u = gt_dir / (gt_dir.norm(dim=-1, keepdim=True) + 1e-12)

    gt_perp = torch.stack([-gt_u[:, 1], gt_u[:, 0]], dim=-1)
    x = (pred_u * gt_u).sum(dim=-1)
    y = (pred_u * gt_perp).sum(dim=-1)
    ang = torch.rad2deg(torch.atan2(y, x))
    return ang.detach().cpu().numpy()


def plot_hist(values: np.ndarray, title: str, xlabel: str, save_path: Path, bins: int = 50):
    plt.figure(figsize=(6, 4))
    plt.hist(values, bins=bins)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("count")
    plt.tight_layout()
    plt.savefig(save_path, dpi=160)
    plt.close()


def plot_patch_with_field(ax, pts: np.ndarray, field: np.ndarray, title: str):
    ax.plot(pts[:, 0], pts[:, 1], "-o", ms=3)
    ax.quiver(pts[:, 0], pts[:, 1], field[:, 0], field[:, 1], angles="xy", scale_units="xy", scale=1.0)
    ax.set_title(title)
    ax.axis("equal")


def save_example_figure(example: dict, out_path: Path, num_neg_vis: int):
    anchor_pts = example["anchor_pts"]
    pos_pts = example["pos_pts"]
    neg_pts = example["neg_pts"][:num_neg_vis]

    anchor_f1 = example["anchor_f1"]
    pos_f1 = example["pos_f1"]
    neg_f1 = example["neg_f1"][:num_neg_vis]

    anchor_f2 = example["anchor_f2"]
    pos_f2 = example["pos_f2"]
    neg_f2 = example["neg_f2"][:num_neg_vis]

    cols = 2 + len(neg_pts)
    fig, axes = plt.subplots(3, cols, figsize=(4 * cols, 11))

    plot_patch_with_field(axes[0, 0], anchor_pts, np.zeros_like(anchor_f1), "Anchor patch")
    plot_patch_with_field(axes[1, 0], anchor_pts, anchor_f1, "Anchor: W(X)X")
    plot_patch_with_field(axes[2, 0], anchor_pts, anchor_f2, "Anchor: W²(X)X")

    plot_patch_with_field(axes[0, 1], pos_pts, np.zeros_like(pos_f1), "Positive patch")
    plot_patch_with_field(axes[1, 1], pos_pts, pos_f1, "Positive: W(TX)TX")
    plot_patch_with_field(axes[2, 1], pos_pts, pos_f2, "Positive: W²(TX)TX")

    for j in range(len(neg_pts)):
        plot_patch_with_field(axes[0, 2 + j], neg_pts[j], np.zeros_like(neg_f1[j]), f"Negative {j}")
        plot_patch_with_field(axes[1, 2 + j], neg_pts[j], neg_f1[j], f"Negative {j}: W(X⁻)X⁻")
        plot_patch_with_field(axes[2, 2 + j], neg_pts[j], neg_f2[j], f"Negative {j}: W²(X⁻)X⁻")

    fig.suptitle(
        f"dataset_idx={example['dataset_idx']} | gt_second_norm={example['gt_second_norm']:.4f}",
        fontsize=14
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


# ----------------------------
# config / model / dataset
# ----------------------------

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


# ----------------------------
# audit for analytic GT consistency
# ----------------------------

def audit_analytic_gt_consistency(dataset: TangentDataset, num_dataset_indices: int = 64) -> dict:
    """
    This audits whether coeffs/t_grid remain geometrically consistent with the actual
    curve_points returned by dataset._get_curve(...).
    This is especially important when warp_sampling_prob > 0 on pregenerated data.
    """
    if dataset.source != "pregenerated" or dataset.bank is None or not dataset.bank.has_coeffs or dataset.bank.t_grid is None:
        return {"available": False, "reason": "pregenerated coeff/t_grid not available"}

    point_errs = []
    first_norms = []
    second_orthogonality = []

    n = min(num_dataset_indices, len(dataset))
    for idx in range(n):
        rng = dataset._make_rng(idx)
        curve_points, coeffs, t_grid = dataset._get_curve(rng, idx)
        if coeffs is None or t_grid is None:
            continue

        curve_points = np.asarray(curve_points, dtype=np.float64)
        t_grid = np.asarray(t_grid, dtype=np.float64)

        # test a few center indices
        test_centers = np.linspace(0, len(curve_points) - 1, num=min(5, len(curve_points)), dtype=int)
        for c in test_centers:
            analytic_pt, analytic_first, analytic_second = compute_single_anchor_fourier_arc_length_derivatives(
                float(t_grid[c]), coeffs
            )

            point_errs.append(float(np.linalg.norm(curve_points[c] - analytic_pt)))
            first_norms.append(float(np.linalg.norm(analytic_first)))
            denom = np.linalg.norm(analytic_first) * np.linalg.norm(analytic_second) + 1e-12
            second_orthogonality.append(float(np.dot(analytic_first, analytic_second) / denom))

    return {
        "available": True,
        "num_checks": len(point_errs),
        "point_match_error": summarize_array(np.asarray(point_errs)),
        "analytic_first_norm": summarize_array(np.asarray(first_norms)),
        "analytic_first_vs_second_cos": summarize_array(np.asarray(second_orthogonality)),
        "warning": (
            "If point_match_error is not ~0, analytic derivatives do not match the actual "
            "sampled curve geometry. This commonly happens if pregenerated curve_points are "
            "warped/noised after loading while coeffs,t_grid remain original."
        ),
    }


# ----------------------------
# main analysis
# ----------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-dir", type=str, required=True)
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--out-dir", type=str, required=True)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num-viz", type=int, default=8)
    parser.add_argument("--num-viz-negatives", type=int, default=2)
    parser.add_argument("--audit-indices", type=int, default=64)
    args = parser.parse_args()

    checkpoint_dir = Path(args.checkpoint_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

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

    # ---- audit analytic derivative consistency ----
    audit = audit_analytic_gt_consistency(dataset, num_dataset_indices=args.audit_indices)
    (out_dir / "analytic_gt_audit.json").write_text(json.dumps(audit, indent=2))

    # ---- accumulators ----
    deriv_first_cos = []
    deriv_second_cos = []
    deriv_first_angle = []
    deriv_second_angle = []
    deriv_first_residual_deg = []
    deriv_second_residual_deg = []
    pred_first_norm = []
    pred_second_norm = []
    gt_first_norm = []
    gt_second_norm = []

    pos_eq1_mse = []
    pos_eq1_mae = []
    pos_eq1_rel = []
    pos_eq1_cos = []

    neg_eq1_mse = []
    neg_eq1_mae = []
    neg_eq1_rel = []
    neg_eq1_cos = []

    pos_eq2_mse = []
    pos_eq2_mae = []
    pos_eq2_rel = []
    pos_eq2_cos = []

    neg_eq2_mse = []
    neg_eq2_mae = []
    neg_eq2_rel = []
    neg_eq2_cos = []

    pos_proj1 = []
    neg_proj1 = []
    pos_proj2 = []
    neg_proj2 = []

    # keep top-curvature examples for viz
    top_examples = []  # min-heap of (score, saved_dict)

    global_sample_index = 0

    with torch.no_grad():
        for batch in loader:
            batch = move_batch(batch, device)

            anchor_out = model(batch.anchor)
            positive_out = model(batch.positive)

            B, M, K, _ = batch.negatives.shape
            flat_neg = batch.negatives.view(B * M, K, 2)
            neg_out = model(flat_neg)

            anchor_f1 = anchor_out["field_first"]                          # [B,K,2]
            pos_f1 = positive_out["field_first"]                           # [B,K,2]
            neg_f1 = neg_out["field_first"].view(B, M, K, 2)              # [B,M,K,2]

            anchor_op = anchor_out["operator"]                             # [B,K,K]
            pos_op = positive_out["operator"]                              # [B,K,K]
            neg_op = neg_out["operator"].view(B, M, K, K)                 # [B,M,K,K]

            anchor_f2 = apply_W(anchor_op, anchor_f1)                      # [B,K,2]
            pos_f2 = apply_W(pos_op, pos_f1)                               # [B,K,2]
            neg_f2 = apply_W(neg_op.view(B * M, K, K), neg_out["field_first"]).view(B, M, K, 2)

            target_f1 = model.apply_linear_map_to_field(batch.transform_matrix, anchor_f1)
            target_f2 = model.apply_linear_map_to_field(batch.transform_matrix, anchor_f2)

            # ---- derivative metrics at center ----
            pred_first = anchor_out["center_first"]
            pred_second = anchor_out["center_second"]
            gt_first = batch.gt_first_anchor
            gt_second = batch.gt_second_anchor

            cos1 = F.cosine_similarity(pred_first, gt_first, dim=-1)
            cos2 = F.cosine_similarity(pred_second, gt_second, dim=-1)
            ang1 = torch.rad2deg(torch.acos(torch.clamp(cos1, -1.0, 1.0)))
            ang2 = torch.rad2deg(torch.acos(torch.clamp(cos2, -1.0, 1.0)))

            deriv_first_cos.extend(cos1.cpu().numpy().tolist())
            deriv_second_cos.extend(cos2.cpu().numpy().tolist())
            deriv_first_angle.extend(ang1.cpu().numpy().tolist())
            deriv_second_angle.extend(ang2.cpu().numpy().tolist())
            deriv_first_residual_deg.extend(frame_residual_angles_deg(pred_first, gt_first).tolist())
            deriv_second_residual_deg.extend(frame_residual_angles_deg(pred_second, gt_second).tolist())
            pred_first_norm.extend(pred_first.norm(dim=-1).cpu().numpy().tolist())
            pred_second_norm.extend(pred_second.norm(dim=-1).cpu().numpy().tolist())
            gt_first_norm.extend(gt_first.norm(dim=-1).cpu().numpy().tolist())
            gt_second_norm.extend(gt_second.norm(dim=-1).cpu().numpy().tolist())

            # ---- W equivariance: positives ----
            diff_pos1 = pos_f1 - target_f1
            pos_eq1_mse.extend(diff_pos1.pow(2).mean(dim=(1, 2)).cpu().numpy().tolist())
            pos_eq1_mae.extend(diff_pos1.abs().mean(dim=(1, 2)).cpu().numpy().tolist())
            pos_eq1_rel.extend(
                (diff_pos1.norm(dim=(1, 2)) / (target_f1.norm(dim=(1, 2)) + 1e-12)).cpu().numpy().tolist()
            )
            pos_eq1_cos.extend(flatten_cos(pos_f1, target_f1).cpu().numpy().tolist())

            # ---- W equivariance: negatives ----
            target_f1_m = target_f1.unsqueeze(1).expand_as(neg_f1)
            diff_neg1 = neg_f1 - target_f1_m
            neg_eq1_mse.extend(diff_neg1.pow(2).mean(dim=(2, 3)).reshape(-1).cpu().numpy().tolist())
            neg_eq1_mae.extend(diff_neg1.abs().mean(dim=(2, 3)).reshape(-1).cpu().numpy().tolist())
            neg_eq1_rel.extend(
                (diff_neg1.norm(dim=(2, 3)) / (target_f1.norm(dim=(1, 2), keepdim=True) + 1e-12)).reshape(-1).cpu().numpy().tolist()
            )
            neg_eq1_cos.extend(
                F.cosine_similarity(neg_f1.reshape(B * M, -1), target_f1_m.reshape(B * M, -1), dim=-1).cpu().numpy().tolist()
            )

            # ---- W² equivariance: positives ----
            diff_pos2 = pos_f2 - target_f2
            pos_eq2_mse.extend(diff_pos2.pow(2).mean(dim=(1, 2)).cpu().numpy().tolist())
            pos_eq2_mae.extend(diff_pos2.abs().mean(dim=(1, 2)).cpu().numpy().tolist())
            pos_eq2_rel.extend(
                (diff_pos2.norm(dim=(1, 2)) / (target_f2.norm(dim=(1, 2)) + 1e-12)).cpu().numpy().tolist()
            )
            pos_eq2_cos.extend(flatten_cos(pos_f2, target_f2).cpu().numpy().tolist())

            # ---- W² equivariance: negatives ----
            target_f2_m = target_f2.unsqueeze(1).expand_as(neg_f2)
            diff_neg2 = neg_f2 - target_f2_m
            neg_eq2_mse.extend(diff_neg2.pow(2).mean(dim=(2, 3)).reshape(-1).cpu().numpy().tolist())
            neg_eq2_mae.extend(diff_neg2.abs().mean(dim=(2, 3)).reshape(-1).cpu().numpy().tolist())
            neg_eq2_rel.extend(
                (diff_neg2.norm(dim=(2, 3)) / (target_f2.norm(dim=(1, 2), keepdim=True) + 1e-12)).reshape(-1).cpu().numpy().tolist()
            )
            neg_eq2_cos.extend(
                F.cosine_similarity(neg_f2.reshape(B * M, -1), target_f2_m.reshape(B * M, -1), dim=-1).cpu().numpy().tolist()
            )

            # ---- projection separation for W ----
            z_anchor1 = model.project_field(target_f1)
            z_pos1 = model.project_field(pos_f1)
            z_neg1 = model.project_field(neg_f1.view(B * M, K, 2)).view(B, M, -1)
            pos_proj1.extend((z_anchor1 * z_pos1).sum(dim=-1).cpu().numpy().tolist())
            neg_proj1.extend(torch.einsum("bd,bmd->bm", z_anchor1, z_neg1).reshape(-1).cpu().numpy().tolist())

            # ---- projection separation for W² ----
            z_anchor2 = model.project_field(target_f2)
            z_pos2 = model.project_field(pos_f2)
            z_neg2 = model.project_field(neg_f2.view(B * M, K, 2)).view(B, M, -1)
            pos_proj2.extend((z_anchor2 * z_pos2).sum(dim=-1).cpu().numpy().tolist())
            neg_proj2.extend(torch.einsum("bd,bmd->bm", z_anchor2, z_neg2).reshape(-1).cpu().numpy().tolist())

            # ---- keep top-curvature examples for visualization ----
            gt_second_norm_batch = gt_second.norm(dim=-1).cpu().numpy()
            for b in range(B):
                sample = {
                    "dataset_idx": global_sample_index + b,
                    "gt_second_norm": float(gt_second_norm_batch[b]),
                    "anchor_pts": batch.anchor[b].cpu().numpy(),
                    "pos_pts": batch.positive[b].cpu().numpy(),
                    "neg_pts": batch.negatives[b, :args.num_viz_negatives].cpu().numpy(),
                    "anchor_f1": anchor_f1[b].cpu().numpy(),
                    "pos_f1": pos_f1[b].cpu().numpy(),
                    "neg_f1": neg_f1[b, :args.num_viz_negatives].cpu().numpy(),
                    "anchor_f2": anchor_f2[b].cpu().numpy(),
                    "pos_f2": pos_f2[b].cpu().numpy(),
                    "neg_f2": neg_f2[b, :args.num_viz_negatives].cpu().numpy(),
                }
                score = sample["gt_second_norm"]
                if len(top_examples) < args.num_viz:
                    heapq.heappush(top_examples, (score, sample))
                else:
                    if score > top_examples[0][0]:
                        heapq.heapreplace(top_examples, (score, sample))

            global_sample_index += B

    # ---- write stats ----
    summary = {
        "checkpoint_dir": str(checkpoint_dir),
        "split": args.split,
        "config_used": cfg,
        "analytic_gt_audit": audit,

        "derivatives": {
            "first_cos": summarize_array(np.asarray(deriv_first_cos)),
            "second_cos": summarize_array(np.asarray(deriv_second_cos)),
            "first_angle_deg": summarize_array(np.asarray(deriv_first_angle)),
            "second_angle_deg": summarize_array(np.asarray(deriv_second_angle)),
            "first_residual_angle_deg": {
                **summarize_array(np.asarray(deriv_first_residual_deg)),
                **circular_summary_deg(np.asarray(deriv_first_residual_deg)),
            },
            "second_residual_angle_deg": {
                **summarize_array(np.asarray(deriv_second_residual_deg)),
                **circular_summary_deg(np.asarray(deriv_second_residual_deg)),
            },
            "pred_first_norm": summarize_array(np.asarray(pred_first_norm)),
            "pred_second_norm": summarize_array(np.asarray(pred_second_norm)),
            "gt_first_norm": summarize_array(np.asarray(gt_first_norm)),
            "gt_second_norm": summarize_array(np.asarray(gt_second_norm)),
        },

        "equivariance_W": {
            "positive_mse": summarize_array(np.asarray(pos_eq1_mse)),
            "positive_mae": summarize_array(np.asarray(pos_eq1_mae)),
            "positive_relative_l2": summarize_array(np.asarray(pos_eq1_rel)),
            "positive_cos": summarize_array(np.asarray(pos_eq1_cos)),
            "negative_mse": summarize_array(np.asarray(neg_eq1_mse)),
            "negative_mae": summarize_array(np.asarray(neg_eq1_mae)),
            "negative_relative_l2": summarize_array(np.asarray(neg_eq1_rel)),
            "negative_cos": summarize_array(np.asarray(neg_eq1_cos)),
        },

        "equivariance_W2": {
            "positive_mse": summarize_array(np.asarray(pos_eq2_mse)),
            "positive_mae": summarize_array(np.asarray(pos_eq2_mae)),
            "positive_relative_l2": summarize_array(np.asarray(pos_eq2_rel)),
            "positive_cos": summarize_array(np.asarray(pos_eq2_cos)),
            "negative_mse": summarize_array(np.asarray(neg_eq2_mse)),
            "negative_mae": summarize_array(np.asarray(neg_eq2_mae)),
            "negative_relative_l2": summarize_array(np.asarray(neg_eq2_rel)),
            "negative_cos": summarize_array(np.asarray(neg_eq2_cos)),
        },

        "projection_similarity": {
            "W_positive": summarize_array(np.asarray(pos_proj1)),
            "W_negative": summarize_array(np.asarray(neg_proj1)),
            "W2_positive": summarize_array(np.asarray(pos_proj2)),
            "W2_negative": summarize_array(np.asarray(neg_proj2)),
        },
    }

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    # ---- histograms ----
    plot_hist(np.asarray(deriv_first_cos), "First derivative cosine", "cos(pred_first, gt_first)", out_dir / "hist_first_cos.png")
    plot_hist(np.asarray(deriv_second_cos), "Second derivative cosine", "cos(pred_second, gt_second)", out_dir / "hist_second_cos.png")
    plot_hist(np.asarray(deriv_first_residual_deg), "First residual angle", "deg", out_dir / "hist_first_residual_angle.png")
    plot_hist(np.asarray(deriv_second_residual_deg), "Second residual angle", "deg", out_dir / "hist_second_residual_angle.png")

    plot_hist(np.asarray(pos_eq1_cos), "W equivariance: positive cosine", "cos(target_W, positive_W)", out_dir / "hist_W_pos_cos.png")
    plot_hist(np.asarray(neg_eq1_cos), "W equivariance: negative cosine", "cos(target_W, negative_W)", out_dir / "hist_W_neg_cos.png")
    plot_hist(np.asarray(pos_eq2_cos), "W² equivariance: positive cosine", "cos(target_W2, positive_W2)", out_dir / "hist_W2_pos_cos.png")
    plot_hist(np.asarray(neg_eq2_cos), "W² equivariance: negative cosine", "cos(target_W2, negative_W2)", out_dir / "hist_W2_neg_cos.png")

    plot_hist(np.asarray(pos_proj1), "Projection similarity on W: positives", "dot(z_anchor_W, z_pos_W)", out_dir / "hist_proj_W_pos.png")
    plot_hist(np.asarray(neg_proj1), "Projection similarity on W: negatives", "dot(z_anchor_W, z_neg_W)", out_dir / "hist_proj_W_neg.png")
    plot_hist(np.asarray(pos_proj2), "Projection similarity on W²: positives", "dot(z_anchor_W2, z_pos_W2)", out_dir / "hist_proj_W2_pos.png")
    plot_hist(np.asarray(neg_proj2), "Projection similarity on W²: negatives", "dot(z_anchor_W2, z_neg_W2)", out_dir / "hist_proj_W2_neg.png")

    # ---- visualizations ----
    top_examples_sorted = sorted(top_examples, key=lambda x: x[0], reverse=True)
    for rank, (_, ex) in enumerate(top_examples_sorted):
        save_example_figure(ex, out_dir / f"example_{rank:02d}.png", args.num_viz_negatives)

    # ---- quick readable text summary ----
    with open(out_dir / "quick_report.txt", "w") as f:
        f.write("=== IMPORTANT ===\n")
        f.write("If analytic_gt_audit.point_match_error is not ~0, derivative emergence metrics are not trustworthy.\n\n")
        f.write("=== derivative summary ===\n")
        f.write(json.dumps(summary["derivatives"], indent=2))
        f.write("\n\n=== W equivariance ===\n")
        f.write(json.dumps(summary["equivariance_W"], indent=2))
        f.write("\n\n=== W^2 equivariance ===\n")
        f.write(json.dumps(summary["equivariance_W2"], indent=2))
        f.write("\n\n=== projection similarity ===\n")
        f.write(json.dumps(summary["projection_similarity"], indent=2))

    print(f"Saved analysis to: {out_dir}")


if __name__ == "__main__":
    main()
