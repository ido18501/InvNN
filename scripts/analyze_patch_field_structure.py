from __future__ import annotations

import argparse
import json
from pathlib import Path
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from datasets.tangent_dataset import TangentDataset
from models.tangent_model import TangentOperatorModel
from training.collate import tangent_collate_fn


# ============================================================
# helpers
# ============================================================

def parse_int_list(text: str) -> list[int]:
    text = str(text).strip()
    if not text:
        return []
    return [int(x.strip()) for x in text.split(",") if x.strip()]


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
    circ_std = np.sqrt(max(0.0, -2.0 * np.log(max(R, 1e-12))))
    return {
        "count": int(angles_deg.size),
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


def save_line_plot(x, ys: dict[str, np.ndarray], title: str, xlabel: str, ylabel: str, path: Path):
    plt.figure(figsize=(7, 4.5))
    for name, y in ys.items():
        plt.plot(x, y, marker="o", label=name)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
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


def cosine_np(a: np.ndarray, b: np.ndarray, eps: float = 1e-12) -> float:
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na <= eps or nb <= eps:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def signed_angle_deg(pred: np.ndarray, ref: np.ndarray) -> float:
    """
    signed angle from ref to pred in degrees, in the local 2D frame.
    """
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


# ============================================================
# exact reconstruction from config
# ============================================================

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


# ============================================================
# numerical geometry on ACTUAL sampled curve
# ============================================================

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


def numerical_geometry_at_dense_points(curve_points: np.ndarray, dense_num_points: int = 4096):
    """
    Returns dense resampled curve and numerical Frenet quantities at all dense points.
    """
    dense = resample_closed_curve_uniform_arc_length(curve_points, num_points=dense_num_points)
    ext = np.vstack([curve_points, curve_points[:1]])
    total = float(np.linalg.norm(np.diff(ext, axis=0), axis=1).sum())
    ds = total / dense_num_points

    T = np.zeros_like(dense)
    K = np.zeros_like(dense)
    N = np.zeros_like(dense)
    kappa = np.zeros((dense_num_points,), dtype=np.float64)

    for i in range(dense_num_points):
        prev_pt = dense[(i - 1) % dense_num_points]
        curr_pt = dense[i]
        next_pt = dense[(i + 1) % dense_num_points]

        first = (next_pt - prev_pt) / (2.0 * ds)
        nf = np.linalg.norm(first)
        if nf > 1e-12:
            t = first / nf
        else:
            t = np.zeros(2, dtype=np.float64)

        second = (next_pt - 2.0 * curr_pt + prev_pt) / (ds ** 2)
        ns = np.linalg.norm(second)
        if ns > 1e-12:
            n = second / ns
        else:
            n = np.zeros(2, dtype=np.float64)

        T[i] = t
        K[i] = second
        N[i] = n
        kappa[i] = ns

    return dense, T, N, K, kappa


# ============================================================
# patch grouping
# ============================================================

def classify_patch_points(kappa_patch: np.ndarray, tip_threshold: float, smooth_threshold: float):
    """
    Patch-local grouping:
    - tip point = support point with max curvature if max >= tip_threshold
    - near_tip = neighbors of tip point (distance 1 in patch order)
    - smooth = all others
    Patch category:
    - tip_patch / curved_patch / smooth_patch
    """
    K = len(kappa_patch)
    max_idx = int(np.argmax(kappa_patch))
    kmax = float(kappa_patch[max_idx])

    point_group = np.array(["smooth"] * K, dtype=object)
    if kmax >= tip_threshold:
        point_group[max_idx] = "tip"
        if max_idx - 1 >= 0:
            point_group[max_idx - 1] = "near_tip"
        if max_idx + 1 < K:
            point_group[max_idx + 1] = "near_tip"
        patch_group = "tip_patch"
    elif kmax >= smooth_threshold:
        patch_group = "curved_patch"
    else:
        patch_group = "smooth_patch"

    return point_group, patch_group, max_idx, kmax


# ============================================================
# main
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-dir", type=str, required=True)
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--out-dir", type=str, required=True)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--fd-dense-points", type=int, default=4096)
    parser.add_argument("--tip-threshold", type=float, default=50.0)
    parser.add_argument("--smooth-threshold", type=float, default=5.0)
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

    patch_size = int(cfg["patch_size"])

    # global accumulators
    pointwise = defaultdict(list)
    by_offset = defaultdict(lambda: defaultdict(list))
    by_point_group = defaultdict(lambda: defaultdict(list))
    by_patch_group = defaultdict(lambda: defaultdict(list))
    patch_stats = defaultdict(list)

    global_dataset_index = 0

    with torch.no_grad():
        for batch in loader:
            batch = move_batch(batch, device)

            anchor_out = model(batch.anchor)
            B, K, _ = batch.anchor.shape
            assert K == patch_size

            field1 = anchor_out["field_first"].detach().cpu().numpy()          # [B,K,2]
            field2 = anchor_out["center_second"].detach().cpu().numpy()        # [B,2]
            full_field2 = apply_W(anchor_out["operator"], anchor_out["field_first"]).detach().cpu().numpy()  # [B,K,2]

            sample_indices = batch.sample_indices.detach().cpu().numpy() if hasattr(batch, "sample_indices") else None
            rel_offsets = batch.relative_offsets.detach().cpu().numpy() if hasattr(batch, "relative_offsets") else None
            center_indices = batch.anchor_center_index.detach().cpu().numpy()

            # If collate currently doesn't carry sample_indices/relative_offsets,
            # reconstruct them from dataset __getitem__ deterministically.
            if sample_indices is None or rel_offsets is None:
                sample_indices = np.zeros((B, K), dtype=np.int64)
                rel_offsets = np.zeros((B, K), dtype=np.int64)
                for b in range(B):
                    ds_idx = global_dataset_index + b
                    item = dataset[ds_idx]
                    sample_indices[b] = item.sample_indices
                    rel_offsets[b] = item.relative_offsets

            for b in range(B):
                ds_idx = global_dataset_index + b

                # reconstruct ACTUAL curve used for this dataset item
                rng = dataset._make_rng(ds_idx)
                curve_points, _, _ = dataset._get_curve(rng, ds_idx)
                curve_points = np.asarray(curve_points, dtype=np.float64)

                dense, T_dense, N_dense, K_dense, kappa_dense = numerical_geometry_at_dense_points(
                    curve_points, dense_num_points=args.fd_dense_points
                )

                inds = sample_indices[b]
                offs = rel_offsets[b]

                # numerical geometry on support points
                T_patch = []
                N_patch = []
                K_patch = []
                kappa_patch = []
                for idx_curve in inds:
                    q = curve_points[int(idx_curve)]
                    j = nearest_index(dense, q)
                    T_patch.append(T_dense[j])
                    N_patch.append(N_dense[j])
                    K_patch.append(K_dense[j])
                    kappa_patch.append(kappa_dense[j])

                T_patch = np.asarray(T_patch)
                N_patch = np.asarray(N_patch)
                K_patch = np.asarray(K_patch)
                kappa_patch = np.asarray(kappa_patch)

                F = field1[b]          # [K,2]
                G = full_field2[b]     # [K,2]

                point_group, patch_group, tip_idx, kmax = classify_patch_points(
                    kappa_patch=kappa_patch,
                    tip_threshold=args.tip_threshold,
                    smooth_threshold=args.smooth_threshold,
                )

                # per-point comparisons
                for i in range(K):
                    off = int(offs[i])
                    pg = str(point_group[i])

                    Fi = F[i]
                    Gi = G[i]
                    Ti = T_patch[i]
                    Ni = N_patch[i]
                    Ki = K_patch[i]
                    kappai = float(kappa_patch[i])

                    vals = {
                        "kappa": kappai,

                        "F_vs_T_cos": cosine_np(Fi, Ti),
                        "F_vs_N_cos": cosine_np(Fi, Ni),
                        "F_vs_K_cos": cosine_np(Fi, Ki),
                        "F_vs_T_angle_deg": signed_angle_deg(Fi, Ti),
                        "F_vs_N_angle_deg": signed_angle_deg(Fi, Ni),
                        "F_norm": float(np.linalg.norm(Fi)),

                        "G_vs_T_cos": cosine_np(Gi, Ti),
                        "G_vs_N_cos": cosine_np(Gi, Ni),
                        "G_vs_K_cos": cosine_np(Gi, Ki),
                        "G_vs_T_angle_deg": signed_angle_deg(Gi, Ti),
                        "G_vs_N_angle_deg": signed_angle_deg(Gi, Ni),
                        "G_norm": float(np.linalg.norm(Gi)),

                        "F_vs_G_cos": cosine_np(Fi, Gi),
                    }

                    for name, v in vals.items():
                        pointwise[name].append(v)
                        by_offset[name][off].append(v)
                        by_point_group[name][pg].append(v)
                        by_patch_group[name][patch_group].append(v)

                # patch-level internal structure
                # pairwise cosines within patch for F and G
                pair_cos_F = []
                pair_cos_G = []
                for i in range(K):
                    for j in range(i + 1, K):
                        pair_cos_F.append(cosine_np(F[i], F[j]))
                        pair_cos_G.append(cosine_np(G[i], G[j]))

                # pointwise F/G similarity within patch
                corr_FG = [cosine_np(F[i], G[i]) for i in range(K)]

                patch_stats["patch_group"].append(patch_group)
                patch_stats["kappa_max"].append(kmax)
                patch_stats["pair_cos_F_mean"].append(float(np.mean(pair_cos_F)) if pair_cos_F else 0.0)
                patch_stats["pair_cos_G_mean"].append(float(np.mean(pair_cos_G)) if pair_cos_G else 0.0)
                patch_stats["FG_same_index_cos_mean"].append(float(np.mean(corr_FG)) if corr_FG else 0.0)

                by_patch_group["pair_cos_F_mean"][patch_group].append(patch_stats["pair_cos_F_mean"][-1])
                by_patch_group["pair_cos_G_mean"][patch_group].append(patch_stats["pair_cos_G_mean"][-1])
                by_patch_group["FG_same_index_cos_mean"][patch_group].append(patch_stats["FG_same_index_cos_mean"][-1])

            global_dataset_index += B

    # =========================
    # save summaries
    # =========================
    summary = {
        "config": cfg,
        "tip_threshold": args.tip_threshold,
        "smooth_threshold": args.smooth_threshold,
        "global_pointwise": {},
        "by_offset": {},
        "by_point_group": {},
        "by_patch_group": {},
    }

    for name, vals in pointwise.items():
        vals_arr = np.asarray(vals, dtype=np.float64)
        entry = summarize_array(vals_arr)
        if "angle_deg" in name:
            entry.update(circular_summary_deg(vals_arr))
        summary["global_pointwise"][name] = entry

    for name, d in by_offset.items():
        summary["by_offset"][name] = {}
        for off, vals in sorted(d.items(), key=lambda kv: kv[0]):
            vals_arr = np.asarray(vals, dtype=np.float64)
            entry = summarize_array(vals_arr)
            if "angle_deg" in name:
                entry.update(circular_summary_deg(vals_arr))
            summary["by_offset"][name][str(off)] = entry

    for name, d in by_point_group.items():
        summary["by_point_group"][name] = {}
        for grp, vals in d.items():
            vals_arr = np.asarray(vals, dtype=np.float64)
            entry = summarize_array(vals_arr)
            if "angle_deg" in name:
                entry.update(circular_summary_deg(vals_arr))
            summary["by_point_group"][name][grp] = entry

    for name, d in by_patch_group.items():
        summary["by_patch_group"][name] = {}
        for grp, vals in d.items():
            vals_arr = np.asarray(vals, dtype=np.float64)
            entry = summarize_array(vals_arr)
            if "angle_deg" in name:
                entry.update(circular_summary_deg(vals_arr))
            summary["by_patch_group"][name][grp] = entry

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    # =========================
    # plots
    # =========================

    # key global histograms
    for name in [
        "F_vs_T_cos", "F_vs_N_cos", "F_vs_K_cos",
        "G_vs_T_cos", "G_vs_N_cos", "G_vs_K_cos",
        "F_vs_G_cos",
        "F_vs_T_angle_deg", "G_vs_T_angle_deg",
        "kappa",
        "pair_cos_F_mean", "pair_cos_G_mean", "FG_same_index_cos_mean",
    ]:
        if name in pointwise:
            save_hist(np.asarray(pointwise[name]), name, name, out_dir / f"hist_{name}.png")

    # offset profiles
    offsets_sorted = sorted(next(iter(by_offset.values())).keys()) if by_offset else []
    x = np.array(offsets_sorted, dtype=np.int64)

    profile_metrics = [
        "F_vs_T_cos", "F_vs_N_cos", "F_vs_K_cos",
        "G_vs_T_cos", "G_vs_N_cos", "G_vs_K_cos",
        "F_vs_G_cos", "kappa"
    ]
    for name in profile_metrics:
        if name not in by_offset:
            continue
        y = []
        for off in offsets_sorted:
            vals = np.asarray(by_offset[name][off], dtype=np.float64)
            y.append(np.mean(vals) if vals.size else np.nan)
        save_line_plot(
            x=x,
            ys={name: np.asarray(y, dtype=np.float64)},
            title=f"{name} vs relative offset",
            xlabel="relative offset in patch",
            ylabel=name,
            path=out_dir / f"profile_{name}.png",
        )

    # combined profiles for main comparisons
    def profile_arr(name):
        y = []
        for off in offsets_sorted:
            vals = np.asarray(by_offset[name][off], dtype=np.float64)
            y.append(np.mean(vals) if vals.size else np.nan)
        return np.asarray(y, dtype=np.float64)

    combined = {
        "F_vs_T": profile_arr("F_vs_T_cos") if "F_vs_T_cos" in by_offset else None,
        "F_vs_N": profile_arr("F_vs_N_cos") if "F_vs_N_cos" in by_offset else None,
        "G_vs_T": profile_arr("G_vs_T_cos") if "G_vs_T_cos" in by_offset else None,
        "G_vs_N": profile_arr("G_vs_N_cos") if "G_vs_N_cos" in by_offset else None,
        "F_vs_G": profile_arr("F_vs_G_cos") if "F_vs_G_cos" in by_offset else None,
    }
    combined = {k: v for k, v in combined.items() if v is not None}
    if combined:
        save_line_plot(
            x=x,
            ys=combined,
            title="Field structure vs relative offset",
            xlabel="relative offset in patch",
            ylabel="mean cosine",
            path=out_dir / "profile_main_cosines.png",
        )

    # point-group bar-like line plots
    groups = ["tip", "near_tip", "smooth"]
    for metric in ["F_vs_T_cos", "F_vs_N_cos", "G_vs_T_cos", "G_vs_N_cos", "F_vs_G_cos", "kappa"]:
        if metric not in by_point_group:
            continue
        y = {}
        for g in groups:
            vals = np.asarray(by_point_group[metric].get(g, []), dtype=np.float64)
            y[g] = np.array([np.mean(vals) if vals.size else np.nan])
        plt.figure(figsize=(6, 4))
        xs = np.arange(len(groups))
        means = [y[g][0] for g in groups]
        plt.bar(xs, means)
        plt.xticks(xs, groups)
        plt.title(f"{metric} by point group")
        plt.ylabel(metric)
        plt.tight_layout()
        plt.savefig(out_dir / f"bar_{metric}_by_point_group.png", dpi=180)
        plt.close()

    patch_groups = ["tip_patch", "curved_patch", "smooth_patch"]
    for metric in ["pair_cos_F_mean", "pair_cos_G_mean", "FG_same_index_cos_mean", "kappa"]:
        if metric not in by_patch_group:
            continue
        plt.figure(figsize=(6, 4))
        xs = np.arange(len(patch_groups))
        means = []
        for g in patch_groups:
            vals = np.asarray(by_patch_group[metric].get(g, []), dtype=np.float64)
            means.append(np.mean(vals) if vals.size else np.nan)
        plt.bar(xs, means)
        plt.xticks(xs, patch_groups)
        plt.title(f"{metric} by patch group")
        plt.ylabel(metric)
        plt.tight_layout()
        plt.savefig(out_dir / f"bar_{metric}_by_patch_group.png", dpi=180)
        plt.close()

    # quick report
    with open(out_dir / "quick_report.txt", "w") as f:
        f.write("=== GLOBAL POINTWISE ===\n")
        for k, v in summary["global_pointwise"].items():
            f.write(f"\n{k}\n{json.dumps(v, indent=2)}\n")
        f.write("\n=== BY POINT GROUP ===\n")
        for k, v in summary["by_point_group"].items():
            f.write(f"\n{k}\n{json.dumps(v, indent=2)}\n")
        f.write("\n=== BY PATCH GROUP ===\n")
        for k, v in summary["by_patch_group"].items():
            f.write(f"\n{k}\n{json.dumps(v, indent=2)}\n")

    print(f"Saved field-structure analysis to: {out_dir}")


if __name__ == "__main__":
    main()
