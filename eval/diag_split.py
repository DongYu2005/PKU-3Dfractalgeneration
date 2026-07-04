"""Diagnose where surface holes come from in fractal generation.

Two views per sample, both on the inference feature path:
  1. Teacher-forced (TF): expand along the GT octree, judge each level's
     P(split) under the actual generation threshold rule. FN here = nodes a
     generation run would drop even with a perfect prefix (pure head error);
     each FN's GT-subtree leaf count estimates the hole it causes.
  2. Free-running: model.generate(), key-align every depth against the GT
     octree, report cumulative missing/extra nodes and depth-stop IoU. The
     gap between free-running missing and TF FN is the cascade / TF-gap
     amplification.

Usage:
  python eval/diag_split.py --config configs/exp/O1_air_struct_best.yaml \
      --ckpt logs/exp/O1_air_struct_best/best_model.pth \
      --out_dir logs/exp/O1_air_struct_best/diag --num_samples 1
  # per-level threshold override, e.g. recall-biased coarse levels:
  #   --split_threshold 0.2,0.3,0.45
"""
import os
import sys
import argparse

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "octgpt"))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from thsolver.config import parse_args
from octgpt.utils import builder
from fractal_models.fractal_generator import FractalGenerator

SYN = {0: "airplane", 1: "car", 2: "chair", 3: "rifle", 4: "table"}


def node_keys(octree, d):
    """Linearized (x, y, z) key per node at depth d (single-sample octree)."""
    x, y, z, _ = octree.xyzb(d)
    S = 2 ** d
    return (x.long() * S + y.long()) * S + z.long()


def subtree_leaf_counts(octree, d, depth_stop):
    """Number of depth_stop descendants for every node at depth d."""
    keys_d = node_keys(octree, d)
    x, y, z, _ = octree.xyzb(depth_stop)
    shift = depth_stop - d
    S = 2 ** d
    anc = ((x.long() >> shift) * S + (y.long() >> shift)) * S + \
        (z.long() >> shift)
    sorted_keys, order = keys_d.sort()
    idx = torch.searchsorted(sorted_keys, anc)
    counts_sorted = torch.bincount(idx, minlength=keys_d.numel())
    counts = torch.zeros_like(counts_sorted)
    counts[order] = counts_sorted
    return counts


def key_membership(query, reference):
    ref_sorted, _ = reference.sort()
    return torch.isin(query, ref_sorted)


def save_point_cloud(octree, d, sel, path):
    """Write selected node centers at depth d as an obj point cloud in
    [-1, 1] coordinates (octree grid frame)."""
    x, y, z, _ = octree.xyzb(d)
    S = 2 ** d
    pts = torch.stack([x, y, z], dim=1).float()[sel]
    pts = (pts + 0.5) / S * 2 - 1
    with open(path, "w") as f:
        for p in pts.tolist():
            f.write(f"v {p[0]:.5f} {p[1]:.5f} {p[2]:.5f}\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--num_samples", type=int, default=8)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--posterior_z", type=int, default=1,
                    help="latent models: use GT-posterior z in both passes "
                         "(isolates split quality from latent mismatch)")
    ap.add_argument("--split_threshold", type=str, default="",
                    help="override, scalar or comma list per level")
    ap.add_argument("--split_close_k", type=int, default=-1,
                    help="override morphological closing (-1 = keep config)")
    ap.add_argument("--save_fn_cloud", type=int, default=2,
                    help="save missing-leaf point clouds for first N samples")
    a, _ = ap.parse_known_args()

    sys.argv = [sys.argv[0], "--config", a.config]
    F = parse_args()
    dev = "cuda"
    os.makedirs(a.out_dir, exist_ok=True)

    cfg = dict(F.MODEL.FractalGen)
    if a.split_threshold:
        vals = [float(v) for v in a.split_threshold.split(",")]
        cfg["split_threshold"] = vals[0] if len(vals) == 1 else vals
    if a.split_close_k >= 0:
        cfg["split_close_k"] = a.split_close_k
    model = FractalGenerator(**cfg).to(dev).eval()
    model.load_state_dict(
        torch.load(a.ckpt, map_location=dev, weights_only=True))
    vq = builder.build_vae_model(F.MODEL.VQVAE).to(dev).eval()
    vq.load_state_dict(
        torch.load(F.MODEL.vqvae_ckpt, weights_only=True, map_location=dev))

    dataset, collate = builder.build_dataset(F.DATA.test)
    n_samples = min(a.num_samples, len(dataset))
    L = model.num_levels
    full_depth, depth_stop = model.full_depth, model.depth_stop
    thresholds = [model._threshold_for(l) for l in range(L)]
    print(f"ckpt={a.ckpt}")
    print(f"samples={n_samples} thresholds={thresholds} "
          f"split_close_k={model.split_close_k} "
          f"posterior_z={bool(a.posterior_z)}")

    agg = {
        "tf_pos": [0] * L, "tf_fn": [0] * L, "tf_fp": [0] * L,
        "tf_lost": [0] * L,
        "fr_gt_nodes": [0] * (L + 1), "fr_missing": [0] * (L + 1),
        "fr_extra": [0] * (L + 1),
        "iou_sum": 0.0, "gt_leaves": 0, "gen_leaves": 0,
    }

    for i in range(n_samples):
        torch.manual_seed(i)
        batch = collate([dataset[i]])
        octree_gt = batch["octree_gt"].cuda()
        label = None
        if model.use_class_cond:
            label = torch.as_tensor(
                batch["label"], device=dev).long().reshape(-1)

        # ---- teacher-forced view ----
        levels = model.teacher_forced_probs(
            octree_gt, vqvae=vq, label=label,
            posterior_z=bool(a.posterior_z))
        for lvl, (probs, gt_split) in enumerate(levels):
            d = full_depth + lvl
            pred = (probs > thresholds[lvl]).long()
            fn = (pred == 0) & (gt_split == 1)
            fp = (pred == 1) & (gt_split == 0)
            sub = subtree_leaf_counts(octree_gt, d, depth_stop)
            agg["tf_pos"][lvl] += int((gt_split == 1).sum())
            agg["tf_fn"][lvl] += int(fn.sum())
            agg["tf_fp"][lvl] += int(fp.sum())
            agg["tf_lost"][lvl] += int(sub[fn].sum())

        # ---- free-running view ----
        z = None
        if model.use_latent and a.posterior_z:
            vq_code = vq.extract_code(octree_gt)
            z, _ = model._encode_latent(
                vq_code, octree_gt, octree_gt.batch_size)
        gen_octree, _ = model.generate(
            batch_size=1, device=dev, temperature=a.temperature,
            vqvae=None, label=label, z=z)
        for lvl in range(L + 1):
            d = full_depth + lvl
            gt_k = node_keys(octree_gt, d)
            if d <= gen_octree.full_depth or gen_octree.nnum[d] > 0:
                gen_k = node_keys(gen_octree, d)
            else:
                gen_k = torch.zeros(0, dtype=torch.long, device=dev)
            miss = ~key_membership(gt_k, gen_k)
            extra = ~key_membership(gen_k, gt_k)
            agg["fr_gt_nodes"][lvl] += gt_k.numel()
            agg["fr_missing"][lvl] += int(miss.sum())
            agg["fr_extra"][lvl] += int(extra.sum())
            if d == depth_stop:
                inter = gt_k.numel() - int(miss.sum())
                union = gt_k.numel() + int(extra.sum())
                agg["iou_sum"] += inter / max(union, 1)
                agg["gt_leaves"] += gt_k.numel()
                agg["gen_leaves"] += gen_k.numel()
                if i < a.save_fn_cloud:
                    tag = SYN.get(int(label[0]), "sample") \
                        if label is not None else "sample"
                    save_point_cloud(
                        octree_gt, d, miss,
                        os.path.join(a.out_dir, f"missing_{tag}_{i}.obj"))

    # ---- report ----
    print("\n== Teacher-forced (threshold rule on inference path) ==")
    print(f"{'lvl':>3} {'depth':>5} {'GT pos':>8} {'FN':>7} {'FN%':>7} "
          f"{'FP':>7} {'lost d6 leaves':>14}")
    for lvl in range(L):
        pos = agg["tf_pos"][lvl]
        fn = agg["tf_fn"][lvl]
        print(f"{lvl:>3} {full_depth + lvl:>5} {pos:>8} {fn:>7} "
              f"{100.0 * fn / max(pos, 1):>6.2f}% {agg['tf_fp'][lvl]:>7} "
              f"{agg['tf_lost'][lvl]:>14}")

    print("\n== Free-running vs GT (cumulative, key-aligned) ==")
    print(f"{'depth':>5} {'GT nodes':>9} {'missing':>8} {'miss%':>7} "
          f"{'extra':>8}")
    for lvl in range(L + 1):
        gtn = agg["fr_gt_nodes"][lvl]
        m = agg["fr_missing"][lvl]
        print(f"{full_depth + lvl:>5} {gtn:>9} {m:>8} "
              f"{100.0 * m / max(gtn, 1):>6.2f}% {agg['fr_extra'][lvl]:>8}")

    print(f"\ndepth-{depth_stop} IoU (mean over {n_samples}): "
          f"{agg['iou_sum'] / max(n_samples, 1):.4f}")
    print(f"GT leaves total {agg['gt_leaves']}, generated leaves total "
          f"{agg['gen_leaves']}")
    print(f"point clouds of missing GT leaves -> {a.out_dir}")


if __name__ == "__main__":
    main()
