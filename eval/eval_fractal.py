"""
Quantitative evaluation for 3D shape generation.

Computes Chamfer Distance (CD), MMD-CD, COV-CD, 1-NN-CD between a directory of
generated .obj meshes and a directory of ground-truth .obj meshes. Reuses
distChamfer / lgan_mmd_cov / knn from octgpt/metrics/evaluation_metrics.py.

Usage:
    python eval/eval_fractal.py \
        --gen_dir logs/fractal/<run>/results \
        --ref_dir data/ShapeNet/datasets_256_test \
        --ref_filelist data/ShapeNet/filelist/test_im_5.txt \
        --out_csv logs/fractal/<run>/eval.csv \
        --n_points 2048 \
        [--by_category]    # group MMD/COV/1-NNA by ShapeNet synset_id
        [--with_emd]       # also compute EMD-based metrics (slow, CPU Hungarian)

Output:
    - <out_csv>: per-sample nearest-CD table
    - stdout: aggregated MMD-CD / COV-CD / 1-NN-CD (overall + per category)
"""

import argparse
import csv
import os
import sys
from collections import defaultdict
from glob import glob

import numpy as np
import torch
import trimesh
from tqdm import tqdm

_THIS = os.path.dirname(os.path.abspath(__file__))
_OCTGPT = os.path.normpath(os.path.join(_THIS, "..", "octgpt"))
if _OCTGPT not in sys.path:
    sys.path.insert(0, _OCTGPT)

from metrics.evaluation_metrics import (
    distChamfer,
    _pairwise_EMD_CD_,
    lgan_mmd_cov,
    knn,
)


def _normalize(pts: np.ndarray) -> np.ndarray:
    pts = pts.astype(np.float64)
    center = (pts.max(0) + pts.min(0)) / 2.0
    scale = (pts.max(0) - pts.min(0)).max()
    if scale < 1e-8:
        raise ValueError("degenerate point cloud (scale~0)")
    return ((pts - center) / scale).astype(np.float32)


def load_and_sample(path: str, n_points: int) -> np.ndarray:
    """Load mesh OR npz pointcloud, normalize to unit cube, return n_points
    surface samples."""
    ext = os.path.splitext(path)[1].lower()
    if ext == ".npz":
        data = np.load(path)
        if "points" not in data:
            raise ValueError(f"npz missing 'points': {path}")
        pts = np.asarray(data["points"], dtype=np.float32)
        if pts.shape[0] > n_points:
            idx = np.random.choice(pts.shape[0], n_points, replace=False)
            pts = pts[idx]
        elif pts.shape[0] < n_points:
            # upsample by repeat with jitter
            reps = (n_points + pts.shape[0] - 1) // pts.shape[0]
            pts = np.tile(pts, (reps, 1))[:n_points]
        return _normalize(pts)
    mesh = trimesh.load(path, force="mesh", process=False)
    if mesh.vertices.shape[0] == 0 or mesh.faces.shape[0] == 0:
        raise ValueError(f"empty mesh: {path}")
    verts = np.asarray(mesh.vertices, dtype=np.float64)
    center = (verts.max(0) + verts.min(0)) / 2.0
    scale = (verts.max(0) - verts.min(0)).max()
    if scale < 1e-8:
        raise ValueError(f"degenerate mesh (scale~0): {path}")
    mesh.vertices = (verts - center) / scale
    pts, _ = trimesh.sample.sample_surface(mesh, n_points)
    return np.asarray(pts, dtype=np.float32)


def collect_obj(d: str) -> list[str]:
    files = sorted(glob(os.path.join(d, "**", "*.obj"), recursive=True))
    if not files:
        # try npz (ShapeNet point cloud cache)
        files = sorted(glob(os.path.join(d, "**", "pointcloud.npz"), recursive=True))
    if not files:
        raise RuntimeError(f"no .obj or pointcloud.npz files found in {d}")
    return files


def category_of(path: str) -> str:
    """ShapeNet synset_id is one path component above the .obj. If no synset
    structure (overfit), return 'all'."""
    parts = os.path.normpath(path).split(os.sep)
    for p in reversed(parts[:-1]):
        if p.isdigit() and len(p) == 8:
            return p
    return "all"


def load_pointclouds(paths: list[str], n_points: int, device: str, desc: str = "sample"
                     ) -> tuple[torch.Tensor, list[str]]:
    pcs, kept = [], []
    for p in tqdm(paths, desc=desc):
        try:
            pcs.append(load_and_sample(p, n_points))
            kept.append(p)
        except Exception as e:
            print(f"  [skip] {p}: {e}", file=sys.stderr)
    if not pcs:
        raise RuntimeError(f"no valid pointclouds loaded from {len(paths)} paths")
    arr = np.stack(pcs, axis=0)
    return torch.from_numpy(arr).to(device), kept


def per_sample_nearest_cd(gen: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    """For each gen sample, find the min Chamfer Distance over all ref samples.
    Returns (N_gen,) tensor."""
    nearest = torch.full((gen.shape[0],), float("inf"), device=gen.device)
    for i in tqdm(range(gen.shape[0]), desc="nearest-CD"):
        g = gen[i:i + 1].expand(ref.shape[0], -1, -1).contiguous()
        dl, dr = distChamfer(g, ref)
        cd = dl.mean(dim=1) + dr.mean(dim=1)  # (N_ref,)
        nearest[i] = cd.min()
    return nearest


def aggregate_mmd_cov_1nna(gen: torch.Tensor, ref: torch.Tensor,
                           pair_batch: int, with_emd: bool) -> dict:
    """Standard 3D generation suite: MMD-CD/EMD, COV-CD/EMD, 1-NN-CD/EMD."""
    out = {}
    # Pairwise distance matrices
    M_rs_cd, M_rs_emd = _pairwise_EMD_CD_(ref, gen, pair_batch, accelerated_cd=False)
    M_rr_cd, M_rr_emd = _pairwise_EMD_CD_(ref, ref, pair_batch, accelerated_cd=False)
    M_ss_cd, M_ss_emd = _pairwise_EMD_CD_(gen, gen, pair_batch, accelerated_cd=False)

    cov_cd = lgan_mmd_cov(M_rs_cd.t())
    out["MMD-CD"] = float(cov_cd["lgan_mmd"])
    out["COV-CD"] = float(cov_cd["lgan_cov"])
    one_nn_cd = knn(M_rr_cd, M_rs_cd, M_ss_cd, 1, sqrt=False)
    out["1-NN-CD-acc"] = float(one_nn_cd["acc"])

    if with_emd:
        cov_emd = lgan_mmd_cov(M_rs_emd.t())
        out["MMD-EMD"] = float(cov_emd["lgan_mmd"])
        out["COV-EMD"] = float(cov_emd["lgan_cov"])
        one_nn_emd = knn(M_rr_emd, M_rs_emd, M_ss_emd, 1, sqrt=False)
        out["1-NN-EMD-acc"] = float(one_nn_emd["acc"])

    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen_dir", required=True, help="dir of generated .obj")
    ap.add_argument("--ref_dir", required=True, help="dir of reference .obj (recursive)")
    ap.add_argument("--ref_filelist", default=None,
                    help="optional: only use ref samples listed here (relative to ref_dir)")
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--n_points", type=int, default=2048)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--by_category", action="store_true",
                    help="group MMD/COV/1-NN by ShapeNet synset_id")
    ap.add_argument("--with_emd", action="store_true",
                    help="also compute EMD metrics (slow, CPU Hungarian)")
    ap.add_argument("--pair_batch", type=int, default=64,
                    help="batch size for pairwise distance matrix")
    ap.add_argument("--max_gen", type=int, default=0, help="cap N_gen (0 = no cap)")
    ap.add_argument("--max_ref", type=int, default=0, help="cap N_ref (0 = no cap)")
    args = ap.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"

    gen_files = collect_obj(args.gen_dir)
    if args.ref_filelist:
        with open(args.ref_filelist) as f:
            relpaths = [ln.strip() for ln in f if ln.strip()]
        ref_files = []
        for rp in relpaths:
            base = os.path.join(args.ref_dir, rp)
            candidates = [
                base if base.endswith((".obj", ".npz")) else None,
                base + ".obj",
                base + ".npz",
                os.path.join(base, "model.obj"),
                os.path.join(base, "normalized_model.obj"),
                os.path.join(base, "pointcloud.npz"),
            ]
            for cand in candidates:
                if cand and os.path.isfile(cand):
                    ref_files.append(cand)
                    break
        if not ref_files:
            raise RuntimeError(
                f"no ref .obj/.npz found via filelist {args.ref_filelist} under {args.ref_dir}")
    else:
        ref_files = collect_obj(args.ref_dir)

    if args.max_gen:
        gen_files = gen_files[:args.max_gen]
    if args.max_ref:
        ref_files = ref_files[:args.max_ref]
    print(f"loaded N_gen={len(gen_files)}, N_ref={len(ref_files)}")

    gen_pcs, gen_files = load_pointclouds(gen_files, args.n_points, device, "sample gen")
    ref_pcs, ref_files = load_pointclouds(ref_files, args.n_points, device, "sample ref")

    # ---- per-sample nearest CD ----
    nearest = per_sample_nearest_cd(gen_pcs, ref_pcs).cpu().numpy()
    os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
    with open(args.out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["gen_obj", "category", "nearest_CD"])
        for p, cd in zip(gen_files, nearest):
            w.writerow([p, category_of(p), f"{cd:.6f}"])
    print(f"wrote per-sample CSV: {args.out_csv}")
    print(f"nearest-CD: mean={nearest.mean():.6f} median={np.median(nearest):.6f}")

    # ---- aggregate MMD/COV/1-NNA ----
    def report(label: str, g: torch.Tensor, r: torch.Tensor):
        if g.shape[0] < 2 or r.shape[0] < 2:
            print(f"[{label}] need >=2 gen and >=2 ref, got {g.shape[0]}/{r.shape[0]}; skip")
            return
        m = aggregate_mmd_cov_1nna(g, r, args.pair_batch, args.with_emd)
        kvs = " ".join(f"{k}={v:.4f}" for k, v in m.items())
        print(f"[{label}] N_gen={g.shape[0]} N_ref={r.shape[0]}  {kvs}")

    report("overall", gen_pcs, ref_pcs)

    if args.by_category:
        gen_by_cat = defaultdict(list)
        ref_by_cat = defaultdict(list)
        for i, p in enumerate(gen_files):
            gen_by_cat[category_of(p)].append(i)
        for i, p in enumerate(ref_files):
            ref_by_cat[category_of(p)].append(i)
        for cat in sorted(set(gen_by_cat) & set(ref_by_cat)):
            gi = torch.tensor(gen_by_cat[cat], device=device)
            ri = torch.tensor(ref_by_cat[cat], device=device)
            report(f"cat={cat}", gen_pcs[gi], ref_pcs[ri])


if __name__ == "__main__":
    main()
