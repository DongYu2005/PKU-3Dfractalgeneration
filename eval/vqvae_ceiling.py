"""VQ-VAE reconstruction ceiling test.

Takes real GT shapes -> octree_gt -> VQ-VAE extract_code -> quantize -> decode
-> marching cubes. This is the BEST our pipeline can ever do at the surface
(perfect GT tokens + real octree). If these recons already have holes, the
holes are the frozen VQ-VAE decoder's ceiling, not our token prediction.

Usage:
  python eval/vqvae_ceiling.py --config configs/exp/V1_latent.yaml \
      --out_dir logs/exp/_vqvae_ceiling --per_class 2
"""
import os
import sys
import copy
import argparse

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "octgpt"))
sys.path.insert(0, _ROOT)

import torch
from thsolver.config import parse_args
from ognn.octreed import OctreeD
from octgpt.utils import utils, builder

SYNSET = {0: "airplane", 1: "car", 2: "chair", 3: "rifle", 4: "table"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out_dir", default="logs/exp/_vqvae_ceiling")
    ap.add_argument("--per_class", type=int, default=2)
    ap.add_argument("--level", type=float, default=0.002)
    ap.add_argument("--res", type=int, default=0, help="0=use SOLVER.resolution")
    ap.add_argument("--clean", type=int, default=1)
    args, _ = ap.parse_known_args()
    sys.argv = [sys.argv[0], "--config", args.config]
    FLAGS = parse_args()
    device = "cuda"
    os.makedirs(args.out_dir, exist_ok=True)

    depth_stop = FLAGS.MODEL.depth_stop
    depth = FLAGS.MODEL.depth

    # frozen VQ-VAE
    vqvae = builder.build_vae_model(FLAGS.MODEL.VQVAE).to(device)
    ckpt = torch.load(FLAGS.MODEL.vqvae_ckpt, weights_only=True,
                      map_location=device)
    vqvae.load_state_dict(ckpt)
    vqvae.eval()
    for p in vqvae.parameters():
        p.requires_grad_(False)
    print(f"loaded VQ-VAE from {FLAGS.MODEL.vqvae_ckpt}")

    dataset, collate = builder.build_dataset(FLAGS.DATA.test)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=1, collate_fn=collate, num_workers=2, shuffle=True)

    seen = {c: 0 for c in SYNSET}
    need = args.per_class * len(SYNSET)
    done = 0
    for batch in loader:
        if done >= need:
            break
        c = int(batch["label"][0])
        if seen.get(c, 0) >= args.per_class:
            continue
        octree_gt = batch["octree_gt"].to(device)
        with torch.no_grad():
            vq_raw = vqvae.extract_code(octree_gt)
            _, idx, _ = vqvae.quantizer(vq_raw)
            vq_code = vqvae.quantizer.extract_code(idx)
            doctree = OctreeD(octree_gt)
            out = vqvae.decode_code(
                vq_code, depth_stop, doctree, copy.deepcopy(doctree),
                update_octree=True)
        name = f"ceiling_{SYNSET[c]}_{seen[c]}.obj"
        path = os.path.join(args.out_dir, name)
        res = args.res if args.res > 0 else FLAGS.SOLVER.resolution
        utils.create_mesh(
            out["neural_mpu"], path, size=res,
            level=args.level, clean=bool(args.clean),
            bbmin=-FLAGS.SOLVER.sdf_scale,
            bbmax=FLAGS.SOLVER.sdf_scale,
            mesh_scale=FLAGS.DATA.test.points_scale, save_sdf=False)
        seen[c] += 1
        done += 1
        print(f"[{done}/{need}] {name}  octree d6 leaves={octree_gt.nnum[depth_stop]}")

    print(f"\nDone. GT-token VQ-VAE recons in {args.out_dir}")
    print("If these have holes, the holes are the VQ-VAE decoder ceiling.")


if __name__ == "__main__":
    main()
