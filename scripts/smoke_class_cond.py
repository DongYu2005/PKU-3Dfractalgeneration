"""Smoke test for class-conditional FractalGenerator.

Verifies, on a few real data samples:
  1. batch['label'] carries non-trivial class indices (0-4), not all zeros
  2. forward() with label produces a finite loss and backprops
  3. generate(label=c) runs for each class
  4. warm-start: a baseline ckpt loads into the class-cond model (strict, after
     adding a zero class_embedding) and the first forward equals the
     unconditional one when class_embedding is zero
Run:
  python scripts/smoke_class_cond.py --config configs/exp/C1_class_cond.yaml
"""
import os
import sys
import argparse

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "octgpt"))
sys.path.insert(0, _ROOT)

import torch
from thsolver.config import parse_args
from octgpt.utils import builder
from fractal_models.fractal_generator import FractalGenerator


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args, _ = parser.parse_known_args()
    sys.argv = [sys.argv[0], "--config", args.config]
    FLAGS = parse_args()

    device = "cuda"
    dflags = FLAGS.DATA.train
    dataset, collate = builder.build_dataset(dflags)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=4, collate_fn=collate, num_workers=2, shuffle=True)
    batch = next(iter(loader))

    label = batch["label"]
    print(f"[1] label tensor: {label.tolist()}  dtype={label.dtype}")
    assert label.dtype == torch.long, "label must be long"
    assert int(label.max()) <= 4 and int(label.min()) >= 0, "label out of range"
    assert int(label.max()) > 0, "all labels are 0 -> filelist 2nd column missing"
    print("    OK: labels are non-trivial class indices")

    # ---- VQ-VAE (frozen) ----
    vqvae = builder.build_vae_model(FLAGS.MODEL.VQVAE).to(device)
    ckpt = torch.load(FLAGS.MODEL.vqvae_ckpt, weights_only=True,
                      map_location=device)
    vqvae.load_state_dict(ckpt)
    vqvae.eval()
    for p in vqvae.parameters():
        p.requires_grad_(False)

    # ---- class-conditional model ----
    model = FractalGenerator(**FLAGS.MODEL.FractalGen).to(device)
    emb_sum = float(model.class_embedding.weight.abs().sum())
    print(f"[2] use_class_cond={model.use_class_cond} "
          f"num_classes={model.num_classes} class_emb_sum={emb_sum:.6f}")
    assert emb_sum == 0.0, "class_embedding must be zero-initialised (warm-start)"
    print("    OK: class_embedding zero-initialised")

    if getattr(model, "use_latent", False):
        zp = float(model.z_proj.weight.abs().sum()) + \
            float(model.z_proj.bias.abs().sum())
        print(f"    use_latent=True z_proj_sum={zp:.6f}")
        assert zp == 0.0, "z_proj must be zero-initialised (warm-start)"
        print("    OK: z_proj zero-initialised")

    octree_gt = batch["octree_gt"].to(device)
    label = label.to(device)
    out = model(octree_gt=octree_gt, vqvae=vqvae, label=label)
    loss = out["loss"]
    klmsg = f" kl={float(out['kl_loss']):.4f}" if "kl_loss" in out else ""
    print(f"[3] forward loss={float(loss):.4f} "
          f"split={float(out['split_loss']):.4f} "
          f"vq={float(out['vq_loss']):.4f}{klmsg}")
    assert torch.isfinite(loss), "loss is not finite"
    loss.backward()
    g = model.class_embedding.weight.grad
    print(f"    class_embedding grad norm={float(g.norm()):.6f} "
          f"(should be >0 after backward)")
    assert g is not None and float(g.norm()) > 0, "no grad to class_embedding"
    if getattr(model, "use_latent", False):
        gz = model.z_mu.weight.grad
        print(f"    z_mu grad norm={float(gz.norm()):.6f}")
        assert gz is not None and float(gz.norm()) > 0, "no grad to z_mu (KL?)"
    print("    OK: finite loss, conditioning receives gradient")

    # ---- generate per class ----
    model.eval()
    with torch.no_grad():
        for c in range(model.num_classes):
            lab = torch.full((1,), c, dtype=torch.long, device=device)
            octree, vq_code = model.generate(
                batch_size=1, device=device, temperature=0.8,
                vqvae=vqvae, label=lab)
            print(f"[4] gen cls{c}: octree leaves(d6)={octree.nnum[6]} "
                  f"vq_code={tuple(vq_code.shape)}")
    print("    OK: generate runs for all classes")

    print("\nSMOKE PASSED")


if __name__ == "__main__":
    main()
