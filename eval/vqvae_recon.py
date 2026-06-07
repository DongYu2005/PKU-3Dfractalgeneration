"""TRUE VQ-VAE reconstruction ceiling, via the VQ-VAE's own forward (mirrors
octgpt/main_vae.py eval_step) so encoding/quantization/decoding stay internally
aligned -- no manual token<->octree pairing (which broke the earlier tests).

Feed a GT shape's octree -> VQ-VAE reconstructs -> mesh. If clean, the VQ-VAE +
depth-6 representation is fine, and our generation holes are our split/token,
not the decoder.
"""
import os, sys, argparse
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "octgpt")); sys.path.insert(0, _ROOT)
import torch
from thsolver.config import parse_args
from ognn.octreed import OctreeD
from octgpt.utils import utils, builder
SYN = {0:"airplane",1:"car",2:"chair",3:"rifle",4:"table"}

ap = argparse.ArgumentParser()
ap.add_argument("--config", required=True)
ap.add_argument("--out_dir", default="logs/exp/_vqvae_recon")
ap.add_argument("--per_class", type=int, default=1)
a,_ = ap.parse_known_args()
sys.argv=[sys.argv[0],"--config",a.config]; F=parse_args()
dev="cuda"; os.makedirs(a.out_dir, exist_ok=True)

vqvae=builder.build_vae_model(F.MODEL.VQVAE).to(dev)
vqvae.load_state_dict(torch.load(F.MODEL.vqvae_ckpt, weights_only=True, map_location=dev))
vqvae.eval()
for p in vqvae.parameters(): p.requires_grad_(False)

ds,coll=builder.build_dataset(F.DATA.test)
loader=torch.utils.data.DataLoader(ds,batch_size=1,collate_fn=coll,num_workers=2,shuffle=True)
seen={c:0 for c in SYN}; need=a.per_class*len(SYN); done=0
for batch in loader:
    if done>=need: break
    c=int(batch["label"][0])
    if seen.get(c,0)>=a.per_class: continue
    octree_in=batch["octree_in"].to(dev)
    octree_out=OctreeD(octree_in)
    with torch.no_grad():
        out=vqvae(octree_in, octree_out, update_octree=True)
    # per-shape bbox like octgpt/main_vae.py eval_step (fall back to +/-sdf_scale)
    if "bbox" in batch:
        bb=batch["bbox"][0].numpy(); bbmin,bbmax=bb[:3],bb[3:]
    else:
        ss=F.DATA.test.get("sdf_scale", F.SOLVER.sdf_scale); bbmin,bbmax=-ss,ss
    path=os.path.join(a.out_dir, f"{SYN[c]}_{seen[c]}.obj")
    utils.create_mesh(out["neural_mpu"], path, size=F.SOLVER.resolution, level=0.002,
        clean=True, bbmin=bbmin, bbmax=bbmax,
        mesh_scale=F.DATA.test.points_scale, save_sdf=False)
    print(f"    bbox_src={'batch' if 'bbox' in batch else 'sdf_scale'}")
    seen[c]+=1; done+=1
    print(f"[{done}/{need}] {SYN[c]} reconstructed -> {path}")
print("done ->", a.out_dir)
