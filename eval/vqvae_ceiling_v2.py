"""Corrected VQ-VAE ceiling: decode GT tokens on a depth_stop octree that is
zero-extended to full depth (exactly the structure distribution the decoder /
OctGPT / our generate use), NOT the real full-depth octree (which is OOD and
created fake holes in the first ceiling test).

If these recons are clean, the VQ-VAE is fine and our generation holes come from
our octree STRUCTURE (split prediction) missing nodes.
"""
import os, sys, copy, argparse
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "octgpt")); sys.path.insert(0, _ROOT)
import torch, ocnn
from thsolver.config import parse_args
from ognn.octreed import OctreeD
from octgpt.utils import utils, builder
SYN = {0:"airplane",1:"car",2:"chair",3:"rifle",4:"table"}

ap = argparse.ArgumentParser()
ap.add_argument("--config", required=True)
ap.add_argument("--out_dir", default="logs/exp/_ceiling_v2")
ap.add_argument("--per_class", type=int, default=1)
a,_ = ap.parse_known_args()
sys.argv=[sys.argv[0],"--config",a.config]; F=parse_args()
dev="cuda"; os.makedirs(a.out_dir, exist_ok=True)
depth_stop=F.MODEL.depth_stop; depth=F.MODEL.depth; full_depth=F.MODEL.full_depth

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
    octree_gt=batch["octree_gt"].to(dev)
    with torch.no_grad():
        vq_raw=vqvae.extract_code(octree_gt)
        _,idx,_=vqvae.quantizer(vq_raw)
        vq_code=vqvae.quantizer.extract_code(idx)
        # rebuild a depth_stop octree from GT points, then zero-extend to full
        pts=octree_gt.to_points()
        oct6=ocnn.octree.Octree(depth_stop, full_depth, device=dev)
        oct6.build_octree(pts)
        oct6.construct_all_neigh()
        for d in range(depth_stop, depth):
            oct6.octree_split(torch.zeros(oct6.nnum[d],device=dev).long(), d)
            oct6.octree_grow(d+1)
        # sanity: depth-6 leaf count must match the codes
        n6_gt=int(octree_gt.nnum[depth_stop]); n6_re=int(oct6.nnum[depth_stop])
        doctree=OctreeD(oct6)
        out=vqvae.decode_code(vq_code, depth_stop, doctree, copy.deepcopy(doctree), update_octree=True)
    path=os.path.join(a.out_dir, f"{SYN[c]}_{seen[c]}.obj")
    utils.create_mesh(out["neural_mpu"], path, size=F.SOLVER.resolution, level=0.002,
        clean=True, bbmin=-F.SOLVER.sdf_scale, bbmax=F.SOLVER.sdf_scale,
        mesh_scale=F.DATA.test.points_scale, save_sdf=False)
    seen[c]+=1; done+=1
    print(f"[{done}/{need}] {SYN[c]}  code_n={vq_code.shape[0]} gt_d6={n6_gt} rebuilt_d6={n6_re} match={n6_gt==n6_re}")
print("done ->", a.out_dir)
