"""Generate meshes from a trained FractalGenerator checkpoint with an explicit
split_sample setting (overrides config), for clean post-hoc evaluation.

Usage:
  python eval/gen_from_ckpt.py --config configs/exp/Q2_scaled.yaml \
      --ckpt logs/exp/Q2_scaled/checkpoints/00020.model.pth \
      --out_dir logs/exp/Q2_scaled/gen_thresh --per_class 3 --split_sample 0
"""
import os, sys, copy, argparse
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "octgpt"))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch, ocnn
from thsolver.config import parse_args
from ognn.octreed import OctreeD
from octgpt.utils import builder, utils
from fractal_models.fractal_generator import FractalGenerator
SYN = {0:"airplane",1:"car",2:"chair",3:"rifle",4:"table"}

ap = argparse.ArgumentParser()
ap.add_argument("--config", required=True)
ap.add_argument("--ckpt", required=True)
ap.add_argument("--out_dir", required=True)
ap.add_argument("--per_class", type=int, default=3)
ap.add_argument("--split_sample", type=int, default=0)
ap.add_argument("--temperature", type=float, default=0.8)
a,_ = ap.parse_known_args()
sys.argv=[sys.argv[0],"--config",a.config]; F=parse_args()
dev="cuda"; os.makedirs(a.out_dir, exist_ok=True)
depth_stop=F.MODEL.depth_stop; depth=F.MODEL.depth

cfg=dict(F.MODEL.FractalGen); cfg["split_sample"]=bool(a.split_sample)
m=FractalGenerator(**cfg).to(dev).eval()
m.load_state_dict(torch.load(a.ckpt, map_location=dev, weights_only=True))
vq=builder.build_vae_model(F.MODEL.VQVAE).to(dev).eval()
vq.load_state_dict(torch.load(F.MODEL.vqvae_ckpt, weights_only=True, map_location=dev))

print(f"ckpt={a.ckpt} split_sample={bool(a.split_sample)} per_class={a.per_class}")
for c in range(len(SYN)):
    for i in range(a.per_class):
        torch.manual_seed(1000*c+i)
        with torch.no_grad():
            oct,vqc=m.generate(batch_size=1,device=dev,temperature=a.temperature,
                               vqvae=vq,label=torch.tensor([c],device=dev))
        if vqc.shape[0]==0:
            print(f"  {SYN[c]}_{i}: empty"); continue
        for d in range(depth_stop,depth):
            oct.octree_split(torch.zeros(oct.nnum[d],device=dev).long(),d); oct.octree_grow(d+1)
        dt=OctreeD(oct); out=vq.decode_code(vqc,depth_stop,dt,copy.deepcopy(dt),update_octree=True)
        p=os.path.join(a.out_dir,f"{SYN[c]}_{i}.obj")
        utils.create_mesh(out["neural_mpu"],p,size=F.SOLVER.resolution,level=0.002,clean=True,
            bbmin=-F.SOLVER.sdf_scale,bbmax=F.SOLVER.sdf_scale,mesh_scale=F.DATA.test.points_scale,save_sdf=False)
        print(f"  {SYN[c]}_{i}: leaves(d6)={oct.nnum[depth_stop].item()}")
print("done ->", a.out_dir)
