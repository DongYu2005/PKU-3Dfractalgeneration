"""
Inference speed benchmark: Fractal vs OctGPT.

Times the transformer generation pass (model.generate(...), excluding VQ-VAE
decode and marching cubes which are identical between methods) and reports:
  - wall clock per sample (median over N runs)
  - VRAM peak
  - theoretical forward count

A trained checkpoint is optional. Without it, the model is randomly initialised;
this still gives a valid architecture-level timing comparison (latency is
weight-independent).

Usage:
    python eval/bench_speed.py \
        --model fractal \
        --config configs/shapenet_frac.yaml \
        [--ckpt logs/fractal/shapenet_fractal_vq/checkpoints/00050.model.pth] \
        --n_samples 50 --warmup 5

    python eval/bench_speed.py \
        --model octgpt \
        --config octgpt/configs/ShapeNet/shapenet_uncond.yaml \
        [--ckpt <path>] \
        --n_samples 50 --warmup 5
"""

import argparse
import os
import statistics
import sys
import time

import torch
import yaml
from yacs.config import CfgNode as CN

_THIS = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.normpath(os.path.join(_THIS, ".."))
_OCTGPT = os.path.normpath(os.path.join(_ROOT, "octgpt"))
for p in (_ROOT, _OCTGPT):
    if p not in sys.path:
        sys.path.insert(0, p)

import ocnn  # noqa: E402


def load_flags(config_path: str) -> CN:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    return CN(cfg, new_allowed=True)


def build_fractal(flags: CN, device: str):
    from fractal_models.fractal_generator import FractalGenerator
    from octgpt.utils import builder

    model_flags = flags.MODEL
    model = FractalGenerator(**model_flags.FractalGen).to(device).eval()
    vqvae = builder.build_vae_model(model_flags.VQVAE).to(device).eval()
    n_forward = model.num_levels + 1  # mid_transformers + leaf_transformer
    return model, vqvae, n_forward


def build_octgpt(flags: CN, device: str):
    from models.octgpt import OctGPT
    from utils import builder

    model_flags = flags.MODEL
    model = OctGPT(vqvae_config=model_flags.VQVAE, **model_flags.OctGPT).to(device).eval()
    vqvae = builder.build_vae_model(model_flags.VQVAE).to(device).eval()
    num_iters = model_flags.OctGPT.get("num_iters", [64, 128, 128, 256])
    n_forward = sum(num_iters) if isinstance(num_iters, list) else int(num_iters)
    return model, vqvae, n_forward


def maybe_load_ckpt(model: torch.nn.Module, ckpt: str | None):
    if not ckpt:
        return False
    sd = torch.load(ckpt, map_location="cuda", weights_only=False)
    if isinstance(sd, dict) and "model_dict" in sd:
        sd = sd["model_dict"]
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"loaded ckpt {ckpt} (missing={len(missing)}, unexpected={len(unexpected)})")
    return True


@torch.no_grad()
def time_fractal(model, vqvae, n_samples: int, warmup: int, device: str):
    times = []
    torch.cuda.reset_peak_memory_stats(device)
    for i in range(warmup + n_samples):
        torch.cuda.synchronize(device)
        t0 = time.perf_counter()
        model.generate(batch_size=1, device=device, temperature=0.8, vqvae=vqvae)
        torch.cuda.synchronize(device)
        dt = time.perf_counter() - t0
        if i >= warmup:
            times.append(dt)
    peak_vram = torch.cuda.max_memory_allocated(device) / 1024**2
    return times, peak_vram


@torch.no_grad()
def time_octgpt(model, vqvae, flags: CN, n_samples: int, warmup: int, device: str):
    depth_low = flags.MODEL.full_depth
    depth_high = flags.MODEL.depth_stop
    full_depth = flags.MODEL.full_depth

    times = []
    torch.cuda.reset_peak_memory_stats(device)
    for i in range(warmup + n_samples):
        octree = ocnn.octree.init_octree(depth_high, full_depth, 1, device)
        torch.cuda.synchronize(device)
        t0 = time.perf_counter()
        model.generate(octree, depth_low=depth_low, depth_high=depth_high, vqvae=vqvae)
        torch.cuda.synchronize(device)
        dt = time.perf_counter() - t0
        if i >= warmup:
            times.append(dt)
    peak_vram = torch.cuda.max_memory_allocated(device) / 1024**2
    return times, peak_vram


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["fractal", "octgpt"])
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--n_samples", type=int, default=50)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"
    flags = load_flags(args.config)

    if args.model == "fractal":
        model, vqvae, n_forward = build_fractal(flags, device)
        maybe_load_ckpt(model, args.ckpt)
        # VQ-VAE weight needed for extract_code in generate(); load from config
        vqvae_ckpt = flags.MODEL.get("vqvae_ckpt", None)
        if vqvae_ckpt and os.path.isfile(os.path.join(_ROOT, vqvae_ckpt)):
            sd = torch.load(os.path.join(_ROOT, vqvae_ckpt),
                            map_location=device, weights_only=False)
            vqvae.load_state_dict(sd)
        times, peak = time_fractal(model, vqvae, args.n_samples, args.warmup, device)
    else:
        model, vqvae, n_forward = build_octgpt(flags, device)
        maybe_load_ckpt(model, args.ckpt)
        vqvae_ckpt = flags.MODEL.get("vqvae_ckpt", None)
        if vqvae_ckpt:
            cand = vqvae_ckpt if os.path.isabs(vqvae_ckpt) else os.path.join(_OCTGPT, vqvae_ckpt)
            if os.path.isfile(cand):
                sd = torch.load(cand, map_location=device, weights_only=False)
                vqvae.load_state_dict(sd)
        times, peak = time_octgpt(model, vqvae, flags, args.n_samples, args.warmup, device)

    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    median = statistics.median(times)
    mean = statistics.mean(times)
    stdev = statistics.stdev(times) if len(times) > 1 else 0.0

    print()
    print(f"=== {args.model} on {device} ===")
    print(f"params:           {n_params:.2f}M")
    print(f"theoretical fwd:  {n_forward}")
    print(f"wall time/sample: median={median*1000:.1f} ms  mean={mean*1000:.1f} ms  std={stdev*1000:.1f} ms")
    print(f"throughput:       {1.0/median:.2f} samples/sec")
    print(f"peak VRAM:        {peak:.1f} MiB")
    print(f"N samples:        {args.n_samples} (warmup {args.warmup})")


if __name__ == "__main__":
    main()
