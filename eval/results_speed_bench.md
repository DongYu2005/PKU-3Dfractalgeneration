# Inference Speed Benchmark — Fractal v4 vs OctGPT

**Date**: 2026-05-15
**Hardware**: 1× NVIDIA A100-SXM4-80GB (CUDA_VISIBLE_DEVICES=4)
**Env**: `ydg-ocnn` conda env
**Note**: Both models randomly initialised (no trained ckpt loaded). Latency is
weight-independent at the architecture level, so these numbers are valid for
comparison even without trained checkpoints. Speed should match within a few
percent once trained ckpts are loaded.

## Configs

- Fractal: `configs/shapenet_frac.yaml` (depth_stop=6, full_depth=3, feature_dim=384)
- OctGPT: `octgpt/configs/ShapeNet/shapenet_uncond.yaml` (depth_stop=6, full_depth=3, num_embed=768, num_iters=[64,128,128,256])

## Results

| Model | Params | Theoretical Fwd | Wall Time / Sample (median) | Throughput | VRAM Peak |
|---|---:|---:|---:|---:|---:|
| OctGPT | 170.29M | 576 | 812,646 ms (~13.5 min) | 0.001 sps | 30,729 MiB |
| Fractal v4 (VQ-VAE) | 19.60M | 4 | 716 ms | 1.40 sps | 12,021 MiB |
| **Speedup** | **8.7× smaller** | **144× fewer** | **~1,134× faster** | — | **2.6× smaller** |

OctGPT N=3 samples + 1 warmup; Fractal N=10 + 2 warmup (could afford more
because it's fast). Std dev reported by script:
- OctGPT std = 14,893 ms across 3 samples
- Fractal std = 167.9 ms across 10 samples

## Why such a gap

- **Forward count**: OctGPT does 576 forwards per sample (MaskGIT-style
  iterative refinement, `num_iters=[64,128,128,256]` cumulative across depths
  3→6), Fractal does 4 (3 mid-transformer levels + 1 leaf transformer).
- **Per-forward cost**: OctGPT's transformer is ~2× wider (feature_dim=768 vs
  384), so each forward is itself more expensive.
- **Sequence length**: at the deepest depth OctGPT's input includes all
  previously generated tokens (cumulative `nnum_split`), Fractal's leaf
  transformer only sees the leaf-depth tokens.

## How to reproduce

```bash
cd /home/dataset-assist-0/usr/lh/ydg/几何计算前沿/3DFractalgen
conda activate ydg-ocnn

# Fractal (fast — finishes in <1 min)
CUDA_VISIBLE_DEVICES=4 python eval/bench_speed.py \
  --model fractal \
  --config configs/shapenet_frac.yaml \
  --n_samples 10 --warmup 2

# OctGPT (~14 min per sample, plan for ~1 hour for 3 samples)
CUDA_VISIBLE_DEVICES=4 python eval/bench_speed.py \
  --model octgpt \
  --config octgpt/configs/ShapeNet/shapenet_uncond.yaml \
  --n_samples 3 --warmup 1
```

## Open questions

1. With a trained Fractal ckpt, does the threshold-based split produce a
   denser octree (more leaves)? If yes, leaf transformer cost grows, and
   per-sample time might increase. Need to re-measure with `ckpt=`.
2. OctGPT's `num_iters` is tunable; lowering it speeds up inference at the cost
   of quality. Need to check what value they actually used for the published
   numbers, and benchmark both at the "quality-equivalent" iter count.
3. Batch generation: Fractal's `batch_size=1` here. With larger batches the
   throughput should scale ~linearly until VRAM saturates; OctGPT may not
   benefit as much because of the iterative refinement.
