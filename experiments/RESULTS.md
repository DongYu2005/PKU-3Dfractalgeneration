# v5 Improvement Experiments — Results Table

Baseline tag: `v4-baseline` (commit 700f410). Branch: `feat/v5-flag-scaffolding`.

All experiments use `configs/shapenet_frac_im5.yaml` as the base (im_5, 5 categories, 20438 train samples, bs=8/GPU × 4 GPUs DDP, 20 epochs unless noted). See `FRACTAL_IMPROVEMENT_PLAN.md` for design rationale.

**Metric naming**: `split_acc` reported below is the **mask-only** metric (建议 6) that aligns with OctGPT's measurement. Higher is better. `split_acc_all` is the legacy v4 metric (all positions) reported alongside for back-compat.

## Stage 1 — Baseline + metric alignment

| ID | Name | Config | split_acc (mask) | split_acc_all | vq_acc | epoch_time(min) | VRAM peak (GB) | Notes |
|---|---|---|---|---|---|---|---|---|
| E0 | baseline_old_metric | E0_baseline_old_metric.yaml | n/a | TBD | TBD | TBD | TBD | reproduce v4 number; sanity check |
| E0' | baseline_new_metric | E0p_baseline_new_metric.yaml | TBD | TBD | TBD | TBD | TBD | identical code path, only metric changed |

## Stage 2 — Single-variable ablation (orthogonal flags)

| ID | Name | Flag | split_acc (mask) | Δ vs E0' | vq_acc | epoch_time(min) | Notes |
|---|---|---|---|---|---|---|---|
| E1 | mask_only | use_masked_training=True | TBD | TBD | TBD | TBD | expect +0.08+ |
| E2 | buffer_only | buffer_size=32 | TBD | TBD | TBD | TBD | expect +0.03+ |
| E3 | blocks_only | mid_blocks_per_level=[4,2,1] | TBD | TBD | TBD | TBD | expect +0.03+; params ↑8M |
| E4 | sibling_only | use_sibling_attn=True | TBD | TBD | TBD | TBD | mesh smoothness focus |
| E5 | ce_only | use_focal_loss=False | TBD | TBD | TBD | TBD | maybe slightly negative alone |

## Stage 3 — Cumulative combinations

| ID | Name | Flags | split_acc (mask) | Δ vs E0' | FID(↓) | infer ms/sample | Notes |
|---|---|---|---|---|---|---|---|
| E6 | mask+buffer | E1+E2 | TBD | TBD | TBD | TBD | expect +0.13+, best ROI |
| E7 | mask+buffer+ce | E6 + use_focal_loss=False | TBD | TBD | TBD | TBD | |
| E8 | mask+buffer+blocks | E6 + mid_blocks=[4,2,1] | TBD | TBD | TBD | TBD | check capacity ceiling |
| E9 | full_stack | E1+E2+E3+E4+E5 | TBD | TBD | TBD | TBD | upper bound |

## Stage 4 — Hyperparameter sweeps (on best combo)

| ID | Name | Variable | Values | split_acc | Best | Notes |
|---|---|---|---|---|---|---|
| H1 | mask_ratio sweep | mask_ratio_min | {0.3, 0.5, 0.7} | TBD | TBD | |
| H2 | buffer_size sweep | buffer_size | {16, 32, 64} | TBD | TBD | |
| H3 | mid_blocks sweep | mid_blocks_per_level | {[2,1,1], [4,2,1], [6,3,1]} | TBD | TBD | |

## Acceptance criteria (v5 final)

| Metric | Target | v4 baseline |
|---|---|---|
| split_acc (mask-only) | ≥ 0.90 | TBD (likely ~0.78 in mask-only metric too) |
| inference time / sample | ≤ 1.0× baseline (≤716ms) | 716ms |
| inference forward count | = 4 | 4 |
| training time / epoch | ≤ 4× baseline | TBD |
| FID vs OctGPT | within 20% | TBD |
