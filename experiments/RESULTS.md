# v5 Improvement Experiments — Results Table

Baseline tag: `v4-baseline` (commit 700f410). Branch: `feat/v5-flag-scaffolding`.

Planned base config: `configs/shapenet_frac_im5.yaml` (im_5, 5 categories, 20438 train samples, 20 epochs unless noted). See `FRACTAL_IMPROVEMENT_PLAN.md` for design rationale.

**Current log audit (2026-05-26)**: existing `logs/exp/*/all_configs.yaml` files show the completed E0-E4 runs were single-GPU runs with `DATA.train.batch_size=16`, not the originally documented 4-GPU DDP setup. E5 did not complete, and E6-E9 failed with CUDA OOM. Do not compare future reruns at a different batch size against the current E0-E4 numbers as a strict ablation unless E0 is rerun with the same runtime settings.

**Metric naming**: current `test/split_accuracy` values below are the reported values from `log.csv`. In the current evaluation code they are not a true OctGPT-style masked metric because eval uses a full mask path, so `test/split_accuracy == test/split_accuracy_all` in the completed logs. Higher is better, but strict OctGPT metric alignment still requires a code change and rerun.

## Stage 1 — Baseline + metric alignment

| ID | Name | Config | split_acc (reported) | split_acc_all | vq_acc | epoch_time(min) | VRAM peak (GB) | Notes |
|---|---|---|---|---|---|---|---|---|
| E0 | baseline_old_metric | E0_baseline_old_metric.yaml | n/a | TBD | TBD | TBD | TBD | reproduce v4 number; sanity check |
| E0' | baseline_new_metric | E0_baseline.yaml | 0.796 | 0.796 | 0.654 | ~60 | TBD | completed, single GPU bs=16 |

## Stage 2 — Single-variable ablation (orthogonal flags)

| ID | Name | Flag | split_acc (reported) | Δ vs E0' | vq_acc | epoch_time(min) | Notes |
|---|---|---|---|---|---|---|---|
| E1 | mask_only | use_masked_training=True | 0.785 | -0.011 | 0.654 | ~60 | completed, worse than E0 under current implementation |
| E2 | buffer_only | buffer_size=32 | 0.808 | +0.012 | 0.655 | ~60 | completed, small positive |
| E3 | blocks_only | mid_blocks_per_level=[4,2,1] | 0.802 | +0.006 | 0.655 | ~60 | completed, small positive; params ↑ |
| E4 | sibling_only | use_sibling_attn=True | 0.792 | -0.004 | 0.654 | ~60 | completed, no metric gain |
| E5 | ce_only | use_focal_loss=False | incomplete | n/a | n/a | n/a | OOM before 20 epochs; current result unusable |

## Stage 3 — Cumulative combinations

| ID | Name | Flags | split_acc (reported) | Δ vs E0' | FID(↓) | infer ms/sample | Notes |
|---|---|---|---|---|---|---|---|
| E6 | mask+buffer | E1+E2 | failed | n/a | TBD | TBD | OOM at bs=16; rerun with lower batch and matched E0 |
| E7 | mask+buffer+ce | E6 + use_focal_loss=False | failed | n/a | TBD | TBD | OOM at bs=16; rerun with lower batch and matched E0 |
| E8 | mask+buffer+blocks | E6 + mid_blocks=[4,2,1] | failed | n/a | TBD | TBD | OOM at bs=16; rerun with lower batch and matched E0 |
| E9 | full_stack | E1+E2+E3+E4+E5 | failed | n/a | TBD | TBD | OOM at bs=16; rerun with lower batch and matched E0 |

## Current Issues Found

1. The current ablation is not complete: E5 has no `done.flag`, and E6-E9 have `failed.flag`.
2. The documented runtime setup did not match the logs: current results are single-GPU batch 16, while the old text said 4-GPU DDP bs=8/GPU.
3. E6-E9 failed because the mask+buffer path is memory-heavy, not because of a metric regression. Existing `scripts/run_all_exp.sh` now defaults to batch 8 and caps the high-memory combinations to batch 8, but a fair comparison requires rerunning E0-E9 under the same batch setting.
4. The current `test/split_accuracy` is not actually the planned OctGPT-style mask-only metric. The equality between `split_accuracy` and `split_accuracy_all` in E0-E4 confirms the metric-alignment claim is not yet validated.
5. E1 being worse than E0 suggests the masked-training implementation or hyperparameters need another check; under the intended hypothesis it should not be treated as validated.

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
