# v5 Improvement Experiments — Results Table

Baseline tag: `v4-baseline` (commit af9afac). Branch: `feat/v5-flag-scaffolding`.

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

## 空洞诊断（2026-07-02，eval/diag_split.py）

问题：过拟合与多类别生成的 mesh 都有空洞，怀疑 split 漏判。逐项定位后结论是两个场景病因完全不同，且都不在原先假设的位置。

### 过拟合（O1，单飞机）：split 无罪，洞在 marching cubes iso-level

| 证据 | 数值 |
|---|---|
| teacher-forced FN（生成阈值 0.45 判决，三层） | 0 / 0 / 0 |
| 自由生成 vs GT，各层 missing | 全部 0，depth-6 IoU 0.993 |
| GT token + GT 结构的 ceiling 解码（`vqvae_ceiling_v2.py`） | 同样满身孔洞 |
| 未量化 code / cond ckpt 解码（ceiling_decomp/） | 同样孔洞（排除量化损失与 ckpt 变体） |
| ceiling 解码 + `mc_level=0.01` 提 mesh（level_sweep/） | 孔洞消失 |
| O1 模型输出 + `mc_level=0.01`（gen_mclevel001/） | 干净完整飞机 |

结论：孔洞是 VQ-VAE 解码 SDF 在 0.002 等值面附近的噪声带，提高 iso-level 到 ~0.01 免费解决。
`SOLVER.mc_level`（main_fractal.py）/ `--mc_level`（eval/gen_from_ckpt.py）已可配置。
注意 HF 上 OctGPT 官方只发布了 vqvae_large_im5（与本地权重一致），论文用的 huge 版不可得，
这个 SDF 噪声带就是当前 frozen VQ-VAE 的固有属性。

### 多类别（Q4，prior z 真实生成域）：粗层 split FN + VQ token 噪声

teacher-forced（posterior z，16 个测试样本，阈值 0.45）：

| lvl | depth | GT pos | FN% | 丢失 depth-6 叶子 |
|---|---|---|---|---|
| 0 | 3 | 690 | **21.45%** | 8400 |
| 1 | 4 | 2231 | 8.43% | 3952 |
| 2 | 5 | 8518 | 5.66% | 3856 |

最大问题在最粗层 depth 3（此前报告聚焦"最后一级 acc ~0.80"是被 accuracy 指标掩盖的错误结论，
新增的 per-level recall/precision 指标可直接暴露）。自由生成 depth-6 缺失 29.3%，IoU 0.33。

免训练修复测试：

| 手段 | 效果 |
|---|---|
| 粗层低阈值（0.2/0.35/0.45 等） | missing 29%→5%，但 extra 爆炸（叶子 4-9 倍），prior z 下渲染成噪声云，**不可用** |
| `split_close_k=3`（最后层闭运算） | missing 17.2%→15.7%（同 8 样本），作用有限：缺失主要是 depth 4/5 级联，不是孤立点 |
| **`temperature=0` + `mc_level=0.01`**（gen_t0_lvl001/） | **质变**：噪声云→实心可辨认物体（chair/car 清晰成形，airplane 仍是过密三角翼团块） |

结论：多类别的"空洞/散点"外观大头是 VQ token 采样噪声 + iso-level，而非结构缺失；
结构侧剩余问题集中在 depth-3 的 21% FN（模型对粗层形体不确定，focal loss 校准差），
这是训练侧问题（Stage 2：粗层加权 / scheduled sampling），阈值旋钮救不了。

## R1b 飞机单类实验结果（2026-07-02，粗层加权 + 空间池化 latent）

warm-start 自 Q4，飞机类 2831 train / 809 test，30 epoch（4h55m 单卡）。

| 指标 | Q4 基线（五类） | R1b（飞机类） |
|---|---|---|
| split_recall_lvl0 (test) | ~0.786（FN 21.4%） | **0.948** |
| split_accuracy | 0.841 | 0.915 |
| vq_accuracy | 0.645 | 0.731 |
| kl_loss | ~0.5 | 1.17（z 携带信息量翻倍） |
| 自由生成 depth-6 IoU（posterior z） | 0.33 | 0.42（阈值 [0.45,0.5,0.6] 时 0.467） |

结论：
- 两个改动方向都有效：粗层 FN 大降、z 开始生效（posterior 与 prior 生成的结构可区分）。
- **但 prior z 生成的飞机仍是菱形团块**：recall 偏置让 precision 掉到 0.81
  （生成叶子 2.4 倍于 GT），细层阈值拉高只能换 IoU 到 0.467，救不回轮廓。
- `split_sample`（全层伯努利采样）生成碎片，排除。
- 根本瓶颈定位：单次前向对每个节点按**独立边缘概率**判决，多模式被平均成团块；
  z 的模式选择能力不足以完全消除。这是与 OctGPT（576 步空间自回归，逐步承诺）
  的本质差距。候选修法：只在 depth-3（≤512 节点）做 4-8 步迭代 refinement
  （成本 +8 次 forward，仍保 ~50x 速度优势），区别于失败的 Q3（全层+错误训练策略）。
