# FractalGenerator 改进方案与对照实验设计

> **目标**: 在保留 1134× 推理加速的前提下，让 FractalGenerator 的训练质量接近 OctGPT 水平（split_acc 0.90+，与 OctGPT 的 0.954 在同等 metric 定义下可比）。
>
> **当前状态**: split_acc 卡在 0.78（im_5 多类别），VQ_acc 0.65。
>
> **交付目标**: 完整的对照实验、可复现的改进版本、清晰的版本管理。

---

## 目录

1. [问题诊断](#1-问题诊断)
2. [改进方案设计](#2-改进方案设计)
3. [对照实验矩阵](#3-对照实验矩阵)
4. [版本管理规约](#4-版本管理规约)
5. [实施 checklist](#5-实施-checklist)
6. [验收标准](#6-验收标准)

---

## 1. 问题诊断

### 1.1 现状

| 指标 | Fractal v4 | OctGPT (im_5) |
|------|-----------|---------------|
| split_acc | 0.78 (epoch 20, plateau) | 0.954 (epoch 4) |
| VQ_acc | ~0.65 | (论文未单独报) |
| 参数量 | 19.6M | 170M |
| Feature dim | 384 | 768 |
| Transformer blocks | 3 mid (1 each) + 6 leaf = 9 | 24 (12 encoder + 12 decoder) |
| 训练目标 | per-level split BCE (Focal) + leaf VQ CE | masked CE on split + vq tokens |
| 推理 forward 次数 | 4 | 576 |
| 推理时间/样本 | 716ms | 13.5min |

### 1.2 根因分析

按重要性从高到低排序：

#### 根因 A：训练时缺少"困难样本"信号（最关键）

当前训练 setup 下，每个节点 feature 来自完美的 teacher forcing 路径，模型几乎没有"难"样本可学。Loss 主要由容易样本（明显的空区域 / 明显的占据区域）贡献，梯度信号稀薄。

OctGPT 通过 random masked training（mask_ratio ∈ [0.5, 1.0]）强制模型从残缺上下文重建预测，每个 batch 都有大量困难位置。

**对应改进：建议 1（Masked Training）**

#### 根因 B：缺少全局上下文通道

Fractal 的 mid_transformer 每层只有 1 个 OctFormer block，节点间通信受 patch_size + dilation 局部窗口限制。整个 octree 的全局信息（如类别、整体形状）很难在 1 block 内聚合传播。

OctGPT 用 buffer_size=64 的可学习全局 token，prepend 到序列开头，24 层中持续作为全局摘要通道。

**对应改进：建议 2（Buffer Tokens）**

#### 根因 C：mid_transformer 容量分布与 fractal philosophy 相反

Kaiming FractalGen Table 1 的容量分布：g1=32 layers / g2=8 / g3=3 / g4=1（**最粗层最深**，因为粗层节点少但需捕获全局语义）。

当前 Fractal 设计：每层 1 block，叶子层 6 blocks（**最粗层最浅**）— 反向。

**对应改进：建议 3（Per-level blocks redistribution）**

#### 根因 D：LocalCrossAttentionExpander 在 K=1 时退化为 MLP

```python
context = parent_features.unsqueeze(1)   # (N, 1, C)
attn_out, _ = self.cross_attn(q, context, context)
```

KV 序列长度为 1，`softmax([single_score]) ≡ 1.0`，attention 退化为 `W_v(context)` 的线性变换。注释里的 "local cross-attention" 实际上是个 MLP。

**对应改进：建议 4（Sibling Self-Attention）**

#### 根因 E：Focal Loss 在多类别下未经调优

α=0.75, γ=2.0 是 overfit 单飞机时调的。im_5 的 split 分布不同，且 Focal 会下调 easy samples 的梯度 — 在已经缺少困难样本信号的情况下进一步压缩信号密度。

OctGPT 用纯 CE + masked loss，loss 自然只在困难位置（被 mask 处）贡献。

**对应改进：建议 5（去 Focal Loss）**

#### 根因 F：Metric 定义不可比

当前 `split_accuracy` 在所有位置算（包括平凡的全空节点）。OctGPT 只在 masked 位置算（困难子集）。

理论上 Fractal 的数字应该天然更高（容易样本多），但实际 0.78 < OctGPT 的 0.954 — **真实结构差距比表面更大**。

**对应改进：建议 6（Metric 对齐）**

### 1.3 速度优势必须保留

任何改动都不能损害推理 single-shot per level 的特性。masked training 在推理时设置 mask=全 True，等价于现有 single-shot inference，**推理路径完全不变**。

---

## 2. 改进方案设计

### 2.1 建议 1：Masked Training（核心改进）

**机制**：每层做 split prediction 之前，随机 mask 一部分节点的 feature。模型必须从邻居 features + buffer + 位置编码重建被 mask 节点的 split decision。

**关键 invariants**：
- Mask 仅作用于 split prediction 的 forward 路径
- Expansion 路径仍走未 mask 的原始 features（保证拓扑结构正确）
- Loss 仅在 mask=True 的位置计算
- 推理时 mask=全 True（等价于 single-shot inference）

**代码位置**：`fractal_models/fractal_generator.py`

**`__init__` 改动**：
```python
import scipy.stats as stats

# 新增：mask token
self.mask_emb = nn.Parameter(torch.zeros(1, feature_dim))
nn.init.normal_(self.mask_emb, std=0.02)

# 新增：mask ratio generator (OctGPT 同款)
self.mask_ratio_min = 0.5  # 可配置
self.mask_ratio_generator = stats.truncnorm(
    (self.mask_ratio_min - 1.0) / 0.25, 0, loc=1.0, scale=0.25)
```

**`forward()` 改动**（fractal expansion 循环）：
```python
for lvl in range(self.num_levels):
    d = self.full_depth + lvl
    nnum_d = octree_gt.nnum[d]

    # === Masked training ===
    if self.training:
        mask_ratio = self.mask_ratio_generator.rvs(1)[0]
        n = features.shape[0]
        num_masked = max(1, int(n * mask_ratio))
        orders = torch.randperm(n, device=features.device)
        mask = torch.zeros(n, dtype=torch.bool, device=features.device)
        mask[orders[:num_masked]] = True

        features_for_pred = torch.where(
            mask.unsqueeze(1),
            self.mask_emb.expand(n, -1),
            features
        )
    else:
        features_for_pred = features
        mask = torch.ones(features.shape[0], dtype=torch.bool,
                          device=features.device)

    # mid_transformer 跑 masked features
    if features.shape[0] > 0:
        features_for_pred = self._run_mid_transformer(
            features_for_pred, octree_gt, d, lvl)

    # split prediction
    logits = self.split_heads[lvl](features_for_pred)
    gt_split = (octree_gt.children[d] >= 0).long()

    # === Loss 仅在 mask 位置 ===
    if mask.any():
        total_split_loss = total_split_loss + F.cross_entropy(
            logits[mask], gt_split[mask])
        with torch.no_grad():
            total_split_acc += (
                logits[mask].argmax(-1) == gt_split[mask]
            ).float().mean().item()

    # === Expand 走原始 features（未 mask）===
    split_mask = (gt_split == 1)
    child_features = self._expand_features(features, split_mask, lvl)
    child_features = child_features + self._get_pos_embed(octree_gt, d + 1)
    features = child_features
```

**踩坑提示**：
- ❌ 不要用 `features_for_pred` 去 expand（会导致下一层 features 全是 mask_token 衍生）
- ❌ 不要在 mid_transformer 之前合并 mask 和 unmask 的 features（mask 必须经过 attention 才能从邻居推断信息）
- ✅ 推理时 mask=全 True 必须保证（不要漏掉这个分支）

**预期效果**：split_acc（mask-only metric）+0.10 ~ +0.15

### 2.2 建议 2：Buffer Tokens

**机制**：一组可学习的全局 token，prepend 到每个 mid_transformer 的输入序列开头。每个空间节点通过 attention 都能看到 buffer，buffer 之间也互相 attend，逐层分化为全局摘要。

**代码位置**：`fractal_models/fractal_generator.py`

**`__init__` 改动**：
```python
self.buffer_size = 32  # 可配置：16 / 32 / 64
self.buffer_emb = nn.Parameter(torch.zeros(self.buffer_size, feature_dim))
nn.init.normal_(self.buffer_emb, std=0.02)
```

**`_run_mid_transformer` 改动**：
```python
def _run_mid_transformer(self, features, octree, depth, lvl):
    B = octree.batch_size

    # Prepend buffer (每个样本一份)
    buffer = self.buffer_emb.unsqueeze(0).expand(B, -1, -1).reshape(
        -1, self.feature_dim)
    feat_with_buf = torch.cat([buffer, features], dim=0)

    # OctreeT 原生支持 buffer_size
    octreeT = OctreeT(
        octree, feat_with_buf.shape[0], self.patch_size, self.dilation,
        nempty=False, depth_list=[depth],
        buffer_size=self.buffer_size,  # ← 关键
        use_swin=self.use_swin)

    feat = depth2batch(feat_with_buf, octreeT.indices)
    feat = self.mid_transformers[lvl](feat, octreeT, context=None)
    feat = batch2depth(feat, octreeT.indices)

    # 剥掉 buffer
    return self.mid_norms[lvl](feat[B * self.buffer_size:])
```

**与 masked training 协同**：buffer 永远是 visible 的，不参与 mask。它是模型的"永远可用全局上下文"，masked 节点通过 attention 从 buffer 拿全局信息。

**预期效果**：split_acc +0.05 ~ +0.10（与建议 1 叠加可推到 0.90+）

### 2.3 建议 3：Per-level Blocks Redistribution

**机制**：把 mid_transformer 的 blocks 从 [1, 1, 1] 改为 [4, 2, 1]（粗层重、细层轻），符合 FractalGen Table 1 的容量分配哲学。

**代码位置**：`fractal_models/fractal_generator.py`

**`__init__` 改动**：
```python
# 原本
self.mid_transformers = nn.ModuleList([
    OctFormerStage(... num_blocks=1, ...) for _ in range(self.num_levels)
])

# 改成接收 list
mid_blocks = kwargs.get('mid_blocks_per_level', [4, 2, 1])
assert len(mid_blocks) == self.num_levels
self.mid_transformers = nn.ModuleList([
    OctFormerStage(
        dim=feature_dim, num_heads=num_heads,
        num_blocks=mid_blocks[i],  # ← per-level
        patch_size=patch_size, dilation=dilation, ...)
    for i in range(self.num_levels)
])
```

**Config 改动**（`configs/shapenet_frac_im5.yaml`）：
```yaml
MODEL:
  FractalGen:
    mid_blocks_per_level: [4, 2, 1]
```

**预期参数变化**：+8M（19.6M → 27.6M）

**预期效果**：split_acc +0.03 ~ +0.06

### 2.4 建议 4：Sibling Self-Attention（修复退化的 Expander）

**机制**：在 LocalCrossAttentionExpander 里，8 个 children query 之间互相 self-attention。便宜（每个 group 只 8 个 token）但补全了"attention" 的真正含义。

**代码位置**：`fractal_models/fractal_generator.py` 的 `LocalCrossAttentionExpander`

**`__init__` 改动**：
```python
# 新增 sibling self-attention
self.norm_sib = nn.LayerNorm(dim)
self.sibling_attn = nn.MultiheadAttention(
    embed_dim=dim, num_heads=num_heads, batch_first=True)
```

**`forward` 改动**：
```python
def forward(self, parent_features):
    N = parent_features.shape[0]
    if N == 0:
        return torch.zeros(0, self.dim, device=parent_features.device)

    context = parent_features.unsqueeze(1)
    q = self.octant_queries.unsqueeze(0).expand(N, -1, -1)
    q = q + self.octant_pos_emb().unsqueeze(0)

    # Cross-attention to parent
    q_normed = self.norm_q(q)
    kv_normed = self.norm_kv(context)
    attn_out, _ = self.cross_attn(q_normed, kv_normed, kv_normed)
    children = q + attn_out

    # === 新增：Sibling self-attention ===
    sib_normed = self.norm_sib(children)  # (N, 8, C)
    sib_out, _ = self.sibling_attn(sib_normed, sib_normed, sib_normed)
    children = children + sib_out

    # FFN
    children = children + self.ffn(self.norm_ffn(children))
    return children.reshape(N * 8, self.dim)
```

**预期效果**：mesh smoothness 明显改善（八个 octant 之间的接缝更平滑），split_acc +0.01 ~ +0.02

### 2.5 建议 5：去 Focal Loss

**机制**：把 `FocalLoss` 替换为标准 `F.cross_entropy`。配合 masked training 已经自动 focus 在困难位置，focal 多余且 α=0.75 在多类别下未经调优。

**代码位置**：`fractal_models/fractal_generator.py` 的 `forward()`

**改动**：
```python
# 原本
total_split_loss = total_split_loss + self.focal_loss(logits, gt_split)

# 改成（配合 masked training）
total_split_loss = total_split_loss + F.cross_entropy(logits[mask], gt_split[mask])
```

**注意**：只在做了建议 1（masked training）的前提下做这条。如果单独去掉 focal 而不加 mask，可能稍微变差。

### 2.6 建议 6：Metric 对齐

把 `split_accuracy` 改成 mask-only 计算，与 OctGPT 一致。Per-level 拆分报告。

**代码位置**：`fractal_models/fractal_generator.py` 的 `forward()`

**改动**：
```python
# Per-level accuracy
per_level_acc = []
for lvl in range(self.num_levels):
    # ... (在循环内累积)
    with torch.no_grad():
        acc_lvl = (logits[mask].argmax(-1) == gt_split[mask]).float().mean()
        per_level_acc.append(acc_lvl)
        output[f'split_acc_lvl{lvl}'] = acc_lvl

output['split_accuracy'] = torch.stack(per_level_acc).mean()
```

**输出多 3 个指标**：`split_acc_lvl0`, `split_acc_lvl1`, `split_acc_lvl2` — 帮助定位瓶颈在粗层还是细层。

### 2.7 不实施的方案（明确排除）

| 方案 | 排除理由 |
|------|---------|
| 加大 feature_dim 384→512 | ROI 低，参数翻倍但效益不如建议 1+2 |
| Leaf transformer 加深 | 当前已 6 blocks，不是瓶颈 |
| Encoder-decoder split（MAE 风格） | 建议 1+2 见效后再考虑，目前优先级低 |
| 端到端版本（去 VQ-VAE）| 另起 issue，本次只优化 VQ 版本 |
| Class conditioning 显式加入 | 先做 mask + buffer，看是否需要再加 |

---

## 3. 对照实验矩阵

### 3.1 实验组定义

每个改进作为一个独立 flag，互相 orthogonal 可组合：

| Flag | 含义 | Default |
|------|------|---------|
| `--use_masked_training` | 启用 masked training（建议 1） | False |
| `--use_buffer_tokens` | 启用 buffer tokens（建议 2） | False |
| `--mid_blocks_per_level` | per-level mid blocks（建议 3）| [1, 1, 1] |
| `--use_sibling_attn` | 启用 sibling self-attention（建议 4） | False |
| `--use_focal_loss` | Focal vs CE（建议 5） | True |
| `--mask_ratio_min` | masked training 的最小 mask ratio | 0.5 |
| `--buffer_size` | buffer token 数量 | 0 (off) |

### 3.2 实验列表

**Stage 1: Baseline + Metric 对齐（2 个实验）**

| ID | 名称 | 配置 | 目的 |
|----|------|------|------|
| E0 | baseline_old_metric | 当前 v4 配置 | 复现 0.78 |
| E0' | baseline_new_metric | 当前 + 仅改 metric 为 mask-only | 验证 metric 改动单独的影响 |

**Stage 2: 单变量消融（5 个实验）**

| ID | 名称 | 配置 | 假设 |
|----|------|------|------|
| E1 | mask_only | E0 + masked_training | 单独 mask 应该 +0.08+ |
| E2 | buffer_only | E0 + buffer_size=32 | 单独 buffer 应该 +0.03+ |
| E3 | blocks_only | E0 + mid_blocks=[4,2,1] | 单独加深应该 +0.03+ |
| E4 | sibling_only | E0 + sibling_attn | 主要影响 mesh quality |
| E5 | ce_only | E0 + 去掉 focal | 单独去 focal 可能微负 |

**Stage 3: 累加实验（找最优组合，4 个实验）**

| ID | 名称 | 配置 | 假设 |
|----|------|------|------|
| E6 | mask + buffer | E1 + buffer_size=32 | 应该 +0.13+，最高 ROI |
| E7 | mask + buffer + ce | E6 + 去 focal | 微调，可能 +0.01 |
| E8 | mask + buffer + blocks | E6 + mid_blocks=[4,2,1] | 看容量是否成为新瓶颈 |
| E9 | full_stack | E1+E2+E3+E4+E5 | 上限验证 |

**Stage 4: Hyper-parameter sweep（仅对最优组合，3-4 个实验）**

基于 E6/E7/E8/E9 选出最优者，做关键超参 sweep：

| ID | 名称 | 变量 | 候选值 |
|----|------|------|--------|
| H1 | mask_ratio_min sweep | mask_ratio_min | {0.3, 0.5, 0.7} |
| H2 | buffer_size sweep | buffer_size | {16, 32, 64} |
| H3 | mid_blocks sweep | mid_blocks_per_level | {[2,1,1], [4,2,1], [6,3,1]} |

### 3.3 实验执行规则

**统一变量**（所有实验保持一致）：
- Dataset: `shapenet_frac_im5.yaml`（im_5, 5 类别）
- Batch size, learning rate, optimizer, AMP 设置不变
- Random seed: 42
- Epochs: 20（足以观察 plateau 行为；选出最优组合后再跑 100+ epoch full run）
- 同一台机器、同一张卡（避免硬件浮点差异）

**Metrics 收集**（每实验必报）：
- `split_acc_total`（mask-only，新 metric）
- `split_acc_lvl0/lvl1/lvl2`（per-level）
- `vq_accuracy`
- `split_loss`, `vq_loss`
- 训练时间（sec/epoch）
- 显存占用（peak GB）
- **生成质量指标**（仅对 E6/E7/E8/E9 + Stage 4）：
  - 生成 100 个样本，计算 FID（用 `metrics/calc_fid.py`）
  - 推理时间/样本（确认 4-forward 速度保留）

**日志规范**：
- 每实验单独 `logs/exp_E{id}_{name}/` 目录
- 保存：tensorboard、stdout、最终 ckpt、配置 yaml 副本
- 每 epoch 末把上述 metrics 写入 `logs/exp_E{id}_{name}/metrics.json`

### 3.4 决策准则

**Stage 2 后**：若某单一改动效果小于 +0.01，剔除该方案不进入 Stage 3。

**Stage 3 后**：选 split_acc 最高且训练时间增加不超 3× baseline 的组合。如果两个组合 split_acc 接近（差 <0.005）但训练时间差异大，优先选快的。

**Stage 4 后**：选 split_acc 最高的超参组合。如果差异在 noise 范围内（多次跑差异 <0.005），选 default 值。

---

## 4. 版本管理规约

### 4.1 Branch 策略

```
main                                  # 永远保持当前 v4 稳定状态
├── exp/baseline-metric-fix           # E0' (轻量改动)
├── exp/masked-training               # E1
├── exp/buffer-tokens                 # E2
├── exp/blocks-redistribute           # E3
├── exp/sibling-attention             # E4
├── exp/ce-loss                       # E5
├── exp/mask-buffer                   # E6 (基于 E1 + E2)
├── exp/mask-buffer-ce                # E7
├── exp/mask-buffer-blocks            # E8
├── exp/full-stack                    # E9
└── feat/v5-final                     # 实验完毕后的最终整合分支
```

### 4.2 Commit 规约

每个 commit message 用 prefix：

| Prefix | 含义 | 举例 |
|--------|------|------|
| `feat:` | 新功能/改进 | `feat(mask): add masked training to fractal forward` |
| `fix:` | bug 修复 | `fix(mask): expansion path should use unmasked features` |
| `exp:` | 实验运行结果 | `exp(E1): split_acc 0.86 on im_5 after 20 epochs` |
| `cfg:` | 配置变更 | `cfg(im5): add mask_ratio_min=0.5 to im_5 config` |
| `doc:` | 文档/注释 | `doc: explain why expand uses unmasked features` |
| `refactor:` | 代码重构（无功能变化） | `refactor: extract masking logic to helper method` |

每个实验分支至少包含：
1. 一个 `feat:` commit 实现核心改动
2. 一个 `cfg:` commit 添加对应 config 文件
3. 一个 `exp:` commit 提交训练 log 和 metrics（用 git-lfs 或单独的 `experiments/` 目录）

### 4.3 配置文件管理

为每个实验创建独立 config：

```
configs/
├── shapenet_frac_im5.yaml             # baseline（不动）
├── exp/
│   ├── E0_baseline_old_metric.yaml
│   ├── E0p_baseline_new_metric.yaml
│   ├── E1_mask_only.yaml
│   ├── E2_buffer_only.yaml
│   ├── E3_blocks_only.yaml
│   ├── E4_sibling_only.yaml
│   ├── E5_ce_only.yaml
│   ├── E6_mask_buffer.yaml
│   ├── E7_mask_buffer_ce.yaml
│   ├── E8_mask_buffer_blocks.yaml
│   ├── E9_full_stack.yaml
│   ├── H1_mask_ratio_sweep_03.yaml
│   ├── H1_mask_ratio_sweep_07.yaml
│   └── ...
```

每个 config 使用 BASE 机制继承 baseline，只 override 必要字段：

```yaml
# configs/exp/E1_mask_only.yaml
BASE:
  - configs/shapenet_frac_im5.yaml

SOLVER:
  logdir: logs/exp_E1_mask_only

MODEL:
  FractalGen:
    use_masked_training: True
    mask_ratio_min: 0.5
```

### 4.4 Code-level Flag 设计

避免每个改动都创建新的 model class。在 `FractalGenerator.__init__` 里全部用 flag 控制：

```python
class FractalGenerator(nn.Module):
    def __init__(
        self,
        feature_dim: int = 384,
        # ... existing args ...
        # === Experiment flags ===
        use_masked_training: bool = False,
        mask_ratio_min: float = 0.5,
        buffer_size: int = 0,  # 0 = disabled
        mid_blocks_per_level: list = None,  # None = [1,1,1]
        use_sibling_attn: bool = False,
        use_focal_loss: bool = True,
        **kwargs,
    ):
        ...
        self.use_masked_training = use_masked_training
        self.buffer_size = buffer_size
        ...
        if buffer_size > 0:
            self.buffer_emb = nn.Parameter(...)

        if use_masked_training:
            self.mask_emb = nn.Parameter(...)
            self.mask_ratio_generator = stats.truncnorm(...)

        mid_blocks = mid_blocks_per_level or [1] * self.num_levels
        ...
```

**好处**：所有实验跑同一份代码、同一个 class，只换 config。可复现性最高，且 baseline 永远是 `flags=all_default`。

### 4.5 Reproducibility 清单

每个 commit 必须满足：
- [ ] `python -c "import torch; print(torch.__version__)"` 输出记录在 `experiments/E{id}/env.txt`
- [ ] `pip freeze > experiments/E{id}/requirements.txt`
- [ ] `nvidia-smi > experiments/E{id}/gpu.txt`
- [ ] Git SHA 记录在 `experiments/E{id}/git_sha.txt`
- [ ] Config 文件 copy 到 `experiments/E{id}/config.yaml`

可以写个 `scripts/start_exp.sh`：
```bash
#!/bin/bash
EID=$1
NAME=$2
DIR=experiments/E${EID}_${NAME}
mkdir -p $DIR
git rev-parse HEAD > $DIR/git_sha.txt
pip freeze > $DIR/requirements.txt
nvidia-smi > $DIR/gpu.txt
cp configs/exp/E${EID}_${NAME}.yaml $DIR/config.yaml
python main_fractal.py --config $DIR/config.yaml 2>&1 | tee $DIR/train.log
```

---

## 5. 实施 Checklist

### 5.1 准备阶段

- [ ] 在 `main` 上 tag 当前版本：`git tag v4-baseline`
- [ ] 创建实验追踪文件 `experiments/RESULTS.md`（表格形式记录每个实验结果）
- [ ] 写 `scripts/start_exp.sh` 启动脚本
- [ ] 验证 `metrics/calc_fid.py` 能跑通（FID 评估管线就绪）

### 5.2 代码改动阶段（每个改动单独分支）

按依赖顺序：

**Phase 1：Metric 对齐（必须先做）**
- [ ] Branch: `exp/baseline-metric-fix`
- [ ] 修改 `forward()` 报告 mask-only accuracy + per-level breakdown
- [ ] Config: `E0p_baseline_new_metric.yaml`
- [ ] 跑 E0 (旧 metric) + E0' (新 metric)，确认 metric 改动不引入功能差异

**Phase 2：单变量改动（互相 orthogonal，可并行）**
- [ ] Branch: `exp/masked-training` — 实现建议 1
- [ ] Branch: `exp/buffer-tokens` — 实现建议 2
- [ ] Branch: `exp/blocks-redistribute` — 实现建议 3
- [ ] Branch: `exp/sibling-attention` — 实现建议 4
- [ ] Branch: `exp/ce-loss` — 实现建议 5

每个分支：实现 + config + 跑 E1-E5

**Phase 3：合并分支跑组合实验**
- [ ] Merge `exp/masked-training` + `exp/buffer-tokens` → `exp/mask-buffer`
- [ ] 跑 E6
- [ ] 基于 E6 创建 E7, E8, E9 分支
- [ ] 跑 E7, E8, E9

**Phase 4：超参 sweep（基于最优组合）**
- [ ] 选出 Stage 3 最优组合，创建 `exp/sweep-{param}` 分支
- [ ] 跑 H1, H2, H3

**Phase 5：最终整合**
- [ ] 创建 `feat/v5-final` 分支
- [ ] Merge 最优配置
- [ ] 跑长 epoch（100+）full run
- [ ] 生成 FID 比较表
- [ ] 更新 README 和 default config

### 5.3 实验执行阶段

对每个实验 E{id}：
- [ ] `bash scripts/start_exp.sh {id} {name}`
- [ ] 监控前 3 epoch 确认 loss 下降（如果 NaN 或不收敛，立即停掉调试）
- [ ] 训练完成后把 final metrics 填入 `experiments/RESULTS.md`
- [ ] Commit log: `exp(E{id}): {summary}`
- [ ] 评审：是否进入下一 Stage

---

## 6. 验收标准

### 6.1 必达指标

最终选定的 v5 版本必须满足：

| 指标 | 目标 | Baseline (v4) | 验证方法 |
|------|------|-------|---------|
| split_acc (mask-only) | ≥ 0.90 | 0.78 (all-pos) / TBD (mask-only) | 训练完最后 epoch |
| 推理时间/样本 | ≤ 1.0× baseline (~716ms) | 716ms | `metrics/` 单样本计时 |
| 推理 forward 次数 | = 4 | 4 | 代码 inspection |
| 训练时间/epoch | ≤ 4× baseline | TBD | 实验记录 |
| FID (vs OctGPT) | 不差过 OctGPT 20% | TBD | `metrics/calc_fid.py` |

### 6.2 可接受妥协

- 训练时间增加到 3× 可接受（4× 是上限）
- 显存增加到 2× 可接受（按需调整 batch_size）
- 参数量增加到 40M 可接受（仍远小于 OctGPT 170M）

### 6.3 报告交付

实验结束后产出：

1. **`experiments/RESULTS.md`**：所有实验的 metrics 表格
2. **`experiments/ANALYSIS.md`**：基于结果的分析，包括：
   - 哪些改动有效、哪些无效
   - 哪个改动 ROI 最高
   - 推理速度是否完整保留
   - 与 OctGPT 在 mask-only metric 下的真实差距
3. **`feat/v5-final` 分支**：可直接用于发表/部署的最终代码
4. **新的 `configs/shapenet_frac_im5_v5.yaml`**：最优配置
5. **更新的 README**：说明 v5 的新功能

### 6.4 如果没达到目标

如果 split_acc 在 mask-only metric 下仍未到 0.90：

1. 检查 metric 是否真的对齐（确认 OctGPT 也是只在 mask 位置算）
2. 检查 buffer_size 是否够大（试 64）
3. 检查 mid_blocks 是否够深（试 [6,3,1]）
4. 考虑引入 encoder-decoder split（MAE 风格，参考 OctGPT `_init_blocks`）
5. 最后才考虑加 feature_dim 384→512

---

## 7. 给 Claude Code 的执行提示

**优先级**：Phase 1 → Phase 2 (并行) → Phase 3 → Phase 4 → Phase 5

**踩坑预警**：
- Masked training 的 expand 路径必须用未 mask 的 features，否则下一层 features 全是 mask_token 衍生
- 推理时 mask=全 True 这个 case 必须测试到，不然推理路径会出 bug
- Buffer 在 `OctreeT` 里要传 `buffer_size` 参数，attention mask 由 `OctreeT` 自动处理；不要自己手搓 mask
- 改 `mid_blocks_per_level` 时 config 的 list 长度必须等于 `num_levels = depth_stop - full_depth`
- `LocalCrossAttentionExpander` 的 sibling attention 加在 cross-attn 之后、FFN 之前

**调试建议**：
- 任何 forward 改动后，先用 `_test_import.py`（项目里已有）做 smoke test
- masked training 实现完后，先用 batch=1, mask_ratio=0 (无 mask) 跑通一次确认数值与 baseline 一致，再开 mask
- 推理路径单独写测试：mask=全 True 时 forward 不报错、shapes 对得上

**Commit 节奏**：
- 每个 sub-feature 一个 commit（不要把 mask + buffer + blocks 打到一个 commit）
- 每跑完一个实验就 commit metrics（不要堆到最后）
- 大改动前 tag 一下方便回滚

---

**完。开始执行吧。**
