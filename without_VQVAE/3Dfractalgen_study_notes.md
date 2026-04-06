# 3Dfractalgen 代码学习笔记（v3 — 去掉中间层 Transformer，只保留叶子层）

## 1. 项目定位

这个项目的核心目标是：**在八叉树表示上做端到端的三维形状生成，不依赖 VQ-VAE，全程连续特征。**

它结合了两篇工作的思路：

- **OctGPT**（组内工作）：多深度八叉树 Transformer，BFS 并行逐层生成
- **FractalGen / MAR**（何恺明团队）：连续特征生成 + per-token Diffusion Loss，绕过离散码本

用一句话概括：

> 在 OctGPT 的八叉树 Transformer 骨架上，用 MAR 式的 Diffusion Loss 替代离散 token 路线，实现"连续特征 + 分形展开 + 端到端训练"的三维生成流程。

## 2. 目录结构

```
3Dfractalgen/
├── main_fractal.py                      # 训练/测试/生成入口
├── configs/shapenet_fractal.yaml        # 实验配置
├── fractal_models/
│   └── fractal_generator.py             # 核心模型（所有创新点）
│
├── octgpt/                              # 复用的底座（只读依赖）
│   ├── models/octformer.py              #   OctFormerStage, OctreeT
│   ├── models/positional_embedding.py   #   SinPosEmb, AbsPosEmb, RMSNorm
│   ├── datasets/shapenet.py             #   ShapeNet 数据加载
│   └── utils/                           #   builder, marching cubes, depth2batch
│
└── (外部) ../mar/                        # MAR 代码库（运行时导入）
    └── diffusion/                       #   GaussianDiffusion, SpacedDiffusion
        ├── __init__.py                  #   create_diffusion()
        ├── gaussian_diffusion.py        #   训练 loss + 采样循环
        └── respace.py                   #   时间步重采样
```

理解方式：**三层架构**

| 层级 | 内容 | 角色 |
|------|------|------|
| 实验入口 | `main_fractal.py` + yaml 配置 | 串联训练/生成流程 |
| 核心创新 | `fractal_generator.py` | 全部模型设计和 4 项任务升级 |
| 底座依赖 | `octgpt/` + `mar/diffusion/` | 八叉树 Transformer + 扩散基础设施 |

## 3. 数据需求

### 3.1 数据格式

本项目 **直接复用 OctGPT 的 ShapeNet 数据处理流程**，不需要额外预处理。

**原始数据**：ShapeNetCore.v1（31GB），每个模型一个 `model.obj`

**预处理后的结构**（由 OctGPT 的 `tools/sample_sdf.py` 生成）：

```
data/ShapeNet/
├── dataset_256/
│   └── {synth_id}/{model_id}/
│       ├── pointcloud.npz     # points (N,3) + normals (N,3), float16
│       └── sdf.npz            # points (M,3) + sdf (M,) + grad (M,3), float16
├── filelist/
│   ├── train_airplane.txt     # 每行一个 model path
│   └── test_airplane.txt
```

**预处理参数**：
- 网格归一化到 `[-0.5, 0.5]`，缩放因子 0.8
- SDF 在 256³ 体素网格上计算（`mesh2sdf` 库）
- 在八叉树节点（深度 3-8）上采样，每个模型最多 40 万点
- 截断距离 `tsdf = 0.05`

### 3.2 训练时数据流

`octgpt/datasets/shapenet.py` 的 `TransformShape` 在每次采样时：

1. 从 `sdf.npz` 采 **10,000 个体采样点**（内外部 SDF 值）
2. 从 `pointcloud.npz` 采 **10,000 个面采样点**（SDF=0，法向作为 grad）
3. 坐标从 `[-0.5, 0.5]` 缩放到 `[-1, 1]`（`points_scale=0.5`）
4. 由点云构建 ground-truth 八叉树 `octree_gt`

**collate 后的 batch 结构**：

| 字段 | 形状 | 含义 |
|------|------|------|
| `octree_gt` | Octree 对象 | GT 八叉树结构（包含每层节点数、children 等） |
| `pos` | (M, 4) | SDF 查询点 `[x, y, z, batch_idx]` |
| `sdf` | (M,) | 对应 SDF 真值 |
| `grad` | (M, 3) | SDF 梯度/法向（当前预留未使用） |

### 3.3 如果你要跑实验

只需要按 OctGPT 的 README 下载 ShapeNet 并跑 `tools/sample_sdf.py` 即可。配置文件 `shapenet_fractal.yaml` 里的路径：

```yaml
location: data/ShapeNet/dataset_256
filelist: data/ShapeNet/filelist/train_airplane.txt
```

确保这两个路径指向正确位置。默认配置是 airplane 单类别，如果想多类别可以换成 `train_im_5.txt`。

## 4. 模型总览：`fractal_generator.py` 中的模块

当前版本包含以下模块（按在文件中出现的顺序）：

### 4.1 MAR Diffusion 基础设施（Task 2）

从 MAR 代码库忠实移植的去噪网络：

| 类 | 作用 |
|----|------|
| `_TimestepEmbedder` | 正弦时间步嵌入 → 2 层 MLP |
| `_ResBlock` | 残差块 + **3-way AdaLN**（shift, scale, **gate**） |
| `_FinalLayer` | 最终层 + 2-way AdaLN（shift, scale） |
| `SimpleMLPAdaLN` | 去噪 MLP 主体：`input_proj → N×ResBlock → FinalLayer` |
| `DiffLoss` | 扩散损失封装：训练用 1000 步 cosine schedule，推理用 respaced steps |

**关键设计**：
- 输出通道数 = `in_channels × 2`（LEARNED_RANGE 方差，不是固定方差）
- AdaLN 层全部 zero-init（DiT 论文的稳定训练技巧）
- 使用 MAR 原版的 `create_diffusion()` 工厂函数创建 `GaussianDiffusion` / `SpacedDiffusion`

### 4.2 局部交叉注意力扩展器（Task 1）

| 类 | 作用 |
|----|------|
| `OctantPositionEmbedding` | 8 个八分体的 3D 固定位置 → 可学习投影 |
| `LocalCrossAttentionExpander` | **替代原 MLP** 的特征扩展模块 |

**LocalCrossAttentionExpander 的核心逻辑**：

```
输入: parent_features (N, C)
  → Context (K/V): 每个父节点 (N, 1, C)
  → Query: 8 个可学习八分体 token + 3D 位置嵌入 (N, 8, C)
  → 局部交叉注意力: 每个父节点独立 attention [N_parents, 8, C]
  → FFN 精炼
输出: (N×8, C)
```

**为什么是"局部"**：attention 在每个父节点内部独立计算（batch 维度是 N_parents），**不是** 8N 个子节点之间的全局注意力。显存安全。

**预留扩展接口**：当 context 从 `(N, 1, C)` 变成 `(N, K, C)`（加入邻居父节点），K/V 长度自然改变，无需修改架构。

### 4.3 SDF 解码器

| 类 | 作用 |
|----|------|
| `LocalImplicitDecoder` | 4 层 MLP：`(feature, local_xyz) → SDF 标量` |

输入是叶节点特征 + 查询点在该体素内的局部坐标（归一化到 `[-1, 1]`）。

### 4.4 主模型 `FractalGenerator`

核心超参：

| 参数 | 默认值 | 含义 |
|------|--------|------|
| `feature_dim` | 384 | 全局特征维度 |
| `full_depth` | 3 | 起始八叉树深度（8³=512 个节点） |
| `depth_stop` | 6 | 终止深度（64³ 分辨率） |
| `num_levels` | 3 | 扩展层数 = depth_stop - full_depth |
| `blocks_per_level` | 6 | 每层 OctFormer 的 block 数 |
| `diffusion_weight` | 0.1 | Diffusion Loss 权重 |
| `diffusion_mlp_width` | 512 | DiffLoss 去噪 MLP 宽度 |
| `diffusion_inference_steps` | 100 | 推理时扩散采样步数 |

**内部组件**：
- `root_embedding`：`(1, C)` 可学习参数，复制到 full_depth 所有节点
- `leaf_transformer`：**仅 1 个** `OctFormerStage`（只在叶子层 depth_stop 使用）
- `leaf_norm`：叶子层 LayerNorm
- `split_heads`：`num_levels` 个线性头（2 分类：split / not split）
- `feature_expanders`：`num_levels` 个 `LocalCrossAttentionExpander`
- `child_pos_emb`：`(8, C)` 共享八分体位置嵌入
- `diffusion_loss`：`DiffLoss` 模块
- `sdf_decoder`：`LocalImplicitDecoder`

## 5. 训练前向传播详解

### 步骤 1：初始化粗层特征

```python
features = self.root_embedding.expand(octree_gt.nnum[full_depth], -1)
```

所有 full_depth 节点共享同一个初始向量。空间差异完全来自后续 OctFormer 的位置编码和八叉树结构注意力。

### 步骤 2：逐层分形展开（循环 `num_levels=3` 次）

每层做 4 件事（无 Transformer，纯靠 cross-attention 扩展传递语义）：

**① 预测 split**
```python
logits = split_heads[lvl](features)          # (N, 2)
gt_split = (octree_gt.children[d] >= 0).long()  # GT: 哪些节点有子节点
loss += cross_entropy(logits, gt_split)
```

**② Scheduled Sampling 选择 split mask（Task 3）**
```python
split_mask = self._scheduled_sampling_mask(gt_split, logits, prob)
```
- `prob=0`（训练早期）：纯 teacher forcing，用 GT mask
- `prob>0`（训练后期）：按概率混合 GT 和模型自己的预测
- 线性 warmup：默认在 50,000 步内从 0 升到 0.5

**③ 局部交叉注意力扩展特征（Task 1）**
```python
child_features = self._expand_features(features, split_mask, lvl)
```
只对 split 的节点做扩展，每个父节点通过 LocalCrossAttentionExpander 生成 8 个子节点特征，再加 `child_pos_emb`。

**④ Diffusion Loss（Task 2）**

在纯 teacher forcing 模式下（`prob=0`），对子节点特征计算 MAR 式扩散损失：
```python
d_loss = self.diffusion_loss(target=child_features, z=child_features.detach())
```
DiffLoss 内部：随机采 timestep → 加噪 → SimpleMLPAdaLN 预测噪声 → MSE loss。

### 步骤 3：叶子层 Transformer 精炼（唯一的 OctFormer）

到达 `depth_stop` 后，运行 **唯一的一个 OctFormerStage**（`leaf_transformer`）。
这是全模型中唯一做空间自注意力的地方，专门为 SDF 解码整理叶节点特征。
中间层不需要全局注意力 — LocalCrossAttentionExpander 已经足够传递语义。

### 步骤 4：SDF Loss

```python
sdf_loss = self._sdf_loss(features, octree_gt, depth_stop, pos, sdf)
```

通过空间哈希（node key = batch×S³ + x×S² + y×S + z）把查询点匹配到叶节点，提取局部坐标，MLP 预测 SDF 值，L1 loss 监督。

### 总 Loss

```python
loss = split_weight × split_loss + sdf_weight × sdf_loss + diffusion_weight × diffusion_loss
```

默认权重：split=1.0, sdf=1.0, diffusion=0.1。

## 6. 生成流程详解

### 6.1 模型生成（`FractalGenerator.generate()`）

和训练同样的逐层结构，但有两个本质区别：

**区别 1：不再 teacher forcing，自己采样结构**

```python
# 采样 split 决策
split = multinomial(softmax(logits / temperature), 1)
# 真的生长八叉树
octree.octree_split(split, d)
octree.octree_grow(d + 1)
```

**区别 2：通过反向扩散采样生成多样特征（Task 2）**

```python
# 交叉注意力扩展后，用扩散采样"刷新"特征
features = self.diffusion_loss.sample(z=features, temperature=diffusion_temperature)
```

`DiffLoss.sample()` 内部执行完整的反向扩散循环（`p_sample_loop`），从高斯噪声出发，以当前特征为条件，逐步去噪。respaced 到 100 步（可配置）。

### 6.2 稀疏 Marching Cubes（Task 4, `main_fractal.py`）

传统方案：查询 256³ = 1600 万个网格点的 SDF → 极慢。

**Task 4 的优化**：只在八叉树叶节点覆盖的区域查询 SDF。

```
1. model.get_leaf_bboxes() → 获取所有叶节点的 AABB
2. 把 bbox 映射到网格坐标，加 padding（1-2 格）
3. 构建布尔占用掩码 occ_mask (size³)
4. 只在 occ_mask=True 的格点上查询 SDF
5. 空区域填默认值 0.1（外部）
6. marching_cubes → trimesh → 导出 .obj
```

典型稀疏率：只需查询 5-20% 的网格点。

## 7. 四项核心升级总结

### Task 1：LocalCrossAttentionExpander

| 项 | 旧版 | 新版 |
|----|------|------|
| 模块 | MLP (`Linear → GELU → Linear`) | 局部交叉注意力 |
| 输入 | 父节点 (N, C) | 同左 |
| 机制 | 直接映射到 8C 再 reshape | 8 个可学习 query attend to 父节点 |
| 空间编码 | 共享 `child_pos_emb` | 同左 + OctantPositionEmbedding |
| 显存 | O(N×8C) | O(N×8×C)，局部 attention 不全局化 |
| 扩展性 | 无 | 天然支持邻居上下文 (N, K, C) |

### Task 2：DiffLoss（MAR 忠实移植）

| 项 | 说明 |
|----|------|
| 训练 | 1000 步 cosine schedule，ε-prediction，LEARNED_RANGE 方差 |
| 推理 | SpacedDiffusion respaced 到 100 步，支持 temperature |
| 去噪网络 | SimpleMLPAdaLN：3 层 ResBlock + 3-way AdaLN (shift/scale/gate) |
| 初始化 | Xavier + AdaLN 零初始化 + timestep embed 小方差初始化 |
| 来源 | 直接使用 `mar/diffusion/` 的 `create_diffusion()` |

### Task 3：Scheduled Sampling

| 项 | 说明 |
|----|------|
| 目的 | 缓解 teacher forcing 导致的 exposure bias |
| 策略 | 每个节点独立随机选用 GT 或模型预测的 split mask |
| 调度 | 线性 warmup：0 → 0.5，over 50,000 steps |
| 细节 | 预测路径 detach，不通过 argmax 反传梯度 |
| 配置 | `SOLVER.ss_max_prob` / `SOLVER.ss_warmup_steps` |

### Task 4：稀疏 Marching Cubes

| 项 | 说明 |
|----|------|
| 目的 | 避免查询 256³ 全网格 SDF |
| 方法 | 只在八叉树叶节点 bbox 覆盖区域采样 |
| 实现 | `get_leaf_bboxes()` + 网格占用掩码 + 稀疏 SDF 查询 |
| 加速 | 典型减少 80-95% 的 SDF 查询量 |
| 兜底 | 叶节点 >10,000 时切换向量化构建路径 |

## 8. 配置文件关键参数

`configs/shapenet_fractal.yaml`：

```yaml
SOLVER:
  max_epoch: 400
  lr: 0.0001              # AdamW
  use_amp: True            # 混合精度
  resolution: 256          # Marching Cubes 分辨率
  sdf_scale: 0.9           # MC 查询范围 [-0.9, 0.9]

DATA:
  depth: 8                 # 八叉树最大深度
  full_depth: 3            # 起始深度
  points_scale: 0.5        # 坐标缩放
  volume_sample_num: 10000 # 每模型体采样数
  surface_sample_num: 10000
  tsdf: 0.05               # SDF 截断
  location: data/ShapeNet/dataset_256
  filelist: data/ShapeNet/filelist/train_airplane.txt
  load_sdf: True           # 加载 SDF（必须 True）
  batch_size: 1

MODEL.FractalGen:
  feature_dim: 384         # 全局特征维度 C
  num_heads: 8             # OctFormer attention heads
  blocks_per_level: 6      # 每层 Transformer blocks
  full_depth: 3            # = DATA.full_depth
  depth_stop: 6            # 最终生成深度
  patch_size: 2048         # OctFormer patch size
  dilation: 2              # OctFormer dilation
  sdf_weight: 1.0
  split_weight: 1.0
  # diffusion_weight/mlp_width/mlp_depth/inference_steps 使用代码默认值
```

**注意**：`depth_stop=6` 意味着模型生成到 64³ 分辨率的八叉树。如果想要更精细（如 128³），改为 `depth_stop=7`。

## 9. 与 OctGPT 的核心区别

| 维度 | OctGPT | 3Dfractalgen |
|------|--------|--------------|
| 特征类型 | 离散 token（VQ-VAE 压缩） | 连续向量（无码本） |
| 训练方式 | 两阶段：先训 VQ-VAE，再训 GPT | 端到端一阶段 |
| 生成分布 | 自回归 next-token 概率 | Diffusion Loss 建模连续分布 |
| 特征扩展 | 不适用（token 序列） | 局部交叉注意力（Task 1） |
| 训练技巧 | 标准 teacher forcing | Scheduled Sampling（Task 3） |
| Mesh 提取 | 全网格 MC | 稀疏 MC（Task 4） |

**优势**：
- 无量化误差，几何细节理论上更好
- 端到端训练更简洁
- Diffusion Loss 避免连续特征坍缩到均值

**代价**：
- 连续特征在大规模数据上组织高层语义的能力有待验证
- 依赖 Diffusion Loss 的额外计算开销
- 训练稳定性更敏感（需要 AMP、Scheduled Sampling 等技巧）

## 10. 已知问题和注意事项

### 10.1 ~~`forward()` 中 DiffLoss 的调用方式~~ ✅ 已修复

已统一为 `self.diffusion_loss(target=child_features, z=child_features.detach())`。

### 10.2 `grad` 未被使用

数据集返回 `grad`（SDF 梯度/法向），`forward()` 也接收但未进入 loss。可作为后续扩展点（eikonal 约束、法向一致性 loss）。

### 10.3 `sys.path` 注入

`main_fractal.py` 和 `fractal_generator.py` 都通过手动改 `sys.path` 导入 octgpt 和 mar。研究原型可以接受，但不利于迁移和协作。

### 10.4 `test_epoch()` 硬编码

配置 `test_every_epoch: 1`，但 `test_epoch()` 内部 `epoch % 5 != 0` 才执行测试。实际行为与配置不一致。

## 11. 推荐阅读顺序

1. **`configs/shapenet_fractal.yaml`** — 了解实验参数
2. **`main_fractal.py`** — 看训练/生成如何串联，重点看 `model_forward()`、`generate_step()`
3. **`fractal_generator.py` 的 `forward()`** — 核心训练流程：init → transformer → split → expand → diffusion loss → SDF loss
4. **`fractal_generator.py` 的 `generate()`** — 推理流程：自主采样结构 + 扩散采样特征
5. **`DiffLoss` + `SimpleMLPAdaLN`** — 理解 MAR 式扩散损失的去噪网络和训练/采样接口
6. **`LocalCrossAttentionExpander`** — 理解交叉注意力如何替代 MLP 做特征扩展
7. **`octgpt/datasets/shapenet.py`** — 数据从 npz 到 batch 的完整流程
8. **`octgpt/models/octformer.py`** — OctreeT 构造、patch attention、dilation/swin 机制

## 12. 后续可做的方向

### 方向 1：纳入 `grad` 作为额外监督
- Eikonal regularization：`‖∇SDF‖ = 1`
- 法向一致性：预测法向 vs GT 法向的 cosine loss
- 可显著提升表面局部几何质量

### 方向 2：引入全局 latent code
- 当前 root_embedding 是固定可学习参数，多样性全靠 split 采样和 diffusion
- 加入显式 latent（如 VAE encoder 生成 z → 条件化 root）可更直接地控制形状多样性

### 方向 3：扩展到条件生成
- 图像条件：sketch/渲染图 → cross-attention 注入每层 Transformer
- 文本条件：CLIP embedding → 调制 root_embedding 或作为额外 KV
- 类别条件：category embedding 加到 root

### 方向 4：提升 depth_stop
- 当前 `depth_stop=6`（64³），几何细节有限
- 提升到 7（128³）或 8（256³）需要更多显存 → 可考虑 gradient checkpointing + 更小的 patch_size
