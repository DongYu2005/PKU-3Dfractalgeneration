# VQ-VAE 分形生成路线汇报分析稿

## 1. 任务目标

本路线希望在 OctGPT 的八叉树/VQ-VAE 表示基础上，探索一种更快的 coarse-to-fine 分形生成器。模型不采用 576 步左右的迭代自回归解码，而是从 `full_depth=3` 开始逐层预测 split，到 `depth_stop=6` 后预测叶子层 VQ token，再交给冻结 VQ-VAE 解码为 SDF 并通过 marching cubes 导出 OBJ。

核心目标可以概括为：

- 保留 OctGPT/VQ-VAE 的网格解码能力；
- 将生成过程压缩为少量逐层 forward；
- 分析这种单次/少步结构预测在三维生成质量上的瓶颈。

## 2. 方法设计

模型包含三部分：

- **split 结构预测**：每一层判断八叉树节点是否继续分裂，决定最终网格的大体拓扑。
- **父到子特征展开**：每个父节点通过 expander 生成 8 个子节点特征，并加入 octant position embedding。
- **叶子 VQ token 预测**：在 depth 6 的叶子节点上预测 frozen VQ-VAE 的 binary quantized token，随后由 VQ-VAE 解码为连续 SDF。

围绕多类别生成质量，我做了几类 ablation：

- **class condition / cond every level**：解决无条件模型生成趋同问题。
- **latent z**：用 pooled GT VQ code 学一个 VAE-style latent，生成时从正态分布采样，提高类内变化。
- **模型扩容 Q2**：将 feature dim 扩到 768，blocks 加深到接近 OctGPT 参数量。
- **split MaskGIT Q3**：尝试用少步迭代方式改善 split 结构。
- **sibling attention Q4**：修复 expander 中 parent-to-children cross-attention 在 K=1 时退化为 MLP 的问题。
- **leaf VQ MaskGIT S1**：只在叶子 VQ token 上做迭代式 mask refinement，尝试提升表面细节。

## 3. 单飞机过拟合结果

单飞机过拟合用于证明 pipeline 本身是可学习、可导出 OBJ 的。已有 baseline：

| 设置 | split acc | VQ acc | loss | 结果 |
|---|---:|---:|---:|---|
| `overfit_thresh050` epoch 300 | 0.998 | 0.847 | 0.319 | 已生成 `logs/fractal/overfit_thresh050/results/300.obj` |

这个结果说明：在单一飞机样本上，split 结构几乎可以完全拟合，VQ token 也能明显高于多类别实验。因此模型框架、VQ-VAE 接口、OBJ 导出流程是通的。

当前为了展示，我补跑两个 warm-start overfit 版本：

| 展示实验 | 配置 | 目的 |
|---|---|---|
| `O1_air_struct_best` | buffer + `[4,2,1]` mid blocks + sibling attention | 展示结构侧改进在单飞机上的效果 |
| `O2_air_surface_best` | O1 + leaf VQ MaskGIT | 展示叶子 token refinement 对表面细节的作用 |

它们从 `overfit_thresh050/checkpoints/00300.model.pth` warm-start，预计结果会写到：

- `logs/exp/O1_air_struct_best/results/`
- `logs/exp/O2_air_surface_best/results/`

汇报时可以把 baseline 的 `300.obj` 与 O1/O2 的最终 OBJ 放在同一页：baseline 证明可过拟合，O1/O2 展示结构和表面细节方向。

## 4. 多类别实验结果

多类别结果需要如实讲：单次/少步分形生成器可以生成不同类别，但与 OctGPT 的迭代式生成质量仍有明显差距，主要瓶颈在 split 结构。

| 实验 | 主要改动 | split acc | depth-6 前最后一级 acc | VQ acc | 观察 |
|---|---|---:|---:|---:|---|
| C2 | 类别条件每层注入 | 0.812 | 0.756 | 0.652 | 类别区分有效，但结构精度不足 |
| V1 | C2 + latent z | 0.836 | 0.780 | 0.647 | 类内变化更好，但 VQ 无明显提升 |
| Q2 | 768 维大模型 + latent | 0.840 | 0.788 | 0.646 | 扩容提升有限，最后一级仍卡住 |
| Q3 | Q2 + split MaskGIT | 0.836 | 0.776 | 0.644 | 没有带来结构提升，生成更稀疏/碎 |
| Q4 | Q2 + sibling attention | 0.841-0.845 | 0.795-0.800 | 0.645-0.647 | 当前结构指标最好，但仍未突破 |
| S1 | V1 + leaf VQ MaskGIT | 0.840 | 0.791 | 0.639 | nearest-CD 略好，但配置是 384 维，不能直接和 Q4 严格对照 |

已有 nearest-CD 粗评估：

| 实验 | 样本数 | nearest-CD mean | 说明 |
|---|---:|---:|---|
| C2 | 10 | 0.018874 | 条件模型 baseline |
| V1 | 15 | 0.012279 | CSV 指向旧生成文件，需谨慎引用 |
| S1 | 15 | 0.013743 | 当前文件可对上，但样本少 |

注意：这些评估样本量很小，且部分 CSV 和当前结果目录不完全一致，所以只能作为趋势参考，不能作为严格最终指标。

## 5. 问题分析

### 5.1 主要瓶颈是 split 结构

生成 OBJ 的大洞、碎片、局部缺失主要来自 depth 6 之前的 split 漏判或误判。多类别中最后一级 split acc 大多只有 0.78-0.80，这一级直接决定叶子节点数量和局部拓扑。一旦某个父节点没有 split，后续 VQ token 再准确也无法恢复那块几何。

### 5.2 单次 forward 与迭代式生成存在质量差距

OctGPT 类方法通过大量 mask/reveal 或自回归步骤逐步修正结构，而本方法为了速度把生成压缩到少数逐层 forward。速度优势明显，但结构错误缺少后验修正机制，这是多类别质量差距的核心原因。

### 5.3 叶子 VQ 不是唯一瓶颈

单飞机过拟合中 VQ acc 能到 0.847，多类别中大约 0.64-0.65。VQ token 的确会影响表面细节，但如果 split 结构已经漏掉叶子，VQ 无法补救拓扑缺失。因此后续应优先提升 split 结构，再处理 VQ refinement。

### 5.4 部分 ablation 结论需要谨慎

S1 的 leaf MaskGIT 结果不能直接与 Q4 比，因为 S1 实际是 384 维配置，而 Q4 是 768 维大模型。严格结论需要补一个 `Q4 + leaf_vq_mask` 的同容量实验。

## 6. 汇报建议

建议按这个逻辑讲：

1. **先展示单飞机 overfit**：证明框架能学习，OBJ 能导出，split 几乎可完全拟合。
2. **再讲多类别扩展**：加入类别条件和 latent 后，不同类别能生成不同形状。
3. **展示 ablation 表**：Q2/Q4 有小幅提升，Q3 split MaskGIT 没达到预期，S1 对 nearest-CD 有趋势但不严格。
4. **诚实说明瓶颈**：多类别质量差距主要来自 split 结构，单次 forward 的速度优势换来了结构修正能力不足。
5. **给出后续方向**：优先做 split refinement、teacher-forcing/free-running mismatch 修正，以及同容量的 `Q4 + leaf_vq_mask` 实验。

一句话总结：

> 本项目完成了从八叉树 split 预测、叶子 VQ token 预测到 VQ-VAE 解码 OBJ 的完整快速生成 pipeline；单飞机过拟合证明模型可学习且导出质量可展示，多类别实验验证了条件生成能力，但也暴露出少步分形生成在 split 结构预测上的主要瓶颈。
