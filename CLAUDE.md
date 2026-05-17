# 3D 分形生成项目

## 项目简介

基于 OctGPT 的 3D 分形生成模型：用一个分形展开的 octree 生成器预测 split，
再用冻结的 VQ-VAE 把叶子节点的 VQ token 解码成 SDF，最后 marching cubes 出 mesh。
当前目标：在 ShapeNet 单个飞机上过拟合，验证整条 fractal + VQ-VAE pipeline。

## ⚠️ 环境（每次启动 CC 后先确认）

**所有命令必须在 `aibasis` conda 环境下执行：**

```bash
conda activate aibasis
```

如果发现 `which python` 的输出不在 aibasis 环境里，**先激活再执行任何操作**，
不要用系统默认的 Python。

## 仓库结构

```
.
├── main_fractal.py                 # 训练/生成入口（VQ-VAE 版）
├── render_obj.py                   # headless 渲染 .obj → 三视图 PNG
├── fractal_models/
│   └── fractal_generator.py        # FractalGenerator v4（focal loss + 阈值 split）
├── without_VQVAE/                  # 不带 VQ-VAE 的对照实现
│   ├── main_fractal_withoutVQVAE.py
│   └── fractal_generator_withoutVQVAE.py
├── configs/
│   ├── shapenet_fractal.yaml       # 当前主 config（overfit 单飞机）
│   ├── shapenet_fractal_debug.yaml
│   └── shapenet_frac.yaml
├── saved_ckpt/                     # 冻结的 VQ-VAE 权重（gitignored）
├── octgpt/                         # 外部依赖，作为 sibling 引入（gitignored）
├── data/ShapeNet/                  # 数据（gitignored）
└── logs/fractal/<run>/results/     # 生成的 .obj（按 epoch 命名，gitignored）
```

注意：`octgpt/`、`data/`、`logs/`、`saved_ckpt/`、`*.obj`、`*.pth` 都在 .gitignore。

## 常用命令

```bash
conda activate aibasis

# 训练（默认就会每 20 epoch 生成一次样本到 logs/.../results/）
python main_fractal.py --config configs/shapenet_fractal.yaml

# 生成（训练完后）
python main_fractal.py --config configs/shapenet_fractal.yaml \
    SOLVER.run generate SOLVER.ckpt <path_to_ckpt>

# 不带 VQ-VAE 的对照
python without_VQVAE/main_fractal_withoutVQVAE.py --config configs/shapenet_fractal.yaml

# 快速渲染某个生成的 .obj 到三视图 PNG（headless，不需要 GUI）
python render_obj.py logs/fractal/<run>/results/<epoch>.obj
```

## 核心参数（我实际会调的）

定义在 `fractal_models/fractal_generator.py` 的 `FractalGenerator.__init__`，
通过 config 里 `MODEL.FractalGen.*` 覆盖。

| 参数 | 作用 | 典型值 | 备注 |
|---|---|---|---|
| `split_threshold` | 生成时 `P(split) > threshold` 才分裂 | 0.3–0.6 | **当前工作值 0.45**，更低会过密 |
| `focal_alpha` | Focal Loss 正样本权重（split=1 很稀疏） | 0.75 | 处理 split 类别极度不均衡 |
| `focal_gamma` | Focal Loss 难样本聚焦系数 | 2.0 | |
| `temperature` | 生成时 VQ token 采样温度 | 0.8 | 0 则 argmax |
| `depth_stop` | 分形展开停止深度（之后交给 VQ-VAE 解码） | 6 | 再由 zero split 扩到 `depth=8` |
| `full_depth` | 分形起始深度（root 所在深度） | 3 | |
| `feature_dim` | Transformer 隐藏维度 | 384 | |
| `blocks_per_level` | 叶子层 OctFormerStage 的 block 数 | 6 | 每个中间层固定 1 block |
| `split_weight` / `vq_weight` | 两路 loss 的权重 | 1.0 / 1.0 | |

Solver 侧的常用参数（`SOLVER.*`）：`max_epoch`、`lr`、`rand_seed`、
`resolution`（marching cubes 分辨率）、`sdf_scale`。

## 已知问题（Known issues）

- **split 阈值偏低 → 表面过密**：0.3–0.4 附近飞机能长出来但叶子层太满；
  0.45 现在看起来是 overfit 下的甜点区，再高（>0.5）之前没系统扫过。
- **argmax 全崩**：纯 argmax 选 split 会因为 50% loss 截断直接输出空节点
  （见 commit `26628fd`），所以生成路径目前是阈值分裂 + VQ 温度采样。
- **VQ-VAE 权重路径**：`saved_ckpt/vqvae_large_im5_{cond,uncond}_bsq32.pth`
  通过 config 的 `vqvae_ckpt` 字段指定，别忘了带路径。
- **octgpt 是 sibling 依赖**：`main_fractal.py` 从 `../octgpt` 做 `sys.path.insert`，
  移动仓库或改目录结构时要同步改。

## 当前状态（v4 / commit 27eb6dc）

- `v1`：argmax 选 split → 输出空节点（commit 26628fd 的问题）
- `v2`：阈值 split，能长出较完整飞机，但阈值偏低表面太密（commit b9c4c18）
- `v3` → `v4`（当前）：`split_threshold=0.45`，在 overfit 单飞机 config 下
  长出"还算比较好"的 obj（commit 27eb6dc），过拟合 pipeline 算走通了
- 过拟合调参到此为止：`0.5` 试过，表面有空缺，`0.45` 仍是唯一可用工作点。
  再调这个参数没有研究价值。
- 200 epoch 已收敛，长训练不要再设 400。

## 当前路线（2026-05-15 起）

完整研究计划写在 `/home/batchcom/.claude/plans/wobbly-swimming-harp.md`。
三阶段并行，**不要再回去调过拟合阈值**：

- **Phase 1（基础设施）**：写 `eval/eval_fractal.py`（Chamfer/MMD-CD/COV/1-NNA）+
  `eval/bench_speed.py`（推理速度对比）。复用 `octgpt/metrics/evaluation_metrics.py`。
- **Phase 2（速度卖点 / Phase A）**：im_5 多类别泛化训练 Fractal + OctGPT baseline，
  出主结果表对比 MMD/COV/1-NNA + 推理 s/sample。卖点是 Fractal 4 次 forward vs
  OctGPT ~576 次 forward。
- **Phase 3（端到端突破 / Phase B）**：先给 `without_VQVAE/` 补齐结构性差异
  （加 mid_transformer + 每层位置编码），定位"端到端崩"的真正原因。

## 评测流程

**量化**（主要）：`python eval/eval_fractal.py --gen_dir logs/<run>/results/ --ref_dir <gt>`
出 MMD-CD / COV / 1-NNA 三个数 + per-sample CSV。
**速度**：`python eval/bench_speed.py --ckpt <fractal_ckpt> --octgpt_ckpt <octgpt_ckpt>`。
**目测**（辅助）：`render_obj.py` 渲三视图，看是否形状合理。

## 在这个项目里工作时

- **永远不要删 `logs/` 和 `saved_ckpt/` 下的文件**。一次训练要跑很久，
  obj 和 ckpt 都要保留作为对照，清理前先问我。
- **Git commit 不要加 `Co-Authored-By: Claude` 等任何 Claude 署名行**。
  正常写 commit message 即可，不要写 trailer。
- 改 `fractal_generator.py` 的默认参数前先问：
  model 的默认值是 fallback，config 没覆盖到时才生效，容易踩坑。
- 参数 sweep 之前，把完整网格列出来让我确认再启动，这些 run 不便宜。
- 一次生成效果好的时候，把 config、ckpt、代表 obj 一起记下来（commit 里
  或单独起文件夹都行），方便后面回退对照。
