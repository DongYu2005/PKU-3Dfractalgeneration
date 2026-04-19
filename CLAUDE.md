# 3D 分形生成项目

<!-- 本文件覆盖并扩展全局 ~/.claude/CLAUDE.md。保持 200 行以内。
     规则膨胀时拆分到 .claude/rules/ 目录做模块化管理。 -->

## 项目简介

一个 3D 分形生长系统，通过迭代累积和一个控制局部密度的截断阈值来生成结构化形状。
当前目标：稳定生成可辨识的目标形状（目前是飞机轮廓），同时避免表面过密。

## ⚠️ 环境（每次启动 CC 后先确认）

**所有命令必须在 `aibasis` conda 环境下执行：**

```bash
conda activate aibasis
```

如果发现 `which python` 的输出不在 aibasis 环境里，**先激活再执行任何操作**，
不要用系统默认的 Python。

## 仓库结构

<!-- TODO：根据实际情况替换路径 -->

```
.
├── src/
│   ├── growth/          # 分形生长核心算法
│   ├── render/          # 3D 可视化（生成你看到的灰色渲染图）
│   ├── config/          # 每个实验对应的 YAML config
│   └── utils/
├── scripts/
│   ├── run_growth.py    # 生成任务的入口
│   └── visualize.py     # 渲染已有的 mesh
├── configs/
│   └── airplane.yaml    # 当前工作 config
└── outputs/             # 生成的 mesh 和渲染图（gitignored）
```

## 常用命令

```bash
# 激活环境（先做这个）
conda activate aibasis

# 从 config 生成形状
python scripts/run_growth.py --config configs/airplane.yaml

# 渲染已有 mesh
python scripts/visualize.py --mesh outputs/airplane_v3.obj

# 快速迭代：生成 + 渲染一气呵成
python scripts/run_growth.py --config configs/airplane.yaml --render
```

## 核心参数（我实际会调的几个）

| 参数 | 作用 | 典型范围 | 备注 |
|---|---|---|---|
| `truncation_threshold` | 局部密度超过此值时截断生长 | 0.3–0.8 | **偏低会导致表面过密**，当前就是这个问题 |
| `growth_rate` | 每步累积概率 | 0.05–0.3 | |
| `max_iterations` | 生长步数上限 | 500–5000 | |
| `seed_shape` | 初始几何形状 | `plane`、`sphere`、`custom` | |

## 已知问题（Known issues）

- **表面过密（当前）**：truncation_threshold 设得偏低（上次是 0.35）。飞机形状
  已经正确长出来了，但表面覆盖了一层小凸起。下次试 `truncation_threshold: 0.55–0.65`。
- `max_iterations > 3000` 跑飞机 config 会开始侵蚀机翼结构，要早点停。
- renderer 存在 y-up / z-up 不一致，导出到 Blender 时部分 mesh 需要手动旋转。

<!-- 每次 debug 完新问题就往这里加一条 -->

## 评测流程

每次生成结束后产出这份报告：
1. 三个规范角度的渲染图（正视、3/4 视角、俯视）
2. 表面密度直方图
3. 体积和 bounding box 尺寸 vs 目标
4. 和上一次最佳结果的并排对比

## 当前状态

- `v1` / `v2`：飞机轮廓初步长出来（关键突破，形状可辨识）
- `v3`（当前）：truncation_threshold 偏低导致表面过密
- 下一步：固定 growth_rate 和 seed，在 `[0.45, 0.55, 0.65, 0.75]` 上扫
  truncation_threshold，对比表面平滑度

## 在这个项目里工作时

- **永远不要删 `outputs/` 里的文件**。每次生成都很贵，需要清理先问我。
- 做参数 sweep 之前，把完整网格列出来让我确认一下再启动，这些 run 不便宜。
- 一次生成效果好的时候，立刻把 config、mesh、渲染图三件套一起存到
  `outputs/snapshots/` 下并命名。
