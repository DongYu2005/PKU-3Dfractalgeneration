# 提交清单（2026-07-04）

## 一、报告（直接提交）

| 文件 | 说明 |
|---|---|
| `report/final_ydg.pdf` | 期末总结报告（编译产物，已含全部图表） |
| `report/final_ydg.tex` | 报告源码（tectonic 编译：`cd report && tectonic final_ydg.tex`） |
| `report/airplane_overfit.png` | 图 1：单飞机 oracle 对照（上 oracle / 下本方法，mc_level 0.01） |
| `report/octgpt_airplane_baseline.png` | 图 2：OctGPT 官方权重 baseline |

注：R2 五类改进段落当前引用 epoch-20 指标；epoch-30 跑完后会刷新一次数字并重新编译。

## 二、代码（GitHub 已推送）

仓库 `DongYu2005/PKU-3Dfractalgeneration`，分支 `feat/v5-flag-scaffolding`（最新 commit `4812c8a`）。
外部依赖 `octgpt/`（sibling 目录）、`data/ShapeNet/`、`saved_ckpt/`、`logs/` 均不入库。

## 三、权重（服务器路径，需自行备份的按优先级排序）

| 优先级 | 文件 | 大小 | 说明 |
|---|---|---|---|
| ★★★ | `saved_ckpt/vqvae_large_im5_uncond_bsq32.pth` | 130M | 冻结 VQ-VAE，整条 pipeline 的解码器，缺它所有模型不能用（可从 HF `wst2001/OctGPT` 重新下载） |
| ★★★ | `logs/exp/Q4_sibling/best_model.pth` | 651M | 多类别最佳基线（报告主结果候选），复现配置 `configs/exp/Q4_sibling.yaml` |
| ★★★ | `logs/exp/R2_im5_full/checkpoints/00030.model.pth`（跑完后） | 683M | 五类改进模型（粗层加权+空间 latent），配置 `configs/exp/R2_im5_full.yaml`；当前最新为 `00025.model.pth` |
| ★★ | `logs/exp/O1_air_struct_best/best_model.pth` | 109M | 单飞机过拟合展示模型，配置 `configs/exp/O1_air_struct_best.yaml` |
| ★★ | `logs/exp/R1b_air_spatialz/best_model.pth` | 683M | 飞机单类消融模型，配置 `configs/exp/R1b_air_spatialz.yaml` |
| ★ | `saved_ckpt/octgpt_airplane.pth` / `octgpt_im5.pth` | 650M×2 | OctGPT baseline（HF 可重下，可不备份） |

## 四、代表性生成结果（渲染图已在 git 里，obj 在服务器）

| 目录 | 内容 |
|---|---|
| `logs/exp/O1_air_struct_best/gen_mclevel001/` | 过拟合飞机最终展示（mc_level 0.01） |
| `logs/exp/O1_air_struct_best/level_sweep/` | iso-level 消融（0→0.02，空洞诊断证据） |
| `logs/exp/Q4_sibling/gen_t0_lvl001/` | Q4 五类（temperature=0 + mc_level 0.01） |
| `logs/exp/R2_im5_full/gen_ep20_t0/` | R2 五类同设置对照 |
| `octgpt/logs/octgpt/airplane_gen/results/` | OctGPT baseline 飞机 ×3（~81 s/样本实测） |

## 五、复现命令速查

```bash
conda activate ydg-ocnn
# 生成（任意 ckpt）
python eval/gen_from_ckpt.py --config <cfg> --ckpt <ckpt> --out_dir <dir> \
    --per_class 2 --temperature 0 --mc_level 0.01
# 结构诊断
python eval/diag_split.py --config <cfg> --ckpt <ckpt> --out_dir <dir> --num_samples 16
# 渲染
python render_obj.py <mesh.obj>
```
