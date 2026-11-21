# 激活函数近似实验 Handoff（2026-10-28）

## 1. 项目与云端环境

- 本地项目：`D:/Desktop/深度思考大模型安全防护/activation_experiments`
- 云端代码目录：`/Code`，用户确认其内部结构与本地项目相同。运行前 `cd /Code`，确认有 `src/`、`configs/`、`data/`。
- 云端模型目录：用户提供 `/home/waas/model`；历史运行记录也出现 `/home/waas/models`。启动前检查二者并确认实际权重目录，不要假定名称一致。
- Qwen2.5-1.5B、3B、7B baseline 均已完成，本阶段不要重跑。
- 云端评测数据路径尚未明确。检查 `/Code/data` 或实际挂载位置中的 HarmBench test、`advbench_raw.jsonl`、`benign_utility.jsonl`。`harmful_prompts.jsonl`、`safety_train.jsonl`、`safety_test.jsonl` 是占位/项目数据，不可直接当正式评测集。

云端环境检查：

```bash
cd /Code
pwd
ls
ls -ld /home/waas/model /home/waas/models 2>/dev/null
find /home/waas/model /home/waas/models -maxdepth 2 -type f -name 'config.json' 2>/dev/null | head
find /Code -maxdepth 3 -type f \( -iname '*harmbench*test*.jsonl' -o -iname '*advbench*.jsonl' -o -iname '*benign*.jsonl' \)
```

## 2. 研究对象与代码边界

研究问题：将 Qwen2.5 SwiGLU MLP 内部的 SiLU 替换为常见近似（当前重点 Hard-Swish）后，安全对齐是否退化，以及近似条件下能否修复。

当前主实现通过 `src.generate` 和激活 hook 替换 MLP 的 `act_fn`。Hard-Swish 是**激活函数近似**，不是权重量化，也不是 MLP 输出量化。过去 Gaussian/Laplace/稀疏化/4、6、8-bit 模拟量化是旧的 MLP 输出扰动实验，单独作为历史探索结果。Python hook 用来验证安全影响，不等于已实现真实硬件加速。

关键入口与配置：`src/generate.py`、`src/activation_hooks.py`、`configs/silu_hard_swish_all.yaml`（Hard-Swish、全层、strength=1.0）。CLI 参数为 `--max-input-tokens`，不是 `--num-input-tokens`。

## 3. 已完成进度

- 1.5B 已完成三种激活近似初筛及 Hard-Swish 层扫描；Hard-Swish 全层影响最大，layer27 是候选敏感后层，但 layer14 也有变化。
- 1.5B Hard-Swish 全层、layer14、layer27 已完成 HarmBench test；用户已对候选样本人工复核并重新评分。
- 最新对比图在 `figures/qwen25_1p5b_hardswish_validation_comparison/`，包括分数百分比、数量、Score 2 比例、`summary.json` 和 README。
- 最新进度日志：`logs/activation_function_experiment_log_v2.md`。
- Hard-Swish 的 AdvBench 和全层 XSTest 已运行；自动初筛汇总在 `results/05_silu_approximation/qwen-1.5b/validation_review/auto_scored/validation_summary.md`。各数据集人工复核状态以对应 review 文件为准。
- Benign 已跑过；设置一致时不重复。

## 4. 最新 HarmBench test 结果

Qwen2.5-1.5B，320 条。统计采用已人工复核标签覆盖自动分数，其余未复核项仍沿用自动评分，因此属于**人工复核修正后的全量估计**，不是 320 条全量人工重标。

| 设置 | Score 0 | Score 1 | Score 2 |
|---|---:|---:|---:|
| Baseline | 272（85.0%） | 5（1.6%） | 43（13.4%） |
| Hard-Swish，全层 | 173（54.1%） | 1（0.3%） | 146（45.6%） |
| Hard-Swish，layer 14 | 266（83.1%） | 5（1.6%） | 49（15.3%） |
| Hard-Swish，layer 27 | 252（78.7%） | 5（1.6%） | 63（19.7%） |

已人工复核：全层159条、layer14 60条、layer27 74条。全层 Score 2 比 baseline 高32.2个百分点，约3.4倍；layer27 高6.3个百分点，layer14 高1.9个百分点。全层退化证据明显，单层差异相对较弱，应作为探索性定位结果；目前更支持多层共同/累积影响，不支持“layer27 是唯一关键层”。早期自动统计的全层48.7%、layer27 21.6%已被上述人工修正估计替代，勿混用。

## 5. 后续路线

### P0：整理标签与归档

1. 保留原始自动评分 CSV，将人工标签统一到清晰的 `人工评分` 列。特别检查 layer27 旧表曾将人工分写入 `人工说明` 的情况。
2. 核对三组 HarmBench test 的人工复核覆盖率；报告区分“人工修正估计”和“全量人工复核”。
3. 优先复核 AdvBench 全层自动 Score 2 和 layer14/27 少量候选。XSTest 按 safe/unsafe 标签分别分析，不把其 Score 2 直接当单一安全失效率。

### P1：有限跨模型验证

在 baseline 已有的 Qwen2.5-3B、7B 上，各跑 Hard-Swish 全层；算力允许时再跑按相对深度映射的后层单层，不照搬1.5B绝对层号27。优先 HarmBench test 与 AdvBench；Benign 条件相同则不重跑。不做新的全层扫描矩阵。

运行模板（路径需按云端实情替换）：

```bash
cd /Code
MODEL=/home/waas/model/Qwen2.5-3B-Instruct
python -m src.generate \
  --model "$MODEL" \
  --data /Code/data/<harmbench_test.jsonl> \
  --output /Code/results/05_silu_approximation/qwen-3b/hard_swish_all_harmbench_test.jsonl \
  --setting hard_swish_all --approximation hard_swish --layers all \
  --strength 1.0 --seed 42 --max-input-tokens 1024 --max-new-tokens 128 \
  --temperature 0 --top-p 1.0 --device cuda --dtype float16
```

按 7B、AdvBench 分别替换模型、数据和输出路径。先核实模型仓库名与数据集文件名；评测后使用项目现有评分流程并人工抽查。

### P2：1.5B 轻量安全修复

确保修复训练数据与 HarmBench test、AdvBench 严格隔离后：先修复 Hard-Swish + layer27，再修复 Hard-Swish 全层；加入原始 SiLU + 同训练数据/设置控制组；在未用于训练的 HarmBench test、AdvBench、Benign 上评估。最低判据是 Score 2 明显下降、Score 0 接近 baseline、Benign 不出现明显过度拒答，且不是记忆训练样本。

当前 `src/train_lora.py` 明确是占位文件，没有可执行训练逻辑。因此修复阶段需要先补齐最小 LoRA/SFT 流程或接入已验证训练框架，再制定训练命令；不要直接尝试运行该占位脚本。

### P3：轻量机制分析（有时间再做）

对 baseline、layer27、全层少量安全/正常 prompt 比较拒答首 token 倾向、输出长度及少量层激活统计，作为辅助解释，不扩大成复杂可解释性项目。

## 6. 交接注意事项

- 不重跑 baseline；不重复相同设置下已完成的 Benign。
- 不把旧 MLP 输出量化/扰动称为 SiLU 激活近似。
- 自动评分不是最终人工安全标签，记录人工覆盖范围。
- 不声称 Python hook 已带来推理加速；若要报告加速，需在目标推理后端、同硬件和控制条件下实测吞吐/延迟。
- 云端启动先记录实际模型权重版本、层数、GPU、dtype、数据路径和代码版本。

## 7. 本地参考文件

- `logs/activation_function_experiment_log_v2.md`
- `logs/activation_function_experiment_plan.md`
- `README_activation_function_track.md`
- `results/05_silu_approximation/qwen-1.5b/validation_review/`
- `figures/qwen25_1p5b_hardswish_validation_comparison/`
