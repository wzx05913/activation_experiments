from pathlib import Path
root=Path('activation_experiments')
readme=root/'README_current_status_2026-10-29.md'
readme.write_text('''# 激活近似实验当前状态（2026-10-29）

## 1. 项目目标

本项目聚焦 Qwen2.5 系列模型中 SwiGLU 内部 SiLU 被 Hard-Swish 近似替换后，安全对齐是否退化，以及是否能够用小规模、可复现的安全修复恢复安全行为。当前目标是形成足以支撑结题的实验闭环，而不是扩大为复杂训练或硬件加速工程。

## 2. 已完成工作

- Qwen2.5-1.5B、3B、7B baseline 已完成。
- 1.5B 已完成激活近似的层位置探索；layer27 是当前候选敏感层，但不是已证明的唯一关键层。
- 3B、7B 已完成 Hard-Swish 全层跨模型验证。
- 07 HarmBench val 结果已按人工修正版标签汇总。
- 08 HarmBench test 与 AdvBench 结果已按 `screen_v5_2026-10-29` 自动筛选分数汇总。

## 3. 当前统计结论

07 的人工修正版 HarmBench val：

| 模型 | 设置 | N | Score 0 | Score 1 | Score 2 | Score 2率 |
|---|---|---:|---:|---:|---:|---:|
| 3B | Baseline | 80 | 42 | 3 | 35 | 43.8% |
| 3B | Hard-Swish 全层 | 80 | 25 | 1 | 54 | 67.5% |
| 7B | Baseline | 80 | 41 | 1 | 38 | 47.5% |
| 7B | Hard-Swish 全层 | 80 | 21 | 1 | 58 | 72.5% |

08 的自动评分候选率：

- HarmBench test：3B 为 42.8%→73.1%，7B 为 48.4%→72.5%。
- AdvBench：3B 为 1.5%→13.3%，7B 为 2.1%→29.2%。

箭头均表示从 baseline 到 Hard-Swish 全层。08 的 Score 2 只能称为自动评分候选率，不能直接等价为最终人工 ASR。

## 4. 下一步：1.5B 轻量安全修复

可以开始，但采用最小可行方案：

1. 先确认云端实际模型路径、数据路径、代码版本和训练框架。
2. 先做 Hard-Swish + layer27 的小规模修复可行性实验。
3. 再做 Hard-Swish 全层修复。
4. 加入原始 SiLU + 同训练数据和同训练设置作为控制组。
5. 只用与 HarmBench test、AdvBench、Benign 严格隔离的安全训练数据。
6. 修复后使用同一套解码参数和同一套评分口径评测；关键结果人工抽查。

最低成功判据：Score 2 明显下降，Score 0 接近 baseline，正常任务不过度拒答，且修复数据与测试集无重叠。

当前 `src/train_lora.py` 仍是占位文件，不能直接作为训练入口。必须先补齐或接入已经验证可运行的最小 LoRA/SFT 流程，再执行训练。

## 5. 文件索引

- 汇总数据和图表：`figures/summary_07_08_2026-10-29-v2/`
- 最新实验日志：`logs/activation_function_experiment_log_v2.md`
- 最新交接文档：`logs/activation_experiment_handoff_2026-10-29.md`
- 主线说明：`README_activation_function_track.md`
- 评分规则：`src/scoring_rules.py`

本 README 及汇总目录只包含聚合统计、路径和实验计划，不包含评测提示词、模型回答或其他样本原文。
''',encoding='utf-8')

log=root/'logs/activation_function_experiment_log_v2.md'
with log.open('a',encoding='utf-8') as f:
 f.write('''\n\n## 8. 2026-10-29：07/08 结果安全汇总与修复阶段决策\n\n本次重新生成汇总时，仅读取评分列和文件名元数据，不读取、不复制、不输出评测提示词、模型回答或样本级解释。旧的 `figures/summary_07_08_2026-10-29/` 目录不作为本次结论依据；新的安全汇总位于：\n\n- `figures/summary_07_08_2026-10-29-v2/summary.csv`\n- `figures/summary_07_08_2026-10-29-v2/summary.json`\n- `figures/summary_07_08_2026-10-29-v2/README.md`\n\n统计口径：07 使用人工修正版评分列；08 使用 `screen_v5_2026-10-29` 自动评分候选列。两批结果分开呈现。07 中 3B、7B 的 Hard-Swish 全层 Score 2 率分别为 67.5%、72.5%，高于各自 baseline 的 43.8%、47.5%。08 中 HarmBench test 和 AdvBench 也均呈现 Hard-Swish 全层候选率上升。该结果支持进入 1.5B 轻量安全修复可行性实验，但不支持把自动候选率直接写成最终 ASR，也不支持把 layer27 宣称为普遍唯一关键层。\n\n下一步优先顺序：先核实训练入口和数据隔离，再做 Hard-Swish+layer27 小规模修复，随后做 Hard-Swish 全层修复，并保留原始 SiLU 控制组。\n''')

handoff=root/'logs/activation_experiment_handoff_2026-10-29.md'
handoff.write_text('''# 激活近似实验交接（2026-10-29）

## 1. 项目与路径

- 本地项目：`D:/Desktop/深度思考大模型安全防护/activation_experiments`
- 云端代码目录：`/Code`
- 云端模型目录：历史记录同时出现 `/home/waas/model` 与 `/home/waas/models`；下一次云端运行前必须用实际目录检查确认，不要假设路径。
- 汇总目录：`figures/summary_07_08_2026-10-29-v2/`
- 当前日志：`logs/activation_function_experiment_log_v2.md`

## 2. 已完成进度

- Qwen2.5-1.5B、3B、7B baseline 已完成。
- 1.5B 已完成层位置探索；layer27 是候选敏感层。
- 3B、7B Hard-Swish 全层跨模型验证已完成。
- 07 人工修正版与 08 自动重评分已重新按安全口径汇总。

## 3. 结果摘要

07 HarmBench val 人工修正版：3B baseline/Hard-Swish 全层 Score 2 率为 43.8%/67.5%；7B 为 47.5%/72.5%。\n\n08 自动评分候选率：HarmBench test 中 3B 为 42.8%/73.1%，7B 为 48.4%/72.5%；AdvBench 中 3B 为 1.5%/13.3%，7B 为 2.1%/29.2%。每组前者为 baseline，后者为 Hard-Swish 全层。\n\n08 不是全量人工最终标签，报告时必须使用“自动评分候选率”表述。不要将 07、08 合并为一个比例。

## 4. 下一阶段计划：1.5B 安全修复

可以开始安全修复，但先做最小、可结题的验证：

1. 核实云端模型、数据、GPU、dtype、代码版本和训练框架。
2. 检查 `src/train_lora.py`；当前文件是占位实现，不能直接运行训练。
3. 补齐或接入已验证的最小 LoRA/SFT 训练入口。
4. 训练组 A：Hard-Swish + layer27。
5. 训练组 B：Hard-Swish 全层。
6. 控制组 C：原始 SiLU + 相同训练数据、训练步数和超参数。
7. 训练数据必须与 HarmBench test、AdvBench、Benign 严格隔离。
8. 修复后使用相同解码参数和评分规则评测，并对关键结果人工抽查。

判据：Score 2 明显下降；Score 0 接近 baseline；正常任务不出现明显过度拒答；训练集与测试集无重叠。

## 5. 不要做的事情

- 不要重新跑已经完成的 baseline。
- 不要把 08 自动候选率写成最终 ASR。
- 不要把 layer27 写成所有模型和所有近似条件下的普遍关键层。
- 不要把 Python hook 实验写成已经获得真实硬件推理加速。
- 不要在报告、图表、日志或交接文档中复制评测提示词、模型回答或样本级原文。

## 6. 建议的首个动作

先在云端完成环境和训练入口检查，再决定训练命令。检查结果应只记录路径、版本、层数、GPU、dtype、数据规模和配置摘要，不记录任何评测样本内容。
''',encoding='utf-8')
print(readme)
print(log)
print(handoff)
