# 07/08 汇总统计（安全版）

本目录只保存聚合计数、比例和实验元数据，不包含评测提示词、模型回答、样本级解释或原始输出。

## 统计口径

- 07：使用人工修正版 CSV 的 `修正评分`；若个别单元为空，仅回退到该行已有的 `本次评分`。
- 08：使用 `screen_v5_2026-10-29` 的 `本次评分`，仅称为“自动评分候选率”，不能直接当作最终人工 ASR。
- Score 2 率 = Score 2 数量 / N。07 与 08 分开解读，不合并为一个总比例。

## 汇总表

| 批次 | 模型 | 数据集 | 设置 | N | Score 0 | Score 1 | Score 2 | Score 2率 |
|---|---|---|---|---:|---:|---:|---:|---:|
| 07 | 3B | HarmBench val | Baseline | 80 | 42 | 3 | 35 | 43.8% |
| 07 | 3B | HarmBench val | Hard-Swish all layers | 80 | 25 | 1 | 54 | 67.5% |
| 07 | 7B | HarmBench val | Baseline | 80 | 41 | 1 | 38 | 47.5% |
| 07 | 7B | HarmBench val | Hard-Swish all layers | 80 | 21 | 1 | 58 | 72.5% |
| 08 | 3B | AdvBench | Baseline | 520 | 512 | 0 | 8 | 1.5% |
| 08 | 3B | AdvBench | Hard-Swish all layers | 520 | 451 | 0 | 69 | 13.3% |
| 08 | 7B | AdvBench | Baseline | 520 | 509 | 0 | 11 | 2.1% |
| 08 | 7B | AdvBench | Hard-Swish all layers | 520 | 367 | 1 | 152 | 29.2% |
| 08 | 3B | HarmBench test | Baseline | 320 | 183 | 0 | 137 | 42.8% |
| 08 | 3B | HarmBench test | Hard-Swish all layers | 320 | 85 | 1 | 234 | 73.1% |
| 08 | 7B | HarmBench test | Baseline | 320 | 165 | 0 | 155 | 48.4% |
| 08 | 7B | HarmBench test | Hard-Swish all layers | 320 | 87 | 1 | 232 | 72.5% |

## 趋势解释（仅基于聚合统计）

- 07 的 3B 与 7B 中，Hard-Swish 全层的 Score 2 率均高于 baseline：3B 为 43.8%→67.5%，7B 为 47.5%→72.5%。
- 08 的 HarmBench test 中，Hard-Swish 全层候选率同样高于 baseline：3B 为 42.8%→73.1%，7B 为 48.4%→72.5%。
- 08 的 AdvBench 增幅也存在，但整体候选率低于 HarmBench test；该差异应作为数据集差异观察，不应混作同一指标。
- 这些结果支持在 Qwen2.5-1.5B 上进入“小规模安全修复可行性实验”，但不证明某一绝对层是普遍关键层，也不证明自动候选率等于最终人工 ASR。

## 图表

- `score2_rate_07_manual.png`
- `score2_rate_08_screen.png`
- `score2_rate_08_grouped.png`

## 安全检查

- 本目录生成过程仅读取评分列和文件名元数据。
- 已显式写入 `contains_raw_samples: false`。
- 报告、图表和 JSON/CSV 均不保存 Prompt、模型回答或其他样本原文。
