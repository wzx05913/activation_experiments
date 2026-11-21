# 激活近似安全修复交接（2026-11-10）

## 项目与边界

- 本地：`D:/Desktop/深度思考大模型安全防护/activation_experiments`；云端代码：`/Code`；云端模型：`/home/waas/models`。
- Qwen2.5-1.5B、3B、7B 已完成基线和部分 Hard-Swish 激活近似对比。1.5B 当前修复针对第 27 层（第 14 层也有早期敏感性观察）；不可直接把该层推广到其他模型。
- 当前方法是固定第 27 层 Hard-Swish 的 LoRA-DPO，以原始 SiLU 模型为冻结参考；不是完整 QuadA，未实现 MVA 搜索、激活表征约束。Python hook 实验不能证明真实推理加速。

## 已完成进度

- 1.5B 先使用 67 对筛选偏好样本，再加入 7 对审核样本，共 74 对。前次记录的配置：LoRA `q_proj,v_proj`，`r=4`、`alpha=8`；DPO `beta=0.1`、sum log-prob；学习率 `1e-4`、梯度累积 8、seed 42；训练至 5 epoch 并评估中间轮次。以云端训练日志为最终依据。
- 74 对训练的 HarmBench val 有 epoch 1/3/5 汇总；epoch 5 的 HarmBench test、AdvBench、XSTest、benign 以及 epoch 3 的 XSTest、benign 逐样本评测已同步本地。epoch 3 的 HarmBench test、AdvBench 未见本地同步。
- `scripts/compare_repair_epoch3_epoch5.py` 对齐 ID/prompt 统计，产物在 `figures/repair_epoch3_epoch5_comparison/`：`summary.csv`、`paired_transitions.csv`、`safety_utility_tradeoff.png` 和 `independent_harmful_benchmarks.png`。epoch 3 未同步的独立测试标为 N/A。

## 主要结果

下表为 `screen_v5` 的 **Score 2 规则筛查候选数/总数**，不是人工核定的 ASR。XSTest safe 的 Score 2 表示过度拒答候选，unsafe 的 Score 2 表示未拒答候选，不应合并。

| 数据集 / 指标 | 原始 SiLU | 未修复 Hard-Swish | 74 对 epoch 3 | 74 对 epoch 5 |
|---|---:|---:|---:|---:|
| HarmBench val 未拒答候选 | 16/80 | 27/80 | 13/80* | 6/80 |
| HarmBench test 未拒答候选 | 83/320 | 103/320 | 未同步 | 30/320 |
| AdvBench 未拒答候选 | 1/520 | 3/520 | 未同步 | 2/520 |
| XSTest unsafe 未拒答候选 | 109/200 | 120/200 | 93/200 | 42/200 |
| XSTest safe 过度拒答候选 | 104/250 | 95/250 | 125/250 | 175/250 |
| Benign 拒答候选 | 3/50 | 3/50 | 6/50 | 7/50 |

\* epoch 3 val 值来自此前用户提供的云端汇总，该轮 74 对的逐样本 val 未见本地同步；其他独立集数字来自本地同步的审核 CSV。epoch 5 val 见本地审核目录。val 与独立 test 不混用。epoch 5 安全候选率下降明显，但 XSTest safe 过度拒答候选比原始 SiLU 增加 71/250；epoch 3 较平衡，仍需确认正常任务质量。自动规则只筛查拒答信号，不能衡量回答是否实质有害或正常任务是否完成。

## 云端文件目录

以下除绝对模型路径外均相对 `/Code`。同步的审核目录在本地核对过；训练数据和权重**未在本地同步核实**，继续运行前须在云端检查。

| 用途 | 云端路径 | 状态 |
|---|---|---|
| 1.5B 模型 | `/home/waas/models/Qwen2.5-1.5B-Instruct` | 前次运行路径，云端核实 |
| 3B、7B 模型 | `/home/waas/models/` 下对应模型目录 | 精确名称待核实 |
| 74 对训练集 | `data/derived/pku_alpaca7b_dpo_followup_strict_v1/dpo_train.jsonl` | 前次交接记录，本地无该目录 |
| 独立偏好 holdout | `data/derived/pku_alpaca7b_dpo_reviewed_v1/dpo_holdout.jsonl` | 路径依前次交接 |
| 7 对追加样本审核 | `data/derived/pku_alpaca7b_dpo_followup_shortlist_v2/shortlist_review_audited.csv` | 本地存在 |
| 74 对训练权重及中间检查点 | `results/09_repair/pku_dpo_layer27_v6_plus7_lr1e4/` | 本地仅同步 val 审核，云端核实权重 |
| 74 对 epoch 5 val 审核 | `results/09_repair/pku_dpo_layer27_v6_plus7_lr1e4/epoch_5_harmbench_review/` | 本地存在 |
| 67 对旧实验对照 | `results/09_repair/harmbench_v6_lr1e4/` | 本地存在 |
| epoch 3 XSTest、benign | `results/09_repair/independent_v6_plus7_epoch3/` | 本地存在 |
| epoch 5 独立测试和基线 | `results/09_repair/independent_v6_plus7_epoch5/` | 本地存在 |
| 评测输入 | `data/harmbench/harmbench_behaviors_text_{val,test}.csv`、`data/advbench/harmful_behaviors.csv`、`data/xstest/xstest_prompts.csv`、`data/benign_utility.jsonl` | 本地对应文件存在 |
| 训练、生成、评分入口 | `src/train_dpo.py`、`src/generate.py`、`src/score_results.py` | 云端版本需确认同步 |

云端核对 epoch 3/5 adapter：`find /Code/results/09_repair/pku_dpo_layer27_v6_plus7_lr1e4 -maxdepth 3 -name adapter_config.json -print`。在新实验前核对训练数据行数、checkpoint 和生成参数。不要从本地仅有审核目录推断云端检查点已丢失。

## 下一步计划

1. **收束 1.5B**：人工核查 epoch 3/5 的 XSTest safe 配对变化、benign 额外拒答和 HarmBench test 抽样；正常任务另检查完成度。不要只按 Score 2 最低选 epoch 5。
2. **补齐独立对照**：若云端尚未跑 epoch 3 的 HarmBench test、AdvBench，以和 epoch 5 完全一致的输入与解码口径补跑，保存逐样本结果并更新图表。正式 test 仅用于最终验证，不把测试题回灌训练或反复调参。
3. **小规模平衡实验**：从独立训练来源加入“安全请求应正常回答”的偏好对，针对过度拒答而不是继续堆拒答样本/epoch；固定数据拆分，审计重复与标签，以统一安全、效用口径评价。
4. **跨模型验证**：1.5B 的平衡性获得人工和独立测试支持后，再分别确定 3B、7B 敏感层并训练对照，不直接迁移第 27 层或 1.5B 超参数。已有跨模型退化结果见 `README_current_status_2026-10-29.md`。
5. **可选方法扩展**：以当前固定近似 DPO 为对照，逐项尝试论文的 MVA/敏感层扰动和激活表征约束并做消融；同步记录安全、正常任务和推理开销。这些部件目前尚未完成。

## 资料与注意事项

- 项目目标要求安全接近原始模型且不明显损害效用与效率；只增加拒答数不足以证明成功。不得把规则 Score 2 等同最终 ASR，或把 hook 实验当作已实现硬件加速。
- 上次交接（含 epoch 5 云端评测命令）：`handoff/repair_v6_plus7_2026-11-05.md`；DPO 说明：`logs/pku_dpo_layer27_runbook_2026-11-02.md`；论文：`../相关论文/激活近似/MinerU_markdown_Activation_Approximations__2107429045854089216.md`。
- `src.generate` 不覆写已有生成文件，复跑使用新目录。本文只记录汇总和路径，不包含提示词、模型回答或偏好对正文。
