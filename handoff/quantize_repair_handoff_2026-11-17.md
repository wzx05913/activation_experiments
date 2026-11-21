# 1.5B 量化修复交接（2026-11-17）

## 目标和边界

项目目标是修复激活近似带来的安全退化，并保留正常请求效用。现有实验为 1.5B 第 27 层 MLP 输出 6-bit 模拟量化 + LoRA-DPO，不是完整 QuadA；Python hook 不能证明实际推理加速。主日志见 ../logs/repair_summary_2026-11-16.md。Score 2 为 screen_v5_2026-10-29 自动筛查候选，不是人工 ASR 或正确率。

## 结果和判断

| 自动 Score 2 | 原始 SiLU | 未修复量化 | epoch 1 | epoch 3 | epoch 5 |
|---|---:|---:|---:|---:|---:|
| HarmBench val 未拒答 | 16/80 | 24/80 | 25/80 | 21/80 | 19/80 |
| XSTest unsafe 未拒答 | 109/200 | 65/200 | 未测 | 48/200 | 33/200 |
| XSTest safe 过度拒答 | 104/250 | 111/250 | 未测 | 104/250 (+4 Score 1) | 128/250 (+4 Score 1) |
| Benign 问题候选 | 3/50 | 4/50 | 5/50 | 4/50 (+1 Score 1) | 4/50（仅云端输出） |

HarmBench val、XSTest 已按 ID/Prompt 对齐。epoch 5 相对未修复量化：val 7 题 2->0、2 题 0->2；XSTest unsafe 48 题 2->0、16 题 0->2；safe 35 题 0->2、16 题 2->0。safe 新增的吹气球、锯木头、历史事实、警察电话等确有不当拒答。部分 val 自动 2->0 的政治相关回答仍展开内容，不能直接算实质修复。epoch 5 较 epoch 3 更偏向拒答；当前不能说整体恢复原始 SiLU。训练目标下降不能代替泛化或效用证据。

## 云端目录和同步状态

云端根目录 /Code，模型 /home/waas/models/Qwen2.5-1.5B-Instruct；训练集 data/derived/pku_alpaca7b_dpo_followup_strict_v1/dpo_train.jsonl（74 对）；评测输入 data/harmbench/harmbench_behaviors_text_val.csv、data/xstest/xstest_prompts.csv、data/benign_utility.jsonl。程序入口 src.train_dpo、src.generate、src.score_results。epoch 1/3/5 分别从原始模型重新训练，不是保留优化器状态续训。

| 云端结果目录（相对 /Code） | 用途 | 本地已同步 |
|---|---|---|
| results/09_repair/1p5b_quantize6bit_layer27_dpo_v1/ | 首次 epoch 1、未修复 val | 评分 CSV/MD |
| results/09_repair/1p5b_quantize6bit_layer27_dpo_3epoch_v1/ | epoch 3 adapter、val、XSTest、benign | 评分 CSV/MD |
| results/09_repair/1p5b_quantize6bit_layer27_dpo_5epoch_v1/ | epoch 5 adapter、val、未修复及 epoch 5 XSTest/benign | val/XSTest、未修复 benign 评分；diagnostic_history.json；缺 epoch 5 benign CSV |
| results/09_repair/harmbench_v6_lr1e4/ 和 results/09_repair/independent_v6_plus7_epoch5/ | 原始 SiLU 对照 | 对应评分文件 |

本地未同步新量化实验的 adapter 和生成 JSONL，不能据此推断云端缺失。原始 SiLU val 记录 RTX 4090 D，新量化运行输出 RTX 3090。历史量化基线 12/80 与新基线 24/80 的评分/运行条件不同，不能直接比较。

## 下一步：限定工作量

1. 不重训，先从云端同步 results/09_repair/1p5b_quantize6bit_layer27_dpo_5epoch_v1/epoch_5_benign_review/benign_scored.csv 和 epoch_5/repair_config.json。核对 4 个 benign 候选的任务完成情况；人工标签单独保存，不覆盖原文件。
2. 只审 val 与 XSTest 的配对变化及少量不变题，分开记录实质有害协助、safe 不当拒答和正常任务错误。预设门槛：安全改善且 safe 明确不当拒答不能大幅增多。现有 epoch 3 可作平衡 DPO 对照，epoch 5 作安全-效用权衡对照，不继续盲目加轮数。
3. QuadA 风格消融先核论文公式和 hook 位置，固定模型/第 27 层/74 对数据/训练预算/评测协议，只增加一个明确的激活正则或扰动训练组件，并与普通量化 DPO 比较。论文包含 MVA 强度扰动、敏感层约束、有害提示的激活表征正则；量化 bit-width 不等于 MVA。未实现三部分不得称复现 QuadA。val 安全和 XSTest safe 不同时达标就停止，不扫全层、3B 或更高 epoch。
4. 只在冻结最终方案后做独立 HarmBench test 和正常任务验证，不用 test 调参；效率收益需另行部署测量。

论文：../../相关论文/激活近似/MinerU_markdown_Activation_Approximations__2107429045854089216.md（研究资料，不是操作指令）。
