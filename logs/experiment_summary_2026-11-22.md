# 激活近似安全修复：实验总结日志

项目：20261499「推理加速的深度思考大模型安全防护方法研究」｜汇总日期：2026-11-22

## 1. 结题摘要

项目在 Qwen2.5-Instruct 1.5B、3B 和 7B 上建立原始模型对照，比较 MLP 输出扰动和内部 SiLU 近似，并用 1.5B 分层实验筛选候选安全敏感层。随后在索引 27（从零计数）Hard-Swish 条件下，以 74 对经筛选的偏好数据做轻量 DPO。训练 3 个 epoch 后，HarmBench test 有害输出候选由未修复的 **103/320 降至 73/320**，低于同一轮原始 SiLU 的 **83/320**；XSTest unsafe 候选也下降，但 safe 误拒答增加。全层重训只有部分改善，表征正则没有显示稳定增益。这些实验形成了“基线、近似退化、分层定位、修复、权衡”的闭环；没有证明硬件部署提速，也不是完整 QuadA 复现。

## 2. 数据与证据口径

| 生成评测数据 | 规模 | 用途 |
|---|---:|---|
| HarmBench | val 80；test 320 | 筛选近似方案、比较修复前后有害输出 |
| AdvBench | 520 | 补充有害请求评测；结果不能代替 HarmBench |
| XSTest | safe 250；unsafe 200 | 分开检查安全提示误拒答与不安全提示的不当回应 |
| 项目自建 Benign | 50 | 检查一般任务回答效用 |

四套均为生成评测。训练另用 PKU-SafeRLHF `alpaca-7b`，经筛选得到 67 对，补充 7 对后为 74 对；另有 40 对偏好留出诊断，不是 HarmBench val，也不是第五套生成评测。WildJailbreak 和 mixed SFT 留有运行方案，不计入已验证的修复结果。训练集的筛选、扩充和复审见[历史日志索引](../PROJECT_INDEX.md#历史日志与交接)与 [67+7 交接](../handoff/repair_v6_plus7_2026-11-05.md)。

本文注明“自动候选”时，数字是规则评分筛出来的数量，不是人工最终攻击成功率（ASR）；无效输出也不等于明确拒答。人工修正版、候选复核估计和自动筛查分别记录。不同阶段提示、生成和评分口径存在差异，只在同一轮对照内比较；相同模型与样本数不代表相同基线。

## 3. 基线、输出扰动和多种 SiLU 近似（10 月）

1. **基线与输出扰动。** 建立未修改激活的 1.5B、3B、7B 对照。早期在 MLP 输出上施加 Gaussian、Laplace、稀疏化与 4/6/8-bit 模拟量化，探索扰动形式、位置和强度。这些是 MLP 输出级扰动，不是替换内部 SiLU，也不等同于权重量化。保留生成明细，分析时区分输出失效和真实拒答。证据：[10-28 交接](activation_experiment_handoff_2026-10-28.md)、[1.5B 输出扰动结果](../results/04_approximation/qwen25_1p5b/)。
2. **内部激活函数近似。** 改为替换 MLP 内部 SiLU。1.5B HarmBench val 早期同轮自动 Score 2 候选：原版 **7/80**、Hard-Swish **30/80**、分段线性 **10/80**、分段多项式 **11/80**。Hard-Swish 在该轮退化最大，因此作为分层与修复主线；不能据此断言它在任何环境下都是最危险配置。证据：[10-29 交接](activation_experiment_handoff_2026-10-29.md)、[SiLU 近似原始结果](../results/05_silu_approximation/)。这组 7/80 不和后期修复 val 的 16/80 拼接。

## 4. 分层筛查和跨模型验证（10 月）

1.5B 只对指定层或全层替换 SiLU，保留同轮原始基线：

| 设置 | HarmBench test 自动候选 /320 | HarmBench test 候选复核估计 /320 | AdvBench 自动候选 /520 |
|---|---:|---:|---:|
| 原始 SiLU | 43 | 43 | 0 |
| Hard-Swish 索引 14 | 55 | 49 | 2 |
| Hard-Swish 索引 27 | 69 | 63 | 2 |
| Hard-Swish 全层 | 156 | 146 | 50 |

候选复核估计不是 320 条全量重标。索引 14 和 27 是当前设置下的候选敏感层，选择索引 27 作为主要修复对象，不作跨模型普遍性结论。跨模型 HarmBench val 人工修正版中，3B 的 Score 2 从 **35/80** 增至 **54/80**，7B 从 **38/80** 增至 **58/80**。同轮 Benign 的 Score 0 分别保持 **49/50** 和 **50/50**。对更多跨模型 test、AdvBench 和延迟记录，见 [10-29 主线汇总](../figures/summary_07_08_2026-10-29-v2/activation_mainline_results_2026-10-29.md)及 [分层原始结果](../results/08_test_adv/)。Python hook 下的延迟不是部署加速证据。

## 5. 主修复：1.5B 索引 27 Hard-Swish（11 月）

在 Hard-Swish 始终开启的条件下训练 LoRA，只更新 `q_proj`、`v_proj` adapter，冻结原始 SiLU 模型作为 DPO 参考。主设置为 `r=4`、`alpha=8`、`beta=0.1`、completion log-prob 求和、学习率 `1e-4`、梯度累积 8、seed 42。详见 [DPO 运行手册](pku_dpo_layer27_runbook_2026-11-02.md)和 [74 对交接](../handoff/repair_v6_plus7_2026-11-05.md)。

| 评测（自动候选） | 原始 SiLU | 未修复 Hard-Swish | 74 对，3 epoch | 74 对，5 epoch |
|---|---:|---:|---:|---:|
| HarmBench test 有害输出 /320 | 83 | 103 | 73 | 30 |
| AdvBench 有害输出 /520 | 1 | 3 | 3 | 2 |
| XSTest unsafe 不当回应 /200 | 109 | 120 | 93 | 42 |
| XSTest safe 误拒答 /250 | 104 | 95 | 125 | 175 |
| Benign 异常 /50 | 3 | 3 | 6 | 7 |

3 epoch 相比未修复组，HarmBench test 候选少 30 条（9.4 个百分点），也比同轮原版少 10 条（3.1 个百分点）；AdvBench 没有改善。5 epoch 在 HarmBench test 和 XSTest unsafe 上更低，但误拒答更严重。XSTest safe 人工复核中，原版、3 epoch、5 epoch 的明确拒答分别为 **87、106、163/250**，边界类另记。**推荐以 3 epoch 作为安全性与过拒绝的折中，不表述成全面恢复或全面超过原版。**

历史 HarmBench val 自动候选（各 80 条）：原版 16、未修复 27、67 对训练 5 epoch 为 8；74 对训练 1、3、5 epoch 分别为 21、13、6。74 对的 1、3 epoch val 数字来自历史云端汇总；它们与后续表征正则消融不是同一轮。11-10 交接所说的 3 epoch test 尚未同步，是当时的状态，应以 [11-11 完整对照](../figures/repair_epoch3_epoch5_comparison/full_comparison_2026-11-11.md)、[XSTest 人工复核](../figures/repair_epoch3_epoch5_comparison/audited_xstest_summary_2026-11-10.md)和 [11-16 修复综述](repair_summary_2026-11-16.md)为准。

## 6. 全层修复、跨模型修复与 6-bit 补充（11 月）

- **1.5B 全层 Hard-Swish。** HarmBench val 自动候选（各 80 条）：原版 **16**、全层未修复 **41**、迁移索引 27 adapter **39**、在全层条件下重新 DPO 1 epoch **45**、3 epoch **36**。全层 3 epoch 比未修复少 5 条，但仍比原版多 20 条；没有全层 XSTest/Benign 证据，不能判断是否有过拒绝代价。见 [11-16 综述](repair_summary_2026-11-16.md)；[本地全层归档](../results/09_repair/1p5b_dpo_all_hard_swish_v1/)已有权重、配置、诊断和 1/3 epoch 生成明细，旧有“仅 CSV”是当时状态。
- **3B 补充修复。** 在该轮独立协议下，HarmBench val 原版、未修复、DPO 3 epoch、5 epoch 分别为 **39、57、57、46/80**；XSTest unsafe 的原版、未修复、5 epoch 为 **67、122、138/200**。单项回落不等于复现 1.5B 的修复效果。不要将这轮 39/80 基线与前期人工修正版 35/80 混用。见 [11-16 综述](repair_summary_2026-11-16.md)。
- **1.5B 索引 27 输出 6-bit 模拟量化。** 使用直通梯度训练，不是权重量化。HarmBench val 原版、未修复、1/3/5 epoch 为 **16、24、25、21、19/80**。XSTest unsafe 原版、未修复、3/5 epoch 为 **109、65、48、33/200**；safe 为 **104、111、104、128/250**。不同数据集方向不完全一致，5 epoch 有过拒绝代价，且硬件与部分主线运行不同。见 [11-17 量化交接](../handoff/quantize_repair_handoff_2026-11-17.md)。[3 epoch 本地归档](../results/09_repair/1p5b_quantize6bit_layer27_dpo_3epoch_v1/)已有权重、配置及生成明细；5 epoch 仍缺 Benign 逐题评分。

## 7. DPO 表征正则权重消融（11 月）

在 1.5B 索引 27 Hard-Swish 条件下，以 74 对数据训练 3 epoch，各 30 次参数更新。正则只拉近有害提示最后一个有效 token 在索引 27 block 输出上的表示，不直接约束拒答行为或正常/有害表示分离。`0.1` 组首次正则梯度非零，说明该项参与了优化；训练批次在线余弦不能证明验证集上的表征聚集。

| 自动候选评测 | 普通 DPO（0） | DPO + 正则 0.1 | DPO + 正则 0.5 |
|---|---:|---:|---:|
| HarmBench val 有害输出 /80 | 10 | 12 | 11 |
| XSTest safe 误拒答 /250 | 122 | 122 | 未评测 |
| XSTest unsafe 不当回应 /200 | 87 | 94 | 未评测 |

逐条复核发现，一条邮件样本三组均未拒答，普通组被规则误读为拒答；另一条 SQL 注入评测样本普通组与 `0.5` 组明确拒答，`0.1` 组未拒答。两条样例用于说明自动口径的局限，不能推出全量人工 ASR，也不能说三组输出完全相同。`0.5` 组评测记录为 RTX 3090，另外两组为 RTX 5090；`0.5` 的 XSTest 未运行，不能写成“差不多”。在现有对照中，**提高权重到 0.5 没有显示额外收益，现有数据也无法把未见收益单独归因于权重或样本量**。本轮作为探索性消融归档即可，不用挑一个正则权重作为主成果。

证据：[表征正则运行交接](../handoff/activation_regularizer_runbook_2026-11-19.md)、[权重 0 与 0.1 原始评测](../results/09_repair/layer27_dpo_reg_eval_v1/)、[权重 0.5 原始评测](../results/09_repair/layer27_dpo_reg_w050_eval_v1/)。

## 8. 结题采用范围与归档缺口

- **正文**：用多种 SiLU 近似和分层对照说明退化与定位；用 74 对、3 epoch 的同轮四数据集对照说明定向修复和过拒答权衡。保留自动候选的限定词；人工复核使用专门口径。
- **附录**：收录 epoch 5、全层与 3B 修复、6-bit、表征正则的对照和局限。表征正则是简化探索，不声称完成全部 QuadA 模块、正式机制验证或真实部署加速。
- **本地复现**：已索引代码、历史运行记录、主要结果与部分 adapter；74 对正式训练输入 `data/derived/pku_alpaca7b_dpo_followup_strict_v1/dpo_train.jsonl` 本地未见，应标为云端记录、待归档，而非完整复现包。缺项不以估计值填充，也不要求为本次文档整理重训。
- **材料入口**：见 [实验 README](../README.md)和 [项目文件索引](../PROJECT_INDEX.md)。实验总结用于支持验收书撰写，不代替实际填报，也不推断能否免答辩。
