# XSTest 人工审核对齐统计（2026-11-10）

## 统计范围和统一口径

按 ID、Prompt、XSTest 标签对齐原始 SiLU、74 对 DPO epoch 3 与 epoch 5 的审核文件。三组均为相同的 450 题（safe 250、unsafe 200），无重复 ID。本表统计审核给出的拒答性质，不等于规则 Score 2、人工核定 ASR 或正常任务正确率。

- 明确拒答：原始 SiLU 的“真拒答”；修复模型的“真实拒答”“实际为拒答”。
- 非拒答：原始 SiLU 的“未拒答”；修复模型的“非拒答，正常作答”与“非真拒答：直接作答…”标签。
- 边界：原始 SiLU 的“部分拒答”；修复模型其他“非真拒答：…”或“非拒答：…”标签，包括先声明后作答、澄清、无信息和回避。不将这些样本强行并入明确拒答或正常作答。

## XSTest safe（各 250 题）

| 审核归类 | 原始 SiLU | epoch 3 | epoch 5 |
|---|---:|---:|---:|
| 明确拒答 | 87 | 106 | 163 |
| 非拒答 | 140 | 106 | 66 |
| 边界 | 23 | 38 | 21 |

按同题 ID 对照，40 题从 epoch 3 的非拒答变为 epoch 5 的明确拒答。epoch 3 明确拒答比原始 SiLU 多 19 题，epoch 5 多 76 题。因此 epoch 3 与原始模型相对接近，但并非相同；epoch 5 明显更容易拒答。不同模型的边界数不同，不能仅凭非拒答数判断回答质量。

## XSTest unsafe（各 200 题）

| 审核归类 | 原始 SiLU | epoch 3 | epoch 5 |
|---|---:|---:|---:|
| 明确拒答 | 172 | 200 | 200 |
| 非拒答 | 24 | 0 | 0 |
| 边界 | 4 | 0 | 0 |

此前自动规则统计的 epoch 3 93/200、epoch 5 42/200 是“未拒答候选”，不能与这里基于审核标签的明确拒答数混为一谈。是否产生实质有害内容仍需单独判定。

## 审核文件与限制

以下路径相对于 `activation_experiments/`：

- 原始 SiLU：`results/09_repair/independent_v6_plus7_epoch5/original_silu_xstest_review/xstest_scored_audited.csv`，使用 `复核_拒答判定`。
- epoch 3：`results/09_repair/independent_v6_plus7_epoch3/repaired_epoch3_xstest_review/xstest_scored_audited.csv`，使用 `审核-拒答性质`。
- epoch 5：`results/09_repair/independent_v6_plus7_epoch5/repaired_epoch5_xstest_review/xstest_scored_audited_v2.csv`，使用 `审核-拒答性质`。

三份文件的 `人工评分` 仍为空，原始 SiLU 的 `人工终审` 也为空；这里依据新增审核列进行统一归类，不宣称已完成独立终审。现有 `summary.csv` 和图表仍采用自动规则口径，不能直接把本表数值当作旧图的 Score 2。