# PKU 拒答数据重筛（2026-10-31）

## 现状

旧版 `data/derived/pku_alpaca7b_pilot_v1/safety_train.jsonl` 主要依据回答的安全标签选择，并非专门的“有害请求 + 拒答”数据；按当前拒答规则，250 条中只有 29 条命中明确拒答。旧版适配器仅作历史对照，不宜把其结果当作拒答专项修复结论。

新版只读取 PKU-SafeRLHF `alpaca-7b` 的 train Arrow：要求一条回答标安全、另一条标不安全且严重程度大于 0；安全回答须在开头明确拒答、总长不超过 300 字符，并排除规则识别的明显有害续写。去重并排除本项目评测提示后，有 267 条自动候选。固定随机种子 42，抽取训练候选 160 条和留出候选 40 条。文件为 `data/derived/pku_alpaca7b_refusal_candidates_v1/train_review.csv`、`holdout_review.csv`；其余 67 条未纳入本轮。

**这些只是待审核候选，不是训练集。** 另一回答“不安全”的标签、危害类别、严重程度均不是请求有害的标签。规则检测到拒答也不保证回答后文安全。两个审核字段目前均为空。

## 审核口径

逐条看 `prompt` 和选中的 `response`，只在请求确实有害时把 `prompt_harmful` 标为 `yes`；正常信息请求、含糊或合法防护请求标 `no`。只在回答明确拒绝该有害请求且没有可执行危害内容时把 `response_safe` 标为 `yes`；含操作建议、绕过方法或虽有歉意但没有拒绝的回答标 `no`。两列都必须填 `yes` 或 `no`；只导出双 `yes`。`unsafe_response_categories` 仅作为审核线索。审核后须确认各类危害请求的覆盖，必要时再从未抽取的候选增补，不要强行凑数。

在 `/Code`（同步脚本和候选 CSV 后）导出：

```bash
python -m src.prepare_pku_refusal export \
  --review-dir data/derived/pku_alpaca7b_refusal_candidates_v1 \
  --eval-root data \
  --output-dir data/derived/pku_alpaca7b_refusal_reviewed_v1
python -m src.audit_repair_data \
  --train data/derived/pku_alpaca7b_refusal_reviewed_v1/safety_train.jsonl \
  --test data/derived/pku_alpaca7b_refusal_reviewed_v1/safety_holdout.jsonl
```

导出会拒绝未填完的审核表、空的训练/留出结果、重复提示与评测提示的精确重叠。最终仍需人工检查训练与评测的语义近重复。不要用 `benign_utility.jsonl` 评测数据训练；正常任务仅用于过度拒答评估，若要加入训练需独立筛选、审核并保证隔离。修复比较沿用 layer27 未修复、修复及原始 SiLU 控制组，先看 HarmBench 有害请求案例，再综合自动评分和人工复核，不能只看训练 loss 或自动分数。
