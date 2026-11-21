# WildJailbreak 修复小样本流程（2026-10-31）

目标：1.5B Hard-Swish layer27 的拒答修复先导实验。只取官方 `train/train.tsv`，不取 `eval/eval.tsv`；有害请求对应拒答，少量正常请求对应正常回答。不能用 HarmBench、AdvBench 或 `benign_utility.jsonl` 的评测样本训练。

先在 WildJailbreak 的 Hugging Face 页面登录并接受 AI2 使用条款。之后在云端 `/Code`，确认新脚本 `src/prepare_wildjailbreak_sft.py` 已同步：

```bash
cd /Code
test -f src/prepare_wildjailbreak_sft.py
hf auth login
hf download allenai/wildjailbreak train/train.tsv --repo-type dataset --local-dir data/wildjailbreak
python -m src.prepare_wildjailbreak_sft screen \
  --tsv data/wildjailbreak/train/train.tsv \
  --eval-root data \
  --output-dir data/derived/wildjailbreak_refusal_pilot_v1 \
  --harmful-train 160 --benign-train 40 \
  --harmful-holdout 40 --benign-holdout 10 --seed 42
```

查看 `selection_report.json`。逐条审核 `train_review.csv`、`holdout_review.csv` 的 `prompt`、`response`：有害请求确实有害、正常请求确实正常才将 `prompt_valid` 填 `yes`；有害回答安全拒答、正常回答真正完成合法任务才将 `response_valid` 填 `yes`，否则填 `no`。每一行两列都填完。若筛选结果不足，脚本会报错，此时降低样本数并记录原因，不能绕过审核。不要只按关键词全填 `yes`。

```bash
python -m src.prepare_wildjailbreak_sft export \
  --review-dir data/derived/wildjailbreak_refusal_pilot_v1 \
  --eval-root data \
  --output-dir data/derived/wildjailbreak_refusal_reviewed_v1
python -m src.audit_repair_data \
  --train data/derived/wildjailbreak_refusal_reviewed_v1/safety_train.jsonl \
  --test data/derived/wildjailbreak_refusal_reviewed_v1/safety_holdout.jsonl
```

先检查 `export_report.json` 的有害/正常比例、训练/留出数量与审核意见；尤其核查训练请求和 HarmBench/AdvBench 评测提示的语义近重复。确认后才运行：

```bash
python -m src.train_lora \
  --model /home/waas/models/Qwen2.5-1.5B-Instruct \
  --train-data data/derived/wildjailbreak_refusal_reviewed_v1/safety_train.jsonl \
  --output-dir results/09_repair/wildjailbreak_layer27_v1 \
  --approximation hard_swish --layers 27 \
  --epochs 2 --batch-size 1 --gradient-accumulation-steps 8 \
  --learning-rate 2e-4 --lora-r 8 --lora-alpha 16 \
  --device cuda --dtype bfloat16
```

训练只是得到 adapter，不代表修复已有效。用相同 HarmBench val 及解码参数比较未修复/修复，重点人工复核有害请求中原来失败的案例，并检查正常请求过度拒答；保持旧 adapter 与结果不覆盖。若效果可信，再做原始 SiLU + 同数据控制组、HarmBench test 和 AdvBench 最终评估。`train_lora.py` 当前不会强制校验 `reviewed` 字段，所以绝对不要把审核 CSV、自动候选或旧版 PKU JSONL 传给它。
