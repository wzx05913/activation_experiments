# PKU mixed repair pilot (2026-11-02)

Goal: retain the 160/40 previously reviewed harmful-request safe responses from
`data/derived/pku_alpaca7b_refusal_candidates_v1/`, and add independently
reviewed normal-task responses from the *train* split of PKU-SafeRLHF.
The earlier refusal-only adapter remains a negative pilot; do not overwrite it.

## What is ready

- `data/derived/pku_alpaca7b_mixed_candidates_v1/benign_train_review.csv`: 160 candidates.
- `data/derived/pku_alpaca7b_mixed_candidates_v1/benign_holdout_review.csv`: 40 candidates.
- Screening found 8,124 dual-safe-response, non-refusal candidates after exact
  evaluation exclusions and excluding the 200 harmful review prompts.
- These are **not** 200 confirmed benign examples. An initial spot check found
  safety-related, borderline, and clearly unsuitable prompts even when both
  responses have PKU safe labels. No mixed training JSONL has been exported.

PKU remains the harmful-refusal source, but its dual-safe subset is a poor
source of ordinary tasks. Prefer an **independent training-only** instruction
JSONL of `{"prompt": "...", "response": "..."}` pairs for the benign side.
Do not use `data/benign_utility.jsonl` or other evaluation data; ensure the
external source has an appropriate license and document its name/version.
Responses should be accurate and useful, ideally checked against the base
Qwen SiLU model's behavior. The script can screen such a file without training:

```bash
python -m src.prepare_pku_mixed screen-jsonl \
  --normal-train-jsonl data/raw/normal_instructions_train.jsonl \
  --source-name normal-instructions:train \
  --harmful-review-dir data/derived/pku_alpaca7b_refusal_candidates_v1 \
  --eval-root data \
  --benign-train 160 --benign-holdout 40 \
  --output-dir data/derived/normal_mixed_candidates_v1
```

The source file must be a separate training split; exact overlap is screened,
semantic overlap and answer quality still require human review. For this
preferred path, use `normal_mixed_candidates_v1` as `--benign-review-dir` in
the export command below. The PKU candidate CSVs remain an optional backup.

## Human review

Fill `prompt_benign` and `response_helpful` with `yes` or `no` for **every**
row in both CSV files. Mark `prompt_benign=yes` only for ordinary, authorized
tasks, not harmful or ambiguous requests. Mark `response_helpful=yes` only if
the response genuinely answers the task, is accurate enough for SFT, and does
not add unsafe material. Do not batch-fill yes. Discard examples with dubious
claims, odd wording, sensitive personal information, or semantic near-duplicates
of HarmBench/AdvBench/XSTest/benign evaluation prompts. Review the existing
harmful CSVs again if the 160 previous yes/yes labels were batch-assigned.

The exporter requires at least 80 approved benign train rows for 160 harmful
rows; training balances the groups by cycling the smaller group. Prefer close
to 160 approved normal tasks. If too few survive review, screen additional
candidates with another seed/new empty output directory, then review them
separately; do not silently relax the review gate.

## Export (after review)

Run from `/Code` after syncing the CSVs and code:

```bash
python -m src.prepare_pku_mixed export \
  --harmful-review-dir data/derived/pku_alpaca7b_refusal_candidates_v1 \
  --benign-review-dir data/derived/pku_alpaca7b_mixed_candidates_v1 \
  --eval-root data \
  --output-dir data/derived/pku_alpaca7b_mixed_reviewed_v1

python -m src.audit_repair_data \
  --train data/derived/pku_alpaca7b_mixed_reviewed_v1/safety_train.jsonl \
  --test data/derived/pku_alpaca7b_mixed_reviewed_v1/safety_holdout.jsonl
```

`export` rechecks exact train/holdout and evaluation overlap. Audit does not
prove semantic non-overlap or answer quality.

## Small mixed SFT (after export and audit)

```bash
python -m src.train_lora \
  --model /home/waas/models/Qwen2.5-1.5B-Instruct \
  --train-data data/derived/pku_alpaca7b_mixed_reviewed_v1/safety_train.jsonl \
  --output-dir results/09_repair/pku_mixed_layer27_v1 \
  --data-mode mixed --approximation hard_swish --layers 27 \
  --target-modules q_proj,v_proj \
  --epochs 1 --batch-size 1 --gradient-accumulation-steps 8 \
  --learning-rate 5e-5 --lora-r 4 --lora-alpha 8 \
  --device cuda --dtype bfloat16
```

This is a *mixed SFT* comparison, not DPO, QuadA, or SiLU logit distillation.
In mixed mode the script requires `kind` and `reviewed=true` on each row,
balances harmful/benign exposure per epoch, and records counts and LoRA target
modules. Historical runs retain the `legacy` default and their old parameters.

Evaluate the new adapter against the same unrepaired Hard-Swish layer27 setting,
with matching generation configuration, on harmful **and** benign held-out
sets. Apply the same manual rubric to both; use HarmBench val for pilot selection
and reserve HarmBench test/AdvBench for final comparison. Do not interpret a
lower training loss or automatic keyword score alone as successful repair.
