#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."
OUTPUT_DIR="${1:?Usage: bash scripts/run_1p5b_dpo_activation_reg.sh OUTPUT_DIR [REG_WEIGHT]}"
REG_WEIGHT="${2:-0.1}"
MODEL_PATH="${MODEL_PATH:-/home/waas/models/Qwen2.5-1.5B-Instruct}"
TRAIN_DATA="${TRAIN_DATA:-data/derived/pku_alpaca7b_dpo_followup_strict_v1/dpo_train.jsonl}"
TRAIN_EPOCHS="${TRAIN_EPOCHS:-3}"
PYTHON_BIN="${PYTHON_BIN:-python}"
"$PYTHON_BIN" -c 'import json,sys; rows=[json.loads(line) for line in open(sys.argv[1], encoding="utf-8") if line.strip()]; assert len(rows)==74, f"Expected 74 pairs, got {len(rows)}"' "$TRAIN_DATA"
EXTRA_ARGS=()
if [[ -n "${TRAIN_MAX_PAIRS:-}" ]]; then
    EXTRA_ARGS+=(--max-pairs "$TRAIN_MAX_PAIRS")
fi
if [[ -n "${DIAGNOSTIC_HOLDOUT:-}" ]]; then
    EXTRA_ARGS+=(--diagnostic-holdout "$DIAGNOSTIC_HOLDOUT")
fi
"$PYTHON_BIN" -m src.train_dpo \
    --model "$MODEL_PATH" --train-data "$TRAIN_DATA" --output-dir "$OUTPUT_DIR" \
    --approximation hard_swish --layers 27 \
    --epochs "$TRAIN_EPOCHS" --learning-rate 1e-4 --gradient-accumulation-steps 8 \
    --beta 0.1 --logprob-reduction sum --lora-r 4 --lora-alpha 8 \
    --target-modules q_proj,v_proj --seed 42 --device cuda --dtype bfloat16 \
    --max-input-tokens 512 --max-completion-tokens 512 --save-every-epoch \
    --activation-reg-weight "$REG_WEIGHT" --activation-reg-layer 27 \
    --activation-reg-batch-size 2 --activation-reg-harmful-only "${EXTRA_ARGS[@]}"
