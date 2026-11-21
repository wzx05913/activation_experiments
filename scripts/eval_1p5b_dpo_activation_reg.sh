#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."
OUT="${1:?Usage: bash scripts/eval_1p5b_dpo_activation_reg.sh OUTPUT_DIR [--regularized-only]}"
MODE="${2:-both}"
if [[ "$MODE" != both && "$MODE" != --regularized-only ]] || (( $# > 2 )); then
    printf 'Usage: bash scripts/eval_1p5b_dpo_activation_reg.sh OUTPUT_DIR [--regularized-only]\n' >&2
    exit 2
fi
MODEL_PATH="${MODEL_PATH:-/home/waas/models/Qwen2.5-1.5B-Instruct}"
PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
CONTROL_ADAPTER="${CONTROL_ADAPTER:-results/09_repair/layer27_dpo_reg_control_v1}"
REG_ADAPTER="${REG_ADAPTER:-results/09_repair/layer27_dpo_reg_w010_v1}"
HARM_VAL=data/harmbench/harmbench_behaviors_text_val.csv
XSTEST=data/xstest/xstest_prompts.csv

adapters=("$REG_ADAPTER")
if [[ "$MODE" == both ]]; then
    adapters+=("$CONTROL_ADAPTER")
fi
for path in "$MODEL_PATH/config.json" "$HARM_VAL" "$XSTEST"; do
    if [[ ! -f "$path" ]]; then
        printf 'Missing required file: %s\n' "$path" >&2
        exit 1
    fi
done
for adapter in "${adapters[@]}"; do
    if [[ ! -f "$adapter/adapter_config.json" ]]; then
        printf 'Missing adapter config: %s\n' "$adapter" >&2
        exit 1
    fi
    if [[ ! -f "$adapter/adapter_model.safetensors" && ! -f "$adapter/adapter_model.bin" ]]; then
        printf 'Missing adapter weights: %s\n' "$adapter" >&2
        exit 1
    fi
done
if [[ "$MODE" == both && "$CONTROL_ADAPTER" -ef "$REG_ADAPTER" ]]; then
    printf 'Control and regularized adapters must be different directories.\n' >&2
    exit 1
fi
"$PYTHON_BIN" -c 'import torch, transformers, peft; torch.ones(1, device="cuda").item()'
mkdir -p -- "$(dirname -- "$OUT")"
mkdir -- "$OUT"

evaluate() {
    local group="$1" adapter="$2" dataset="$3" data_path="$4" kind="$5"
    local responses="$OUT/$group/$dataset.jsonl"
    "$PYTHON_BIN" -m src.generate --model "$MODEL_PATH" --adapter "$adapter" \
        --data "$data_path" --output "$responses" --setting "${group}_${dataset}" \
        --approximation hard_swish --layers 27 --strength 1.0 \
        --max-input-tokens 1024 --max-new-tokens 128 \
        --temperature 0 --top-p 1 --seed 42 --device cuda --dtype bfloat16
    "$PYTHON_BIN" -m src.score_results --input "$responses" \
        --output-dir "$OUT/$group" --name "$dataset" --kind "$kind"
}

if [[ "$MODE" == both ]]; then
    evaluate dpo_control "$CONTROL_ADAPTER" harmbench_val "$HARM_VAL" harmful
fi
evaluate dpo_regularized "$REG_ADAPTER" harmbench_val "$HARM_VAL" harmful
if [[ "$MODE" == both ]]; then
    evaluate dpo_control "$CONTROL_ADAPTER" xstest "$XSTEST" xstest
fi
evaluate dpo_regularized "$REG_ADAPTER" xstest "$XSTEST" xstest
printf 'Evaluation outputs: %s\n' "$OUT"
