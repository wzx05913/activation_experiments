#!/usr/bin/env bash
# Run with: bash scripts/run_3b_single_layer_dpo.sh
set -euo pipefail
cd /Code

MODEL=/home/waas/models/Qwen2.5-3B-Instruct
TRAIN=data/derived/pku_alpaca7b_dpo_followup_strict_v1/dpo_train.jsonl
HOLDOUT=data/derived/pku_alpaca7b_dpo_reviewed_v1/dpo_holdout.jsonl
VAL=data/harmbench/harmbench_behaviors_text_val.csv
OUT=results/09_repair/3b_single_layer_dpo_v1
if [[ -z ${SCREENED_LAYERS:-} ]]; then
  echo "Set SCREENED_LAYERS to 1-2 validated layer indices, e.g. SCREENED_LAYERS=22" >&2
  exit 1
fi
read -r -a LAYERS <<< "$SCREENED_LAYERS"
if (( ${#LAYERS[@]} < 1 || ${#LAYERS[@]} > 2 )); then
  echo "Select 1-2 layers after screening." >&2
  exit 1
fi

for path in "$MODEL/config.json" "$TRAIN" "$HOLDOUT" "$VAL"; do
  if [[ ! -f "$path" ]]; then
    printf 'Missing: %s\n' "$path" >&2
    exit 1
  fi
done
if [[ $(wc -l < "$TRAIN") -ne 74 ]]; then
  echo 'Training data is not 74 lines; stop and check the reviewed dataset.' >&2
  exit 1
fi
if [[ -e "$OUT" ]]; then
  echo "Output exists: $OUT; choose a new OUT instead of overwriting results." >&2
  exit 1
fi
python - "$MODEL/config.json" "${LAYERS[@]}" <<'PY'
import json
import sys
with open(sys.argv[1], encoding='utf-8') as handle:
    n = json.load(handle)['num_hidden_layers']
layers = [int(value) for value in sys.argv[2:]]
if len(set(layers)) != len(layers) or any(layer < 0 or layer >= n for layer in layers):
    raise SystemExit(f'Invalid layer selection {layers}; model has {n} layers')
print(f'Selected exploratory layers: {layers} / {n} layers')
PY

mkdir -p "$OUT"
generate_val() {
  local name=$1
  shift
  python -m src.generate --model "$MODEL" --data "$VAL" \
    --output "$OUT/${name}.jsonl" --setting "3b_${name}_val" \
    --max-input-tokens 1024 --max-new-tokens 128 \
    --temperature 0 --top-p 1 --seed 42 --device cuda --dtype bfloat16 "$@"
  python -m src.score_results --input "$OUT/${name}.jsonl" \
    --output-dir "$OUT/${name}_review" --name harmbench_val --kind harmful
}

generate_val original_silu
for layer in "${LAYERS[@]}"; do
  generate_val "layer_${layer}_unrepaired" --approximation hard_swish --layers "$layer"
  adapter="$OUT/layer_${layer}_adapter"
  python -m src.train_dpo --model "$MODEL" --train-data "$TRAIN" \
    --diagnostic-holdout "$HOLDOUT" --output-dir "$adapter" \
    --approximation hard_swish --layers "$layer" \
    --epochs 5 --save-every-epoch --gradient-accumulation-steps 8 \
    --learning-rate 1e-4 --beta 0.1 --logprob-reduction sum \
    --lora-r 4 --lora-alpha 8 --target-modules q_proj,v_proj \
    --seed 42 --device cuda --dtype bfloat16
  for epoch in 3 5; do
    generate_val "layer_${layer}_epoch_${epoch}" \
      --adapter "$adapter/epoch_${epoch}" --approximation hard_swish --layers "$layer"
  done
done
