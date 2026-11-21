# 1.5B layer-27 Hard-Swish DPO repair handoff (2026-11-05)

## Current result and limits

- Model: `/home/waas/models/Qwen2.5-1.5B-Instruct`; code root: `/Code`.
- Training: 67 reviewed pairs plus 7 audited additions, total 74. DPO with original SiLU frozen reference, Hard-Swish at layer 27, LoRA q_proj/v_proj, r=4, alpha=8, beta=0.1, sum log-probabilities, learning rate 1e-4, accumulation 8, seed 42, 5 epochs.
- HarmBench val (80 prompts), automatic screen Score 0/1/2: original SiLU 64/0/16; unrepaired Hard-Swish 53/0/27; 67-pair epoch 5 72/0/8; 74-pair epochs 1, 3, 5: 59/0/21, 67/0/13, 74/0/6.
- The 74-pair epoch-5 CSV is synced locally and agrees with 74/0/6. Epoch-1/3 counts are user-provided cloud summaries; their per-row files are not synced here. Score 2 is a keyword-screening candidate count, not human-verified ASR. The 74-pair 74 Score-0 cases are not yet manually audited; a 2-sample gain over 67 pairs is insufficient to establish improved generalization.
- Training log reported improving reference-relative holdout objective while raw preference accuracy stayed 36/40. This is a preference diagnostic, not an independent generation safety metric.

## Cloud paths

| Purpose | `/Code`-relative path |
|---|---|
| 74-pair train | `data/derived/pku_alpaca7b_dpo_followup_strict_v1/dpo_train.jsonl` |
| Independent DPO holdout | `data/derived/pku_alpaca7b_dpo_reviewed_v1/dpo_holdout.jsonl` |
| Audited shortlist | `data/derived/pku_alpaca7b_dpo_followup_shortlist_v2/shortlist_review_audited.csv` |
| 74-pair adapter and epoch checkpoints | `results/09_repair/pku_dpo_layer27_v6_plus7_lr1e4/` |
| 74-pair val evaluation | `results/09_repair/pku_dpo_layer27_v6_plus7_lr1e4/epoch_5_harmbench_review/` |
| 67-pair comparison | `results/09_repair/harmbench_v6_lr1e4/` |
| New independent evaluation | `results/09_repair/independent_v6_plus7_epoch5/` |

The adapter/checkpoint directory was reported by the cloud training process. Only the epoch-5 scored val review is present in the local synced directory; confirm exact checkpoint name with `find results/09_repair/pku_dpo_layer27_v6_plus7_lr1e4 -maxdepth 2 -name adapter_config.json -print` before running the commands below. The cloud train file was made with a one-off command, not a committed data-export script.

## Next evaluation

1. Freeze the 74-pair epoch-5 checkpoint and the same decoding settings. Evaluate HarmBench **test** and AdvBench against original SiLU and unrepaired layer-27 Hard-Swish. Use fresh outputs if older baselines differ in input protocol or generation parameters.
2. Evaluate XSTest safe/unsafe separately and benign utility for overrefusal and task completion. The automatic Score 0 on benign only means non-refusal, not correctness.
3. Audit all Score-1/2 cases and a sample of Score-0 responses, especially paired changes; retain manual labels separately. Do not select a checkpoint based on the held-out test sets.
4. Report paired per-ID changes, coverage, confidence intervals, and the remaining limitations. Only then decide whether more independently sourced training data or cross-model repair is warranted. Latency/throughput claims require a measured implementation, not merely replacing an activation in a hook.

## Cloud commands: epoch 5 only

```bash
cd /Code
MODEL=/home/waas/models/Qwen2.5-1.5B-Instruct
ADAPTER=results/09_repair/pku_dpo_layer27_v6_plus7_lr1e4
OUT=results/09_repair/independent_v6_plus7_epoch5
find "$ADAPTER" -maxdepth 2 -name adapter_config.json -print
test -f "$ADAPTER/adapter_config.json" || { echo 'Set ADAPTER to the epoch-5 checkpoint shown above'; exit 1; }
mkdir -p "$OUT"

for SPEC in \
  harmbench_test:data/harmbench/harmbench_behaviors_text_test.csv:harmful \
  advbench:data/advbench/harmful_behaviors.csv:harmful \
  xstest:data/xstest/xstest_prompts.csv:xstest \
  benign:data/benign_utility.jsonl:benign; do
  IFS=: read -r NAME DATA KIND <<< "$SPEC"
  for SETTING in original_silu unrepaired_hard_swish repaired_epoch5; do
    EXTRA=()
    case "$SETTING" in
      original_silu) ;;
      unrepaired_hard_swish) EXTRA=(--approximation hard_swish --layers 27) ;;
      repaired_epoch5) EXTRA=(--adapter "$ADAPTER" --approximation hard_swish --layers 27) ;;
    esac
    python -m src.generate \
      --model "$MODEL" --data "$DATA" \
      --output "$OUT/${SETTING}_${NAME}.jsonl" \
      --setting "${SETTING}_${NAME}" "${EXTRA[@]}" \
      --max-input-tokens 1024 --max-new-tokens 128 \
      --temperature 0 --top-p 1 --seed 42 --device cuda --dtype bfloat16
    python -m src.score_results \
      --input "$OUT/${SETTING}_${NAME}.jsonl" \
      --output-dir "$OUT/${SETTING}_${NAME}_review" \
      --name "$NAME" --kind "$KIND"
  done
done
```

`src.generate` refuses existing output files, and `src.score_results` preserves existing review files. Set `OUT` to a fresh directory on rerun. If input validation fails, determine a shared `--max-input-tokens` for *all* three conditions on that dataset and rerun uniformly; do not truncate one condition. HarmBench test is the held-out 320-sample set; the test results must not be fed back into training.
