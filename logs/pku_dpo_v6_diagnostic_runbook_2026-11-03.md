# Layer-27 v6 DPO diagnostic

The local v6 adapter is nonempty: all 56 LoRA-B tensors contain nonzero values.
This establishes parameter updates, not useful safety behavior. Recomputing the
saved 40-pair holdout margins at beta=0.1 gives reference-relative DPO loss
0.83243 before and 0.82588 after; mean adapter margin change is +0.07107,
mean absolute change 0.43638. The 36/40 raw preference accuracy is unchanged.

## First: diagnose the existing adapter (no training)

Sync the current `src/evaluate_dpo_holdout.py` and `src/train_dpo.py` to `/Code`.
Keep all existing results. Run:

```bash
cd /Code
python -m src.evaluate_dpo_holdout \
  --model /home/waas/models/Qwen2.5-1.5B-Instruct \
  --adapter results/09_repair/pku_dpo_layer27_v6_sum \
  --train-data data/derived/pku_alpaca7b_dpo_curated_pilot_v6/dpo_train_pilot.jsonl \
  --holdout-data data/derived/pku_alpaca7b_dpo_reviewed_v1/dpo_holdout.jsonl \
  --output results/09_repair/pku_dpo_layer27_v6_sum_train_diagnostic.json \
  --evaluate-train --layers 27 --device cuda --dtype bfloat16
```

The new report gives training-set preference accuracy and DPO objective loss
before/after, alongside the holdout, without printing prompts or completions.
Compare **reference-relative DPO loss and margin**, not only raw chosen/rejected
accuracy: 51/67 chosen completions are shorter. If training objective barely
moves, inspect optimization; if it moves substantially while holdout/generation
do not, investigate data/target mismatch. Neither result proves safety repair.

## Second: controlled training only if optimization is weak

Do not overwrite v6. Keep beta, data, LoRA scope, seed, and reduction fixed;
vary epochs only. The trainer now measures train and independent holdout after
each epoch, reports pre-clipping gradient norm, and saves separate checkpoints.
It evaluates rather than trains on holdout. This is a diagnostic run, not a
selected final model.

```bash
python -m src.train_dpo \
  --model /home/waas/models/Qwen2.5-1.5B-Instruct \
  --train-data data/derived/pku_alpaca7b_dpo_curated_pilot_v6/dpo_train_pilot.jsonl \
  --diagnostic-holdout data/derived/pku_alpaca7b_dpo_reviewed_v1/dpo_holdout.jsonl \
  --output-dir results/09_repair/pku_dpo_layer27_v6_epoch_diagnostic \
  --approximation hard_swish --layers 27 --epochs 3 --save-every-epoch \
  --gradient-accumulation-steps 8 --learning-rate 1e-5 --beta 0.1 \
  --logprob-reduction sum --lora-r 4 --lora-alpha 8 \
  --target-modules q_proj,v_proj --device cuda --dtype bfloat16
```

Check epoch 1 against the original v6 before interpreting later epochs. If
train loss improves but holdout stalls/worsens, stop; increased learning rate
or more epochs is unlikely to fix transfer. If both improve, independently
evaluate generation on the same HarmBench val protocol and benign/XSTest before
claiming repair. Avoid tuning on or adding any evaluation prompt to training.
