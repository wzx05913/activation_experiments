# PKU preference DPO pilot, Qwen2.5-1.5B layer 27 Hard-Swish

This replaces neither the original SFT adapters nor their evaluation. The
experiment is **fixed-approximation LoRA DPO**, inspired by the DPO term of
QuadA (Eq. 7/9), with an original SiLU reference and a layer-27 Hard-Swish
policy. It does not implement QuadA's MVA search, sensitive-layer perturbation,
or activation cosine regularizer. No training result has been obtained yet.

## Data already prepared locally

The data is in `data/derived/pku_alpaca7b_dpo_reviewed_v1/`. It was
recovered from the PKU-SafeRLHF `alpaca-7b` train Arrow and previously
human-reviewed refusal CSVs:
160 train pairs and 40 holdout pairs, no exact overlap with existing eval
sets. Each `chosen` is the reviewed safe refusal; `rejected` is the *same
prompt's* PKU-labeled unsafe response. The rejected response has **not**
been separately human-reviewed. Never train on holdout or use HarmBench/AdvBench
data for training. Do not print raw paired responses in reports.

To reproduce the export on `/Code` after syncing code, Arrow and review CSVs:

```bash
python -m src.prepare_pku_dpo \
  --arrow data/PKU-Alignment___pku-safe_rlhf/alpaca-7b/0.0.0/9421ffafec3fa40a1f1a7d567b4d525079477ecb/pku-safe_rlhf-train.arrow \
  --review-dir data/derived/pku_alpaca7b_refusal_candidates_v1 \
  --eval-root data \
  --output-dir data/derived/pku_alpaca7b_dpo_reviewed_v1
```

Use a new empty output directory if the export already exists. Transfer the
two exported JSONL files (and ideally the selection report) to the cloud.

## Train on /Code

```bash
cd /Code
python -m src.train_dpo \
  --model /home/waas/models/Qwen2.5-1.5B-Instruct \
  --train-data data/derived/pku_alpaca7b_dpo_reviewed_v1/dpo_train.jsonl \
  --output-dir results/09_repair/pku_dpo_layer27_v1 \
  --approximation hard_swish --layers 27 \
  --epochs 1 --gradient-accumulation-steps 8 \
  --learning-rate 1e-5 --beta 0.1 \
  --lora-r 4 --lora-alpha 8 --target-modules q_proj,v_proj \
  --device cuda --dtype bfloat16
```

The trainer first caches the original SiLU model's completion log-probabilities,
then fits only the LoRA policy with the approximation active. The reference
model has no LoRA or approximation; this avoids a second resident 1.5B model.
It uses gradient checkpointing, batch size 1, and no answer truncation. If a
pair exceeds the default token cap, increase the corresponding limit; do not
silently truncate or drop it. This is a single pilot, not a tuned result.

## Evaluate

Generate with the same HarmBench val prompts and decoding configuration as
the saved *unrepaired layer-27 Hard-Swish* file, with `--adapter` pointing to
`results/09_repair/pku_dpo_layer27_v1`. Keep the activation hook on layer 27. Compare
both to the original SiLU baseline (different control) and the previous SFT
pilots. Also run the existing benign/XSTest checks for over-refusal and
manually review harmful-case scoring. The 40 PKU holdout pairs are only a
preference diagnostic; they are not a replacement for HarmBench or utility.
Do not claim success from DPO loss alone. If the 160 pairs still fail to cover
HarmBench inducement types, changing the loss cannot repair that data gap.

```bash
python -m src.generate \
  --model /home/waas/models/Qwen2.5-1.5B-Instruct \
  --adapter results/09_repair/pku_dpo_layer27_v1 \
  --data data/harmbench/harmbench_behaviors_text_val.csv \
  --output results/09_repair/pku_dpo_layer27_v1_harmbench_val.jsonl \
  --setting pku_dpo_layer27_v1_harmbench_val \
  --approximation hard_swish --layers 27 \
  --max-input-tokens 1024 --max-new-tokens 128 \
  --temperature 0 --top-p 1 --device cuda --dtype bfloat16

python -m src.score_results \
  --input results/09_repair/pku_dpo_layer27_v1_harmbench_val.jsonl \
  --output-dir results/09_repair/pku_dpo_layer27_v1_review \
  --name harmbench_val --kind harmful
```
