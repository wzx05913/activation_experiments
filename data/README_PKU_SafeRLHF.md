# PKU-SafeRLHF Download and Preparation

## Dataset format

PKU-SafeRLHF is preference data, not directly compatible with `src/train_lora.py`. The training script expects `prompt` and `response`. First download the source dataset, select a safe response using the safety and helpfulness labels, convert it to SFT JSONL, then audit it.

## Download

Run this in the cloud training environment:

```powershell
pip install -U datasets huggingface_hub
python -c "from datasets import load_dataset; ds=load_dataset('PKU-Alignment/PKU-SafeRLHF'); print(ds)"
```

If access requires authentication, sign in with the Hugging Face CLI. Do not commit access tokens or cache directories. Record dataset revision/configuration, split, download date, and selection rules. First inspect the returned split names and column names because dataset revisions/exports can differ; do not assume a particular response-field order. Use the training split for training, not the evaluation splits.

## Convert for SFT

Select the safer response, while checking that it is not substantially less helpful. Preserve the original pair and labels for possible future DPO use. Convert selected examples to:

```json
{"id":"pku_000001","category":"...","prompt":"...","response":"...","source":"PKU-SafeRLHF","reviewed":true}
```

Manually inspect a sample to ensure refusals do not contain actionable harmful steps, code, or materials. Do not train on evaluation prompts or their reference answers.

For a future DPO experiment, retain both responses and convert to `prompt`, `chosen`, and `rejected`; DPO is not the current mainline.

## Leakage checks

Audit the converted files with:

```powershell
python -m activation_experiments.src.audit_repair_data `
  --train activation_experiments/data/safety_train.jsonl `
  --test activation_experiments/data/safety_test.jsonl
```

Also check training prompts for overlap with `data/advbench/harmful_behaviors.csv`, `data/harmbench/harmbench_behaviors_text_test.csv`, `data/xstest/xstest_prompts.csv`, and `data/benign_utility.jsonl`.

Keep raw source files under `data/raw/pku_saferlhf/` and derived SFT data under `data/derived/`. Logs should contain counts and filtering statistics, not full prompts or answers.

## Project decision

Mainline: PKU-SafeRLHF -> safe-response selection -> SFT JSONL -> Hard-Swish-conditioned LoRA/SFT.

Optional later extension: use the preserved preference pairs for DPO comparison.