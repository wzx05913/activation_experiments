"""Measure held-out chosen/rejected preference under SiLU and Hard-Swish."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch

from .activation_hooks import ActivationApproximation, HookConfig
from .train_dpo import completion_logprob, encode_pair, validate_pairs
from .utils import device_and_dtype, read_jsonl


def summarize(margins: list[float]) -> dict:
    if not margins:
        raise ValueError("No preference margins")
    return {"chosen_preferred": sum(x > 0 for x in margins),
            "rejected_preferred": sum(x < 0 for x in margins),
            "ties": sum(x == 0 for x in margins),
            "preference_accuracy": sum(x > 0 for x in margins) / len(margins),
            "mean_margin": sum(margins) / len(margins)}


def objective_diagnostic(reference: list[float], before: list[float], after: list[float], beta: float) -> dict:
    if not reference or len(reference) != len(before) or len(before) != len(after) or beta <= 0:
        raise ValueError("Expected matched nonempty margins and positive beta")
    def metrics(policy):
        logits = [beta * (value - ref) for value, ref in zip(policy, reference)]
        return {"mean_logit": sum(logits) / len(logits),
                "mean_loss": sum(max(0, -value) + math.log1p(math.exp(-abs(value))) for value in logits) / len(logits),
                "positive_relative_to_reference": sum(value > 0 for value in logits)}
    deltas = [new - old for old, new in zip(before, after)]
    return {"unrepaired": metrics(before), "repaired": metrics(after),
            "mean_adapter_margin_delta": sum(deltas) / len(deltas),
            "mean_absolute_adapter_margin_delta": sum(abs(value) for value in deltas) / len(deltas),
            "margin_increased": sum(value > 0 for value in deltas)}


def check_split(holdout: list[dict], train: list[dict]) -> None:
    validate_pairs(holdout)
    validate_pairs(train)
    train_ids = {row["id"] for row in train}
    train_prompts = {" ".join(row["prompt"].split()).casefold() for row in train}
    for row in holdout:
        if row["id"] in train_ids or " ".join(row["prompt"].split()).casefold() in train_prompts:
            raise ValueError(f"Train/holdout overlap: {row['id']}")


def measure(model, pairs, device) -> tuple[list[float], list[float]]:
    model.eval()
    margins, per_token_margins = [], []
    with torch.inference_mode():
        for chosen, rejected in pairs:
            good = float(completion_logprob(model, chosen, device))
            bad = float(completion_logprob(model, rejected, device))
            good_len = chosen[0].shape[1] - chosen[1]
            bad_len = rejected[0].shape[1] - rejected[1]
            margins.append(good - bad)
            per_token_margins.append(good / good_len - bad / bad_len)
    return margins, per_token_margins


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True)
    p.add_argument("--adapter", required=True)
    p.add_argument("--holdout-data", required=True)
    p.add_argument("--train-data", required=True)
    p.add_argument("--evaluate-train", action="store_true", help="Measure training-set preference without response text")
    p.add_argument("--output", required=True)
    p.add_argument("--layers", type=int, default=27)
    p.add_argument("--max-input-tokens", type=int, default=512)
    p.add_argument("--max-completion-tokens", type=int, default=512)
    p.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    p.add_argument("--dtype", choices=["auto", "float16", "bfloat16", "float32"], default="auto")
    args = p.parse_args()
    out = Path(args.output)
    if out.exists():
        raise FileExistsError(f"Preserving existing evaluation: {out}")
    if min(args.max_input_tokens, args.max_completion_tokens) <= 0:
        p.error("Token limits must be positive")
    adapter = Path(args.adapter)
    config = json.loads((adapter / "repair_config.json").read_text(encoding="utf-8"))
    reduction = config.get("logprob_reduction", "sum")
    beta = config.get("beta", 0.1)
    if reduction not in {"sum", "mean"}:
        raise ValueError("Unknown adapter log-probability reduction")
    if (config.get("model") != args.model or config.get("approximation") != "hard_swish"
            or config.get("layers") != args.layers or config.get("reference") != "original_silu_frozen"):
        raise ValueError("Adapter training configuration does not match model/layer/reference")
    holdout = read_jsonl(args.holdout_data)
    train = read_jsonl(args.train_data)
    check_split(holdout, train)

    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    if tokenizer.eos_token_id is None:
        raise ValueError("Tokenizer needs an EOS token")
    pairs = [encode_pair(tokenizer, row, args.max_input_tokens, args.max_completion_tokens)
             for row in holdout]
    train_pairs = ([encode_pair(tokenizer, row, args.max_input_tokens, args.max_completion_tokens)
                    for row in train] if args.evaluate_train else [])
    device, dtype = device_and_dtype(args.device, args.dtype)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype).to(device)
    model.config.use_cache = False
    silu, silu_per_token = measure(model, pairs, device)
    if args.evaluate_train:
        train_silu, train_silu_per_token = measure(model, train_pairs, device)
    with ActivationApproximation(model, HookConfig(kind="hard_swish", layers=(args.layers,))):
        unrepaired, unrepaired_per_token = measure(model, pairs, device)
        if args.evaluate_train:
            train_unrepaired, train_unrepaired_per_token = measure(model, train_pairs, device)
    model = PeftModel.from_pretrained(model, str(adapter), is_trainable=False)
    with ActivationApproximation(model, HookConfig(kind="hard_swish", layers=(args.layers,))) as hook:
        if len(hook.applied_modules) != 1:
            raise RuntimeError("Expected exactly one approximate activation")
        repaired, repaired_per_token = measure(model, pairs, device)
        if args.evaluate_train:
            train_repaired, train_repaired_per_token = measure(model, train_pairs, device)

    report = {
        "holdout_data": args.holdout_data, "train_data": args.train_data,
        "adapter": str(adapter), "reference": "original_silu_frozen",
        "approximation": "hard_swish", "layers": [args.layers], "num_pairs": len(holdout),
        "metric": "sum of completion token log-probabilities (including EOS); chosen > rejected",
        "training_logprob_reduction": reduction,
        "chosen_shorter_pairs": sum(chosen[0].shape[1] - chosen[1] <
                                     rejected[0].shape[1] - rejected[1]
                                     for chosen, rejected in pairs),
        "summary": {
            "original_silu": summarize(silu),
            "unrepaired_hard_swish": summarize(unrepaired),
            "repaired_hard_swish": summarize(repaired),
            "per_token_diagnostic": {
                "original_silu": summarize(silu_per_token),
                "unrepaired_hard_swish": summarize(unrepaired_per_token),
                "repaired_hard_swish": summarize(repaired_per_token),
            },
            "training_objective_metric": {
                "original_silu": summarize(silu if reduction == "sum" else silu_per_token),
                "unrepaired_hard_swish": summarize(unrepaired if reduction == "sum" else unrepaired_per_token),
                "repaired_hard_swish": summarize(repaired if reduction == "sum" else repaired_per_token),
            },
            "objective_diagnostic": objective_diagnostic(
                silu if reduction == "sum" else silu_per_token,
                unrepaired if reduction == "sum" else unrepaired_per_token,
                repaired if reduction == "sum" else repaired_per_token, beta),
            "repaired_vs_unrepaired": {
                "rejected_to_chosen": sum(a <= 0 < b for a, b in zip(unrepaired, repaired)),
                "chosen_to_rejected": sum(a > 0 >= b for a, b in zip(unrepaired, repaired)),
                "margin_increased": sum(b > a for a, b in zip(unrepaired, repaired)),
                "margin_decreased": sum(b < a for a, b in zip(unrepaired, repaired)),
            },
        },
        "pairs": [{"id": row["id"], "original_silu_margin": a,
                   "unrepaired_hard_swish_margin": b, "repaired_hard_swish_margin": c,
                   "repaired_minus_unrepaired": c - b}
                  for row, a, b, c in zip(holdout, silu, unrepaired, repaired)],
    }
    if args.evaluate_train:
        reference = train_silu if reduction == "sum" else train_silu_per_token
        before = train_unrepaired if reduction == "sum" else train_unrepaired_per_token
        after = train_repaired if reduction == "sum" else train_repaired_per_token
        report["training_set"] = {"num_pairs": len(train),
            "summary": {"original_silu": summarize(train_silu),
                        "unrepaired_hard_swish": summarize(train_unrepaired),
                        "repaired_hard_swish": summarize(train_repaired),
                        "per_token_diagnostic": {
                            "unrepaired_hard_swish": summarize(train_unrepaired_per_token),
                            "repaired_hard_swish": summarize(train_repaired_per_token)},
                        "objective_diagnostic": objective_diagnostic(reference, before, after, beta)},
            "pairs": [{"id": row["id"], "adapter_margin_delta": new - old}
                      for row, old, new in zip(train, before, after)]}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report["summary"], indent=2))
    if args.evaluate_train:
        print(json.dumps({"training_set": report["training_set"]["summary"]}, indent=2))
    print(f"Saved {len(holdout)} pairwise margins to {out.resolve()}")


if __name__ == "__main__":
    main()
