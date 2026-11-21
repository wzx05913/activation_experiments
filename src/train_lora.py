"""Minimal activation-aware LoRA/SFT repair training.

The approximation hook stays enabled during training, so the adapter is
optimized for the same deployment condition that will be evaluated later.
Training data must contain JSONL records with ``prompt`` and ``response``.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import torch
from torch.nn.utils.rnn import pad_sequence

from .activation_hooks import ActivationApproximation, HookConfig
from .repair_sampling import epoch_order, validate_mixed_rows
from .utils import device_and_dtype, parse_layers, read_jsonl, set_seed


def build_prompt(tokenizer, prompt: str) -> str:
    messages = [{"role": "user", "content": prompt}]
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return prompt


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--train-data", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--data-mode", choices=["legacy", "mixed"], default="legacy",
                   help="mixed requires reviewed harmful and benign examples and balances them per epoch")
    p.add_argument("--approximation", choices=["none", "hard_swish", "piecewise_linear", "piecewise_polynomial"], default="hard_swish")
    p.add_argument("--layers", default="27", help="Comma-separated zero-based layers, or all")
    p.add_argument("--strength", type=float, default=1.0)
    p.add_argument("--num-bins", type=int, default=16)
    p.add_argument("--domain", type=float, default=4.0)
    p.add_argument("--max-input-tokens", type=int, default=512)
    p.add_argument("--max-target-tokens", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--gradient-accumulation-steps", type=int, default=8)
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--learning-rate", type=float, default=2e-4)
    p.add_argument("--lora-r", type=int, default=8)
    p.add_argument("--lora-alpha", type=float, default=16.0)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument("--target-modules", default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
                   help="Comma-separated LoRA projection names; mixed pilot may use q_proj,v_proj")
    p.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    p.add_argument("--dtype", choices=["auto", "float16", "bfloat16", "float32"], default="auto")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def make_example(tokenizer, row: dict[str, Any], max_input: int, max_target: int) -> dict[str, torch.Tensor]:
    prompt = build_prompt(tokenizer, str(row["prompt"]))
    answer = str(row["response"])
    prompt_ids = tokenizer(prompt, add_special_tokens=True, truncation=True, max_length=max_input)["input_ids"]
    answer_ids = tokenizer(answer, add_special_tokens=False, truncation=True, max_length=max_target)["input_ids"]
    if tokenizer.eos_token_id is not None:
        answer_ids = answer_ids + [tokenizer.eos_token_id]
    return {
        "input_ids": torch.tensor(prompt_ids + answer_ids, dtype=torch.long),
        "labels": torch.tensor([-100] * len(prompt_ids) + answer_ids, dtype=torch.long),
    }


def collate(examples: list[dict[str, torch.Tensor]], pad_id: int) -> dict[str, torch.Tensor]:
    lengths = [x["input_ids"].numel() for x in examples]
    ids = pad_sequence([x["input_ids"] for x in examples], batch_first=True, padding_value=pad_id)
    labels = pad_sequence([x["labels"] for x in examples], batch_first=True, padding_value=-100)
    positions = torch.arange(ids.shape[1]).unsqueeze(0)
    attention_mask = positions < torch.tensor(lengths).unsqueeze(1)
    return {"input_ids": ids, "attention_mask": attention_mask.long(), "labels": labels}


def main() -> None:
    args = parse_args()
    if args.epochs <= 0 or args.batch_size <= 0 or args.gradient_accumulation_steps <= 0:
        raise ValueError("epochs, batch-size, and gradient-accumulation-steps must be positive")
    raw_rows = read_jsonl(args.train_data)
    invalid = [i + 1 for i, r in enumerate(raw_rows)
               if not str(r.get("prompt", "")).strip() or not str(r.get("response", "")).strip()]
    if invalid:
        raise ValueError(f"Training JSONL has empty prompt/response at line(s): {invalid[:10]}")
    rows = raw_rows
    if not rows:
        raise ValueError("Training JSONL must contain at least one non-empty prompt/response pair")
    kind_counts = validate_mixed_rows(rows) if args.data_mode == "mixed" else {}
    allowed_modules = {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"}
    target_modules = [part.strip() for part in args.target_modules.split(",")]
    if not target_modules or any(part not in allowed_modules for part in target_modules) or len(set(target_modules)) != len(target_modules):
        raise ValueError(f"target-modules must be unique members of {sorted(allowed_modules)}")
    out = Path(args.output_dir)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {out}")
    out.mkdir(parents=True, exist_ok=True)

    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    set_seed(args.seed)
    device, dtype = device_and_dtype(args.device, args.dtype)
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype)
    model.to(device)
    model.config.use_cache = False
    model.enable_input_require_grads()
    model = get_peft_model(model, LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
        bias="none", task_type=TaskType.CAUSAL_LM,
        target_modules=target_modules,
    ))
    model.print_trainable_parameters()

    base = model.get_base_model()
    if args.layers.strip().lower() == "all":
        layers = list(range(len(ActivationApproximation._get_layers(base))))
    else:
        layers = parse_layers(args.layers)
    hook = ActivationApproximation(base, HookConfig(kind=args.approximation, layers=tuple(layers), strength=args.strength, num_bins=args.num_bins, domain=args.domain))
    hook.install()
    examples = [make_example(tokenizer, r, args.max_input_tokens, args.max_target_tokens) for r in rows]
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=args.learning_rate)
    model.train(); optimizer.zero_grad(set_to_none=True)
    losses: list[float] = []
    rng = random.Random(args.seed)
    try:
        for epoch in range(args.epochs):
            order = epoch_order(rows, args.data_mode, rng)
            epoch_losses = []
            for step, start in enumerate(range(0, len(order), args.batch_size)):
                batch = collate([examples[i] for i in order[start:start + args.batch_size]], tokenizer.pad_token_id)
                batch = {k: v.to(device) for k, v in batch.items()}
                raw_loss = model(**batch).loss
                (raw_loss / args.gradient_accumulation_steps).backward()
                epoch_losses.append(float(raw_loss.detach().cpu()))
                if (step + 1) % args.gradient_accumulation_steps == 0 or start + args.batch_size >= len(order):
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step(); optimizer.zero_grad(set_to_none=True)
            losses.extend(epoch_losses)
            print(json.dumps({"epoch": epoch + 1, "mean_loss": sum(epoch_losses) / len(epoch_losses)}, ensure_ascii=False))
    finally:
        hook.remove()
    model.save_pretrained(out); tokenizer.save_pretrained(out)
    (out / "repair_config.json").write_text(json.dumps({**vars(args), "layers_resolved": layers,
        "target_modules_resolved": target_modules, "kind_counts": kind_counts,
        "examples_per_epoch": len(epoch_order(rows, args.data_mode, random.Random(args.seed))),
        "num_train_rows": len(rows), "mean_loss": sum(losses) / len(losses)}, indent=2), encoding="utf-8")
    print(f"Saved LoRA adapter to {out.resolve()}")


if __name__ == "__main__":
    main()
