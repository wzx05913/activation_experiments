"""Minimal LoRA DPO: Hard-Swish policy versus frozen original SiLU reference."""
from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

import torch
import torch.nn.functional as F

from .activation_hooks import ActivationApproximation, HookConfig
from .activation_regularizer import (
    pairwise_cosine_loss, prompt_representations, regularizer_gradient_norm,
    select_regularizer_indices,
)
from .train_lora import build_prompt
from .utils import device_and_dtype, read_jsonl, set_seed


def validate_pairs(rows: list[dict]) -> None:
    if not rows:
        raise ValueError("No preference pairs")
    ids, prompts = set(), set()
    for n, row in enumerate(rows, 1):
        identifier = str(row.get("id", "")).strip()
        prompt = " ".join(str(row.get("prompt", "")).split()).casefold()
        chosen, rejected = str(row.get("chosen", "")).strip(), str(row.get("rejected", "")).strip()
        if row.get("reviewed") is not True or not identifier or not prompt or not chosen or not rejected or chosen == rejected:
            raise ValueError(f"Invalid/unreviewed preference pair at line {n}")
        if identifier in ids or prompt in prompts:
            raise ValueError(f"Duplicate id/prompt at line {n}")
        ids.add(identifier)
        prompts.add(prompt)


def encode_pair(tokenizer, row: dict, max_input: int, max_completion: int):
    prompt = build_prompt(tokenizer, row["prompt"])
    prompt_ids = tokenizer(prompt, add_special_tokens=not bool(getattr(tokenizer, "chat_template", None)))["input_ids"]
    if not prompt_ids or len(prompt_ids) > max_input:
        raise ValueError(f"Prompt exceeds {max_input} tokens: {row['id']}")
    result = []
    for name in ("chosen", "rejected"):
        completion = tokenizer(row[name], add_special_tokens=False)["input_ids"]
        if not completion or len(completion) + 1 > max_completion:
            raise ValueError(f"{name} exceeds {max_completion} tokens: {row['id']}; increase --max-completion-tokens")
        result.append((torch.tensor([prompt_ids + completion + [tokenizer.eos_token_id]], dtype=torch.long), len(prompt_ids)))
    return result


def completion_logprob(model, encoded, device, reduction="sum"):
    ids, prompt_length = encoded
    ids = ids.to(device)
    logits = model(input_ids=ids, use_cache=False).logits[:, :-1].float()
    token_logps = F.log_softmax(logits, dim=-1).gather(-1, ids[:, 1:, None]).squeeze(-1)
    completion_logps = token_logps[:, prompt_length - 1:]
    if reduction == "sum":
        return completion_logps.sum()
    if reduction == "mean":
        return completion_logps.mean()
    raise ValueError(f"Unsupported log-probability reduction: {reduction}")


def dpo_loss(policy_chosen, policy_rejected, reference_chosen, reference_rejected, beta):
    logit = beta * ((policy_chosen - policy_rejected) - (reference_chosen - reference_rejected))
    return -F.logsigmoid(logit), logit


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True)
    p.add_argument("--train-data", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--approximation", choices=["hard_swish", "quantize"], default="hard_swish")
    p.add_argument("--strength", type=float, default=1.0, help="Quantization bit width when --approximation quantize")
    p.add_argument("--layers", default="27", help="Zero-based layer indices (e.g. 27 or 12,24), or all")
    p.add_argument("--max-input-tokens", type=int, default=512)
    p.add_argument("--max-completion-tokens", type=int, default=512)
    p.add_argument("--max-pairs", type=int, default=None)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--gradient-accumulation-steps", type=int, default=8)
    p.add_argument("--learning-rate", type=float, default=1e-5)
    p.add_argument("--beta", type=float, default=0.1)
    p.add_argument("--logprob-reduction", choices=["sum", "mean"], default="sum",
                   help="sum is standard DPO; mean is an experimental length-normalized control")
    p.add_argument("--lora-r", type=int, default=4)
    p.add_argument("--lora-alpha", type=float, default=8)
    p.add_argument("--target-modules", default="q_proj,v_proj")
    p.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    p.add_argument("--dtype", choices=["auto", "float16", "bfloat16", "float32"], default="auto")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--diagnostic-holdout", help="Independent reviewed pairs, used for diagnostics only")
    p.add_argument("--save-every-epoch", action="store_true")
    p.add_argument("--activation-reg-weight", type=float, default=0.0)
    p.add_argument("--activation-reg-layer", type=int, default=27,
                   help="Zero-based block output, before final model normalization")
    p.add_argument("--activation-reg-batch-size", type=int, default=2)
    p.add_argument("--activation-reg-harmful-only", action="store_true",
                   help="Confirm all selected training prompts are harmful; required for regularization")
    args = p.parse_args()
    if not math.isfinite(args.activation_reg_weight) or args.activation_reg_weight < 0:
        p.error("activation-reg-weight must be finite and nonnegative")
    if args.activation_reg_weight > 0:
        if args.activation_reg_batch_size < 2 or args.activation_reg_layer < 0:
            p.error("Regularization needs batch-size >= 2 and layer >= 0")
        if not args.activation_reg_harmful_only:
            p.error("Confirm harmful-only training prompts with --activation-reg-harmful-only")
    if args.approximation == "quantize" and (args.strength != int(args.strength) or not 2 <= args.strength <= 16):
        p.error("quantize requires --strength to be an integer bit width in [2, 16]")
    if args.epochs <= 0 or args.gradient_accumulation_steps <= 0 or args.beta <= 0 or args.learning_rate <= 0:
        p.error("epochs, gradient-accumulation-steps, beta and learning-rate must be positive")
    if args.max_pairs is not None and args.max_pairs <= 0:
        p.error("max-pairs must be positive")
    if args.max_input_tokens <= 0 or args.max_completion_tokens <= 0 or args.lora_r <= 0:
        p.error("token limits and lora-r must be positive")
    modules = [s.strip() for s in args.target_modules.split(",")]
    allowed = {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"}
    if not modules or len(set(modules)) != len(modules) or any(s not in allowed for s in modules):
        p.error("Invalid target-modules")
    rows = read_jsonl(args.train_data)
    validate_pairs(rows)
    rng = random.Random(args.seed)
    rng.shuffle(rows)
    rows = rows[:args.max_pairs]
    if args.activation_reg_weight > 0 and len(rows) < 2:
        p.error("Regularization needs at least two training pairs")
    regularizer_rng = random.Random(args.seed + 104729)
    out = Path(args.output_dir)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {out}")

    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    set_seed(args.seed)
    device, dtype = device_and_dtype(args.device, args.dtype)
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    if tokenizer.eos_token_id is None:
        raise ValueError("Tokenizer needs an EOS token")
    encoded = [encode_pair(tokenizer, row, args.max_input_tokens, args.max_completion_tokens) for row in rows]
    from .evaluate_dpo_holdout import check_split, objective_diagnostic, summarize
    holdout_rows = read_jsonl(args.diagnostic_holdout) if args.diagnostic_holdout else []
    if args.diagnostic_holdout:
        check_split(holdout_rows, rows)
    holdout_encoded = [encode_pair(tokenizer, row, args.max_input_tokens, args.max_completion_tokens)
                       for row in holdout_rows]
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype).to(device)
    model.config.use_cache = False
    model.eval()
    # The base model is the fixed, unapproximated reference. Cache its two
    # sequence log-probabilities so no second 1.5B model occupies GPU memory.
    references = []
    with torch.no_grad():
        for pair in encoded:
            references.append(tuple(float(completion_logprob(model, seq, device, args.logprob_reduction)) for seq in pair))
        holdout_references = [tuple(float(completion_logprob(model, seq, device, args.logprob_reduction))
                                   for seq in pair) for pair in holdout_encoded]
    print(json.dumps({"reference": "original_silu_frozen", "cached_pairs": len(references)}))

    decoder_layers = ActivationApproximation._get_layers(model)
    if args.activation_reg_weight > 0 and args.activation_reg_layer >= len(decoder_layers):
        raise ValueError("activation-reg-layer is outside decoder layers")
    if args.layers.strip().lower() == "all":
        layer_indices = tuple(range(len(decoder_layers)))
    else:
        try:
            layer_indices = tuple(int(part.strip()) for part in args.layers.split(","))
        except ValueError as exc:
            raise ValueError("--layers must be comma-separated integers or all") from exc
        if not layer_indices or len(set(layer_indices)) != len(layer_indices):
            raise ValueError("--layers must contain distinct layer indices")
    if any(index < 0 or index >= len(decoder_layers) for index in layer_indices):
        raise ValueError(f"--layers must be within [0, {len(decoder_layers) - 1}]")
    model = get_peft_model(model, LoraConfig(r=args.lora_r, lora_alpha=args.lora_alpha,
                           lora_dropout=0.0, bias="none", task_type=TaskType.CAUSAL_LM,
                           target_modules=modules))
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model.print_trainable_parameters()
    hook = ActivationApproximation(model.get_base_model(), HookConfig(
        kind=args.approximation, layers=layer_indices, strength=args.strength,
        quantize_ste=args.approximation == "quantize"))
    hook.install()
    if len(hook.applied_modules) != len(layer_indices):
        raise RuntimeError(f"Expected {len(layer_indices)} approximate activations, got {len(hook.applied_modules)}")
    model.eval()
    with torch.no_grad():
        initial_margins = [float(completion_logprob(model, chosen, device, args.logprob_reduction)
                                 - completion_logprob(model, rejected, device, args.logprob_reduction))
                           for chosen, rejected in encoded]
    lengths = [(chosen[0].shape[1] - chosen[1], rejected[0].shape[1] - rejected[1])
               for chosen, rejected in encoded]
    diagnostic = {"logprob_reduction": args.logprob_reduction,
                  "initial_policy_chosen_preferred": sum(margin > 0 for margin in initial_margins),
                  "initial_policy_ties": sum(margin == 0 for margin in initial_margins),
                  "chosen_shorter_pairs": sum(good < bad for good, bad in lengths),
                  "mean_chosen_tokens": sum(good for good, _ in lengths) / len(lengths),
                  "mean_rejected_tokens": sum(bad for _, bad in lengths) / len(lengths)}
    print(json.dumps({"training_data_diagnostic": diagnostic}))
    def margins(pairs):
        model.eval()
        with torch.no_grad():
            result = [float(completion_logprob(model, chosen, device, args.logprob_reduction)
                            - completion_logprob(model, rejected, device, args.logprob_reduction))
                      for chosen, rejected in pairs]
        return result
    initial_holdout_margins = margins(holdout_encoded)
    reference_margins = [chosen - rejected for chosen, rejected in references]
    holdout_reference_margins = [chosen - rejected for chosen, rejected in holdout_references]
    def snapshot(epoch):
        current = margins(encoded)
        result = {"epoch": epoch, "train_preference": summarize(current),
                  "train_objective": objective_diagnostic(reference_margins, initial_margins, current, args.beta)}
        if holdout_encoded:
            current_holdout = margins(holdout_encoded)
            result["holdout_preference"] = summarize(current_holdout)
            result["holdout_objective"] = objective_diagnostic(
                holdout_reference_margins, initial_holdout_margins, current_holdout, args.beta)
        model.train()
        return result
    history = [snapshot(0)]
    print(json.dumps(history[-1]), flush=True)
    def save_checkpoint(destination, completed_epochs):
        destination.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(destination)
        tokenizer.save_pretrained(destination)
        config = {**vars(args), "reference": "original_silu_frozen",
                  "policy": f"{args.approximation}_layers_{args.layers}", "layer_indices": list(layer_indices),
                  "approximation_location": "mlp_output_legacy" if args.approximation == "quantize" else "activation_function",
                  "quantize_ste": args.approximation == "quantize", "num_train_pairs": len(rows),
                  "target_modules_resolved": modules, "completed_epochs": completed_epochs,
                  "optimizer_steps_total": completed_epochs * math.ceil(len(rows) / args.gradient_accumulation_steps),
                  "mean_dpo_loss": sum(losses) / len(losses) if losses else None,
                  "training_data_diagnostic": diagnostic, "diagnostic_history": history}
        config["activation_regularizer"] = {
            "enabled": args.activation_reg_weight > 0,
            "objective": "dpo + weight * (1 - mean_off_diagonal_cosine)",
            "representation": "block_output_last_prompt_token_before_final_norm",
            "sampling": "current_accumulation_group; singleton supplemented from training only",
            "rng_seed": args.seed + 104729,
            "gradient_norm_first_step": regularizer_first_gradient_norm,
            "method_scope": "fixed-approximation QuadA-style adaptation, not full QuadA",
        }
        (destination / "repair_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    model.train()
    optimizer = torch.optim.AdamW((param for param in model.parameters() if param.requires_grad), lr=args.learning_rate)
    optimizer.zero_grad(set_to_none=True)
    losses = []
    regularizer_first_gradient_norm = None
    try:
        for epoch in range(args.epochs):
            order = list(range(len(encoded)))
            rng.shuffle(order)
            epoch_losses, correct = [], 0
            gradient_norms = []
            regularizer_losses, regularizer_similarities, regularizer_batch_sizes = [], [], []
            for step, index in enumerate(order):
                chosen, rejected = encoded[index]
                ref_chosen, ref_rejected = references[index]
                pol_chosen = completion_logprob(model, chosen, device, args.logprob_reduction)
                pol_rejected = completion_logprob(model, rejected, device, args.logprob_reduction)
                loss, logit = dpo_loss(pol_chosen, pol_rejected, ref_chosen, ref_rejected, args.beta)
                # Scale the final partial accumulation group by its actual size.
                group_start = (step // args.gradient_accumulation_steps) * args.gradient_accumulation_steps
                group_size = min(args.gradient_accumulation_steps, len(order) - group_start)
                (loss / group_size).backward()
                epoch_losses.append(float(loss.detach()))
                correct += int(logit.detach() > 0)
                if (step + 1) % args.gradient_accumulation_steps == 0 or step + 1 == len(order):
                    if args.activation_reg_weight > 0:
                        selected = select_regularizer_indices(
                            order[group_start:step + 1], range(len(encoded)),
                            args.activation_reg_batch_size, regularizer_rng)
                        prompts = [encoded[item][0][0][0, :encoded[item][0][1]] for item in selected]
                        representations = prompt_representations(
                            model, prompts, args.activation_reg_layer,
                            tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,
                            device)
                        reg_loss, similarity = pairwise_cosine_loss(representations)
                        weighted_reg_loss = args.activation_reg_weight * reg_loss
                        if regularizer_first_gradient_norm is None:
                            regularizer_first_gradient_norm = regularizer_gradient_norm(weighted_reg_loss, model)
                            print(json.dumps({"activation_reg_gradient_norm": regularizer_first_gradient_norm}), flush=True)
                        weighted_reg_loss.backward()
                        regularizer_losses.append(float(reg_loss.detach()))
                        regularizer_similarities.append(float(similarity.detach()))
                        regularizer_batch_sizes.append(len(selected))
                        del representations, reg_loss, similarity, weighted_reg_loss
                    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
                    gradient_norms.append(float(norm))
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
            losses.extend(epoch_losses)
            record = snapshot(epoch + 1)
            record.update({"online_mean_dpo_loss": sum(epoch_losses) / len(epoch_losses),
                           "online_reference_relative_accuracy": correct / len(order),
                           "mean_gradient_norm_before_clip": sum(gradient_norms) / len(gradient_norms),
                           "optimizer_steps_total": (epoch + 1) * len(gradient_norms)})
            if regularizer_losses:
                record["activation_regularizer"] = {
                    "updates": len(regularizer_losses),
                    "mean_loss": sum(regularizer_losses) / len(regularizer_losses),
                    "mean_weighted_loss": args.activation_reg_weight * sum(regularizer_losses) / len(regularizer_losses),
                    "mean_cosine": sum(regularizer_similarities) / len(regularizer_similarities),
                    "batch_sizes": regularizer_batch_sizes,
                    "gradient_norm_first_step": regularizer_first_gradient_norm,
                }
            history.append(record)
            print(json.dumps(record), flush=True)
            out.mkdir(parents=True, exist_ok=True)
            (out / "diagnostic_history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
            if args.save_every_epoch:
                save_checkpoint(out / f"epoch_{epoch + 1}", epoch + 1)
            print(json.dumps({"epoch": epoch + 1, "mean_dpo_loss": sum(epoch_losses) / len(epoch_losses),
                              "preference_accuracy": correct / len(order), "optimizer_steps": math.ceil(len(order) / args.gradient_accumulation_steps)}))
    finally:
        hook.remove()
    save_checkpoint(out, args.epochs)
    print(f"Saved DPO adapter to {out.resolve()}")


if __name__ == "__main__":
    main()
