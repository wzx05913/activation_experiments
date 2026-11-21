"""Generate model responses for baseline and activation approximation settings."""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch

from .activation_hooks import ActivationApproximation, HookConfig
from .utils import device_and_dtype, device_metadata, parse_layers, read_dataset, set_seed, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--adapter", default=None, help="Optional PEFT/LoRA adapter directory")
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--setting", default="baseline")
    parser.add_argument("--check-inputs-only", action="store_true", help="Validate all prompts with the tokenizer without loading model weights")
    parser.add_argument("--approximation", default="none", choices=["none", "piecewise_linear", "piecewise_polynomial", "hard_swish", "gaussian", "laplace", "sparsify", "quantize"])
    parser.add_argument("--layers", default="", help="Comma-separated zero-based layer indices, or all for every decoder layer")
    parser.add_argument("--strength", type=float, default=1.0, help="Approximation blend/strength; main function approximations use 1.0")
    parser.add_argument("--num-bins", type=int, default=16, help="Number of intervals for piecewise SiLU approximation")
    parser.add_argument("--domain", type=float, default=4.0, help="Central approximation domain [-domain, domain]")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--max-input-tokens", type=int, default=1024)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto", help="auto uses CUDA when available")
    parser.add_argument("--dtype", choices=["auto", "float16", "bfloat16", "float32"], default="auto")
    return parser.parse_args()


def build_prompt(tokenizer, prompt: str) -> str:
    messages = [{"role": "user", "content": prompt}]
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return prompt


def validate_input_lengths(tokenizer, rows, max_input_tokens, max_new_tokens):
    """Check every chat-formatted prompt before any model generation."""
    if max_input_tokens <= 0 or max_new_tokens <= 0:
        raise ValueError("Input and output token limits must be positive")
    lengths = []
    context_limit = getattr(tokenizer, "model_max_length", None)
    for row in rows:
        formatted = build_prompt(tokenizer, str(row["prompt"]))
        ids = tokenizer(formatted, truncation=False)["input_ids"]
        count = len(ids)
        if count > max_input_tokens:
            raise ValueError(f"Input {row.get('id')} needs {count} tokens; --max-input-tokens={max_input_tokens}. "
                             "Increase the same limit for both baseline and approximation; no input was truncated.")
        if isinstance(context_limit, int) and count + max_new_tokens > context_limit:
            raise ValueError(f"Input {row.get('id')} plus output budget exceeds tokenizer context limit {context_limit}")
        lengths.append(count)
    return lengths


def main() -> None:
    args = parse_args()
    from transformers import AutoModelForCausalLM, AutoTokenizer

    rows = read_dataset(args.data, args.max_samples)
    if not rows:
        raise ValueError("Dataset has no samples")
    if not args.check_inputs_only and Path(args.output).exists():
        raise FileExistsError(f"Preserving existing output: {args.output}; choose a new filename")
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is None:
            raise ValueError("Tokenizer has neither pad_token nor eos_token; configure a pad token first.")
        tokenizer.pad_token = tokenizer.eos_token
    lengths = validate_input_lengths(tokenizer, rows, args.max_input_tokens, args.max_new_tokens)
    print(f"[inputs] samples={len(rows)} contextual={sum(r['context_included'] for r in rows)} max_tokens={max(lengths)}; truncation disabled")
    if args.check_inputs_only:
        return
    set_seed(args.seed)
    device, dtype = device_and_dtype(args.device, args.dtype)
    runtime = device_metadata(device, dtype)
    print(f"[runtime] device={runtime['device']} dtype={runtime['dtype']} cuda_available={runtime['cuda_available']} gpu={runtime['gpu_name']}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype,
        device_map="auto" if device.type == "cuda" else None,
    )
    if device.type == "cpu":
        model.to(device)
    if args.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.adapter, is_trainable=False)
        print(f"[adapter] loaded={Path(args.adapter).resolve()}")
    model.eval()
    input_device = next(model.parameters()).device

    if args.layers.strip().lower() == "all":
        layers = list(range(len(ActivationApproximation._get_layers(model))))
    else:
        layers = parse_layers(args.layers)
    hook_config = HookConfig(
        kind=args.approximation,
        layers=tuple(layers),
        strength=args.strength,
        location="activation_function" if args.approximation in {"piecewise_linear", "piecewise_polynomial", "hard_swish"} else "mlp_output_legacy",
        num_bins=args.num_bins,
        domain=args.domain,
    )
    outputs = []
    context = ActivationApproximation(model, hook_config) if layers and args.approximation != "none" else None
    if context is not None:
        context.install()

    try:
        for row in rows:
            prompt = str(row["prompt"])
            formatted = build_prompt(tokenizer, prompt)
            inputs = tokenizer(formatted, return_tensors="pt", truncation=False)
            input_tokens = inputs["input_ids"].shape[1]
            if input_tokens > args.max_input_tokens:
                raise ValueError(
                    f"Input {row.get('id', '<unknown>')} has {input_tokens} tokens, above "
                    f"--max-input-tokens={args.max_input_tokens}. Refusing silent truncation "
                    "because it may remove benchmark context or the task."
                )
            inputs = {key: value.to(input_device) for key, value in inputs.items()}
            generation_kwargs = {
                "max_new_tokens": args.max_new_tokens,
                "do_sample": args.temperature > 0,
                "temperature": args.temperature if args.temperature > 0 else 1.0,
                "top_p": args.top_p,
                "pad_token_id": tokenizer.pad_token_id,
            }
            start = time.perf_counter()
            with torch.inference_mode():
                generated = model.generate(**inputs, **generation_kwargs)
            elapsed = time.perf_counter() - start
            new_tokens = generated[0, inputs["input_ids"].shape[1]:]
            response = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
            outputs.append({
                **row,
                "response": response,
                "model": args.model,
                "adapter": str(Path(args.adapter).resolve()) if args.adapter else None,
                "setting": args.setting,
                "approximation": args.approximation,
                "approximation_location": hook_config.location,
                "approximation_num_bins": args.num_bins,
                "approximation_domain": args.domain,
                "input_token_count": input_tokens,
                "input_truncated": False,
                "max_input_tokens": args.max_input_tokens,
                "max_new_tokens": args.max_new_tokens,
                "temperature": args.temperature,
                "top_p": args.top_p,
                "generated_token_count": int(new_tokens.shape[0]),
                "output_token_limit_reached": int(new_tokens.shape[0]) >= args.max_new_tokens,
                "generation_protocol_version": "context_v2_no_truncation",
                "layers": layers,
                "strength": args.strength,
                "seed": args.seed,
                "hooked_modules": context.applied_modules if context is not None else [],
                "latency_seconds": elapsed,
                **runtime,
            })
    finally:
        if context is not None:
            context.remove()

    write_jsonl(args.output, outputs)
    print(f"Wrote {len(outputs)} responses to {Path(args.output).resolve()}")


if __name__ == "__main__":
    main()





