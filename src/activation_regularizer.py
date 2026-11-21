"""Prompt-only block-output clustering for fixed-approximation DPO."""
from __future__ import annotations

import torch
import torch.nn.functional as F

from .activation_hooks import ActivationApproximation


def select_regularizer_indices(group, population, batch_size, rng):
    if batch_size < 2 or len(population) < 2:
        raise ValueError("Regularization needs at least two distinct prompts")
    selected = rng.sample(list(group), min(batch_size, len(group)))
    if len(selected) < 2:
        remaining = [index for index in population if index not in selected]
        selected.extend(rng.sample(remaining, 2 - len(selected)))
    return selected


def pairwise_cosine_loss(representations):
    if representations.ndim != 2 or representations.shape[0] < 2:
        raise ValueError("Expected at least two prompt vectors [batch, hidden]")
    values = representations.float()
    if not torch.isfinite(values).all() or (values.norm(dim=-1) <= 1e-8).any():
        raise ValueError("Prompt vectors must be finite and nonzero")
    normalized = F.normalize(values, dim=-1, eps=1e-8)
    similarities = normalized @ normalized.T
    off_diagonal = ~torch.eye(len(values), dtype=torch.bool, device=values.device)
    similarity = similarities[off_diagonal].mean()
    return 1 - similarity, similarity


def prompt_representations(model, prompts, layer_index, pad_token_id, device):
    if len(prompts) < 2 or any(prompt.ndim != 1 or prompt.numel() == 0 for prompt in prompts):
        raise ValueError("Expected at least two nonempty one-dimensional token sequences")
    layers = ActivationApproximation._get_layers(model)
    if not 0 <= layer_index < len(layers):
        raise ValueError("Regularizer layer is outside decoder layers")
    lengths = torch.tensor([prompt.numel() for prompt in prompts], device=device)
    input_ids = torch.full((len(prompts), int(lengths.max())), pad_token_id,
                           dtype=torch.long, device=device)
    attention_mask = torch.zeros_like(input_ids)
    for index, prompt in enumerate(prompts):
        input_ids[index, :prompt.numel()] = prompt.to(device)
        attention_mask[index, :prompt.numel()] = 1
    captured = []

    def capture(_module, _inputs, output):
        hidden = output[0] if isinstance(output, tuple) else output
        captured.append(hidden[torch.arange(len(prompts), device=hidden.device), lengths - 1])

    checkpointing = bool(getattr(model, "is_gradient_checkpointing", False))
    handle = layers[layer_index].register_forward_hook(capture)
    try:
        if checkpointing:
            model.gradient_checkpointing_disable()
        model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
    finally:
        handle.remove()
        if checkpointing:
            model.gradient_checkpointing_enable()
    if len(captured) != 1:
        raise RuntimeError(f"Expected one block output, captured {len(captured)}")
    return captured[0]


def regularizer_gradient_norm(loss, model):
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    gradients = torch.autograd.grad(loss, parameters, retain_graph=True, allow_unused=True)
    squared_norm = torch.zeros((), device=loss.device)
    for gradient in gradients:
        if gradient is not None:
            squared_norm = squared_norm + gradient.detach().float().square().sum()
    norm = squared_norm.sqrt()
    if not torch.isfinite(norm) or norm <= 0:
        raise RuntimeError("Regularizer has zero or non-finite gradient on trainable parameters")
    return float(norm)
