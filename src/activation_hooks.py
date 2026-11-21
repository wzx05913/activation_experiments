"""Install activation-function approximations inside decoder MLPs."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from .approximation import ApproximateSiLU, apply_approximation


@dataclass
class HookConfig:
    # New main-track kinds: piecewise_linear, piecewise_polynomial, lookup.
    # Legacy kinds are retained for reproducibility of historical runs.
    kind: str = "none"
    layers: tuple[int, ...] = ()
    strength: float = 1.0
    location: str = "activation_function"
    num_bins: int = 16
    domain: float = 4.0
    quantize_ste: bool = False


class ActivationApproximation:
    """Temporarily replace MLP activation functions, not MLP outputs."""

    FUNCTION_KINDS = {"piecewise_linear", "piecewise_polynomial", "hard_swish"}
    LEGACY_KINDS = {"gaussian", "laplace", "sparsify", "quantize"}

    def __init__(self, model: torch.nn.Module, config: HookConfig):
        self.model = model
        self.config = config
        self.handles: list[Any] = []
        self.applied_modules: list[str] = []
        self._original_activations: list[tuple[torch.nn.Module, str, Any]] = []

    @staticmethod
    def _get_layers(model: torch.nn.Module) -> list[torch.nn.Module]:
        # PEFT wraps the base model in PeftModel/LoraModel.
        pending = [model]
        seen: set[int] = set()
        while pending:
            candidate = pending.pop(0)
            if not isinstance(candidate, torch.nn.Module) or id(candidate) in seen:
                continue
            seen.add(id(candidate))
            for path in ("model.layers", "transformer.h", "transformer.layers", "model.decoder.layers"):
                current: Any = candidate
                try:
                    for part in path.split("."):
                        current = getattr(current, part)
                    if isinstance(current, (torch.nn.ModuleList, list, tuple)):
                        return list(current)
                except AttributeError:
                    continue
            for name in ("base_model", "model", "module"):
                child = getattr(candidate, name, None)
                if isinstance(child, torch.nn.Module):
                    pending.append(child)
        raise ValueError("Could not find decoder layers in model")

    @staticmethod
    def _find_mlp(block: torch.nn.Module) -> torch.nn.Module:
        for name in ("mlp", "MLP", "feed_forward", "ffn", "ffn_layer"):
            if hasattr(block, name):
                return getattr(block, name)
        for name, module in block.named_children():
            if any(token in name.lower() for token in ("mlp", "ffn", "feed")):
                return module
        raise ValueError(f"Could not find MLP submodule in {type(block).__name__}")

    @staticmethod
    def _find_activation(mlp: torch.nn.Module) -> tuple[str, Any]:
        for name in ("act_fn", "activation_fn", "activation"):
            if hasattr(mlp, name):
                return name, getattr(mlp, name)
        raise ValueError(
            f"No replaceable activation found in {type(mlp).__name__}. "
            "Expected act_fn/activation_fn/activation; inspect the model MLP."
        )

    def install(self) -> "ActivationApproximation":
        if self.config.kind in {"none", "identity"} or not self.config.layers:
            return self
        layers = self._get_layers(self.model)
        for index in self.config.layers:
            if index < 0 or index >= len(layers):
                raise IndexError(f"Layer {index} is outside [0, {len(layers) - 1}]")
            mlp = self._find_mlp(layers[index])
            if self.config.kind in self.FUNCTION_KINDS:
                name, original = self._find_activation(mlp)
                if not isinstance(original, nn.Module):
                    raise TypeError(f"{type(mlp).__name__}.{name} is not a torch module")
                if "silu" not in type(original).__name__.lower():
                    raise TypeError(
                        f"Expected a SiLU activation at layer {index}, got {type(original).__name__}; "
                        "refusing to silently approximate a different activation."
                    )
                replacement = ApproximateSiLU(
                    self.config.kind,
                    strength=self.config.strength,
                    num_bins=self.config.num_bins,
                    domain=self.config.domain,
                ).to(device=next(mlp.parameters()).device, dtype=next(mlp.parameters()).dtype)
                self._original_activations.append((mlp, name, original))
                setattr(mlp, name, replacement)
                self.applied_modules.append(f"layer[{index}].{type(mlp).__name__}.{name}")
            elif self.config.kind in self.LEGACY_KINDS:
                # Historical path: output perturbation retained only for old results.
                def hook(_module: nn.Module, _inputs: Any, output: Any, kind=self.config.kind, strength=self.config.strength, ste=self.config.quantize_ste) -> Any:
                    def approximate(value: torch.Tensor) -> torch.Tensor:
                        quantized = apply_approximation(value, kind, strength)
                        if kind == "quantize" and ste:
                            return value + (quantized - value).detach()
                        return quantized
                    if isinstance(output, tuple):
                        return (approximate(output[0]), *output[1:])
                    if isinstance(output, torch.Tensor):
                        return approximate(output)
                    raise TypeError(f"Unsupported MLP output type: {type(output)}")
                self.handles.append(mlp.register_forward_hook(hook))
                self.applied_modules.append(f"layer[{index}].{type(mlp).__name__}.output(legacy)")
            else:
                raise ValueError(f"Unknown approximation kind: {self.config.kind}")
        return self

    def remove(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        for mlp, name, original in reversed(self._original_activations):
            setattr(mlp, name, original)
        self._original_activations.clear()

    def __enter__(self) -> "ActivationApproximation":
        return self.install()

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.remove()
