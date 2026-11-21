"""Activation-function approximations used by the main experiment track.

The legacy tensor perturbation operators remain available for historical runs, but
new experiments should use the activation-function replacements below.  They are
implemented as regular PyTorch modules so they can replace Qwen/Llama-style MLP
``act_fn`` modules during inference.
"""
from __future__ import annotations

import torch
from torch import nn


def _relative_scale(x: torch.Tensor) -> torch.Tensor:
    scale = x.detach().float().std().clamp_min(1e-6)
    return scale.to(dtype=x.dtype, device=x.device)


def gaussian_noise(x: torch.Tensor, strength: float) -> torch.Tensor:
    if strength == 0:
        return x
    return x + torch.randn_like(x) * (strength * _relative_scale(x))


def laplace_noise(x: torch.Tensor, strength: float) -> torch.Tensor:
    if strength == 0:
        return x
    scale = _relative_scale(x) * strength
    uniform = torch.rand_like(x).clamp_(1e-6, 1 - 1e-6) - 0.5
    noise = -scale * torch.sign(uniform) * torch.log1p(-2 * uniform.abs())
    return x + noise


def sparsify(x: torch.Tensor, sparsity: float) -> torch.Tensor:
    if not 0 <= sparsity < 1:
        raise ValueError("sparsity must be in [0, 1)")
    if sparsity == 0:
        return x
    threshold = torch.quantile(x.detach().abs().float(), sparsity).to(x.device)
    return torch.where(x.abs() <= threshold, torch.zeros_like(x), x)


def quantize_activation(x: torch.Tensor, bits: int) -> torch.Tensor:
    if bits < 2 or bits > 16:
        raise ValueError("activation quantization bits must be between 2 and 16")
    qmax = float(2 ** (bits - 1) - 1)
    max_abs = x.detach().abs().amax().clamp_min(1e-6)
    scale = max_abs / qmax
    return torch.clamp(torch.round(x / scale), -qmax, qmax) * scale


def _silu(x: torch.Tensor) -> torch.Tensor:
    return x * torch.sigmoid(x)


class ApproximateSiLU(nn.Module):
    """A deployment-oriented approximation of SiLU.

    ``kind`` is one of ``piecewise_linear``, ``piecewise_polynomial`` or
    ``hard_swish``.  The tails use SiLU's stable asymptotes (0 on the negative side
    and x on the positive side), while the central interval is approximated.
    """

    def __init__(self, kind: str, strength: float = 1.0, num_bins: int = 16, domain: float = 4.0):
        super().__init__()
        self.kind = kind.lower()
        self.strength = float(strength)
        self.num_bins = int(num_bins)
        self.domain = float(domain)
        if self.kind not in {"piecewise_linear", "piecewise_polynomial", "hard_swish"}:
            raise ValueError(f"Unknown SiLU approximation kind: {kind}")
        if self.num_bins < 2 or self.domain <= 0:
            raise ValueError("num_bins must be >= 2 and domain must be positive")
        if self.kind == "piecewise_polynomial":
            self._register_polynomial_coefficients()

    def _register_polynomial_coefficients(self) -> None:
        # Fit one cubic on each interval. Coefficients are computed once in
        # float64 and stored as buffers, so inference remains ordinary tensor ops.
        edges = torch.linspace(-self.domain, self.domain, self.num_bins + 1, dtype=torch.float64)
        coeffs = []
        for left, right in zip(edges[:-1], edges[1:]):
            x = torch.linspace(left.item(), right.item(), 17, dtype=torch.float64)
            design = torch.stack([x**3, x**2, x, torch.ones_like(x)], dim=1)
            coeffs.append(torch.linalg.lstsq(design, _silu(x).unsqueeze(1)).solution[:, 0])
        self.register_buffer("poly_edges", edges.float())
        self.register_buffer("poly_coeffs", torch.stack(coeffs).float())

    def _piecewise_linear(self, x: torch.Tensor) -> torch.Tensor:
        edges = torch.linspace(-self.domain, self.domain, self.num_bins + 1, device=x.device, dtype=x.dtype)
        values = _silu(edges)
        idx = torch.bucketize(x, edges[1:-1])
        idx = idx.clamp(0, self.num_bins - 1)
        left, right = edges[idx], edges[idx + 1]
        v_left, v_right = values[idx], values[idx + 1]
        t = ((x - left) / (right - left)).clamp(0, 1)
        central = torch.lerp(v_left, v_right, t)
        return torch.where(x < -self.domain, torch.zeros_like(x), torch.where(x > self.domain, x, central))

    def _hard_swish(self, x: torch.Tensor) -> torch.Tensor:
        # Common deployment-friendly piecewise approximation to SiLU.
        return x * torch.clamp(x + 3.0, min=0.0, max=6.0) / 6.0

    def _piecewise_polynomial(self, x: torch.Tensor) -> torch.Tensor:
        edges = self.poly_edges.to(device=x.device, dtype=x.dtype)
        coeffs = self.poly_coeffs.to(device=x.device, dtype=x.dtype)
        idx = torch.bucketize(x, edges[1:-1]).clamp(0, self.num_bins - 1)
        c = coeffs[idx]
        central = ((c[..., 0] * x + c[..., 1]) * x + c[..., 2]) * x + c[..., 3]
        return torch.where(x < -self.domain, torch.zeros_like(x), torch.where(x > self.domain, x, central))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.kind == "piecewise_linear":
            approx = self._piecewise_linear(x)
        elif self.kind == "hard_swish":
            approx = self._hard_swish(x)
        else:
            approx = self._piecewise_polynomial(x)
        return approx


def apply_approximation(x: torch.Tensor, kind: str, strength: float) -> torch.Tensor:
    kind = kind.lower()
    if kind in {"none", "identity"}:
        return x
    if kind == "gaussian":
        return gaussian_noise(x, strength)
    if kind == "laplace":
        return laplace_noise(x, strength)
    if kind in {"sparsify", "sparsity"}:
        return sparsify(x, strength)
    if kind in {"quantize", "activation_quantization"}:
        return quantize_activation(x, int(strength))
    raise ValueError(f"Unknown tensor approximation kind: {kind}")


