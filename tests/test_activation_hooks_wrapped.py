"""Regression tests for activation hooks under PEFT-style wrappers."""
import unittest

import torch
from torch import nn

from src.activation_hooks import ActivationApproximation, HookConfig


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = nn.Module()
        self.mlp.gate_proj = nn.Linear(2, 2)
        self.mlp.act_fn = nn.SiLU()


class BaseModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList(Decoder() for _ in range(28))


class LoraModel(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model


class PeftModel(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.base_model = LoraModel(model)


class WrappedActivationTests(unittest.TestCase):
    def test_hard_swish_layer_27_with_peft_wrapper(self):
        base = BaseModel()
        wrapped = PeftModel(base)
        target = base.model.layers[27].mlp
        original = target.act_fn
        with ActivationApproximation(wrapped, HookConfig(kind="hard_swish", layers=(27,))) as hook:
            self.assertEqual(len(hook.applied_modules), 1)
            self.assertNotEqual(target.act_fn, original)
            self.assertIs(base.model.layers[14].mlp.act_fn.__class__, nn.SiLU)
        self.assertIs(target.act_fn, original)

    def test_unwrapped_model_still_resolves(self):
        base = BaseModel()
        self.assertEqual(len(ActivationApproximation._get_layers(base)), 28)

    def test_quantize_ste_preserves_forward_and_gradient(self):
        base = BaseModel()
        target = base.model.layers[27].mlp
        target.forward = lambda value: value
        x = torch.tensor([[0.13, 0.72]], requires_grad=True)
        with ActivationApproximation(base, HookConfig(kind="quantize", layers=(27,), strength=6, quantize_ste=True)):
            actual = target(x)
            expected = torch.clamp(torch.round(x.detach() / (x.detach().abs().amax() / 31)), -31, 31) * (x.detach().abs().amax() / 31)
            self.assertTrue(torch.allclose(actual, expected))
            actual.sum().backward()
        self.assertTrue(torch.allclose(x.grad, torch.ones_like(x)))


if __name__ == "__main__":
    unittest.main()
