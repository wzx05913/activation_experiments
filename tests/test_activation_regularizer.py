import random
import unittest

import torch
from torch import nn

from src.activation_regularizer import (
    pairwise_cosine_loss, prompt_representations, regularizer_gradient_norm,
    select_regularizer_indices,
)


class TinyDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(16, 4)
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([nn.Linear(4, 4), nn.Linear(4, 4)])
        self.is_gradient_checkpointing = True
        self.fail_forward = False
        self.last_mask = None

    def gradient_checkpointing_disable(self):
        self.is_gradient_checkpointing = False

    def gradient_checkpointing_enable(self):
        self.is_gradient_checkpointing = True

    def forward(self, input_ids, attention_mask, use_cache=False):
        self.last_mask = attention_mask
        hidden = self.embedding(input_ids)
        for layer in self.model.layers:
            hidden = layer(hidden)
        if self.fail_forward:
            raise RuntimeError("injected failure")
        return hidden * 7


class RegularizerTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)

    def test_cosine_excludes_diagonal(self):
        loss, similarity = pairwise_cosine_loss(torch.eye(2))
        self.assertEqual(float(loss), 1.0)
        self.assertEqual(float(similarity), 0.0)

    def test_identical_and_opposite_vectors(self):
        for vectors, expected in [([[1., 0.], [2., 0.]], 0.), ([[1., 0.], [-1., 0.]], 2.)]:
            loss, _ = pairwise_cosine_loss(torch.tensor(vectors))
            self.assertAlmostEqual(float(loss), expected)

    def test_invalid_vectors(self):
        for vectors in [torch.ones(1, 4), torch.zeros(2, 4), torch.full((2, 4), float("nan"))]:
            with self.assertRaises(ValueError):
                pairwise_cosine_loss(vectors)

    def test_last_real_token_and_block_output(self):
        model = TinyDecoder()
        prompts = [torch.tensor([1, 2, 3]), torch.tensor([4])]
        actual = prompt_representations(model, prompts, 1, 0, "cpu")
        expected = model.embedding(torch.tensor([3, 4]))
        for layer in model.model.layers:
            expected = layer(expected)
        torch.testing.assert_close(actual, expected)
        self.assertEqual(model.last_mask.tolist(), [[1, 1, 1], [1, 0, 0]])
        self.assertTrue(model.is_gradient_checkpointing)
        self.assertFalse(model.model.layers[1]._forward_hooks)

    def test_regularizer_backpropagates_and_probe_preserves_existing_gradients(self):
        model = TinyDecoder()
        model.embedding.weight.requires_grad_(False)
        representations = prompt_representations(model, [torch.tensor([1, 2]), torch.tensor([3])], 1, 0, "cpu")
        loss, _ = pairwise_cosine_loss(representations)
        parameter = model.model.layers[0].weight
        parameter.grad = torch.ones_like(parameter)
        self.assertGreater(regularizer_gradient_norm(loss, model), 0)
        torch.testing.assert_close(parameter.grad, torch.ones_like(parameter))
        loss.backward()
        self.assertFalse(torch.equal(parameter.grad, torch.ones_like(parameter)))
        self.assertIsNone(model.embedding.weight.grad)

    def test_hook_and_checkpointing_restored_on_error(self):
        model = TinyDecoder()
        model.fail_forward = True
        with self.assertRaisesRegex(RuntimeError, "injected failure"):
            prompt_representations(model, [torch.tensor([1]), torch.tensor([2])], 1, 0, "cpu")
        self.assertTrue(model.is_gradient_checkpointing)
        self.assertFalse(model.model.layers[1]._forward_hooks)

    def test_singleton_group_uses_distinct_training_partner(self):
        selected = select_regularizer_indices([3], range(5), 4, random.Random(42))
        self.assertEqual(len(selected), 2)
        self.assertIn(3, selected)
        self.assertEqual(len(set(selected)), 2)
        self.assertTrue(set(selected) <= set(range(5)))

    def test_sampling_keeps_dpo_rng_unchanged(self):
        dpo_rng = random.Random(42)
        state = dpo_rng.getstate()
        selected = select_regularizer_indices([1, 2, 3, 4], range(8), 2, random.Random(17))
        self.assertEqual(dpo_rng.getstate(), state)
        self.assertTrue(set(selected) <= {1, 2, 3, 4})


if __name__ == "__main__":
    unittest.main()
