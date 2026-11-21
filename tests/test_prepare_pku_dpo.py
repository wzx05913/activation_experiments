"""Offline checks for preference recovery and DPO sequence log probabilities."""
import unittest

import torch

from src.prepare_pku_dpo import recover
from src.train_dpo import completion_logprob, dpo_loss, validate_pairs


class PairTests(unittest.TestCase):
    def test_recover_checks_reviewed_response_and_label(self):
        source = [{"prompt": "request", "response_0": "unsafe", "response_1": "refusal",
                   "is_response_0_safe": False, "is_response_1_safe": True}]
        reviewed = [{"id": "pku_alpaca7b_train_00000", "prompt": "request",
                     "response": "refusal", "source": "PKU"}]
        pair = recover(reviewed, source)[0]
        self.assertEqual((pair["chosen"], pair["rejected"]), ("refusal", "unsafe"))
        self.assertTrue(pair["reviewed"])
        with self.assertRaisesRegex(ValueError, "mismatch"):
            recover([{**reviewed[0], "response": "another refusal"}], source)

    def test_unreviewed_and_duplicate_pairs_are_rejected(self):
        pair = {"id": "one", "prompt": "request", "chosen": "refusal",
                "rejected": "unsafe", "reviewed": True}
        validate_pairs([pair])
        with self.assertRaises(ValueError):
            validate_pairs([pair, {**pair, "id": "two"}])
        with self.assertRaises(ValueError):
            validate_pairs([{**pair, "reviewed": False}])

    def test_loss_prefers_chosen_relative_to_reference(self):
        lower, _ = dpo_loss(torch.tensor(-2.), torch.tensor(-3.), -2., -2., 0.1)
        higher, _ = dpo_loss(torch.tensor(-2.), torch.tensor(-2.), -2., -2., 0.1)
        self.assertLess(lower.item(), higher.item())
        self.assertAlmostEqual(higher.item(), torch.log(torch.tensor(2.)).item(), places=5)

    def test_logprob_masks_prompt_and_keeps_first_answer_token(self):
        class TinyModel:
            def __call__(self, input_ids, use_cache):
                logits = torch.zeros(1, 4, 5)
                logits[0, 0, 1] = 100  # Prompt token must not count.
                logits[0, 1, 3] = 2    # First completion token.
                return type("Output", (), {"logits": logits})()

        ids = torch.tensor([[0, 1, 3, 4]])
        value = completion_logprob(TinyModel(), (ids, 2), torch.device("cpu"))
        expected = torch.log_softmax(torch.tensor([0., 0., 0., 2., 0.]), 0)[3]
        expected += torch.log(torch.tensor(0.2))
        self.assertAlmostEqual(value.item(), expected.item(), places=5)


if __name__ == "__main__":
    unittest.main()
