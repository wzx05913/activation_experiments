"""Focused checks for held-out DPO preference evaluation."""
import unittest
from types import SimpleNamespace

import torch

from src.evaluate_dpo_holdout import check_split, objective_diagnostic, summarize
from src.train_dpo import completion_logprob


class HoldoutEvaluationTests(unittest.TestCase):
    def test_completion_logprob_reductions_include_eos(self):
        class FixedModel:
            def __call__(self, input_ids, use_cache):
                return SimpleNamespace(logits=torch.zeros((1, input_ids.shape[1], 4)))

        pair = (torch.tensor([[0, 1, 2, 3]]), 2)
        summed = completion_logprob(FixedModel(), pair, "cpu", "sum")
        averaged = completion_logprob(FixedModel(), pair, "cpu", "mean")
        self.assertAlmostEqual(summed.item(), 2 * averaged.item())
        with self.assertRaises(ValueError):
            completion_logprob(FixedModel(), pair, "cpu", "invalid")

    def test_summary_treats_ties_as_not_preferred(self):
        result = summarize([2.0, -1.0, 0.0])
        self.assertEqual(result["chosen_preferred"], 1)
        self.assertEqual(result["rejected_preferred"], 1)
        self.assertEqual(result["ties"], 1)
        self.assertAlmostEqual(result["preference_accuracy"], 1 / 3)

    def test_objective_diagnostic_detects_adapter_progress(self):
        result = objective_diagnostic([2.0, 3.0], [2.0, 3.0], [3.0, 5.0], 0.1)
        self.assertEqual(result["repaired"]["positive_relative_to_reference"], 2)
        self.assertLess(result["repaired"]["mean_loss"], result["unrepaired"]["mean_loss"])
        self.assertAlmostEqual(result["mean_adapter_margin_delta"], 1.5)

    def test_split_rejects_matching_prompts_even_with_different_ids(self):
        train = [{"id": "train", "prompt": " Harmful request ", "chosen": "no",
                  "rejected": "yes", "reviewed": True}]
        holdout = [{**train[0], "id": "test", "prompt": "harmful  request"}]
        with self.assertRaisesRegex(ValueError, "overlap"):
            check_split(holdout, train)
        check_split([{**holdout[0], "prompt": "other request"}], train)


if __name__ == "__main__":
    unittest.main()
