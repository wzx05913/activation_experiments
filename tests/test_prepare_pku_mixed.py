"""Offline tests for mixed PKU selection, review gates and sampling."""
import tempfile
import unittest
from pathlib import Path
import random
import json

from src.prepare_pku_mixed import approved_benign, benign_candidates, check_unique, external_candidates, save_review
from src.repair_sampling import epoch_order, validate_mixed_rows


def pair(prompt, response="A useful explanation."):
    return {"prompt": prompt, "response_0": response, "response_1": "A second answer.",
            "is_response_0_safe": True, "is_response_1_safe": True,
            "better_response_id": 0, "prompt_source": "test", "response_0_source": "test"}


class MixedPreparationTests(unittest.TestCase):
    def test_candidate_filter_and_review_gate(self):
        rows = [pair("normal task"), pair("evaluation task"), pair("normal task"),
                pair("refusal", "I cannot help with that."),
                {**pair("one safe"), "is_response_1_safe": False}]
        candidates = benign_candidates(rows, {"evaluation task"})
        self.assertEqual([row["prompt"] for row in candidates], ["normal task"])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "review.csv"
            save_review(path, candidates)
            with self.assertRaises(ValueError):
                approved_benign(path)
            candidates[0]["prompt_benign"] = "yes"
            candidates[0]["response_helpful"] = "no"
            save_review(path, candidates)
            self.assertEqual(approved_benign(path), [])
            candidates[0]["response_helpful"] = "yes"
            save_review(path, candidates)
            self.assertEqual(approved_benign(path)[0]["kind"], "benign")

    def test_overlap_and_reviewed_mixed_validation(self):
        harmful = {"id": "h", "prompt": "unsafe request", "response": "I cannot help.",
                   "kind": "harmful", "reviewed": True}
        benign = {"id": "b", "prompt": "ordinary request", "response": "An answer.",
                  "kind": "benign", "reviewed": True}
        check_unique([harmful], [benign], set())
        self.assertEqual(validate_mixed_rows([harmful, benign]), {"harmful": 1, "benign": 1})
        with self.assertRaises(ValueError):
            check_unique([harmful], [benign], {"ordinary request"})
        with self.assertRaises(ValueError):
            validate_mixed_rows([harmful, {**benign, "reviewed": False}])
        with self.assertRaises(ValueError):
            validate_mixed_rows([harmful, {**benign, "prompt": "UNSAFE request"}])

    def test_external_training_source_is_screened_and_provenance_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "normal_train.jsonl"
            rows = [{"prompt": "Explain gravity.", "response": "Gravity attracts masses."},
                    {"prompt": "Evaluation task", "response": "An answer."},
                    {"prompt": "Refused", "response": "I cannot help."}]
            path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
            candidates = external_candidates(path, {"evaluation task"}, "independent-instructions:train")
            self.assertEqual([row["prompt"] for row in candidates], ["Explain gravity."])
            candidates[0].update(prompt_benign="yes", response_helpful="yes")
            review = Path(directory) / "review.csv"
            save_review(review, candidates)
            self.assertEqual(approved_benign(review, "independent-instructions:train")[0]["source"],
                             "independent-instructions:train")
            with self.assertRaises(ValueError):
                external_candidates(path, set(), "HarmBench-test")

    def test_imbalanced_rows_are_balanced_by_sampling(self):
        rows = [{"kind": "harmful"}] * 3 + [{"kind": "benign"}]
        order = epoch_order(rows, "mixed", random.Random(42))
        self.assertEqual(len(order), 6)
        self.assertEqual([rows[i]["kind"] for i in order], ["harmful", "benign"] * 3)


if __name__ == "__main__":
    unittest.main()
