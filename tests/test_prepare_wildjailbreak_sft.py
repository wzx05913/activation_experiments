"""Offline screening tests without restricted dataset access."""
import tempfile
import unittest
from pathlib import Path

from src.prepare_wildjailbreak_sft import approved, check_unique, collect, read_tsv, save_csv


class WildJailbreakTests(unittest.TestCase):
    def test_type_balance_and_eval_exclusion(self):
        rows = [
            {"vanilla": "harmful candidate", "completion": "I cannot help with that.", "data_type": "vanilla_harmful"},
            {"vanilla": "benign candidate", "completion": "Here is a helpful answer.", "data_type": "vanilla_benign"},
            {"vanilla": "excluded", "completion": "I cannot help.", "data_type": "vanilla_harmful"},
            {"vanilla": "not refusal", "completion": "Here is an answer.", "data_type": "vanilla_harmful"},
            {"vanilla": "adversarial", "completion": "I cannot help.", "data_type": "adversarial_harmful"},
        ]
        groups, counts = collect(rows, {"excluded"})
        self.assertEqual(len(groups["harmful"]), 1)
        self.assertEqual(len(groups["benign"]), 1)
        self.assertEqual(counts["eligible_harmful"], 1)
        check_unique(groups["harmful"], groups["benign"], {"excluded"})
        with self.assertRaises(ValueError):
            check_unique(groups["harmful"], groups["harmful"], set())

    def test_real_tsv_shape_and_review_gate(self):
        with tempfile.TemporaryDirectory() as directory:
            tsv = Path(directory) / "train.tsv"
            tsv.write_text("vanilla\tcompletion\tdata_type\nrequest\tI cannot help.\tvanilla_harmful\n", encoding="utf-8")
            group, _ = collect(read_tsv(tsv), set())
            candidate = group["harmful"][0]
            review = Path(directory) / "review.csv"
            save_csv(review, [candidate])
            with self.assertRaises(ValueError):
                approved(review)
            candidate["prompt_valid"] = "yes"
            candidate["response_valid"] = "yes"
            save_csv(review, [candidate])
            self.assertEqual(approved(review)[0]["reviewed"], True)


if __name__ == "__main__":
    unittest.main()
