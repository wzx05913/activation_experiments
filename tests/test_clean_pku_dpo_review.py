"""Tests for review staging and rejection bookkeeping."""
import csv
import json
import tempfile
import unittest
from pathlib import Path

from src.clean_pku_dpo_review import normalize_flag, stage_review
from src.expand_pku_dpo import read_review, review_row, save_review


class CleaningTests(unittest.TestCase):
    def test_normalize_annotated_no_only(self):
        self.assertEqual(normalize_flag("no\uFF08not a refusal\uFF09"), ("no", "not a refusal"))
        with self.assertRaises(ValueError):
            normalize_flag("maybe")

    def test_stage_preserves_rejection_and_needs_categories(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            review = root / "review"
            review.mkdir()
            original = {"id": "old", "prompt": "Prompt A", "chosen": "Refusal",
                        "rejected": "Unsafe", "prompt_source": "PKU", "pku_categories": "Cybercrime",
                        "domain_hint": "cybercrime_intrusion"}
            extra = {**original, "id": "new", "prompt": "Prompt B"}
            old_row = review_row(original)
            old_row.update(prompt_harmful="yes", chosen_safe="yes", rejected_unsafe="yes")
            extra_row = review_row(extra)
            extra_row.update(prompt_harmful="yes", chosen_safe="no\uFF08not a refusal\uFF09",
                             rejected_unsafe="yes")
            save_review(review / "existing_train_review.csv", [old_row])
            save_review(review / "additional_review.csv", [extra_row])
            (review / "coverage_screen_report.json").write_text(
                json.dumps({"old_train": 1, "review_candidate_ids": ["new"]}), encoding="utf-8")
            report = stage_review(review, root / "staged")
            self.assertEqual((report["safety_approved_candidates"], report["excluded"],
                              report["classification_pending"]), (1, 1, 1))
            with (root / "staged" / "excluded_review.csv").open(encoding="utf-8-sig", newline="") as stream:
                rejected = list(csv.DictReader(stream))
            self.assertEqual(rejected[0]["chosen_safe_raw"], "no\uFF08not a refusal\uFF09")
            with (root / "staged" / "existing_train_review.csv").open(encoding="utf-8-sig", newline="") as stream:
                staged_old = list(csv.DictReader(stream))
            self.assertEqual(staged_old[0]["domain"], "")
            self.assertEqual(staged_old[0]["suggested_domain"], "cybercrime_intrusion")
            with self.assertRaisesRegex(ValueError, "domain/attack_form"):
                read_review(root / "staged" / "existing_train_review.csv")
            self.assertEqual(extra_row["chosen_safe"], "no\uFF08not a refusal\uFF09")


if __name__ == "__main__":
    unittest.main()
