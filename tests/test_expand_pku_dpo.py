"""Offline checks for coverage-review and expansion boundaries."""
import csv
import tempfile
import unittest
from pathlib import Path

from src.expand_pku_dpo import approved, parse_domains, read_review, review_row, save_review, validate_pairs


class ExpansionReviewTests(unittest.TestCase):
    def setUp(self):
        self.source = {"pku_alpaca7b_train_00001": {
            "id": "pku_alpaca7b_train_00001", "prompt": "Example request",
            "chosen": "I cannot assist.", "rejected": "Unsafe answer",
            "prompt_source": "PKU", "pku_categories": "Cybercrime",
            "domain_hint": "cybercrime_intrusion"}}

    def test_unreviewed_cannot_be_exported(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "review.csv"
            save_review(path, [review_row(next(iter(self.source.values())))])
            with self.assertRaisesRegex(ValueError, "yes/no"):
                read_review(path)

    def test_approved_requires_domain_and_verifies_original_pair(self):
        row = review_row(next(iter(self.source.values())))
        row.update(prompt_harmful="yes", chosen_safe="yes", rejected_unsafe="yes")
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "review.csv"
            save_review(path, [row])
            with self.assertRaisesRegex(ValueError, "domain/attack_form"):
                read_review(path)
            row.update(domain="cybercrime_intrusion", attack_form="direct")
            save_review(path, [row])
            pair = approved(read_review(path), self.source)[0]
            validate_pairs([pair])
            row["chosen"] = "Modified answer"
            with self.assertRaisesRegex(ValueError, "differs from Arrow"):
                approved([row], self.source)

    def test_no_review_does_not_approve(self):
        row = review_row(next(iter(self.source.values())))
        row.update(prompt_harmful="yes", chosen_safe="no", rejected_unsafe="yes")
        self.assertEqual(approved([row], self.source), [])

    def test_multilabel_domain_and_reviewed_form(self):
        row = review_row(next(iter(self.source.values())))
        row.update(prompt_harmful="yes", chosen_safe="yes", rejected_unsafe="yes",
                   domain="cybercrime_intrusion; illegal", attack_form="disguised_intent")
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "review.csv"
            save_review(path, [row])
            self.assertEqual(approved(read_review(path), self.source)[0]["domain"], row["domain"])
            for invalid in ("illegal; illegal", "illegal; nonsense", "illegal; other"):
                row["domain"] = invalid
                save_review(path, [row])
                with self.assertRaisesRegex(ValueError, "domain/attack_form"):
                    read_review(path)
        self.assertEqual(parse_domains("illegal; harmful"), ["illegal", "harmful"])

    def test_partial_export_requires_both_labels_and_skips_pending(self):
        row = review_row(next(iter(self.source.values())))
        row.update(prompt_harmful="yes", chosen_safe="yes", rejected_unsafe="yes")
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "review.csv"
            save_review(path, [row])
            pending = read_review(path, allow_unclassified=True)
            self.assertEqual(approved(pending, self.source, allow_unclassified=True), [])
            row["domain"] = "cybercrime_intrusion"
            save_review(path, [row])
            with self.assertRaisesRegex(ValueError, "domain/attack_form"):
                read_review(path, allow_unclassified=True)
            row["attack_form"] = "direct"
            save_review(path, [row])
            classified = read_review(path, allow_unclassified=True)
            self.assertEqual(len(approved(classified, self.source, allow_unclassified=True)), 1)


if __name__ == "__main__":
    unittest.main()
