"""Offline checks for the reviewed-refusal data boundary."""
import tempfile
import unittest
from pathlib import Path

from src.prepare_pku_refusal import approved, candidates, ensure_unique, read_review, save_csv


def row(prompt, safe_answer="I cannot help with that request."):
    return {"prompt": prompt, "response_0": safe_answer, "response_1": "unsafe",
            "is_response_0_safe": True, "is_response_1_safe": False,
            "response_1_severity_level": 2, "response_1_harm_category": {"Violence": True},
            "prompt_source": "test", "response_0_source": "test"}


class RefusalSelectionTests(unittest.TestCase):
    def test_selects_only_short_labeled_refusal_and_excludes_overlap(self):
        rows = [row("harmful request"), row("evaluation prompt"), row("harmful request"),
                row("benign request", "Here is the information."), row("long response", "I cannot help. " * 50)]
        rows.append({**row("both safe"), "is_response_1_safe": True})
        rows.append({**row("no severity"), "response_1_severity_level": 0})
        selected, counts = candidates(rows, {"evaluation prompt"})
        self.assertEqual([r["prompt"] for r in selected], ["harmful request"])
        self.assertEqual(selected[0]["unsafe_response_categories"], "Violence")
        self.assertEqual(counts["eval_overlap"], 1)
        self.assertEqual(counts["duplicate"], 1)

    def test_review_is_required_and_only_two_yes_values_export(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "review.csv"
            selected, _ = candidates([row("first"), row("second")], set())
            save_csv(path, selected)
            with self.assertRaises(ValueError):
                read_review(path)
            selected[0]["prompt_harmful"] = "yes"
            selected[0]["response_safe"] = "yes"
            selected[1]["prompt_harmful"] = "no"
            selected[1]["response_safe"] = "yes"
            save_csv(path, selected)
            reviewed = read_review(path)
            self.assertEqual([r["prompt"] for r in approved(reviewed)], ["first"])
            self.assertTrue(approved(reviewed)[0]["reviewed"])

    def test_overlap_checked_again_at_export(self):
        selected, _ = candidates([row("first"), row("second")], set())
        with self.assertRaises(ValueError):
            ensure_unique(selected[:1], selected[:1], set())
        with self.assertRaises(ValueError):
            ensure_unique(selected[:1], selected[1:], {"second"})


if __name__ == "__main__":
    unittest.main()
