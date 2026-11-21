import csv
import json
import tempfile
import unittest
from pathlib import Path

from src.shortlist_pku_dpo import shortlist


class ShortlistTests(unittest.TestCase):
    def test_excludes_train_holdout_and_requires_email_fraud_context(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            reviews = root / "review.csv"
            fields = ("id", "prompt", "chosen", "rejected", "domain", "attack_form",
                      "prompt_harmful", "chosen_safe", "rejected_unsafe")
            rows = [
                ("train", "existing", "cybercrime_intrusion"),
                ("holdout", "reserved", "cybercrime_intrusion"),
                ("cyber", "unauthorized account access", "cybercrime_intrusion"),
                ("email", "fake email scam", "illegal"),
                ("privacy", "personal email address", "illegal"),
                ("blackmail", "blackmail request", "illegal"),
            ]
            with reviews.open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=fields)
                writer.writeheader()
                for identifier, prompt, domain in rows:
                    writer.writerow({"id": identifier, "prompt": prompt, "chosen": "refusal",
                                     "rejected": "unsafe", "domain": domain,
                                     "attack_form": "direct_request", "prompt_harmful": "yes",
                                     "chosen_safe": "yes", "rejected_unsafe": "yes"})
            train, holdout = root / "train.jsonl", root / "holdout.jsonl"
            train.write_text(json.dumps({"id": "train", "prompt": "existing"}) + "\n", encoding="utf-8")
            holdout.write_text(json.dumps({"id": "holdout", "prompt": "reserved"}) + "\n", encoding="utf-8")
            selected, report = shortlist([reviews], train, holdout,
                                         {"cybercrime_intrusion": 2, "illegal_email_fraud": 2})
            self.assertEqual({row["id"] for row in selected}, {"cyber", "email"})
            self.assertEqual(report["shortfalls"], {"cybercrime_intrusion": 1,
                                                    "illegal_email_fraud": 1})
            self.assertTrue(all(row["manual_approve"] == "" for row in selected))


if __name__ == "__main__":
    unittest.main()
