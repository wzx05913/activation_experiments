"""Shortlist unused, reviewed PKU pairs for a follow-up safety audit."""
from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter
from pathlib import Path

from .expand_pku_dpo import parse_domains, read_jsonl
from .prepare_pku_sft import key

TARGETS = {"cybercrime_intrusion": 12, "chemical_biological": 8,
           "illegal_email_fraud": 4, "misinformation_disinformation": 8,
           "copyright": 3}
EMAIL_TERMS = ("email", "e-mail", "e mail", "phishing")
FRAUD_TERMS = ("scam", "phish", "fraud", "fake", "deceiv", "trick")
FIELDS = ("id", "stratum", "domain", "attack_form", "prompt", "chosen", "rejected",
          "manual_approve", "review_note")


def shortlist(review_files, train_file, holdout_file, targets=TARGETS, seed=42):
    train, holdout = read_jsonl(train_file), read_jsonl(holdout_file)
    blocked_ids = {row["id"] for row in train + holdout}
    blocked_prompts = {key(row["prompt"]) for row in train + holdout}
    if len(blocked_ids) != len(train) + len(holdout):
        raise ValueError("Train and holdout IDs overlap")
    pool, seen_ids, seen_prompts = [], set(), set()
    for path in review_files:
        with path.open(encoding="utf-8-sig", newline="") as stream:
            for row in csv.DictReader(stream):
                identifier, normalized = row["id"], key(row["prompt"])
                if identifier in seen_ids or normalized in seen_prompts:
                    raise ValueError(f"Duplicate review pair: {identifier}")
                seen_ids.add(identifier)
                seen_prompts.add(normalized)
                if identifier in blocked_ids or normalized in blocked_prompts:
                    continue
                if all(row[field].strip().lower() == "yes" for field in
                       ("prompt_harmful", "chosen_safe", "rejected_unsafe")):
                    if not row["chosen"].strip() or not row["rejected"].strip():
                        raise ValueError(f"Empty approved response: {identifier}")
                    pool.append(row)

    rng = random.Random(seed)
    rng.shuffle(pool)
    selected, counts, selected_ids = [], Counter(), set()
    for stratum, quota in targets.items():
        if quota < 0:
            raise ValueError("Targets must be nonnegative")
        candidates = [row for row in pool if row["id"] not in selected_ids
                      and (stratum in parse_domains(row["domain"])
                           if stratum != "illegal_email_fraud" else
                           "illegal" in parse_domains(row["domain"])
                           and any(term in row["prompt"].casefold() for term in EMAIL_TERMS)
                           and any(term in row["prompt"].casefold() for term in FRAUD_TERMS))]
        forms = Counter()
        while counts[stratum] < quota and candidates:
            row = min(candidates, key=lambda item: forms[item["attack_form"]])
            candidates.remove(row)
            forms[row["attack_form"]] += 1
            counts[stratum] += 1
            selected_ids.add(row["id"])
            selected.append({field: row[field] for field in ("id", "domain", "attack_form",
                                                              "prompt", "chosen", "rejected")}
                            | {"stratum": stratum, "manual_approve": "", "review_note": ""})
    report = {"seed": seed, "base_train_pairs": len(train), "holdout_pairs": len(holdout),
              "eligible_unused": len(pool), "requested": targets,
              "selected": dict(counts), "shortfalls": {domain: n - counts[domain]
                                                  for domain, n in targets.items() if counts[domain] < n},
              "note": "Review-only shortlist, not a training file. Recheck chosen quality, subtopics, and semantic evaluation overlap before export."}
    return selected, report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review-dir", type=Path, required=True)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--holdout", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error("Output directory must be empty")
    files = [args.review_dir / name for name in ("existing_train_review.csv", "additional_review.csv")]
    rows, report = shortlist(files, args.train, args.holdout, seed=args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "shortlist_review.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    (args.output_dir / "shortlist_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
