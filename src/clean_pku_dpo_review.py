"""Stage reviewed PKU pairs without overwriting the original review."""
from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
from collections import Counter
from pathlib import Path

from .expand_pku_dpo import DOMAINS, FIELDS, FORMS

REVIEW_FILES = ("existing_train_review.csv", "additional_review.csv")
STAGED_FIELDS = (*FIELDS, "suggested_domain")
EXCLUDED_FIELDS = ("review_file", *FIELDS, "chosen_safe_raw", "exclusion_reason")


def normalize_flag(value: str) -> tuple[str, str]:
    raw = value.strip()
    if raw.casefold() in ("yes", "no"):
        return raw.casefold(), ""
    # A note entered in the yes/no cell is not a positive safety judgment.
    match = re.fullmatch(r"no\s*[(\uFF08]([^()\uFF08\uFF09]+)[)\uFF09]", raw, re.IGNORECASE)
    if match:
        return "no", match.group(1).strip()
    raise ValueError(f"Unrecognized review flag: {value!r}")


def stage_review(review_dir: Path, output_dir: Path) -> dict:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError("Output directory must be empty")
    source_report = json.loads((review_dir / "coverage_screen_report.json").read_text(encoding="utf-8"))
    staged, excluded = {}, []
    seen_ids = set()
    seen_prompts = set()
    suggestions = Counter()
    pending = 0
    for filename in REVIEW_FILES:
        path = review_dir / filename
        with path.open(encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            if not set(FIELDS).issubset(reader.fieldnames or []):
                raise ValueError(f"Missing review columns: {path}")
            rows = list(reader)
        if filename == REVIEW_FILES[0]:
            if len(rows) != source_report["old_train"]:
                raise ValueError("Existing review count changed")
        elif [r["id"] for r in rows] != source_report["review_candidate_ids"]:
            raise ValueError("Additional review IDs/order changed")
        result = []
        for row in rows:
            identifier = row["id"].strip()
            prompt_key = " ".join(row["prompt"].split()).casefold()
            if not identifier or not prompt_key or identifier in seen_ids or prompt_key in seen_prompts:
                raise ValueError(f"Duplicate or empty ID/prompt: {identifier}")
            seen_ids.add(identifier)
            seen_prompts.add(prompt_key)
            flags = {field: normalize_flag(row[field]) for field in
                     ("prompt_harmful", "chosen_safe", "rejected_unsafe")}
            original_chosen_flag = row["chosen_safe"]
            for field, (value, _) in flags.items():
                row[field] = value
            if flags["chosen_safe"][1]:
                note = f"chosen_safe cell: {flags['chosen_safe'][1]}"
                row["review_note"] = f"{row['review_note'].strip()} | {note}".strip(" |")
            accepted = all(value == "yes" for value, _ in flags.values())
            if not accepted:
                excluded.append({"review_file": filename, **{f: row[f] for f in FIELDS},
                                 "chosen_safe_raw": original_chosen_flag,
                                 "exclusion_reason": "; ".join(f for f, (v, _) in flags.items() if v != "yes")})
            elif not row["chosen"].strip() or not row["rejected"].strip():
                raise ValueError(f"Empty response: {identifier}")
            hint = row["domain_hint"].strip()
            row["suggested_domain"] = hint if hint in DOMAINS else ""
            if accepted:
                suggestions[row["suggested_domain"] or "ambiguous_or_missing"] += 1
                if not row["domain"].strip() or not row["attack_form"].strip():
                    pending += 1
            if row["domain"].strip() and row["domain"].strip() not in DOMAINS:
                raise ValueError(f"Invalid domain: {identifier}")
            if row["attack_form"].strip() and row["attack_form"].strip() not in FORMS:
                raise ValueError(f"Invalid attack form: {identifier}")
            result.append({f: row[f] for f in STAGED_FIELDS})
        staged[filename] = result
    output_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(review_dir / "coverage_screen_report.json", output_dir / "coverage_screen_report.json")
    for filename, rows in staged.items():
        with (output_dir / filename).open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, STAGED_FIELDS)
            writer.writeheader()
            writer.writerows(rows)
    with (output_dir / "excluded_review.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, EXCLUDED_FIELDS)
        writer.writeheader()
        writer.writerows(excluded)
    report = {"total_reviewed": len(seen_ids), "safety_approved_candidates": len(seen_ids) - len(excluded),
              "excluded": len(excluded), "excluded_by_file": dict(Counter(r["review_file"] for r in excluded)),
              "candidate_hint_counts_not_verified": dict(suggestions), "classification_pending": pending,
              "note": "Suggestions derive from response harm labels, not request classifications. Do not train until manual categories and export validation are complete."}
    (output_dir / "cleaning_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(stage_review(args.review_dir, args.output_dir), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
