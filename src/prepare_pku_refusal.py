"""Select PKU refusal candidates and export only human-approved examples."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from pathlib import Path

from .prepare_pku_sft import exclusions, key, write
from .scoring_rules import refusal_match, score

FIELDS = ["id", "prompt", "response", "prompt_source", "response_source",
          "unsafe_response_severity", "unsafe_response_categories", "prompt_harmful", "response_safe"]
SOURCE = "PKU-Alignment/PKU-SafeRLHF:alpaca-7b:train"


def candidates(rows, blocked, max_response_chars=300):
    eligible, counts, seen = [], {}, set()
    for index, row in enumerate(rows):
        def skip(reason):
            counts[reason] = counts.get(reason, 0) + 1

        a, b = row["is_response_0_safe"], row["is_response_1_safe"]
        if a == b:
            skip("not_one_safe_one_unsafe")
            continue
        selected = 0 if a else 1
        other = 1 - selected
        severity = row[f"response_{other}_severity_level"]
        if severity is None or severity <= 0:
            skip("unsafe_response_not_severe")
            continue
        prompt = str(row["prompt"] or "").strip()
        response = str(row[f"response_{selected}"] or "").strip()
        normalized = key(prompt)
        if not normalized or not response:
            skip("empty")
            continue
        if normalized in blocked:
            skip("eval_overlap")
            continue
        if normalized in seen:
            skip("duplicate")
            continue
        if len(response) > max_response_chars or not refusal_match(response[:160])[0] or score({"response": response}, "harmful")[0] != "0":
            skip("no_short_clean_refusal")
            continue
        seen.add(normalized)
        categories = row[f"response_{other}_harm_category"] or {}
        eligible.append({"id": f"pku_alpaca7b_train_{index:05d}", "prompt": prompt,
                         "response": response, "prompt_source": row["prompt_source"],
                         "response_source": row[f"response_{selected}_source"],
                         "unsafe_response_severity": severity,
                         "unsafe_response_categories": "; ".join(k for k, v in categories.items() if v),
                         "prompt_harmful": "", "response_safe": ""})
    return eligible, counts


def save_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def read_review(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if not set(FIELDS).issubset(reader.fieldnames or []):
            raise ValueError(f"Missing review columns: {path}")
        rows = list(reader)
    if not rows:
        raise ValueError(f"Empty review: {path}")
    for row in rows:
        if row["prompt_harmful"].strip().lower() not in {"yes", "no"} or row["response_safe"].strip().lower() not in {"yes", "no"}:
            raise ValueError(f"Review both columns with yes/no for every candidate: {path} ({row['id']})")
    return rows


def approved(rows):
    return [{"id": r["id"], "prompt": r["prompt"], "response": r["response"],
             "source": SOURCE, "selection": "human_verified_harmful_refusal", "reviewed": True}
            for r in rows if r["prompt_harmful"].strip().lower() == "yes"
            and r["response_safe"].strip().lower() == "yes"]


def ensure_unique(train, holdout, blocked):
    all_rows = train + holdout
    keys = [key(r["prompt"]) for r in all_rows]
    ids = [r["id"] for r in all_rows]
    if not all(keys) or not all(ids) or len(keys) != len(set(keys)) or len(ids) != len(set(ids)) or set(keys) & blocked:
        raise ValueError("Empty, duplicate, or evaluation-overlapping prompt/ID")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    screen = sub.add_parser("screen")
    screen.add_argument("--arrow", type=Path, required=True, help="PKU alpaca-7b train Arrow stream")
    screen.add_argument("--eval-root", type=Path, required=True)
    screen.add_argument("--output-dir", type=Path, required=True)
    screen.add_argument("--train", type=int, default=160)
    screen.add_argument("--holdout", type=int, default=40)
    screen.add_argument("--seed", type=int, default=42)
    export = sub.add_parser("export")
    export.add_argument("--review-dir", type=Path, required=True)
    export.add_argument("--eval-root", type=Path, required=True)
    export.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error("Output directory must be empty; never overwrite previous data")
    if args.action == "screen":
        import pyarrow.ipc as ipc
        if args.train <= 0 or args.holdout <= 0 or "train" not in args.arrow.name or "test" in args.arrow.name:
            parser.error("Use a train Arrow file and positive train/holdout sizes")
        with args.arrow.open("rb") as stream:
            rows = ipc.open_stream(stream).read_all().to_pylist()
        blocked = exclusions(args.eval_root)
        eligible, excluded = candidates(rows, blocked)
        if len(eligible) < args.train + args.holdout:
            parser.error(f"Only {len(eligible)} eligible candidates, requested {args.train + args.holdout}")
        random.Random(args.seed).shuffle(eligible)
        train = eligible[:args.train]
        holdout = eligible[args.train:args.train + args.holdout]
        ensure_unique(train, holdout, blocked)
        save_csv(args.output_dir / "train_review.csv", train)
        save_csv(args.output_dir / "holdout_review.csv", holdout)
        with args.arrow.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        report = {"source": SOURCE, "source_sha256": digest, "seed": args.seed,
                  "eligible": len(eligible), "train_candidates": len(train),
                  "holdout_candidates": len(holdout), "excluded": excluded,
                  "note": "Candidates only: unsafe response labels do not establish harmful prompts; review both yes/no columns before export."}
        (args.output_dir / "selection_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    else:
        train = approved(read_review(args.review_dir / "train_review.csv"))
        holdout = approved(read_review(args.review_dir / "holdout_review.csv"))
        if not train or not holdout:
            parser.error("No approved samples in train or holdout")
        ensure_unique(train, holdout, exclusions(args.eval_root))
        write(args.output_dir / "safety_train.jsonl", train)
        write(args.output_dir / "safety_holdout.jsonl", holdout)
        report = {"train_approved": len(train), "holdout_approved": len(holdout),
                  "note": "Human review recorded; do not train on unreviewed CSV candidates."}
        (args.output_dir / "export_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
