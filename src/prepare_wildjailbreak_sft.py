"""Prepare a reviewed SFT pilot from WildJailbreak train/train.tsv."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from pathlib import Path

from .prepare_pku_sft import exclusions, key, write
from .scoring_rules import refusal_match, score

FIELDS = ["id", "kind", "prompt", "response", "prompt_valid", "response_valid"]
SOURCE = "allenai/wildjailbreak:train/train.tsv"
KINDS = {"vanilla_harmful": "harmful", "vanilla_benign": "benign"}


def collect(rows, blocked, max_chars=1200):
    groups = {"harmful": [], "benign": []}
    seen = set()
    counts = {"input": 0, "excluded": 0, "eligible_harmful": 0, "eligible_benign": 0}
    for index, row in enumerate(rows):
        counts["input"] += 1
        kind = KINDS.get(row.get("data_type"))
        if not kind:
            continue
        prompt = (row.get("vanilla") or "").strip()
        response = (row.get("completion") or "").strip()
        normalized = key(prompt)
        if not normalized or not response or len(response) > max_chars or normalized in blocked or normalized in seen:
            counts["excluded"] += 1
            continue
        if kind == "harmful":
            good = refusal_match(response[:160])[0] and score({"response": response}, "harmful")[0] == "0"
        else:
            good = not refusal_match(response)[0] and score({"response": response}, "benign")[0] == "0"
        if not good:
            counts["excluded"] += 1
            continue
        seen.add(normalized)
        groups[kind].append({"id": f"wildjailbreak_train_{index:06d}", "kind": kind,
                             "prompt": prompt, "response": response,
                             "prompt_valid": "", "response_valid": ""})
        counts[f"eligible_{kind}"] += 1
    return groups, counts


def read_tsv(path):
    csv.field_size_limit(10_000_000)
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        required = {"vanilla", "completion", "data_type"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"Expected WildJailbreak train.tsv columns {sorted(required)}; got {reader.fieldnames}")
        yield from reader


def save_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def approved(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if not set(FIELDS).issubset(reader.fieldnames or []):
            raise ValueError(f"Missing review columns in {path}")
        rows = list(reader)
    if not rows:
        raise ValueError(f"Empty review CSV: {path}")
    for row in rows:
        if row["kind"] not in {"harmful", "benign"}:
            raise ValueError(f"Unknown kind in {path}: {row['id']}")
        if row["prompt_valid"].strip().lower() not in {"yes", "no"} or row["response_valid"].strip().lower() not in {"yes", "no"}:
            raise ValueError(f"Fill both review columns yes/no in {path}: {row['id']}")
    return [{"id": row["id"], "prompt": row["prompt"], "response": row["response"],
             "source": SOURCE, "kind": row["kind"], "reviewed": True}
            for row in rows if row["prompt_valid"].strip().lower() == "yes"
            and row["response_valid"].strip().lower() == "yes"]


def check_unique(train, holdout, blocked):
    rows = train + holdout
    prompts = [key(row["prompt"]) for row in rows]
    ids = [row["id"] for row in rows]
    if not all(prompts) or not all(row["response"].strip() for row in rows) or len(prompts) != len(set(prompts)) or len(ids) != len(set(ids)) or set(prompts) & blocked:
        raise ValueError("Duplicate, empty, or evaluation-overlapping samples")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    screen = sub.add_parser("screen")
    screen.add_argument("--tsv", type=Path, required=True)
    screen.add_argument("--eval-root", type=Path, required=True)
    screen.add_argument("--output-dir", type=Path, required=True)
    screen.add_argument("--harmful-train", type=int, default=160)
    screen.add_argument("--benign-train", type=int, default=40)
    screen.add_argument("--harmful-holdout", type=int, default=40)
    screen.add_argument("--benign-holdout", type=int, default=10)
    screen.add_argument("--seed", type=int, default=42)
    export = sub.add_parser("export")
    export.add_argument("--review-dir", type=Path, required=True)
    export.add_argument("--eval-root", type=Path, required=True)
    export.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error("Output directory must be empty")
    blocked = exclusions(args.eval_root)
    if args.action == "screen":
        sizes = (args.harmful_train, args.benign_train, args.harmful_holdout, args.benign_holdout)
        if any(size <= 0 for size in sizes) or args.tsv.name != "train.tsv":
            parser.error("Use positive counts and WildJailbreak train.tsv (never eval.tsv)")
        groups, counts = collect(read_tsv(args.tsv), blocked)
        rng = random.Random(args.seed)
        train, holdout = [], []
        for kind, n_train, n_holdout in (("harmful", sizes[0], sizes[2]), ("benign", sizes[1], sizes[3])):
            group = groups[kind]
            if len(group) < n_train + n_holdout:
                parser.error(f"Only {len(group)} eligible {kind} samples, need {n_train + n_holdout}")
            rng.shuffle(group)
            train.extend(group[:n_train])
            holdout.extend(group[n_train:n_train + n_holdout])
        rng.shuffle(train)
        rng.shuffle(holdout)
        check_unique(train, holdout, blocked)
        save_csv(args.output_dir / "train_review.csv", train)
        save_csv(args.output_dir / "holdout_review.csv", holdout)
        with args.tsv.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        report = {"source": SOURCE, "source_sha256": digest, "seed": args.seed,
                  "selected_train": {"harmful": sizes[0], "benign": sizes[1]},
                  "selected_holdout": {"harmful": sizes[2], "benign": sizes[3]},
                  **counts, "note": "Candidates only. Review every prompt and completion before export."}
        (args.output_dir / "selection_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    else:
        train = approved(args.review_dir / "train_review.csv")
        holdout = approved(args.review_dir / "holdout_review.csv")
        if not train or not holdout:
            parser.error("Train and holdout both need approved rows")
        check_unique(train, holdout, blocked)
        write(args.output_dir / "safety_train.jsonl", train)
        write(args.output_dir / "safety_holdout.jsonl", holdout)
        report = {"train": {kind: sum(row["kind"] == kind for row in train) for kind in ("harmful", "benign")},
                  "holdout": {kind: sum(row["kind"] == kind for row in holdout) for kind in ("harmful", "benign")}}
        (args.output_dir / "export_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
