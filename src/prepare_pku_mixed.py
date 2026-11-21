"""Prepare reviewed PKU harmful refusals plus separately reviewed benign tasks."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from pathlib import Path

from .prepare_pku_refusal import approved as approved_harmful, read_review as read_harmful
from .prepare_pku_sft import exclusions, key, write
from .scoring_rules import refusal_match

SOURCE = "PKU-Alignment/PKU-SafeRLHF:alpaca-7b:train"
FIELDS = ["id", "prompt", "response", "prompt_source", "response_source",
          "prompt_benign", "response_helpful"]


def benign_candidates(rows, blocked, max_response_chars=800):
    selected, seen = [], set()
    for index, row in enumerate(rows):
        if not (row["is_response_0_safe"] and row["is_response_1_safe"]):
            continue
        chosen = row["better_response_id"]
        if chosen not in (0, 1):
            continue
        prompt = str(row["prompt"] or "").strip()
        response = str(row[f"response_{chosen}"] or "").strip()
        normalized = key(prompt)
        if (not normalized or normalized in blocked or normalized in seen or not response
                or len(response) > max_response_chars or refusal_match(response)[0]):
            continue
        seen.add(normalized)
        selected.append({"id": f"pku_alpaca7b_benign_{index:05d}", "prompt": prompt,
                         "response": response, "prompt_source": row["prompt_source"],
                         "response_source": row[f"response_{chosen}_source"],
                         "prompt_benign": "", "response_helpful": ""})
    return selected


def save_review(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def approved_benign(path, source=SOURCE):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if not set(FIELDS).issubset(reader.fieldnames or []):
            raise ValueError(f"Missing review columns: {path}")
        rows = list(reader)
    if not rows:
        raise ValueError(f"Empty review: {path}")
    for row in rows:
        if (row["prompt_benign"].strip().lower() not in {"yes", "no"}
                or row["response_helpful"].strip().lower() not in {"yes", "no"}):
            raise ValueError(f"Review both columns yes/no: {path} ({row['id']})")
    return [{"id": row["id"], "prompt": row["prompt"], "response": row["response"],
             "source": source, "kind": "benign", "reviewed": True}
            for row in rows if row["prompt_benign"].strip().lower() == "yes"
            and row["response_helpful"].strip().lower() == "yes"]


def harmful_rows(path):
    return [{**row, "kind": "harmful"} for row in approved_harmful(read_harmful(path))]


def external_candidates(path, blocked, source, max_response_chars=1200):
    if not source.strip() or any(word in source.casefold() for word in ("harmbench", "advbench", "xstest", "eval", "test")):
        raise ValueError("Provide an independent training-only source name")
    selected, seen = [], set()
    with path.open(encoding="utf-8-sig") as stream:
        for index, line in enumerate(stream):
            if not line.strip():
                continue
            row = json.loads(line)
            prompt = str(row.get("prompt") or "").strip()
            response = str(row.get("response") or "").strip()
            normalized = key(prompt)
            if (not normalized or normalized in blocked or normalized in seen or not response
                    or len(response) > max_response_chars or refusal_match(response)[0]):
                continue
            seen.add(normalized)
            selected.append({"id": f"normal_train_{index:06d}", "prompt": prompt,
                             "response": response, "prompt_source": source,
                             "response_source": source, "prompt_benign": "", "response_helpful": ""})
    return selected


def check_unique(train, holdout, blocked):
    rows = train + holdout
    prompts = [key(row["prompt"]) for row in rows]
    ids = [row["id"] for row in rows]
    if (not all(prompts) or not all(ids) or not all(str(row["response"]).strip() for row in rows)
            or len(prompts) != len(set(prompts)) or len(ids) != len(set(ids))
            or set(prompts) & blocked):
        raise ValueError("Empty, duplicate, or evaluation-overlapping ID/prompt/response")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    for name in ("screen", "screen-jsonl"):
        screen = sub.add_parser(name)
        if name == "screen":
            screen.add_argument("--arrow", type=Path, required=True)
        else:
            screen.add_argument("--normal-train-jsonl", type=Path, required=True)
            screen.add_argument("--source-name", required=True)
        screen.add_argument("--harmful-review-dir", type=Path, required=True)
        screen.add_argument("--eval-root", type=Path, required=True)
        screen.add_argument("--output-dir", type=Path, required=True)
        screen.add_argument("--benign-train", type=int, default=160)
        screen.add_argument("--benign-holdout", type=int, default=40)
        screen.add_argument("--seed", type=int, default=42)
    export = sub.add_parser("export")
    export.add_argument("--harmful-review-dir", type=Path, required=True)
    export.add_argument("--benign-review-dir", type=Path, required=True)
    export.add_argument("--eval-root", type=Path, required=True)
    export.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error("Output directory must be empty; never overwrite prior data")
    blocked = exclusions(args.eval_root)
    harmful_train = harmful_rows(args.harmful_review_dir / "train_review.csv")
    harmful_holdout = harmful_rows(args.harmful_review_dir / "holdout_review.csv")
    if not harmful_train or not harmful_holdout:
        parser.error("Both harmful splits need approved samples")
    check_unique(harmful_train, harmful_holdout, blocked)
    if args.action in {"screen", "screen-jsonl"}:
        if min(args.benign_train, args.benign_holdout) <= 0:
            parser.error("Use positive split sizes")
        excluded = blocked | {key(row["prompt"]) for row in harmful_train + harmful_holdout}
        if args.action == "screen":
            if args.arrow.name != "pku-safe_rlhf-train.arrow":
                parser.error("Use PKU train Arrow only")
            import pyarrow.ipc as ipc
            with args.arrow.open("rb") as stream:
                rows = ipc.open_stream(stream).read_all().to_pylist()
            candidates = benign_candidates(rows, excluded)
            source, source_file = SOURCE, args.arrow
        else:
            if "train" not in args.normal_train_jsonl.name.casefold() or any(
                    word in args.normal_train_jsonl.name.casefold() for word in ("eval", "test")):
                parser.error("Use an independent train JSONL only, never evaluation data")
            source, source_file = args.source_name, args.normal_train_jsonl
            candidates = external_candidates(source_file, excluded, source)
        if len(candidates) < args.benign_train + args.benign_holdout:
            parser.error(f"Only {len(candidates)} benign candidates available")
        random.Random(args.seed).shuffle(candidates)
        train = candidates[:args.benign_train]
        holdout = candidates[args.benign_train:args.benign_train + args.benign_holdout]
        check_unique(harmful_train + train, harmful_holdout + holdout, blocked)
        save_review(args.output_dir / "benign_train_review.csv", train)
        save_review(args.output_dir / "benign_holdout_review.csv", holdout)
        with source_file.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        report = {"source": source, "source_sha256": digest, "seed": args.seed,
                  "harmful_train_approved": len(harmful_train), "harmful_holdout_approved": len(harmful_holdout),
                  "dual_safe_non_refusal_candidates": len(candidates), "benign_train_candidates": len(train),
                  "benign_holdout_candidates": len(holdout),
                  "note": "Source labels and non-refusal filtering do NOT establish a benign prompt. Review every prompt and answer before export."}
    else:
        report_path = args.benign_review_dir / "selection_report.json"
        source = json.loads(report_path.read_text(encoding="utf-8"))["source"]
        benign_train = approved_benign(args.benign_review_dir / "benign_train_review.csv", source)
        benign_holdout = approved_benign(args.benign_review_dir / "benign_holdout_review.csv", source)
        if len(benign_train) * 2 < len(harmful_train) or not benign_holdout:
            parser.error("Need at least 80 approved benign train rows for 160 harmful rows, plus benign holdout; screen more if needed")
        train = harmful_train + benign_train
        holdout = harmful_holdout + benign_holdout
        check_unique(train, holdout, blocked)
        random.Random(42).shuffle(train)
        random.Random(43).shuffle(holdout)
        write(args.output_dir / "safety_train.jsonl", train)
        write(args.output_dir / "safety_holdout.jsonl", holdout)
        report = {split: {kind: sum(row["kind"] == kind for row in group)
                          for kind in ("harmful", "benign")}
                  for split, group in (("train", train), ("holdout", holdout))}
    (args.output_dir / ("selection_report.json" if args.action == "screen" else "export_report.json")).write_text(
        json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
