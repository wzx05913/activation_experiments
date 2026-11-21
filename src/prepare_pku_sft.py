"""Prepare a small SFT pilot from PKU-SafeRLHF Arrow data."""
import argparse
import csv
import hashlib
import json
import random
from pathlib import Path

import pyarrow.ipc as ipc


def key(text):
    return " ".join(text.split()).casefold()


def exclusions(root):
    blocked = set()
    for filename, field in [("advbench/harmful_behaviors.csv", "goal"), ("harmbench/harmbench_behaviors_text_val.csv", "Behavior"), ("harmbench/harmbench_behaviors_text_test.csv", "Behavior"), ("xstest/xstest_prompts.csv", "prompt")]:
        with (root / filename).open(encoding="utf-8-sig", newline="") as stream:
            blocked.update(key(row[field]) for row in csv.DictReader(stream) if row.get(field))
    with (root / "benign_utility.jsonl").open(encoding="utf-8-sig") as stream:
        blocked.update(key(json.loads(line)["prompt"]) for line in stream if line.strip())
    return blocked


def choose(row):
    a, b = row["is_response_0_safe"], row["is_response_1_safe"]
    if a and not b:
        return 0, "one_safe"
    if b and not a:
        return 1, "one_safe"
    if a and b and row["better_response_id"] in (0, 1):
        return row["better_response_id"], "both_safe"
    return None, "no_safe_response"


def write(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arrow", type=Path, required=True)
    parser.add_argument("--eval-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--one-safe", type=int, default=150)
    parser.add_argument("--both-safe", type=int, default=100)
    parser.add_argument("--holdout", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if min(args.one_safe, args.both_safe, args.holdout) < 0 or not args.one_safe + args.both_safe:
        parser.error("Invalid sample counts")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error("Output directory must be empty")
    with args.arrow.open("rb") as stream:
        table = ipc.open_stream(stream).read_all()
    blocked = exclusions(args.eval_root)
    groups = {"one_safe": [], "both_safe": []}
    skipped = {}
    seen = set()
    for index, row in enumerate(table.to_pylist()):
        selected, group = choose(row)
        if selected is None:
            skipped[group] = skipped.get(group, 0) + 1
            continue
        prompt = row["prompt"].strip()
        response = row[f"response_{selected}"].strip()
        normalized = key(prompt)
        reason = "empty" if not normalized or not response else "eval_overlap" if normalized in blocked else "duplicate" if normalized in seen else None
        if reason:
            skipped[reason] = skipped.get(reason, 0) + 1
            continue
        seen.add(normalized)
        groups[group].append({"id": f"pku_alpaca7b_train_{index:05d}", "prompt": prompt, "response": response, "source": "PKU-Alignment/PKU-SafeRLHF:alpaca-7b:train", "selection": group, "reviewed": False})
    rng = random.Random(args.seed)
    for rows in groups.values():
        rng.shuffle(rows)
    if len(groups["one_safe"]) < args.one_safe or len(groups["both_safe"]) < args.both_safe:
        parser.error("Not enough eligible rows")
    train = groups["one_safe"][:args.one_safe] + groups["both_safe"][:args.both_safe]
    rest = groups["one_safe"][args.one_safe:] + groups["both_safe"][args.both_safe:]
    rng.shuffle(rest)
    if len(rest) < args.holdout:
        parser.error("Not enough holdout rows")
    rng.shuffle(train)
    write(args.output_dir / "safety_train.jsonl", train)
    write(args.output_dir / "safety_holdout.jsonl", rest[:args.holdout])
    report = {"source": str(args.arrow), "source_sha256": hashlib.file_digest(args.arrow.open("rb"), "sha256").hexdigest(), "seed": args.seed, "train": len(train), "holdout": args.holdout, "eligible": {k: len(v) for k, v in groups.items()}, "excluded": skipped, "note": "Safe response is not a prompt harm label; inspect examples and benign balance manually"}
    (args.output_dir / "selection_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
