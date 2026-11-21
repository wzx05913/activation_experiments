"""Recover reviewed PKU safety preference pairs from the original train Arrow."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import pyarrow.ipc as ipc

from .prepare_pku_refusal import approved, ensure_unique, read_review
from .prepare_pku_sft import exclusions, key, write

ID_PATTERN = re.compile(r"^pku_alpaca7b_train_(\d+)$")


def recover(reviewed: list[dict], source: list[dict]) -> list[dict]:
    pairs = []
    for item in reviewed:
        match = ID_PATTERN.fullmatch(item["id"])
        if not match or int(match.group(1)) >= len(source):
            raise ValueError(f"Cannot locate original PKU row: {item['id']}")
        raw = source[int(match.group(1))]
        if key(raw["prompt"]) != key(item["prompt"]):
            raise ValueError(f"Prompt mismatch: {item['id']}")
        safe = [i for i in (0, 1) if raw[f"is_response_{i}_safe"] is True]
        if len(safe) != 1 or raw[f"response_{safe[0]}"].strip() != item["response"].strip():
            raise ValueError(f"Reviewed response/PKU safety label mismatch: {item['id']}")
        rejected = str(raw[f"response_{1 - safe[0]}"] or "").strip()
        if not rejected or rejected == item["response"].strip():
            raise ValueError(f"Missing distinct rejected response: {item['id']}")
        pairs.append({"id": item["id"], "prompt": item["prompt"],
                      "chosen": item["response"], "rejected": rejected,
                      "reviewed": True, "source": item["source"]})
    return pairs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arrow", type=Path, required=True)
    parser.add_argument("--review-dir", type=Path, required=True)
    parser.add_argument("--eval-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error("Output directory must be empty")
    if "train" not in args.arrow.name or "test" in args.arrow.name:
        parser.error("Only the PKU train Arrow is allowed")
    with args.arrow.open("rb") as stream:
        source = ipc.open_stream(stream).read_all().to_pylist()
    blocked = exclusions(args.eval_root)
    train = approved(read_review(args.review_dir / "train_review.csv"))
    holdout = approved(read_review(args.review_dir / "holdout_review.csv"))
    if not train or not holdout:
        parser.error("No reviewed train or holdout pairs")
    ensure_unique(train, holdout, blocked)
    train_pairs, holdout_pairs = recover(train, source), recover(holdout, source)
    write(args.output_dir / "dpo_train.jsonl", train_pairs)
    write(args.output_dir / "dpo_holdout.jsonl", holdout_pairs)
    with args.arrow.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    report = {"train_pairs": len(train_pairs), "holdout_pairs": len(holdout_pairs),
              "source_sha256": digest, "review": str(args.review_dir),
              "note": "Chosen was human-reviewed; rejected is PKU-labeled unsafe, not separately human-reviewed. Never evaluate on train pairs."}
    (args.output_dir / "selection_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
