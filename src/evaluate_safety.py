"""Lightweight safety screening metrics for generated JSONL outputs."""
from __future__ import annotations

import argparse
import re
from pathlib import Path

from .utils import read_jsonl, write_json

from .scoring_rules import RULE_VERSION, is_ref


def looks_like_refusal(text: str) -> bool:
    return is_ref(text)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    rows = read_jsonl(args.input)
    refusals = sum(looks_like_refusal(str(row.get("response", ""))) for row in rows)
    total = len(rows)
    metrics = {
        "input": str(Path(args.input).resolve()),
        "num_samples": total,
        "rule_version": RULE_VERSION,
        "empty_responses": sum(not str(row.get("response") or "").strip() for row in rows),
        "refusal_rate_rule_based": refusals / total if total else None,
        "screened_non_refusal_rate": (total - refusals) / total if total else None,
        "warning": "Rule-based screening is not a validated harmfulness classifier; manually review samples before reporting ASR.",
    }
    write_json(args.output, metrics)
    print(metrics)


if __name__ == "__main__":
    main()
