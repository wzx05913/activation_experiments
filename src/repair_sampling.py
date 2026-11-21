"""Validation and balanced ordering for reviewed mixed repair data."""
from __future__ import annotations

import random
from collections import Counter
from typing import Any


def validate_mixed_rows(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts = Counter()
    ids, prompts = set(), set()
    for index, row in enumerate(rows, 1):
        kind = row.get("kind")
        identifier = str(row.get("id", "")).strip()
        prompt = " ".join(str(row.get("prompt", "")).split()).casefold()
        if kind not in {"harmful", "benign"} or row.get("reviewed") is not True:
            raise ValueError(f"Mixed row {index} requires kind=harmful/benign and reviewed=true")
        if not identifier or not prompt or identifier in ids or prompt in prompts:
            raise ValueError(f"Mixed row {index} has missing/duplicate id or prompt")
        ids.add(identifier)
        prompts.add(prompt)
        counts[kind] += 1
    if not counts["harmful"] or not counts["benign"]:
        raise ValueError("Mixed mode requires both harmful and benign training rows")
    return dict(counts)


def epoch_order(rows: list[dict[str, Any]], mode: str, rng: random.Random) -> list[int]:
    if mode == "legacy":
        order = list(range(len(rows)))
        rng.shuffle(order)
        return order
    groups = {kind: [i for i, row in enumerate(rows) if row["kind"] == kind]
              for kind in ("harmful", "benign")}
    for group in groups.values():
        rng.shuffle(group)
    order = []
    # Cycle the smaller group so both kinds have equal exposure each epoch.
    for offset in range(max(map(len, groups.values()))):
        for kind in ("harmful", "benign"):
            order.append(groups[kind][offset % len(groups[kind])])
    return order
