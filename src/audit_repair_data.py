"""Audit safety repair JSONL data without printing sample text."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def load(path: Path):
    rows = []
    with path.open(encoding="utf-8-sig") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_no}: expected JSON object")
            rows.append(row)
    return rows

def fingerprint(row: dict) -> str:
    value = str(row.get("prompt", "")).strip()
    return hashlib.sha256(value.encode("utf-8")).hexdigest()

def audit(path: Path):
    rows = load(path)
    ids = [str(r.get("id", "")).strip() for r in rows if str(r.get("id", "")).strip()]
    missing_id = sum(not str(r.get("id", "")).strip() for r in rows)
    duplicate_ids = len(ids) - len(set(ids))
    empty_prompt = sum(not str(r.get("prompt", "")).strip() for r in rows)
    empty_response = sum(not str(r.get("response", "")).strip() for r in rows)
    placeholder = sum("<replace_with_" in (str(r.get("prompt", "")) + str(r.get("response", ""))).lower() for r in rows)
    fps = [fingerprint(r) for r in rows if str(r.get("prompt", "")).strip()]
    duplicate_prompts = len(fps) - len(set(fps))
    return {"path": str(path.resolve()), "rows": len(rows), "missing_id": missing_id, "duplicate_ids": duplicate_ids, "empty_prompt": empty_prompt, "empty_response": empty_response, "placeholder_rows": placeholder, "duplicate_prompts": duplicate_prompts}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--test")
    args = ap.parse_args()
    train = Path(args.train)
    report = {"train": audit(train)}
    if args.test:
        test = Path(args.test)
        report["test"] = audit(test)
        train_fp = {fingerprint(r) for r in load(train) if str(r.get("prompt", "")).strip()}
        test_fp = {fingerprint(r) for r in load(test) if str(r.get("prompt", "")).strip()}
        report["prompt_overlap_train_test"] = len(train_fp & test_fp)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    bad = any(v for section in report.values() if isinstance(section, dict) for k, v in section.items() if k not in {"path", "rows"} and isinstance(v, int) and v > 0)
    raise SystemExit(2 if bad else 0)

if __name__ == "__main__":
    main()
