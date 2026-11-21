"""Create a CSV summary from generated-output JSONL files."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .utils import read_jsonl


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    records = []
    for path in args.inputs:
        rows = read_jsonl(path)
        for row in rows:
            records.append({
                "source_file": str(Path(path)),
                "id": row.get("id"),
                "setting": row.get("setting"),
                "approximation": row.get("approximation"),
                "layers": ",".join(map(str, row.get("layers", []))),
                "strength": row.get("strength"),
                "latency_seconds": row.get("latency_seconds"),
                "response": row.get("response"),
            })
    pd.DataFrame(records).to_csv(args.output, index=False, encoding="utf-8-sig")
    print(f"Wrote {len(records)} rows to {Path(args.output).resolve()}")


if __name__ == "__main__":
    main()
