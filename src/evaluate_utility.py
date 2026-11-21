"""Basic utility statistics for benign generated responses."""
from __future__ import annotations

import argparse
import numpy as np

from .utils import read_jsonl, write_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    rows = read_jsonl(args.input)
    latencies = [float(row["latency_seconds"]) for row in rows if "latency_seconds" in row]
    lengths = [len(str(row.get("response", "")).split()) for row in rows]
    metrics = {
        "input": args.input,
        "num_samples": len(rows),
        "mean_latency_seconds": float(np.mean(latencies)) if latencies else None,
        "p50_latency_seconds": float(np.median(latencies)) if latencies else None,
        "mean_response_words": float(np.mean(lengths)) if lengths else None,
    }
    write_json(args.output, metrics)
    print(metrics)


if __name__ == "__main__":
    main()
