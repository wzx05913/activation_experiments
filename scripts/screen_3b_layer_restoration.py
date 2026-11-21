"""Screen one-layer SiLU restoration against an existing all-Hard-Swish baseline."""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

from src.utils import read_dataset

MODEL = Path("/home/waas/models/Qwen2.5-3B-Instruct")
VAL = Path("data/harmbench/harmbench_behaviors_text_val.csv")
ROOT = Path("results/09_repair/3b_layer_restoration_v1")


def run(*args: str) -> None:
    subprocess.run([sys.executable, "-m", *args], check=True)


def scored(path: Path) -> dict[str, str]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 80 or len({row["ID"] for row in rows}) != 80:
        raise ValueError(f"Expected 80 distinct scored IDs: {path}")
    return {row["ID"]: row["本次评分"] for row in rows}


def validate_baseline(path: Path, rows: list[dict], depth: int) -> None:
    from src.score_results import load

    baseline = load(path)
    if len(baseline) != 80 or [(str(r["id"]), r["prompt"]) for r in baseline] != [
        (str(r["id"]), r["prompt"]) for r in rows
    ]:
        raise ValueError("Baseline IDs/prompts/order do not match the 80 val inputs")
    for row in baseline:
        if (row.get("approximation") != "hard_swish"
                or row.get("layers") != list(range(depth))
                or row.get("max_input_tokens") != 1024
                or row.get("max_new_tokens") != 128
                or row.get("temperature") != 0
                or row.get("top_p") != 1
                or row.get("seed") != 42):
            raise ValueError("Baseline generation configuration differs from restoration protocol")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layers", type=int, nargs="+", default=[14, 22],
                        help="Zero-based layers to restore to SiLU (default: 14 22)")
    args = parser.parse_args()
    baseline_path = ROOT / "all_hard_swish.jsonl"
    if not (MODEL / "config.json").is_file() or not VAL.is_file() or not baseline_path.is_file():
        parser.error("Check cloud model, val data, and existing all_hard_swish.jsonl")
    with (MODEL / "config.json").open(encoding="utf-8") as handle:
        depth = json.load(handle)["num_hidden_layers"]
    if not 1 <= len(args.layers) <= 4 or len(set(args.layers)) != len(args.layers):
        parser.error("Choose 1-4 distinct layers")
    if any(layer < 0 or layer >= depth for layer in args.layers):
        parser.error(f"Layer indices must be within [0, {depth - 1}]")
    rows = read_dataset(VAL)
    if len(rows) != 80:
        parser.error(f"Expected 80 val rows, got {len(rows)}")
    validate_baseline(baseline_path, rows, depth)
    baseline_review = ROOT / "all_hard_swish_review"
    if not baseline_review.exists():
        run("src.score_results", "--input", str(baseline_path),
            "--output-dir", str(baseline_review), "--name", "harmbench_val", "--kind", "harmful")
    baseline = scored(baseline_review / "harmbench_val_scored.csv")
    print(f"All Hard-Swish Score 2: {Counter(baseline.values())['2']}/80", flush=True)
    for restored in args.layers:
        name = f"restore_silu_{restored}"
        output = ROOT / f"{name}.jsonl"
        review = ROOT / f"{name}_review"
        if output.exists() or review.exists():
            parser.error(f"Preserving existing output: {output} or {review}")
        layers = [i for i in range(depth) if i != restored]
        run("src.generate", "--model", str(MODEL), "--data", str(VAL),
            "--output", str(output), "--setting", f"3b_{name}",
            "--approximation", "hard_swish", "--layers", ",".join(map(str, layers)),
            "--max-input-tokens", "1024", "--max-new-tokens", "128",
            "--temperature", "0", "--top-p", "1", "--seed", "42",
            "--device", "cuda", "--dtype", "bfloat16")
        run("src.score_results", "--input", str(output), "--output-dir", str(review),
            "--name", "harmbench_val", "--kind", "harmful")
        current = scored(review / "harmbench_val_scored.csv")
        if set(current) != set(baseline):
            raise ValueError(f"Restored layer {restored}: IDs do not match baseline")
        changes = Counter((baseline[key], current[key]) for key in baseline)
        print(f"Restore SiLU {restored}: Score 2 {Counter(current.values())['2']}/80; "
              f"all-Hard-Swish 2->0 {changes['2', '0']}, 0->2 {changes['0', '2']}", flush=True)
    print("Automatic screening only; inspect paired responses before selecting a repair layer.")


if __name__ == "__main__":
    main()
