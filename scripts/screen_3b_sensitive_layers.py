"""Two-stage, paired 3B single-layer HarmBench-val sensitivity screen."""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

from src.utils import read_dataset, write_jsonl

MODEL = "/home/waas/models/Qwen2.5-3B-Instruct"
VAL = Path("data/harmbench/harmbench_behaviors_text_val.csv")
ROOT = Path("results/09_repair/3b_layer_screen_v1")


def run(*args: str) -> None:
    subprocess.run([sys.executable, "-m", *args], check=True)


def generate(name: str, data: Path, output: Path, layer: int | None) -> dict[str, str]:
    path = output / f"{name}.jsonl"
    review = output / f"{name}_review"
    if path.exists() or review.exists():
        raise FileExistsError(f"Preserving existing results: {path} or {review}")
    extra = ["--approximation", "hard_swish", "--layers", str(layer)] if layer is not None else []
    run("src.generate", "--model", MODEL, "--data", str(data), "--output", str(path),
        "--setting", f"3b_screen_{name}", "--max-input-tokens", "1024", "--max-new-tokens", "128",
        "--temperature", "0", "--top-p", "1", "--seed", "42", "--device", "cuda",
        "--dtype", "bfloat16", *extra)
    run("src.score_results", "--input", str(path), "--output-dir", str(review),
        "--name", "harmbench_val", "--kind", "harmful")
    with (review / "harmbench_val_scored.csv").open(encoding="utf-8-sig", newline="") as handle:
        return {row["ID"]: row["本次评分"] for row in csv.DictReader(handle)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("pilot", "full"), required=True)
    parser.add_argument("--layers", type=int, nargs="+", help="Full stage: 1-2 pilot-selected layers")
    args = parser.parse_args()
    if not Path(MODEL, "config.json").is_file() or not VAL.is_file():
        parser.error("Check model and HarmBench val paths on the cloud host")
    with Path(MODEL, "config.json").open(encoding="utf-8") as handle:
        depth = json.load(handle)["num_hidden_layers"]
    layers = (14, 22, 27, 32) if args.stage == "pilot" else args.layers
    if not layers or len(layers) > (4 if args.stage == "pilot" else 2):
        parser.error("Full stage needs 1-2 selected layers")
    if len(set(layers)) != len(layers) or any(layer < 0 or layer >= depth for layer in layers):
        parser.error(f"Invalid layers for model depth {depth}")
    rows = read_dataset(VAL)
    if args.stage == "pilot":
        if len(rows) != 80:
            parser.error(f"Expected 80 val rows, got {len(rows)}")
        # Evenly spaced, deterministic IDs; identical subset for every condition.
        rows = [rows[index] for index in range(0, 80, 3)][:24]
    output = ROOT / args.stage
    if output.exists():
        parser.error(f"Output exists; choose a new ROOT in this script: {output}")
    output.mkdir(parents=True)
    data = output / "inputs.jsonl"
    write_jsonl(data, rows)
    baseline = generate("original_silu", data, output, None)
    print(f"Baseline Score 2: {Counter(baseline.values())['2']}/{len(baseline)}", flush=True)
    for layer in layers:
        current = generate(f"layer_{layer}", data, output, layer)
        if set(current) != set(baseline):
            raise ValueError(f"Layer {layer}: IDs do not match baseline")
        transitions = Counter((baseline[key], current[key]) for key in baseline)
        print(f"Layer {layer}: Score 2 {Counter(current.values())['2']}/{len(current)}; "
              f"baseline 0->2 {transitions['0', '2']}, 2->0 {transitions['2', '0']}", flush=True)
    print("Automatic screening only; inspect paired responses before selecting a repair layer.")


if __name__ == "__main__":
    main()
