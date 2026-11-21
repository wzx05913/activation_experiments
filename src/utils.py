"""Shared utilities for the activation approximation experiments."""
from __future__ import annotations

import csv
import json
import random
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def read_jsonl(path: str | Path, max_samples: int | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
            if max_samples is not None and len(rows) >= max_samples:
                break
    return rows


def _first_nonempty(row: dict[str, Any], keys: tuple[str, ...]) -> str:
    for key in keys:
        value = row.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    return ""


def _build_prompt(row: dict[str, Any], prompt: str) -> str:
    """Idempotent HarmBench direct request: context + separator + behavior.

    Matches upstream baselines/direct_request/direct_request.py. Only known
    direct-request forms are normalized; custom prompts are never discarded.
    """
    prompt = str(prompt or "").strip()
    context = str(row.get("ContextString", "") or "").strip()
    behavior = str(row.get("Behavior", "") or "").strip()
    if not context or not behavior:
        return prompt
    expected = f"{context}\n\n---\n\n{behavior}"
    legacy = f"Context:\n{context}\n\nTask:\n{behavior}"
    duplicate = f"Context:\n{context}\n\nTask:\n{legacy}"
    if prompt in {behavior, expected, legacy, duplicate}:
        return expected
    raise ValueError("Ambiguous HarmBench prompt: explicit prompt differs from Behavior; "
                     "provide the original Behavior or the canonical contextual prompt.")


def read_dataset(path: str | Path, max_samples: int | None = None) -> list[dict[str, Any]]:
    """Read JSONL or CSV datasets and normalize rows to records with a prompt."""
    if max_samples is not None and max_samples <= 0:
        raise ValueError("max_samples must be positive")
    source = Path(path)
    if source.suffix.lower() == ".jsonl":
        rows = read_jsonl(source, max_samples)
    elif source.suffix.lower() == ".json":
        payload = json.loads(source.read_text(encoding="utf-8-sig"))
        rows = payload if isinstance(payload, list) else [payload]
        if max_samples is not None:
            rows = rows[:max_samples]
    elif source.suffix.lower() == ".csv":
        rows = []
        with source.open("r", encoding="utf-8-sig", newline="") as f:
            for index, raw in enumerate(csv.DictReader(f)):
                row = dict(raw)
                prompt = _first_nonempty(row, ("prompt", "goal", "Behavior", "behavior", "question", "text"))
                if not prompt:
                    raise ValueError(f"No prompt-like column found in {source} row {index}")
                normalized: dict[str, Any] = {
                    "id": row.get("id") or row.get("BehaviorID") or f"{source.stem}_{index:05d}",
                    "prompt": prompt,
                    "source_file": str(source),
                }
                normalized.update(row)
                # The common cleanup path adds HarmBench context exactly once.
                normalized["prompt"] = prompt
                if "goal" in row:
                    normalized["category"] = "harmful_request"
                    normalized["source"] = "AdvBench"
                elif "Behavior" in row:
                    normalized["category"] = row.get("SemanticCategory") or "harmful_request"
                    normalized["source"] = "HarmBench"
                elif "label" in row:
                    normalized["category"] = "xstest_" + str(row.get("label", "unknown"))
                    normalized["source"] = "XSTest"
                rows.append(normalized)
                if max_samples is not None and len(rows) >= max_samples:
                    break
    else:
        raise ValueError(f"Unsupported dataset format: {source.suffix}. Use .jsonl or .csv")

    cleaned: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"Expected object at row {index} in {source}")
        prompt = _first_nonempty(row, ("prompt", "goal", "Behavior", "behavior", "question", "text"))
        if not prompt and isinstance(row.get("raw"), dict):
            content = row["raw"].get("content")
            if isinstance(content, list) and content:
                prompt = str(content[0]).strip()
        if not prompt:
            raise ValueError(f"Empty prompt at row {index} in {source}")
        row["prompt"] = _build_prompt(row, prompt)
        row["id"] = row.get("id") or row.get("BehaviorID") or f"{source.stem}_{index:05d}"
        row["prompt_format_version"] = "harmbench_direct_request_v2" if row.get("Behavior") else "plain_v1"
        row["context_included"] = bool(row.get("Behavior") and str(row.get("ContextString") or "").strip())
        cleaned.append(row)
    return cleaned


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def device_and_dtype(device_name: str = "auto", dtype_name: str = "auto") -> tuple[torch.device, torch.dtype]:
    """Resolve requested device/dtype; fail clearly instead of silently using CPU."""
    if device_name not in {"auto", "cuda", "cpu"}:
        raise ValueError("device must be auto, cuda, or cpu")
    use_cuda = torch.cuda.is_available() if device_name == "auto" else device_name == "cuda"
    if use_cuda and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is False. Check the active Python environment and CUDA PyTorch installation.")
    device = torch.device("cuda" if use_cuda else "cpu")
    if dtype_name == "auto":
        dtype = torch.float16 if device.type == "cuda" else torch.float32
    elif dtype_name == "float16":
        dtype = torch.float16
    elif dtype_name == "bfloat16":
        dtype = torch.bfloat16
    elif dtype_name == "float32":
        dtype = torch.float32
    else:
        raise ValueError("dtype must be auto, float16, bfloat16, or float32")
    return device, dtype


def device_metadata(device: torch.device, dtype: torch.dtype) -> dict[str, Any]:
    return {
        "device": str(device),
        "dtype": str(dtype).replace("torch.", ""),
        "cuda_available": bool(torch.cuda.is_available()),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }


def parse_layers(value: str | None) -> list[int]:
    if not value:
        return []
    return [int(item.strip()) for item in value.split(",") if item.strip()]
