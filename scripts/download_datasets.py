from __future__ import annotations
import json
import os
from pathlib import Path
from typing import Any
from datasets import load_dataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)
ADVBENCH_ID = "AlignmentResearch/AdvBench"
XSTEST_ID = "purpcode/XSTest"

def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")

def message_prompt(messages: Any) -> str:
    if not isinstance(messages, list): return ""
    for msg in reversed(messages):
        if isinstance(msg, dict) and msg.get("role") in {"user", "human"}:
            return str(msg.get("content", ""))
    return ""

def first_text(raw: dict[str, Any], keys: tuple[str, ...]) -> str:
    for key in keys:
        value = raw.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
        if isinstance(value, list) and value:
            return str(value[0]).strip()
    return message_prompt(raw.get("messages"))

def download_advbench() -> None:
    print(f"[1/2] 正在加载 {ADVBENCH_ID} ...")
    dataset = load_dataset(ADVBENCH_ID, "default", split="train")
    print(f"AdvBench split: train（{len(dataset)} 条）")
    rows = []
    for index, raw in enumerate(dataset):
        prompt = first_text(raw, ("goal", "prompt", "instruction", "instructions", "text", "question"))
        rows.append({"id": f"advbench_{index:05d}", "prompt": prompt, "category": "harmful_request", "source": ADVBENCH_ID, "source_split": "train", "raw": dict(raw)})
    output = DATA_DIR / "advbench_raw.jsonl"
    write_jsonl(output, rows)
    print(f"AdvBench 已保存：{output}（{len(rows)} 条）")

def download_xstest() -> None:
    print(f"[2/2] 正在加载 {XSTEST_ID} ...")
    dataset = load_dataset(XSTEST_ID, split="test")
    print(f"XSTest split: test（{len(dataset)} 条）")
    rows = []
    for index, raw in enumerate(dataset):
        rows.append({"id": str(raw.get("task_id") or f"xstest_{index:05d}"), "prompt": first_text(raw, ("prompt", "question", "text", "instruction")), "category": raw.get("focus", "unknown"), "type": raw.get("type", "unknown"), "note": raw.get("note", ""), "label": raw.get("label", ""), "source": XSTEST_ID, "source_split": "test", "raw": dict(raw)})
    output = DATA_DIR / "xstest_raw.jsonl"
    write_jsonl(output, rows)
    print(f"XSTest 已保存：{output}（{len(rows)} 条）")

def main() -> None:
    print(f"项目目录：{PROJECT_ROOT}")
    print(f"HF_TOKEN 已设置：{'是' if os.environ.get('HF_TOKEN') else '否（公开数据集可先匿名尝试）'}")
    errors = []
    for name, func in (("AdvBench", download_advbench), ("XSTest", download_xstest)):
        try: func()
        except Exception as exc:
            errors.append(f"{name}: {type(exc).__name__}: {exc}")
            print(f"{name} 下载失败：{exc}")
    if errors: raise SystemExit("\n".join(errors))
    print("两个数据集均已下载并转换完成。")

if __name__ == "__main__": main()
