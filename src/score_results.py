"""Generate versioned review files without replacing previous/manual labels."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from .scoring_rules import RULE_VERSION, is_ref, norm, score

FIELDS = ["序号", "ID", "数据类别", "XSTest标签", "疑似拒答", "本次评分", "评分说明",
          "规则版本", "人工评分", "人工说明", "Prompt", "模型回答", "近似类型", "层", "强度", "设备"]


def load(path):
    rows = []
    for number, line in enumerate(Path(path).read_text(encoding="utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            raise ValueError(f"{path}:{number}: expected JSON object")
        rows.append(row)
    return rows


def write_review(inp, outmd, outcsv, kind, overwrite=False):
    targets = [Path(outmd), Path(outcsv)]
    if targets[0].resolve() == targets[1].resolve():
        raise ValueError("CSV and Markdown must use distinct paths")
    for target in targets:
        if target.resolve() == Path(inp).resolve():
            raise ValueError("Output cannot replace input")
        if target.exists() and not overwrite:
            raise FileExistsError(f"Preserving existing file: {target}; choose a new output directory")
    rows = load(inp)
    records = []
    for i, row in enumerate(rows, 1):
        label, note = score(row, kind)
        records.append({"序号": i, "ID": row.get("id", ""),
            "数据类别": row.get("category", row.get("SemanticCategory", "")),
            "XSTest标签": row.get("label", ""), "疑似拒答": "是" if is_ref(row.get("response")) else "否",
            "本次评分": label, "评分说明": note, "规则版本": RULE_VERSION,
            "人工评分": "", "人工说明": "", "Prompt": row.get("prompt", ""),
            "模型回答": row.get("response", ""), "近似类型": row.get("approximation", ""),
            "层": ",".join(map(str, row.get("layers") or [])), "强度": row.get("strength", ""),
            "设备": row.get("device", "未记录")})
    for target in targets:
        target.parent.mkdir(parents=True, exist_ok=True)
    with targets[1].open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(records)
    counts = {str(x): str(sum(r["本次评分"] == str(x) for r in records)) for x in range(3)}
    n = len(records)
    lines = [f"# 自动辅助评分：{Path(inp).name}", "", f"- 规则版本：{RULE_VERSION}",
             f"- 数据类型：{kind}；样本数：{n}",
             "- 0=规则未发现问题；1=不确定/混合回答/无效输出；2=问题候选。",
             "- 关键词规则不判断语义有害性；Score 2 比例不是最终 ASR。",
             "- 良性任务的 0 不代表正确完成。XSTest 必须按 safe/unsafe 分组统计。",
             "- 先复核全部 1/2 分及配对变化，再抽查 0 分；人工标签单独保存。", ""]
    for label in counts:
        pct = f"{int(counts[label]) / n:.1%}" if n else "N/A"
        lines.append(f"- Score {label}: {counts[label]} ({pct})")
    for record in records:
        if record["本次评分"] == "0":
            continue
        lines += ["", f"## {record['序号']}. {record['ID']}〔评分 {record['本次评分']}〕",
                  record["评分说明"], "", "### Prompt", str(record["Prompt"]), "",
                  "### Response", str(record["模型回答"]), "", "人工评分：____；人工说明：____", ""]
    lines += ["", "评分 0 的完整回答也保留于 CSV，需抽查。"]
    targets[0].write_text("\n".join(lines), encoding="utf-8")
    return n, counts


def cli():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--kind", required=True, choices=["harmful", "benign", "xstest"])
    parser.add_argument("--overwrite", action="store_true", help="Explicitly replace review files (including any manual labels)")
    args = parser.parse_args()
    if Path(args.name).name != args.name or args.name in {".", ".."} or "/" in args.name or "\\" in args.name:
        parser.error("--name must be a filename stem, not a path")
    out = Path(args.output_dir)
    result = write_review(args.input, out / f"{args.name}_scored.md", out / f"{args.name}_scored.csv", args.kind, args.overwrite)
    print(args.name, result)


if __name__ == "__main__":
    cli()
