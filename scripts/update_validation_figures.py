from __future__ import annotations

import json
from pathlib import Path
from xml.sax.saxutils import escape


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "figures" / "qwen25_1p5b_hardswish_validation_comparison"
OUT.mkdir(parents=True, exist_ok=True)

ROWS = [
    {"name": "Baseline", "counts": {"0": 272, "1": 5, "2": 43}, "n": 320, "source": "existing scored baseline"},
    {"name": "Hard-Swish all layers", "counts": {"0": 173, "1": 1, "2": 146}, "n": 320, "source": "manual corrections + automatic remainder"},
    {"name": "Hard-Swish layer 14", "counts": {"0": 266, "1": 5, "2": 49}, "n": 320, "source": "manual corrections + automatic remainder"},
    {"name": "Hard-Swish layer 27", "counts": {"0": 252, "1": 5, "2": 63}, "n": 320, "source": "manual corrections + automatic remainder"},
]


def text(x: float, y: float, value: str, size: int = 12, anchor: str = "start", weight: str = "normal") -> str:
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" font-family="Arial, sans-serif" '
        f'font-size="{size}px" text-anchor="{anchor}" font-weight="{weight}">{escape(str(value))}</text>'
    )


def save_stacked(path: Path, title: str, percentage: bool = True) -> None:
    width, height = 1180, 720
    left, right, top, bottom = 90, 40, 90, 170
    plot_w, plot_h = width - left - right, height - top - bottom
    max_value = 100 if percentage else 320
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        text(width / 2, 38, title, 20, "middle", "bold"),
    ]
    for i in range(6):
        value = max_value * i / 5
        y = top + plot_h - plot_h * i / 5
        parts.append(f'<line x1="{left}" y1="{y:.1f}" x2="{width-right}" y2="{y:.1f}" stroke="#dddddd"/>')
        parts.append(text(left - 10, y + 5, f"{value:.0f}{'%' if percentage else ''}", 11, "end"))

    colors = {"0": "#4daf4a", "1": "#ffb000", "2": "#e41a1c"}
    labels = {"0": "Score 0: safe/refusal", "1": "Score 1: uncertain", "2": "Score 2: unsafe"}
    group_w = plot_w / len(ROWS)
    bar_w = min(110, group_w * 0.48)
    for index, row in enumerate(ROWS):
        center = left + group_w * (index + 0.5)
        y_base = top + plot_h
        for score in ("0", "1", "2"):
            count = row["counts"][score]
            value = count / row["n"] * 100 if percentage else count
            bar_h = plot_h * value / max_value
            y_base -= bar_h
            parts.append(f'<rect x="{center-bar_w/2:.1f}" y="{y_base:.1f}" width="{bar_w:.1f}" height="{bar_h:.1f}" fill="{colors[score]}" stroke="white" stroke-width="1"/>')
            label = f"{count} ({count / row['n']:.1%})"
            if bar_h >= 30:
                parts.append(text(center, y_base + bar_h / 2 + 4, label, 12, "middle", "bold"))
            elif count:
                parts.append(text(center + bar_w / 2 + 6, y_base + bar_h / 2 + 4, label, 11))
        label_y = top + plot_h + 28
        label = text(center, label_y, row["name"], 11, "end")
        parts.append(label.replace(f'<text x="{center:.1f}"', f'<text transform="rotate(-42 {center:.1f} {label_y})" x="{center:.1f}"'))

    parts.append(text(left, height - 48, "Each bar totals 320 HarmBench test samples; labels show count (percentage)", 12))
    legend_x, legend_y = left, top - 25
    for score in ("0", "1", "2"):
        parts.append(f'<rect x="{legend_x}" y="{legend_y-12}" width="14" height="14" fill="{colors[score]}"/>')
        parts.append(text(legend_x + 21, legend_y, labels[score], 12))
        legend_x += 220
    parts.append("</svg>")
    path.write_text("".join(parts), encoding="utf-8")


def save_score2(path: Path) -> None:
    width, height = 1180, 620
    left, right, top, bottom = 90, 40, 80, 150
    plot_w, plot_h = width - left - right, height - top - bottom
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        text(width / 2, 38, "HarmBench test: unsafe candidate rate (Score 2)", 20, "middle", "bold"),
    ]
    for i in range(6):
        value = 20 * i
        y = top + plot_h - plot_h * i / 5
        parts.append(f'<line x1="{left}" y1="{y:.1f}" x2="{width-right}" y2="{y:.1f}" stroke="#dddddd"/>')
        parts.append(text(left - 10, y + 5, f"{value}%", 11, "end"))
    group_w = plot_w / len(ROWS)
    bar_w = min(110, group_w * 0.48)
    for index, row in enumerate(ROWS):
        center = left + group_w * (index + 0.5)
        pct = row["counts"]["2"] / row["n"] * 100
        bar_h = plot_h * pct / 100
        y = top + plot_h - bar_h
        parts.append(f'<rect x="{center-bar_w/2:.1f}" y="{y:.1f}" width="{bar_w:.1f}" height="{bar_h:.1f}" fill="#e41a1c"/>')
        parts.append(text(center, y - 8, f"{row['counts']['2']} ({pct:.1f}%)", 12, "middle", "bold"))
        label_y = top + plot_h + 28
        label = text(center, label_y, row["name"], 11, "end")
        parts.append(label.replace(f'<text x="{center:.1f}"', f'<text transform="rotate(-42 {center:.1f} {label_y})" x="{center:.1f}"'))
    parts.append(text(left, height - 45, "Lower is safer for HarmBench harmful-request evaluation", 12))
    parts.append("</svg>")
    path.write_text("".join(parts), encoding="utf-8")


save_stacked(OUT / "harmbench_test_score_percentages.svg", "HarmBench test: score distribution (%)")
save_stacked(OUT / "harmbench_test_score_counts.svg", "HarmBench test: score counts", percentage=False)
save_score2(OUT / "harmbench_test_unsafe_percentage.svg")
(OUT / "summary.json").write_text(json.dumps({"dataset": "HarmBench test", "rows": ROWS, "note": "Manual labels replace automatic scores only for reviewed candidates; remaining rows retain automatic scores."}, ensure_ascii=False, indent=2), encoding="utf-8")
(OUT / "README.md").write_text(
    "# Qwen2.5-1.5B Hard-Swish validation comparison\n\n"
    "These figures use the 320-sample HarmBench test set. Baseline uses the existing scored result. "
    "For Hard-Swish all/layer14/layer27, manually reviewed candidates replace automatic scores and unreviewed rows retain the automatic score. "
    "The figures are therefore corrected full-set estimates, not full manual relabels.\n",
    encoding="utf-8",
)
print(OUT)

