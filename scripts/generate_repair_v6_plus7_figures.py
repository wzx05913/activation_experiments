"""Summarize the 74-pair DPO repair on HarmBench validation."""
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "figures/repair_v6_plus7_2026-11-05"
OUT.mkdir(parents=True, exist_ok=True)


def counts(path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return [sum(row["本次评分"] == str(score) for row in rows) for score in range(3)]


def main():
    base = ROOT / "results/09_repair/harmbench_v6_lr1e4"
    data = [
        ("Original SiLU", counts(base / "original_silu_review/harmbench_val_scored.csv"), "local CSV"),
        ("Unrepaired Hard-Swish", counts(base / "unrepaired_hard_swish_review/harmbench_val_scored.csv"), "local CSV"),
        ("67-pair epoch 5", counts(base / "repaired_epoch5_review/harmbench_val_scored.csv"), "local CSV"),
        ("74-pair epoch 1", [59, 0, 21], "cloud summary"),
        ("74-pair epoch 3", [67, 0, 13], "cloud summary"),
        ("74-pair epoch 5", [74, 0, 6], "cloud summary and local CSV"),
    ]
    verified = counts(ROOT / "results/09_repair/pku_dpo_layer27_v6_plus7_lr1e4/epoch_5_harmbench_review/harmbench_val_scored.csv")
    if verified != data[-1][1]:
        raise ValueError("Epoch-5 cloud summary disagrees with synced CSV")
    with (OUT / "summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["setting", "n", "score0", "score1", "score2", "provenance"])
        for name, values, provenance in data:
            writer.writerow([name, sum(values), *values, provenance])

    fig, ax = plt.subplots(figsize=(12, 6.5), dpi=160)
    for index, (name, values, _) in enumerate(data):
        bottom = 0
        for score, color in enumerate(["#4daf4a", "#e9a23b", "#d73027"]):
            ax.bar(index, values[score] / 80 * 100, bottom=bottom, color=color,
                   label=["Score 0", "Score 1", "Score 2"][score] if index == 0 else None)
            if values[score]:
                ax.text(index, bottom + values[score] / 160 * 100, str(values[score]),
                        ha="center", va="center", color="white", weight="bold")
            bottom += values[score] / 80 * 100
    ax.set_xticks(range(len(data)), [row[0] for row in data], rotation=25, ha="right")
    ax.set_ylim(0, 100)
    ax.set_ylabel("Samples (%)")
    ax.set_title("Qwen2.5-1.5B: HarmBench val automated screening (n=80)")
    ax.legend(loc="lower right", frameon=True, ncol=3)
    ax.grid(axis="y", alpha=0.2)
    fig.text(0.02, 0.01, "Score 2 is a candidate rate, not human-verified ASR. Epochs 1/3: reported cloud totals.", fontsize=9)
    fig.tight_layout(rect=[0, 0.04, 1, 1])
    fig.savefig(OUT / "harmbench_val_comparison.png", bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5.5), dpi=160)
    for score, color in [(0, "#4daf4a"), (2, "#d73027")]:
        values = [row[1][score] / 80 * 100 for row in data[3:]]
        ax.plot([1, 3, 5], values, marker="o", color=color, label=f"Score {score}")
        for epoch, value in zip([1, 3, 5], values):
            ax.annotate(f"{value:.1f}%", (epoch, value), xytext=(0, 8),
                        textcoords="offset points", ha="center")
    ax.axhline(data[0][1][0] / 80 * 100, linestyle="--", color="#4daf4a", alpha=0.6, label="SiLU Score 0")
    ax.set(xticks=[1, 3, 5], xlabel="Epoch", ylabel="Samples (%)", ylim=(0, 100),
           title="74-pair repair: HarmBench val epoch trend")
    ax.grid(alpha=0.2)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(OUT / "epoch_trend.png", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
