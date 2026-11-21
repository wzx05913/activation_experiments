"""Compare synced DPO checkpoints and matched baseline reviews."""
from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results/09_repair"
OUT = ROOT / "figures/repair_epoch3_epoch5_comparison"
MODELS = (("original_silu", "Original SiLU", "#477b73"), ("unrepaired_hard_swish", "Unrepaired Hard-Swish", "#a9a8a3"), ("repaired_epoch3", "Repair epoch 3", "#dc8b46"), ("repaired_epoch5", "Repair epoch 5", "#695b9e"))
DATASETS = ("harmbench_test", "advbench", "xstest", "benign")

def review_path(model, dataset):
    epoch = "epoch3" if model == "repaired_epoch3" else "epoch5"
    return RESULTS / f"independent_v6_plus7_{epoch}" / f"{model}_{dataset}_review" / f"{dataset}_scored.csv"

def read_review(path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    by_id = {row["ID"]: row for row in rows}
    if not rows or len(by_id) != len(rows):
        raise ValueError(f"Empty review or duplicate IDs: {path}")
    return by_id

def counts(rows, label=None):
    selected = [row for row in rows.values() if label is None or row["XSTest标签"] == label]
    values = Counter(row["本次评分"] for row in selected)
    return {"n": len(selected), **{f"score{score}": values[str(score)] for score in range(3)}}

def plot_groups(filename, title, groups, note):
    fig, axes = plt.subplots(1, len(groups), figsize=(5.3 * len(groups), 5.2), dpi=180, sharey=True)
    if len(groups) == 1:
        axes = [axes]
    for axis, (label, subset) in zip(axes, groups):
        for index, (model, name, color) in enumerate(MODELS):
            values = subset.get(model)
            if values is None:
                axis.text(index, 5, "N/A", ha="center", va="bottom", fontsize=10, color="#666666")
                continue
            pct = values["score2"] / values["n"] * 100
            axis.bar(index, pct, width=0.68, color=color)
            axis.text(index, pct + 2, f'{pct:.1f}%\n({values["score2"]}/{values["n"]})', ha="center", va="bottom", fontsize=9)
        axis.set_title(label, fontsize=12)
        axis.set_xticks(range(len(MODELS)), [name.replace(" ", "\n", 1) for _, name, _ in MODELS], fontsize=8)
        axis.set_ylim(0, 104)
        axis.grid(axis="y", alpha=0.16)
        axis.set_axisbelow(True)
    axes[0].set_ylabel("Score 2 candidates (%)")
    fig.suptitle(title, fontsize=14, y=0.98)
    fig.text(0.01, 0.015, note, fontsize=8, color="#444444")
    fig.tight_layout(rect=(0, 0.065, 1, 0.94), w_pad=2.1)
    fig.savefig(OUT / filename, bbox_inches="tight")
    plt.close(fig)

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    reviews, summary = {}, []
    for dataset in DATASETS:
        baseline = None
        for model, _, _ in MODELS:
            path = review_path(model, dataset)
            if not path.exists():
                raise FileNotFoundError(path)
            rows = read_review(path)
            if baseline is None:
                baseline = rows
            elif set(rows) != set(baseline) or any(rows[key]["Prompt"] != baseline[key]["Prompt"] for key in rows):
                raise ValueError(f"Unmatched ID or prompt: {path}")
            reviews[model, dataset] = rows
            labels = ("safe", "unsafe") if dataset == "xstest" else (None,)
            for label in labels:
                values = counts(rows, label)
                summary.append([dataset, label or "all", model, values["n"], values["score0"], values["score1"], values["score2"], "synced review CSV", str(path.relative_to(ROOT))])
    with (OUT / "summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["dataset", "split", "model", "n", "score0", "score1", "score2", "provenance", "source"])
        writer.writerows(summary)
    with (OUT / "paired_transitions.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["dataset", "split", "comparison", "before", "after", "count"])
        for dataset in DATASETS:
            after = reviews["repaired_epoch3", dataset]
            for model in ("original_silu", "unrepaired_hard_swish", "repaired_epoch5"):
                before = reviews[model, dataset]
                labels = ("safe", "unsafe") if dataset == "xstest" else (None,)
                for label in labels:
                    transitions = Counter((before[key]["本次评分"], row["本次评分"]) for key, row in after.items() if label is None or row["XSTest标签"] == label)
                    for (old, new), count in sorted(transitions.items()):
                        writer.writerow([dataset, label or "all", f"{model}->repaired_epoch3", old, new, count])
    def group(dataset, label=None):
        return {model: counts(reviews[model, dataset], label) for model, _, _ in MODELS if (model, dataset) in reviews}
    plot_groups("safety_utility_tradeoff.png", "DPO repair: safety and over-refusal", [("XSTest unsafe: non-refusal", group("xstest", "unsafe")), ("XSTest safe: over-refusal", group("xstest", "safe")), ("Benign: refusal", group("benign"))], "Lower is better in each panel. Rule-based screening, not human-verified outcome.")
    plot_groups("independent_harmful_benchmarks.png", "Independent harmful benchmark screening", [("HarmBench test (n=320)", group("harmbench_test")), ("AdvBench (n=520)", group("advbench"))], "Both checkpoints are synced. Score 2 is a rule-based candidate, not verified ASR.")


def audited_xstest(model):
    paths = {
        "original_silu": RESULTS / "independent_v6_plus7_epoch5/original_silu_xstest_review/xstest_scored_audited.csv",
        "repaired_epoch3": RESULTS / "independent_v6_plus7_epoch3/repaired_epoch3_xstest_review/xstest_scored_audited.csv",
        "repaired_epoch5": RESULTS / "independent_v6_plus7_epoch5/repaired_epoch5_xstest_review/xstest_scored_audited_v2.csv",
    }
    if model not in paths:
        return None
    rows = read_review(paths[model])
    def classify(row):
        value = row["复核_拒答判定"] if model == "original_silu" else row["审核-拒答性质"]
        if value in {"真拒答", "真实拒答", "实际为拒答"}:
            return "refusal"
        if value in {"未拒答", "非拒答，正常作答"} or value.startswith("非真拒答：直接作答"):
            return "non_refusal"
        if value == "部分拒答" or value.startswith(("非真拒答：", "非拒答：")):
            return "boundary"
        raise ValueError(f"Unknown audited XSTest label: {value}")
    return {key: (row, classify(row)) for key, row in rows.items()}


def plot_current(filename, title, panels, note):
    fig, axes = plt.subplots(1, len(panels), figsize=(5.1 * len(panels), 4.8), dpi=180)
    if len(panels) == 1:
        axes = [axes]
    for ax, (label, items, denominator) in zip(axes, panels):
        for index, (name, _, color) in enumerate(MODELS):
            if name not in items:
                ax.text(index, 4, "N/A", ha="center")
                continue
            count = items[name]
            pct = count / denominator * 100
            ax.bar(index, pct, width=0.65, color=color)
            ax.text(index, pct + 2, f"{count}/{denominator}\n({pct:.1f}%)", ha="center", fontsize=8)
        ax.set_title(label, fontsize=11)
        ax.set_xticks(range(len(MODELS)), [name.replace("_", "\n") for name, _, _ in MODELS], fontsize=8)
        ax.set_ylim(0, 112)
        ax.grid(axis="y", alpha=0.15)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("Percentage of prompts")
    fig.suptitle(title, fontsize=13)
    fig.text(0.01, 0.01, note, fontsize=8)
    fig.tight_layout(rect=(0, 0.08, 1, 0.94))
    fig.savefig(OUT / filename, bbox_inches="tight")
    plt.close(fig)


def write_current_comparison():
    reviews = {(model, dataset): read_review(review_path(model, dataset))
               for dataset in DATASETS for model, _, _ in MODELS}
    audits = {model: audited_xstest(model) for model, _, _ in MODELS}
    summary, transitions = [], []
    for dataset in DATASETS:
        reference = reviews["original_silu", dataset]
        for model, _, _ in MODELS:
            rows = reviews[model, dataset]
            if set(rows) != set(reference) or any(rows[key]["Prompt"] != reference[key]["Prompt"] for key in rows):
                raise ValueError(f"Unmatched ID or prompt: {model} {dataset}")
            if dataset == "xstest" and audits[model] is not None:
                audit = audits[model]
                if set(audit) != set(rows) or any(audit[key][0]["Prompt"] != rows[key]["Prompt"] or
                                                 audit[key][0]["XSTest标签"] != rows[key]["XSTest标签"] for key in rows):
                    raise ValueError(f"Unmatched XSTest audit: {model}")
            for split in (("safe", "unsafe") if dataset == "xstest" else ("all",)):
                selected = {key: row for key, row in rows.items() if split == "all" or row["XSTest标签"] == split}
                if dataset == "xstest" and audits[model] is not None:
                    scores = Counter(audits[model][key][1] for key in selected)
                    source = "audited refusal category; boundary separate"
                else:
                    manual = [row["人工评分"] for row in selected.values()]
                    if dataset in ("advbench", "benign") and all(value in {"0", "1", "2"} for value in manual):
                        scores = Counter(manual)
                        source = "manual score (0/1/2); see reviewer rubric"
                    else:
                        scores = Counter(row["本次评分"] for row in selected.values())
                        source = "automatic screen_v5 Score 0/1/2; not verified ASR"
                summary.append([dataset, split, model, source, len(selected),
                                scores["0"], scores["1"], scores["2"],
                                scores["refusal"], scores["non_refusal"], scores["boundary"]])
    with (OUT / "current_review_summary.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["dataset", "split", "model", "source", "n", "score0", "score1", "score2",
                         "refusal", "non_refusal", "boundary"])
        writer.writerows(summary)
    for dataset in DATASETS:
        for before, after in (("original_silu", "repaired_epoch3"), ("unrepaired_hard_swish", "repaired_epoch3"),
                              ("repaired_epoch3", "repaired_epoch5")):
            for split in (("safe", "unsafe") if dataset == "xstest" else ("all",)):
                before_rows, after_rows = reviews[before, dataset], reviews[after, dataset]
                if dataset == "xstest" and (audits[before] is None or audits[after] is None):
                    continue  # No comparable manual audit for the unrepaired model.
                def value(model, key):
                    if dataset == "xstest":
                        return audits[model][key][1]
                    row = reviews[model, dataset][key]
                    return row["人工评分"] if dataset in ("advbench", "benign") else row["本次评分"]
                counts = Counter((value(before, key), value(after, key)) for key, row in before_rows.items()
                                 if split == "all" or row["XSTest标签"] == split)
                for (old, new), count in sorted(counts.items()):
                    transitions.append([dataset, split, f"{before}->{after}", old, new, count,
                                        "audited refusal category" if dataset == "xstest" else
                                        "manual score" if dataset in ("advbench", "benign") else "automatic screen_v5"])
    with (OUT / "current_review_transitions.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["dataset", "split", "comparison", "before", "after", "count", "source"])
        writer.writerows(transitions)
    def selected(dataset, split, field):
        return {row[2]: int(row[field]) for row in summary
                if row[0] == dataset and row[1] == split and row[field] != ""
                and (dataset != "xstest" or row[3].startswith("audited"))}
    plot_current("current_review_safety_utility.png", "Safety and benign behavior: available reviews", [
        ("XSTest unsafe: non-refusal (audited)", selected("xstest", "unsafe", 9), 200),
        ("XSTest safe: refusal (audited)", selected("xstest", "safe", 8), 250),
        ("Benign: manual score 2", selected("benign", "all", 7), 50),
    ], "XSTest boundary cases excluded from refusal/non-refusal; benign scores follow reviewer labels.")
    plot_current("current_review_harmful_benchmarks.png", "Harmful benchmarks: evidence by source", [
        ("HarmBench test: automatic Score 2", selected("harmbench_test", "all", 7), 320),
        ("AdvBench: manual score 2", selected("advbench", "all", 7), 520),
    ], "Different scoring sources; automatic HarmBench candidates are not verified ASR.")

if __name__ == "__main__":
    main()
    write_current_comparison()
