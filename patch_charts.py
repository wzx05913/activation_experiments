from pathlib import Path
p=Path('make_comparison.py')
s=p.read_text(encoding='utf-8')
s=s.replace('import csv, re, json\nfrom pathlib import Path\nfrom collections import Counter\n', 'import csv, re, json\nfrom pathlib import Path\nfrom collections import Counter\n\nimport matplotlib\nmatplotlib.use("Agg")\nimport matplotlib.pyplot as plt\n')
needle="(out/'layer27_quantize_baseline_comparison.md').write_text(md2,encoding='utf-8')\n\n"
insert='''(out/'layer27_quantize_baseline_comparison.md').write_text(md2,encoding='utf-8')

# -----------------------------------------------------------------------------
# Charts generated directly from the scored result CSV files.
# -----------------------------------------------------------------------------
def save_score_count_chart(rows, path, title):
    names = [r['配置'] for r in rows]
    x = list(range(len(names)))
    width = 0.24
    fig, ax = plt.subplots(figsize=(max(10, len(names) * 1.15), 6.5))
    for i, (key, label, color) in enumerate([
        ('0分', 'Score 0 (safe/refusal)', '#4daf4a'),
        ('1分', 'Score 1 (uncertain)', '#ffb000'),
        ('2分', 'Score 2 (unsafe)', '#e41a1c'),
    ]):
        vals = [int(r[key]) for r in rows]
        ax.bar([v + (i - 1) * width for v in x], vals, width, label=label, color=color)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha='right')
    ax.set_ylabel('Number of samples')
    ax.set_title(title)
    ax.grid(axis='y', alpha=.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches='tight')
    plt.close(fig)

def save_percent_stacked_chart(rows, path, title):
    names = [r['配置'] for r in rows]
    fig, ax = plt.subplots(figsize=(max(10, len(names) * 1.15), 6.5))
    bottom = [0.0] * len(rows)
    for key, label, color in [
        ('0分', 'Score 0', '#4daf4a'),
        ('1分', 'Score 1', '#ffb000'),
        ('2分', 'Score 2', '#e41a1c'),
    ]:
        vals = [int(r[key]) / max(int(r['样本数']), 1) * 100 for r in rows]
        ax.bar(names, vals, bottom=bottom, label=label, color=color)
        bottom = [a + b for a, b in zip(bottom, vals)]
    ax.set_ylabel('Percentage (%)')
    ax.set_ylim(0, 100)
    ax.set_title(title)
    ax.tick_params(axis='x', labelrotation=45)
    for label in ax.get_xticklabels(): label.set_ha('right')
    ax.grid(axis='y', alpha=.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches='tight')
    plt.close(fig)

def save_change_chart(rows, path, title):
    names = [r['配置'] for r in rows]
    x = list(range(len(names)))
    width = 0.25
    series = [
        ('相对baseline新增2分(0→2)', 'New 0->2 degradation', '#d73027'),
        ('相对baseline恶化总数', 'All worsened', '#fc8d59'),
        ('相对baseline改善总数', 'Improved', '#4575b4'),
    ]
    fig, ax = plt.subplots(figsize=(max(10, len(names) * 1.15), 6.5))
    for i, (key, label, color) in enumerate(series):
        vals = [int(r[key]) for r in rows]
        ax.bar([v + (i - 1) * width for v in x], vals, width, label=label, color=color)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha='right')
    ax.set_ylabel('Number of paired samples')
    ax.set_title(title)
    ax.grid(axis='y', alpha=.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches='tight')
    plt.close(fig)

save_score_count_chart(focus, out / 'layer27_quantize_score_counts.png', 'Layer 27 quantization: score counts on HarmBench val')
save_percent_stacked_chart(focus, out / 'layer27_quantize_score_percentages.png', 'Layer 27 quantization: score distribution (%)')
save_change_chart(focus, out / 'layer27_quantize_baseline_changes.png', 'Layer 27 quantization: paired changes versus baseline')
save_score_count_chart(summary, out / 'all_approximation_score_counts.png', 'All scored approximation configurations: score counts')

'''
if needle not in s: raise SystemExit('needle not found')
s=s.replace(needle,insert)
s=s.replace('\n# chart patch marker\n','\n')
p.write_text(s,encoding='utf-8')
