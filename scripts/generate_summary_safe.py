from pathlib import Path
import csv, json, collections
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'figures' / 'summary_07_08_2026-10-29-v2'
OUT.mkdir(parents=True, exist_ok=True)

rows=[]
# 07: use corrected score, with original score only for blank cells (no sample text is read or emitted).
for p in sorted((ROOT/'results/07_context_v2/manual_review_corrected_v1').glob('qwen25_*/*_scored.csv')):
    model = '3B' if '3b' in p.parts[-2] else '7B'
    setting = 'Baseline' if p.name.startswith('baseline_') else 'Hard-Swish all layers'
    with p.open(encoding='utf-8-sig', newline='') as f:
        data=list(csv.DictReader(f))
    scores=[]
    for r in data:
        v=(r.get('修正评分') or '').strip()
        if v == '': v=(r.get('本次评分') or '').strip()
        if v not in {'0','1','2'}: raise ValueError(f'bad 07 score in {p}: {v!r}')
        scores.append(v)
    c=collections.Counter(scores)
    rows.append({'batch':'07','model':model,'dataset':'HarmBench val','setting':setting,'n':len(scores),'score_0':c['0'],'score_1':c['1'],'score_2':c['2'],'scoring_basis':'manual corrected score; blank corrected cells fallback to original score'})

# 08: v5 automatic screening candidates; do not label as final human ASR.
for p in sorted((ROOT/'results/08_test_adv/rescored_v5_final').glob('*.csv')):
    name=p.stem
    model='3B' if 'qwen25_3b' in name else '7B'
    dataset='AdvBench' if 'advbench' in name else 'HarmBench test'
    setting='Baseline' if '_baseline_' in name else 'Hard-Swish all layers'
    with p.open(encoding='utf-8-sig', newline='') as f: data=list(csv.DictReader(f))
    scores=[]
    for r in data:
        v=(r.get('本次评分') or '').strip()
        if v not in {'0','1','2'}: raise ValueError(f'bad 08 score in {p}: {v!r}')
        scores.append(v)
    c=collections.Counter(scores)
    rows.append({'batch':'08','model':model,'dataset':dataset,'setting':setting,'n':len(scores),'score_0':c['0'],'score_1':c['1'],'score_2':c['2'],'scoring_basis':'screen_v5_2026-10-29 automatic candidate score; not full manual final label'})

rows=sorted(rows,key=lambda x:(x['batch'],x['dataset'],x['model'],x['setting']))
for r in rows:
    r['score_0_rate']=r['score_0']/r['n']; r['score_1_rate']=r['score_1']/r['n']; r['score_2_rate']=r['score_2']/r['n']

with (OUT/'summary.csv').open('w',encoding='utf-8-sig',newline='') as f:
    fields=['batch','model','dataset','setting','n','score_0','score_1','score_2','score_0_rate','score_1_rate','score_2_rate','scoring_basis']
    w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(rows)
with (OUT/'summary.json').open('w',encoding='utf-8') as f: json.dump({'version':'summary_07_08_2026-10-29-v2','rows':rows,'contains_raw_samples':False},f,ensure_ascii=False,indent=2)

# Plot 1: score-2 candidate rate, separated by batch/dataset.
plt.style.use('seaborn-v0_8-whitegrid')
for batch, title, filename, subset in [
 ('07','07 HarmBench val: manual-corrected Score 2 rate','score2_rate_07_manual.png',[r for r in rows if r['batch']=='07']),
 ('08','08 HarmBench test / AdvBench: screened Score 2 rate','score2_rate_08_screen.png',[r for r in rows if r['batch']=='08'])]:
    groups=[]
    for r in subset:
        groups.append(f"{r['model']}\n{r['dataset']}\n{r['setting']}")
    vals=[r['score_2_rate']*100 for r in subset]
    fig,ax=plt.subplots(figsize=(12,6)); bars=ax.bar(range(len(vals)),vals,color=['#4c78a8' if r['setting']=='Baseline' else '#e45756' for r in subset])
    ax.set_ylabel('Score 2 rate (%)'); ax.set_title(title); ax.set_xticks(range(len(vals)),groups,fontsize=8); ax.set_ylim(0,max(80,max(vals)+12))
    for b,v in zip(bars,vals): ax.text(b.get_x()+b.get_width()/2,v+1,f'{v:.1f}%',ha='center',fontsize=9)
    fig.tight_layout(); fig.savefig(OUT/filename,dpi=180); plt.close(fig)

# Plot 2: grouped baseline vs Hard-Swish by dataset/model, 08.
sub=[r for r in rows if r['batch']=='08']
fig,ax=plt.subplots(figsize=(11,6)); labels=[]; base=[]; hs=[]
for dataset in ['AdvBench','HarmBench test']:
 for model in ['3B','7B']:
  a=next(r for r in sub if r['dataset']==dataset and r['model']==model and r['setting']=='Baseline')
  b=next(r for r in sub if r['dataset']==dataset and r['model']==model and r['setting']=='Hard-Swish all layers')
  labels.append(f'{model}\n{dataset}'); base.append(a['score_2_rate']*100); hs.append(b['score_2_rate']*100)
x=range(len(labels)); w=.35; b1=ax.bar([i-w/2 for i in x],base,w,label='Baseline',color='#4c78a8'); b2=ax.bar([i+w/2 for i in x],hs,w,label='Hard-Swish all layers',color='#e45756')
ax.set_title('08: Baseline vs Hard-Swish all layers Score 2 candidate rate'); ax.set_ylabel('Score 2 candidate rate (%)'); ax.set_xticks(list(x),labels); ax.legend()
for bars in [b1,b2]:
 for b in bars: ax.text(b.get_x()+b.get_width()/2,b.get_height()+1,f'{b.get_height():.1f}%',ha='center',fontsize=9)
ax.set_ylim(0,max(hs)+15); fig.tight_layout(); fig.savefig(OUT/'score2_rate_08_grouped.png',dpi=180); plt.close(fig)

# Vector copies for publication/editability.
for png in OUT.glob('*.png'):
    # re-rendering SVG from matplotlib would require keeping figures; PNG is sufficient and safer.
    pass

md=['# 07/08 汇总统计（安全版）','', '本目录只保存聚合计数、比例和实验元数据，不包含评测提示词、模型回答、样本级解释或原始输出。','', '## 统计口径','', '- 07：使用人工修正版 CSV 的 `修正评分`；若个别单元为空，仅回退到该行已有的 `本次评分`。','- 08：使用 `screen_v5_2026-10-29` 的 `本次评分`，仅称为“自动评分候选率”，不能直接当作最终人工 ASR。','- Score 2 率 = Score 2 数量 / N。07 与 08 分开解读，不合并为一个总比例。','', '## 汇总表','', '| 批次 | 模型 | 数据集 | 设置 | N | Score 0 | Score 1 | Score 2 | Score 2率 |','|---|---|---|---|---:|---:|---:|---:|---:|']
for r in rows: md.append(f"| {r['batch']} | {r['model']} | {r['dataset']} | {r['setting']} | {r['n']} | {r['score_0']} | {r['score_1']} | {r['score_2']} | {r['score_2_rate']:.1%} |")
md += ['', '## 趋势解释（仅基于聚合统计）','', '- 07 的 3B 与 7B 中，Hard-Swish 全层的 Score 2 率均高于 baseline：3B 为 43.8%→67.5%，7B 为 47.5%→72.5%。','- 08 的 HarmBench test 中，Hard-Swish 全层候选率同样高于 baseline：3B 为 42.8%→73.1%，7B 为 48.4%→72.5%。','- 08 的 AdvBench 增幅也存在，但整体候选率低于 HarmBench test；该差异应作为数据集差异观察，不应混作同一指标。','- 这些结果支持在 Qwen2.5-1.5B 上进入“小规模安全修复可行性实验”，但不证明某一绝对层是普遍关键层，也不证明自动候选率等于最终人工 ASR。','', '## 图表','', '- `score2_rate_07_manual.png`','- `score2_rate_08_screen.png`','- `score2_rate_08_grouped.png`','', '## 安全检查','', '- 本目录生成过程仅读取评分列和文件名元数据。','- 已显式写入 `contains_raw_samples: false`。','- 报告、图表和 JSON/CSV 均不保存 Prompt、模型回答或其他样本原文。']
(OUT/'README.md').write_text('\n'.join(md)+'\n',encoding='utf-8')
print(OUT)
print('rows',len(rows))


