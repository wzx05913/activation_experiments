"""Audit cross-model outputs without changing any original results."""
import argparse, csv, hashlib, json, statistics, sys
from collections import Counter
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.score_results import score, is_ref
from src.scoring_rules import RULE_VERSION
OUT = ROOT / f'results/06_cross_model/analysis_{RULE_VERSION}'

def load(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8-sig').splitlines() if line.strip()]

def save_csv(path, rows):
    if not rows: return
    with path.open('w', encoding='utf-8-sig', newline='') as f:
        w=csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

def main():
    global OUT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=OUT)
    args = parser.parse_args()
    OUT = args.output_dir
    OUT.mkdir(parents=True, exist_ok=False)
    summary=[]; records=[]; pairs=[]; integrity=[]; categories=[]; data={}
    for model in ['qwen25_3b','qwen25_7b']:
        for dataset, expected in [('harmbench_val',80),('benign',50)]:
            for arm, folder in [('baseline','01_baseline'),('hard_swish_all','06_cross_model')]:
                p=ROOT/'results'/folder/model/'raw_jsonl'/f'{arm}_{dataset}.jsonl'
                rows=load(p)
                assert len(rows)==expected, (p,len(rows))
                ids=[r['id'] for r in rows]; assert len(set(ids))==len(ids), p
                data[model,dataset,arm]={r['id']:r for r in rows}
                cp=p.parent.parent/'scored'/f'{arm}_{dataset}_scored.csv'
                saved={r['ID']:r for r in csv.DictReader(cp.open(encoding='utf-8-sig'))}
                assert set(saved)==set(ids), cp
                scores=[]; diffs=[]; content_diffs=[]
                for r in rows:
                    s,note=score(r,'harmful' if dataset=='harmbench_val' else 'benign'); s=int(s); scores.append(s)
                    old=saved[r['id']]
                    if s!=int(old['本次评分']): diffs.append(r['id'])
                    if old['Prompt']!=r['prompt'] or old['模型回答']!=r['response']: content_diffs.append(r['id'])
                    records.append(dict(model=model,dataset=dataset,arm=arm,id=r['id'],category=r.get('category',''),functional_category=r.get('FunctionalCategory',''),rule_version=RULE_VERSION,rule_score=s,rule_reason=note,review_status='rule_only',context_missing=bool(r.get('ContextString') and r['ContextString'].strip() not in r['prompt']),suspected_refusal=is_ref(r['response']),response_empty=not r['response'].strip(),response_words=len(r['response'].split()),prompt=r['prompt'],response=r['response']))
                lengths=[len(r['response'].split()) for r in rows]; lat=[r['latency_seconds'] for r in rows if isinstance(r.get('latency_seconds'),(int,float))]
                counts=Counter(scores)
                summary.append(dict(rule_version=RULE_VERSION,model=model,dataset=dataset,arm=arm,n=len(rows),score0=counts[0],score1=counts[1],score2=counts[2],score2_pct=100*counts[2]/len(rows),empty=sum(not r['response'].strip() for r in rows),mean_response_words=statistics.mean(lengths),mean_latency_seconds=statistics.mean(lat) if lat else None,median_latency_seconds=statistics.median(lat) if lat else None))
                meta={k:dict(Counter(json.dumps(r.get(k,'MISSING'),ensure_ascii=False) for r in rows)) for k in ['model','approximation','approximation_location','layers','hooked_modules','strength','seed','dtype','gpu_name','max_new_tokens','temperature','finish_reason']}
                integrity.append(dict(model=model,dataset=dataset,arm=arm,path=str(p.relative_to(ROOT)),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),n=len(rows),unique_ids=len(set(ids)),rule_version=RULE_VERSION,existing_score_differences=diffs,existing_text_differences=content_diffs,metadata=meta,context_missing_n=sum(bool(r.get('ContextString') and r['ContextString'].strip() not in r['prompt']) for r in rows),contextual_n=sum(r.get('FunctionalCategory')=='contextual' for r in rows),context_nonempty=sum(bool(r.get('ContextString','')) for r in rows)))
            base=data[model,dataset,'baseline']; approx=data[model,dataset,'hard_swish_all']; assert set(base)==set(approx)
            for id,b in base.items():
                h=approx[id]; assert b['prompt']==h['prompt'], id
                kind='harmful' if dataset=='harmbench_val' else 'benign'
                bs=int(score(b,kind)[0]); hs=int(score(h,kind)[0])
                pairs.append(dict(model=model,dataset=dataset,id=id,category=b.get('category',''),functional_category=b.get('FunctionalCategory',''),baseline_rule_score=bs,hard_swish_rule_score=hs,transition=f'{bs}->{hs}',prompt=b['prompt'],baseline_response=b['response'],hard_swish_response=h['response']))
            for cat in sorted({r.get('category','') for r in base.values()}):
                matched=[p for p in pairs if p['model']==model and p['dataset']==dataset and p['category']==cat]
                b2=sum(r['baseline_rule_score']==2 for r in matched); h2=sum(r['hard_swish_rule_score']==2 for r in matched)
                categories.append(dict(model=model,dataset=dataset,category=cat,n=len(matched),baseline_score2=b2,hard_swish_score2=h2,delta_pp=100*(h2-b2)/len(matched)))
    save_csv(OUT/'rule_summary.csv',summary); save_csv(OUT/'rule_scored_all.csv',records); save_csv(OUT/'paired_rule_scores.csv',pairs); save_csv(OUT/'rule_category_summary.csv',categories)
    matrices=[]
    for model in ['qwen25_3b','qwen25_7b']:
        for ds in ['harmbench_val','benign']:
            sub=[p for p in pairs if p['model']==model and p['dataset']==ds]
            for b in range(3):
                for h in range(3): matrices.append(dict(model=model,dataset=ds,baseline_score=b,hard_swish_score=h,count=sum(p['baseline_rule_score']==b and p['hard_swish_rule_score']==h for p in sub)))
    save_csv(OUT/'rule_transition_matrix.csv',matrices)
    (OUT/'data_integrity.json').write_text(json.dumps(integrity,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'summary':summary,'harmful_transitions':[m for m in matrices if m['dataset']=='harmbench_val' and m['count']], 'integrity':[{k:r[k] for k in ['model','dataset','arm','existing_score_differences','existing_text_differences','contextual_n','context_nonempty']} for r in integrity]}, ensure_ascii=False,indent=2))
if __name__=='__main__': main()
