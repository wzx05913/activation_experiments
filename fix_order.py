from pathlib import Path
p=Path('make_comparison.py')
s=p.read_text(encoding='utf-8')
old="focus=[r for r in summary if r['配置'] in focus_names]"
new="focus=[next(r for r in summary if r['配置']==name) for name in focus_names]"
if old not in s: raise SystemExit('not found')
p.write_text(s.replace(old,new),encoding='utf-8')
