from pathlib import Path
p=Path('make_comparison.py')
s=p.read_text(encoding='utf-8')
s=s.replace('\nimport matplotlib\nmatplotlib.use("Agg")\nimport matplotlib.pyplot as plt\n','\nfrom xml.sax.saxutils import escape\n')
start=s.index('# -----------------------------------------------------------------------------\n# Charts generated directly')
end=s.index("print(json.dumps", start)
block=r'''# -----------------------------------------------------------------------------
# Charts generated directly from the scored result CSV files.
# SVG is used intentionally: it needs no extra plotting dependency, remains
# editable/vector-based, and can be opened directly in a browser or Word.
# -----------------------------------------------------------------------------
def svg_text(x, y, text, size=12, anchor='start', weight='normal'):
    return f'<text x="{x}" y="{y}" font-size="{size}px" text-anchor="{anchor}" font-family="Arial, Microsoft YaHei, sans-serif" font-weight="{weight}">{escape(str(text))}</text>'

def save_svg(path, title, rows, mode='counts'):
    W = max(1100, 230 * len(rows)); H = 680
    left, right, top, bottom = 85, 35, 80, 150
    plot_w, plot_h = W-left-right, H-top-bottom
    maxv = 100 if mode == 'percent' else max(10, max(int(r.get('样本数', 0)) for r in rows))
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">', '<rect width="100%" height="100%" fill="white"/>']
    parts.append(svg_text(W/2, 35, title, 20, 'middle', 'bold'))
    # grid and y labels
    for i in range(6):
        val = maxv * i / 5
        y = top + plot_h - plot_h * i / 5
        parts.append(f'<line x1="{left}" y1="{y:.1f}" x2="{W-right}" y2="{y:.1f}" stroke="#dddddd"/>')
        parts.append(svg_text(left-10, y+5, f'{val:.0f}' + ('%' if mode == 'percent' else ''), 11, 'end'))
    n=len(rows); group_w=plot_w/max(n,1); bar_w=min(42, group_w/5)
    colors=['#4daf4a','#ffb000','#e41a1c']; keys=['0分','1分','2分']; labels=['Score 0','Score 1','Score 2']
    for idx,r in enumerate(rows):
        cx=left+group_w*(idx+0.5)
        if mode == 'stacked':
            ybase=top+plot_h
            for key,color in zip(keys,colors):
                v=int(r[key])/max(int(r['样本数']),1)*100
                h=plot_h*v/100
                ybase-=h
                parts.append(f'<rect x="{cx-bar_w/2:.1f}" y="{ybase:.1f}" width="{bar_w:.1f}" height="{h:.1f}" fill="{color}"/>')
        else:
            for j,(key,color) in enumerate(zip(keys,colors)):
                v=(int(r[key])/max(int(r['样本数']),1)*100) if mode == 'percent' else int(r[key])
                h=plot_h*v/maxv
                x=cx+(j-1)*(bar_w+4)-bar_w/2
                y=top+plot_h-h
                parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w:.1f}" height="{h:.1f}" fill="{color}"/>')
        name=str(r['配置'])
        parts.append(svg_text(cx, top+plot_h+22, name, 11, 'end'))
        # rotate the labels for readability
        parts[-1]=parts[-1].replace(f'<text x="{cx}"', f'<text transform="rotate(-45 {cx} {top+plot_h+22})" x="{cx}"')
    parts.append(svg_text(left, H-35, 'Number of samples' if mode=='counts' else 'Percentage (%)', 12))
    legend_x=left; legend_y=top-25
    for color,label in zip(colors,labels):
        parts.append(f'<rect x="{legend_x}" y="{legend_y-12}" width="14" height="14" fill="{color}"/>')
        parts.append(svg_text(legend_x+20, legend_y, label, 12)); legend_x+=125
    parts.append('</svg>')
    Path(path).write_text(''.join(parts), encoding='utf-8')

def save_change_svg(path, title, rows):
    W=max(1100,230*len(rows)); H=680; left,right,top,bottom=85,35,80,150; plot_w,plot_h=W-left-right,H-top-bottom
    keys=[('相对baseline新增2分(0→2)','New 0->2','#d73027'),('相对baseline恶化总数','All worsened','#fc8d59'),('相对baseline改善总数','Improved','#4575b4')]
    maxv=max(10,max(int(r[k[0]]) for r in rows for k in keys)); group_w=plot_w/max(len(rows),1); bar_w=min(42,group_w/5)
    parts=[f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}"><rect width="100%" height="100%" fill="white"/>',svg_text(W/2,35,title,20,'middle','bold')]
    for i in range(6):
        val=maxv*i/5; y=top+plot_h-plot_h*i/5; parts.append(f'<line x1="{left}" y1="{y}" x2="{W-right}" y2="{y}" stroke="#ddd"/>'); parts.append(svg_text(left-10,y+5,f'{val:.0f}',11,'end'))
    for idx,r in enumerate(rows):
        cx=left+group_w*(idx+.5)
        for j,(key,label,color) in enumerate(keys):
            v=int(r[key]); h=plot_h*v/maxv; x=cx+(j-1)*(bar_w+4)-bar_w/2; y=top+plot_h-h; parts.append(f'<rect x="{x}" y="{y}" width="{bar_w}" height="{h}" fill="{color}"/>')
        parts.append(svg_text(cx,top+plot_h+22,r['配置'],11,'end').replace(f'<text x="{cx}"',f'<text transform="rotate(-45 {cx} {top+plot_h+22})" x="{cx}"'))
    lx=left
    for _,label,color in keys: parts.append(f'<rect x="{lx}" y="{top-37}" width="14" height="14" fill="{color}"/>'); parts.append(svg_text(lx+20,top-25,label,12)); lx+=180
    parts.append('</svg>'); Path(path).write_text(''.join(parts),encoding='utf-8')

save_svg(out/'layer27_quantize_score_counts.svg','Layer 27 quantization: score counts on HarmBench val',focus,'counts')
save_svg(out/'layer27_quantize_score_percentages.svg','Layer 27 quantization: score distribution (%)',focus,'percent')
save_change_svg(out/'layer27_quantize_baseline_changes.svg','Layer 27 quantization: paired changes versus baseline',focus)
save_svg(out/'all_approximation_score_counts.svg','All scored approximation configurations: score counts',summary,'counts')

'''
s=s[:start]+block+s[end:]
p.write_text(s,encoding='utf-8')
