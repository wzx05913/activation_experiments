from pathlib import Path
p=Path('make_comparison.py')
s=p.read_text(encoding='utf-8')
start=s.index('def save_svg(')
end=s.index('def save_change_svg(', start)
new=r'''def save_svg(path, title, rows, mode='counts'):
    """One stacked bar per experiment; each bar totals the sample count."""
    W = max(1100, 230 * len(rows)); H = 720
    left, right, top, bottom = 85, 35, 80, 170
    plot_w, plot_h = W-left-right, H-top-bottom
    # All current HarmBench configurations contain 80 samples, but calculate
    # this dynamically so the chart remains correct for future datasets.
    maxv = max(10, max(int(r.get('样本数', 0)) for r in rows))
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">', '<rect width="100%" height="100%" fill="white"/>']
    parts.append(svg_text(W/2, 35, title, 20, 'middle', 'bold'))
    for i in range(6):
        val = maxv * i / 5
        y = top + plot_h - plot_h * i / 5
        parts.append(f'<line x1="{left}" y1="{y:.1f}" x2="{W-right}" y2="{y:.1f}" stroke="#dddddd"/>')
        parts.append(svg_text(left-10, y+5, f'{val:.0f}', 11, 'end'))

    n=len(rows); group_w=plot_w/max(n,1); bar_w=min(70, group_w*0.42)
    colors=['#4daf4a','#ffb000','#e41a1c']; keys=['0分','1分','2分']; labels=['Score 0: safe/refusal','Score 1: uncertain','Score 2: unsafe']
    for idx,r in enumerate(rows):
        cx=left+group_w*(idx+0.5)
        ybase=top+plot_h
        total=max(int(r.get('样本数', 0)),1)
        for key,color in zip(keys,colors):
            count=int(r[key]); h=plot_h*count/maxv
            ybase-=h
            parts.append(f'<rect x="{cx-bar_w/2:.1f}" y="{ybase:.1f}" width="{bar_w:.1f}" height="{h:.1f}" fill="{color}" stroke="white" stroke-width="1"/>')
            # Show both count and percentage inside each segment. For very
            # small segments, place the label just above the segment.
            pct=count/total*100
            if h >= 28:
                text_y=ybase+h/2+4
                text_color='white' if key=='2分' else '#222222'
                parts.append(svg_text(cx, text_y, f'{count} ({pct:.1f}%)', 12, 'middle', 'bold'))
            elif count > 0:
                parts.append(svg_text(cx+bar_w/2+5, ybase+h/2+4, f'{count} ({pct:.1f}%)', 11, 'start'))
        name=str(r['配置'])
        label=svg_text(cx, top+plot_h+25, name, 11, 'end')
        parts.append(label.replace(f'<text x="{cx}"', f'<text transform="rotate(-45 {cx} {top+plot_h+25})" x="{cx}"'))

    parts.append(svg_text(left, H-45, 'Each bar totals the sample count; segment labels show count (percentage)', 12))
    legend_x=left; legend_y=top-25
    for color,label in zip(colors,labels):
        parts.append(f'<rect x="{legend_x}" y="{legend_y-12}" width="14" height="14" fill="{color}"/>')
        parts.append(svg_text(legend_x+20, legend_y, label, 12)); legend_x+=210
    parts.append('</svg>')
    Path(path).write_text(''.join(parts), encoding='utf-8')

'''
s=s[:start]+new+s[end:]
s=s.replace("save_svg(out/'layer27_quantize_score_counts.svg','Layer 27 quantization: score counts on HarmBench val',focus,'counts')", "save_svg(out/'layer27_quantize_score_counts.svg','Layer 27 quantization: score counts (stacked, total=80)',focus,'stacked')")
s=s.replace("save_svg(out/'layer27_quantize_score_percentages.svg','Layer 27 quantization: score distribution (%)',focus,'percent')", "save_svg(out/'layer27_quantize_score_percentages.svg','Layer 27 quantization: score distribution (stacked)',focus,'stacked')")
s=s.replace("save_svg(out/'all_approximation_score_counts.svg','All scored approximation configurations: score counts',summary,'counts')", "save_svg(out/'all_approximation_score_counts.svg','All scored approximation configurations: score counts (stacked)',summary,'stacked')")
p.write_text(s,encoding='utf-8')
