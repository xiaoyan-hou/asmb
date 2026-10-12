import csv, json, statistics
from pathlib import Path

def main():
    root=Path('results/analysis'); files=sorted(root.glob('memory_curve_*.json')); rows=[]
    for f in files:
        try: rows.extend(json.loads(f.read_text(encoding='utf-8')))
        except Exception: pass
    summary=[]
    for selector in sorted({r['selector'] for r in rows}):
        for ratio in sorted({r['ratio'] for r in rows}):
            vals=[r for r in rows if r['selector']==selector and r['ratio']==ratio]
            if not vals: continue
            def stat(k):
                x=[r[k] for r in vals]; return {'mean':sum(x)/len(x),'std':statistics.stdev(x) if len(x)>1 else 0.0}
            summary.append({'selector':selector,'ratio':ratio,'categories':len(vals),'image_auroc_mean':stat('image_auroc')['mean'],'image_auroc_std':stat('image_auroc')['std'],'pixel_auroc_mean':stat('pixel_auroc')['mean'],'pixel_auroc_std':stat('pixel_auroc')['std'],'memory_bank_mean':sum(r['memory_bank_size'] for r in vals)/len(vals)})
    out=root/'memory_efficiency_summary.json'; out.write_text(json.dumps(summary,indent=2),encoding='utf-8')
    with (root/'memory_efficiency_summary.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=summary[0].keys()); w.writeheader(); w.writerows(summary)
    print(json.dumps(summary,indent=2)); print(f'Loaded {len(files)} category curve files, {len(rows)} rows')
if __name__=='__main__': main()
