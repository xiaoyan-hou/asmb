import csv, json, statistics
from pathlib import Path

CATS=['bottle','cable','capsule','carpet','grid','hazelnut','leather','metal_nut','pill','screw','tile','toothbrush','transistor','wood','zipper']
FRACS=['01','03','05','07']
def main():
    root=Path('results/raw'); rows=[]
    for c in CATS:
        base_path=root/f'patchcore_{c}_baseline.json'
        if not base_path.exists(): continue
        base=json.loads(base_path.read_text())
        for f in FRACS:
            p=root/f'patchcore_{c}_sensitivity_f{f}.json'
            if p.exists():
                x=json.loads(p.read_text()); rows.append({'category':c,'fraction':int(f)/10,'image_auroc':x['image_auroc'],'pixel_auroc':x['pixel_auroc'],'baseline_image_auroc':base['image_auroc'],'baseline_pixel_auroc':base['pixel_auroc'],'delta_image_auroc':x['image_auroc']-base['image_auroc'],'delta_pixel_auroc':x['pixel_auroc']-base['pixel_auroc'],'memory_bank_size':x['memory_bank_size']})
    summary=[]
    for frac in sorted({r['fraction'] for r in rows}):
        x=[r for r in rows if r['fraction']==frac]
        def stat(k):
            v=[r[k] for r in x]; return sum(v)/len(v), statistics.stdev(v) if len(v)>1 else 0
        im,ims=stat('image_auroc'); px,pxs=stat('pixel_auroc'); di,dis=stat('delta_image_auroc'); dp,dps=stat('delta_pixel_auroc')
        summary.append({'fraction':frac,'categories':len(x),'image_mean':im,'image_std':ims,'pixel_mean':px,'pixel_std':pxs,'delta_image_mean':di,'delta_image_std':dis,'delta_pixel_mean':dp,'delta_pixel_std':dps,'memory_bank_mean':sum(r['memory_bank_size'] for r in x)/len(x)})
    best=[]
    for c in CATS:
        x=[r for r in rows if r['category']==c]
        if x: best.append(min(x,key=lambda r:-r['image_auroc']))
    out=Path('results/analysis'); out.mkdir(parents=True,exist_ok=True)
    (out/'sensitivity_ablation_summary.json').write_text(json.dumps({'summary':summary,'best_by_category':best},indent=2),encoding='utf-8')
    with (out/'sensitivity_ablation_rows.csv').open('w',newline='',encoding='utf-8') as f: w=csv.DictWriter(f,fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
    with (out/'sensitivity_ablation_summary.csv').open('w',newline='',encoding='utf-8') as f: w=csv.DictWriter(f,fieldnames=summary[0].keys()); w.writeheader(); w.writerows(summary)
    print(json.dumps({'summary':summary,'best_by_category':best},indent=2))
if __name__=='__main__': main()
