import argparse, csv, json, sys, time
from pathlib import Path
import torch, yaml
from torch.utils.data import DataLoader
import torch.nn.functional as F
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.mvtec import MVTecDataset
from src.data.transforms import image_transform
from src.models.backbone import build_backbone
from src.models.feature_extractor import FeatureExtractor
from src.models.patchcore.coreset import CoresetSelector
from src.models.patchcore.hybrid import HybridStabilitySelector
from src.models.patchcore.scorer import Scorer
from src.evaluation.evaluator import Evaluator
from src.utils.seed import seed_everything

CATEGORIES = ['bottle','cable','capsule','carpet','grid','hazelnut','leather','metal_nut','pill','screw','tile','toothbrush','transistor','wood','zipper']

@torch.no_grad()
def extract(loader, extractor, augmented=False):
    feats, augs, labels, masks = [], [], [], []
    for batch in loader:
        f = extractor.extract(batch['image']); feats.append(f.cpu())
        if augmented:
            af = extractor.extract(torch.flip(batch['image'], dims=[3])); side = int(af.shape[1] ** .5)
            af = af.reshape(af.shape[0], side, side, -1).flip(dims=[2]).reshape_as(af); augs.append(af.cpu())
        if 'label' in batch: labels.extend(batch['label'].tolist()); masks.extend(batch['mask'])
    return torch.cat(feats), (torch.cat(augs) if augs else None), labels, (torch.stack(masks) if masks else None)

def main():
    p=argparse.ArgumentParser(); p.add_argument('--config',default='configs/patchcore_mvtec.yaml'); p.add_argument('--categories',nargs='*',default=CATEGORIES); p.add_argument('--ratios',nargs='*',type=float,default=[.01,.1,.2]); a=p.parse_args()
    cfg=yaml.safe_load(open(a.config)); seed_everything(cfg['runtime']['seed']); device='cuda' if cfg['runtime']['use_cuda'] and torch.cuda.is_available() else 'cpu'
    tfm=image_transform(cfg['input']['resize'],cfg['input']['image_size']); extractor=FeatureExtractor(build_backbone(cfg['model']['pretrained']),cfg['features']['layers'],device); scorer=Scorer(cfg['anomaly']['num_neighbors']); evaluator=Evaluator(); rows=[]
    out=Path('results/analysis'); out.mkdir(parents=True,exist_ok=True)
    for cat in a.categories:
        print(f'[{cat}] extracting features...',flush=True)
        train=MVTecDataset(cfg['dataset']['root'],cat,'train',tfm); test=MVTecDataset(cfg['dataset']['root'],cat,'test',tfm)
        tr=DataLoader(train,cfg['runtime']['batch_size'],False,num_workers=cfg['runtime']['num_workers']); te=DataLoader(test,cfg['runtime']['batch_size'],False,num_workers=cfg['runtime']['num_workers'])
        train_f, train_aug, _, _=extract(tr,extractor,augmented=True); test_f, _, labels, masks=extract(te,extractor,augmented=False); total=train_f.shape[0]*train_f.shape[1]
        for ratio in a.ratios:
            for name, selector in [('coreset', CoresetSelector(ratio,cfg['runtime']['seed'])), ('hybrid_stability', HybridStabilitySelector(ratio,.3,cfg['runtime']['seed']))]:
                bank=selector.fit(train_f,train_aug) if name.startswith('hybrid') else selector.fit(train_f)
                # Keep distance computation on CUDA when available; scorer
                # still processes queries in bounded chunks.
                bank_obj = type('B',(),{'features':bank})()
                start=time.perf_counter(); patch_scores=scorer.score(test_f, bank_obj); elapsed=time.perf_counter()-start
                side=int(patch_scores.shape[1]**.5); ps=patch_scores[:,:side*side].reshape(patch_scores.shape[0],side,side); maps=F.interpolate(ps[:,None],size=masks.shape[-2:],mode='bilinear',align_corners=False)[:,0]
                metrics=evaluator.evaluate(ps.flatten(1).max(1).values.tolist(),labels,maps,masks)
                rows.append({'category':cat,'selector':name,'ratio':ratio,'memory_bank_size':len(bank),'total_train_patches':total,'compression_ratio':len(bank)/total,**metrics,'inference_time':elapsed/len(test)})
                (out/'memory_efficiency_curve.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')
                print(f'  {name} ratio={ratio}: image={metrics["image_auroc"]:.4f} pixel={metrics["pixel_auroc"]:.4f}',flush=True)
        (out / f'memory_curve_{cat}.json').write_text(json.dumps([r for r in rows if r['category'] == cat], indent=2), encoding='utf-8')
    (out/'memory_efficiency_curve.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')
    fields=list(rows[0]);
    with (out/'memory_efficiency_curve.csv').open('w',newline='',encoding='utf-8') as f: w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(rows)
    print(f'Saved {len(rows)} rows to {out}')
if __name__=='__main__': raise SystemExit(main())
