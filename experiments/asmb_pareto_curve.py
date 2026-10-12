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
from src.models.patchcore.coreset import GreedyCoresetSelector
from src.models.patchcore.sensitivity import GreedySensitivitySelector
from src.models.patchcore.scorer import Scorer
from src.evaluation.evaluator import Evaluator
from src.utils.seed import seed_everything

CATEGORIES=['bottle','cable','capsule','carpet','grid','hazelnut','leather','metal_nut','pill','screw','tile','toothbrush','transistor','wood','zipper']

@torch.no_grad()
def extract(loader, extractor):
    feats, labels, masks = [], [], []
    for batch in loader:
        feats.append(extractor.extract(batch['image']).cpu())
        if 'label' in batch:
            labels.extend(batch['label'].tolist()); masks.extend(batch['mask'])
    return torch.cat(feats), labels, torch.stack(masks) if masks else None

def main():
    p=argparse.ArgumentParser();p.add_argument('--config',default='configs/patchcore_mvtec.yaml');p.add_argument('--categories',nargs='*',default=CATEGORIES);p.add_argument('--ratios',nargs='*',type=float,default=[.01,.03,.05,.10]);a=p.parse_args()
    cfg=yaml.safe_load(open(a.config));seed_everything(cfg['runtime']['seed']);device='cuda' if cfg['runtime']['use_cuda'] and torch.cuda.is_available() else 'cpu'
    tfm=image_transform(cfg['input']['resize'],cfg['input']['image_size']);extractor=FeatureExtractor(build_backbone(cfg['model']['pretrained']),cfg['features']['layers'],device);scorer=Scorer(cfg['anomaly']['num_neighbors']);evaluator=Evaluator();rows=[]
    out=Path('results/analysis');out.mkdir(parents=True,exist_ok=True);progress=out/'asmb_pareto_rows.json'
    for cat in a.categories:
        train=MVTecDataset(cfg['dataset']['root'],cat,'train',tfm);test=MVTecDataset(cfg['dataset']['root'],cat,'test',tfm)
        tr=DataLoader(train,cfg['runtime']['batch_size'],False,num_workers=cfg['runtime']['num_workers']);te=DataLoader(test,cfg['runtime']['batch_size'],False,num_workers=cfg['runtime']['num_workers'])
        train_f,_,_=extract(tr,extractor);test_f,labels,masks=extract(te,extractor);total=train_f.shape[0]*train_f.shape[1]
        for ratio in a.ratios:
            methods=[('greedy',GreedyCoresetSelector(ratio,cfg['runtime']['seed'])),('greedy_sensitivity',GreedySensitivitySelector(ratio,.3,cfg['runtime']['seed']))]
            for name,selector in methods:
                bank=selector.fit(train_f);bank_obj=type('B',(),{'features':bank})();start=time.perf_counter();patch_scores=scorer.score(test_f,bank_obj);elapsed=time.perf_counter()-start
                side=int(patch_scores.shape[1]**.5);ps=patch_scores[:,:side*side].reshape(patch_scores.shape[0],side,side);maps=F.interpolate(ps[:,None],size=masks.shape[-2:],mode='bilinear',align_corners=False)[:,0]
                metrics=evaluator.evaluate(ps.flatten(1).max(1).values.tolist(),labels,maps,masks)
                rows.append({'category':cat,'selector':name,'ratio':ratio,'memory_bank_size':len(bank),'total_train_patches':total,'compression_ratio':len(bank)/total,**metrics,'inference_time':elapsed/len(test)})
                progress.write_text(json.dumps(rows,indent=2));print(cat,name,ratio,metrics['image_auroc'],metrics['pixel_auroc'],flush=True)
        adaptive=json.load(open(f'results/raw/patchcore_{cat}_adaptive_sensitivity.json'))
        rows.append({'category':cat,'selector':'adaptive_sensitivity','ratio':adaptive['selected_fraction'],'memory_bank_size':adaptive['memory_bank_size'],'total_train_patches':total,'compression_ratio':adaptive['memory_bank_size']/total,'image_auroc':adaptive['image_auroc'],'pixel_auroc':adaptive['pixel_auroc'],'inference_time':adaptive['inference_time']})
        progress.write_text(json.dumps(rows,indent=2))
    with (out/'asmb_pareto_rows.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
    return 0
if __name__=='__main__':raise SystemExit(main())
