import argparse, csv, json, sys
from pathlib import Path
import torch
import yaml
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.mvtec import MVTecDataset
from src.data.transforms import image_transform
from src.models.backbone import build_backbone
from src.models.feature_extractor import FeatureExtractor
from src.utils.seed import seed_everything

DEFAULT_CATEGORIES = ['bottle','cable','capsule','carpet','grid','hazelnut','leather','metal_nut','pill','screw','tile','toothbrush','transistor','wood','zipper']

@torch.no_grad()
def profile_category(category, cfg, extractor, device):
    tfm = image_transform(cfg['input']['resize'], cfg['input']['image_size'])
    dataset = MVTecDataset(cfg['dataset']['root'], category, 'train', tfm)
    loader = DataLoader(dataset, batch_size=cfg['runtime']['batch_size'], shuffle=False, num_workers=cfg['runtime']['num_workers'])
    values = []
    for batch in loader:
        images = batch['image']
        original = extractor.extract(images)
        flipped = extractor.extract(torch.flip(images, dims=[3]))
        side = int(flipped.shape[1] ** 0.5)
        flipped = flipped.reshape(flipped.shape[0], side, side, -1).flip(dims=[2]).reshape_as(flipped)
        values.append(torch.nn.functional.cosine_similarity(original, flipped, dim=-1).flatten().cpu())
    s = torch.cat(values)
    thresholds = {f'fraction_ge_{t}': float((s >= t).float().mean()) for t in (0.8, 0.9, 0.95, 0.98)}
    return {'category': category, 'num_patches': int(s.numel()), 'mean': float(s.mean()), 'std': float(s.std()), 'min': float(s.min()), 'max': float(s.max()), 'q05': float(torch.quantile(s, .05)), 'q25': float(torch.quantile(s, .25)), 'q50': float(torch.quantile(s, .50)), 'q75': float(torch.quantile(s, .75)), 'q95': float(torch.quantile(s, .95)), **thresholds}

def main():
    p = argparse.ArgumentParser(); p.add_argument('--config', default='configs/patchcore_mvtec.yaml'); p.add_argument('--categories', nargs='*', default=DEFAULT_CATEGORIES); a = p.parse_args()
    cfg = yaml.safe_load(open(a.config)); seed_everything(cfg['runtime']['seed'])
    device = 'cuda' if cfg['runtime']['use_cuda'] and torch.cuda.is_available() else 'cpu'
    extractor = FeatureExtractor(build_backbone(cfg['model']['pretrained']), cfg['features']['layers'], device)
    profiles = []
    for category in a.categories:
        print(f'Analyzing {category}...', flush=True)
        profiles.append(profile_category(category, cfg, extractor, device))
    for row in profiles:
        base_path = Path('results/raw') / f"patchcore_{row['category']}_baseline.json"
        hybrid_path = Path('results/raw') / f"patchcore_{row['category']}_hybrid.json"
        if base_path.exists() and hybrid_path.exists():
            base, hybrid = json.loads(base_path.read_text()), json.loads(hybrid_path.read_text())
            row['baseline_image_auroc'] = base['image_auroc']; row['hybrid_image_auroc'] = hybrid['image_auroc']; row['delta_image_auroc'] = hybrid['image_auroc'] - base['image_auroc']
            row['baseline_pixel_auroc'] = base['pixel_auroc']; row['hybrid_pixel_auroc'] = hybrid['pixel_auroc']; row['delta_pixel_auroc'] = hybrid['pixel_auroc'] - base['pixel_auroc']
    out = Path('results/analysis'); out.mkdir(parents=True, exist_ok=True)
    (out / 'stability_profiles.json').write_text(json.dumps(profiles, indent=2), encoding='utf-8')
    fields = list(dict.fromkeys(key for row in profiles for key in row.keys()))
    with (out / 'stability_profiles.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fields); writer.writeheader(); writer.writerows(profiles)
    print(json.dumps(profiles, indent=2)); print(f'Saved to {out}')

if __name__ == '__main__': raise SystemExit(main())
