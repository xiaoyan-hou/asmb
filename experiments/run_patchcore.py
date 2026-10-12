import argparse, json, sys, time
from pathlib import Path
try:
    import torch, yaml
except ImportError as exc:
    print(f"Missing runtime dependency: {exc}. Create the documented conda environment first (conda env create -f environment.yml).")
    raise SystemExit(2)
from torch.utils.data import DataLoader
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.mvtec import MVTecDataset
from src.data.transforms import image_transform
from src.models.backbone import build_backbone
from src.models.feature_extractor import FeatureExtractor
from src.models.patchcore.coreset import CoresetSelector, GreedyCoresetSelector
from src.models.patchcore.stability import StabilitySelector
from src.models.patchcore.diversity_stability import DiversityStabilitySelector
from src.models.patchcore.feature_diversity import FeatureDiversityStabilitySelector
from src.models.patchcore.hybrid import HybridStabilitySelector
from src.models.patchcore.adaptive_gate import AdaptiveGateSelector
from src.models.patchcore.sensitivity import CoresetSensitivitySelector, GreedySensitivitySelector
from src.models.patchcore.adaptive_coverage import AdaptiveCoverageSelector, AdaptiveSensitivitySelector
from src.models.patchcore.memory_bank import MemoryBank
from src.models.patchcore.scorer import Scorer
from src.models.patchcore.anomaly_map import AnomalyMap
from src.models.patchcore.patchcore import PatchCore
from src.evaluation.evaluator import Evaluator
from src.utils.seed import seed_everything
from src.utils.visualization import save_anomaly_figure
import torch.nn.functional as F

def main():
    p = argparse.ArgumentParser(); p.add_argument("--config", default="configs/patchcore_mvtec.yaml"); p.add_argument("--category", default="bottle"); p.add_argument("--batch-size", type=int); p.add_argument("--output-suffix", default=""); p.add_argument("--seed", type=int); p.add_argument("--skip-figures", action="store_true"); a = p.parse_args()
    cfg = yaml.safe_load(open(a.config)); cfg["runtime"]["seed"] = a.seed if a.seed is not None else cfg["runtime"]["seed"]; seed_everything(cfg["runtime"]["seed"])
    device = "cuda" if cfg["runtime"]["use_cuda"] and torch.cuda.is_available() else "cpu"
    tfm = image_transform(cfg["input"]["resize"], cfg["input"]["image_size"])
    try:
        train = MVTecDataset(cfg["dataset"]["root"], a.category, "train", tfm); test = MVTecDataset(cfg["dataset"]["root"], a.category, "test", tfm)
    except FileNotFoundError as e: print(e); return 2
    bs = a.batch_size or cfg["runtime"]["batch_size"]; tr = DataLoader(train, bs, shuffle=False, num_workers=cfg["runtime"]["num_workers"]); te = DataLoader(test, bs, shuffle=False, num_workers=cfg["runtime"]["num_workers"])
    selector_name = cfg["memory"].get("selector", "coreset")
    selector_cls = {"stability": StabilitySelector, "diversity_stability": DiversityStabilitySelector, "feature_diversity": FeatureDiversityStabilitySelector, "hybrid_stability": HybridStabilitySelector, "adaptive_gate": AdaptiveGateSelector, "coreset_sensitivity": CoresetSensitivitySelector, "greedy_sensitivity": GreedySensitivitySelector, "greedy_coreset": GreedyCoresetSelector, "adaptive_coverage": AdaptiveCoverageSelector, "adaptive_sensitivity": AdaptiveSensitivitySelector}.get(selector_name, CoresetSelector)
    ratio = cfg["memory"]["coreset_sampling_ratio"]
    if selector_cls is AdaptiveSensitivitySelector:
        opts = cfg["memory"].get("adaptive", {}).copy(); opts["sensitivity_fraction"] = cfg["memory"].get("sensitivity_fraction", 0.3)
        selector = selector_cls(**opts, seed=cfg["runtime"]["seed"])
    elif selector_cls is AdaptiveCoverageSelector:
        selector = selector_cls(**cfg["memory"].get("adaptive", {}), seed=cfg["runtime"]["seed"])
    elif selector_cls in (CoresetSensitivitySelector, GreedySensitivitySelector):
        selector = selector_cls(ratio, cfg["memory"].get("sensitivity_fraction", 0.3), cfg["runtime"]["seed"],
                                knn_k=cfg["memory"].get("knn_k", 10))
    elif selector_cls is AdaptiveGateSelector:
        selector = selector_cls(ratio, seed=cfg["runtime"]["seed"])
    elif selector_cls is HybridStabilitySelector:
        selector = selector_cls(ratio, cfg["memory"].get("stable_fraction", 0.3), cfg["runtime"]["seed"])
    elif selector_cls is FeatureDiversityStabilitySelector:
        selector = selector_cls(ratio, cfg["memory"].get("num_projections", 8), cfg["runtime"]["seed"])
    elif selector_cls is DiversityStabilitySelector:
        selector = selector_cls(ratio, cfg["memory"].get("grid_bins", 8))
    elif selector_cls is StabilitySelector:
        selector = selector_cls(ratio)
    elif selector_cls is GreedyCoresetSelector:
        selector = selector_cls(ratio, cfg["runtime"]["seed"], cfg["memory"].get("candidate_size", 8192))
    else:
        selector = selector_cls(ratio, cfg["runtime"]["seed"])
    model = PatchCore(FeatureExtractor(build_backbone(cfg["model"]["pretrained"]), cfg["features"]["layers"], device), selector, MemoryBank(), Scorer(cfg["anomaly"]["num_neighbors"], stability_weight=cfg.get("scoring", {}).get("stability_weight", 0.0)), AnomalyMap())
    model.fit(tr); start = time.perf_counter(); scores, maps, labels, masks, images = model.predict(te); elapsed = time.perf_counter() - start
    # Save a small, deterministic sample for qualitative inspection.
    for i in range(0 if a.skip_figures else min(4, len(images))):
        image = images[i].permute(1, 2, 0).numpy()
        image = image * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]
        image = image.clip(0, 1)
        amap = F.interpolate(maps[i][None, None], size=image.shape[:2], mode="bilinear", align_corners=False)[0, 0].numpy()
        mask = F.interpolate(masks[i][None, None], size=image.shape[:2], mode="nearest")[0, 0].numpy()
        save_anomaly_figure(image, mask, amap, Path("results/figures") / f"patchcore_{a.category}_{i}.png")
    metrics = Evaluator().evaluate(scores, labels, maps, masks) | {"method":"patchcore","selector":selector_name,"selected_fraction":getattr(selector, "selected_fraction", None),"dataset":"mvtec","category":a.category,"seed":cfg["runtime"]["seed"],"backbone":cfg["model"]["backbone"],"layers":cfg["features"]["layers"],"coreset_ratio":cfg["memory"]["coreset_sampling_ratio"],"memory_bank_size":len(model.memory_bank.features),"inference_time":elapsed / len(test),"peak_gpu_memory_mb":torch.cuda.max_memory_allocated()/2**20 if device == "cuda" else 0,"coverage_history":getattr(selector, "coverage_history", None)}
    out = Path("results/raw"); out.mkdir(parents=True, exist_ok=True); json.dump(metrics, open(out / f"patchcore_{a.category}{a.output_suffix}.json", "w"), indent=2); print(json.dumps(metrics, indent=2)); return 0
if __name__ == "__main__": raise SystemExit(main())
