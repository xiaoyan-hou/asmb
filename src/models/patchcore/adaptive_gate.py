import torch
import torch.nn.functional as F
from .hybrid import HybridStabilitySelector

class AdaptiveGateSelector(HybridStabilitySelector):
    """Choose the stability fraction using a training-only coverage proxy.

    No test labels are used: candidates are scored on a subsample by nearest
    distance from normal features (coverage) and mean stability.
    """
    def __init__(self, ratio=0.1, fractions=(0.0, 0.1, 0.3, 0.5), alpha=0.02, seed=0):
        super().__init__(ratio, 0.3, seed)
        self.fractions, self.alpha = fractions, alpha

    def fit(self, features, augmented_features=None, validation_features=None):
        if augmented_features is None: raise ValueError("AdaptiveGateSelector requires augmented features")
        x = features.detach().cpu().reshape(-1, features.shape[-1])
        aug = augmented_features.detach().cpu().reshape_as(x)
        stability = F.cosine_similarity(x, aug, dim=1)
        n = max(1, min(len(x), int(len(x) * self.ratio)))
        g = torch.Generator().manual_seed(self.seed)
        if validation_features is None:
            validation_features = features
        validation = validation_features.detach().cpu().reshape(-1, features.shape[-1])
        eval_ids = torch.randperm(len(validation), generator=g)[:min(2048, len(validation))]
        stable_order = torch.argsort(stability, descending=True)
        candidates = []
        for fraction in self.fractions:
            stable_n = min(n, max(0, int(n * fraction)))
            chosen = stable_order[:stable_n].tolist()
            chosen_set = set(chosen)
            for idx in torch.randperm(len(x), generator=g).tolist():
                if idx not in chosen_set:
                    chosen.append(idx); chosen_set.add(idx)
                    if len(chosen) == n: break
            bank = x[chosen[:n]]
            coverage = torch.cdist(validation[eval_ids], bank).min(dim=1).values.mean()
            stability_term = stability[chosen[:n]].mean()
            candidates.append((float(coverage - self.alpha * stability_term), fraction, chosen[:n]))
        _, self.selected_fraction, selected = min(candidates, key=lambda z: z[0])
        self.selected_stability = stability[selected]
        return x[selected]
