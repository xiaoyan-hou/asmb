import torch
import torch.nn.functional as F
from .coreset import MemorySelector

class HybridStabilitySelector(MemorySelector):
    """Keep a random coverage pool plus stable representatives."""
    requires_augmented = True

    def __init__(self, ratio=0.1, stable_fraction=0.3, seed=0):
        self.ratio, self.stable_fraction, self.seed = ratio, stable_fraction, seed

    def fit(self, features, augmented_features=None):
        if augmented_features is None:
            raise ValueError("HybridStabilitySelector requires augmented features")
        x = features.detach().cpu().reshape(-1, features.shape[-1])
        aug = augmented_features.detach().cpu().reshape_as(x)
        n = max(1, min(len(x), int(len(x) * self.ratio)))
        stable_n = min(n, max(1, int(n * self.stable_fraction)))
        g = torch.Generator().manual_seed(self.seed)
        stable = F.cosine_similarity(x, aug, dim=1)
        stable_ids = torch.topk(stable, stable_n, largest=True).indices.tolist()
        chosen = set(stable_ids)
        remaining = n - len(stable_ids)
        if remaining:
            perm = torch.randperm(len(x), generator=g).tolist()
            chosen.update(idx for idx in perm if idx not in chosen)
        chosen = list(chosen)[:n]
        self.selected_stability = stable[chosen]
        return x[chosen]
