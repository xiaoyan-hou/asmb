import torch
import torch.nn.functional as F
from .coreset import MemorySelector

class StabilitySelector(MemorySelector):
    """Select normal patches that remain consistent under a horizontal flip."""
    requires_augmented = True

    def __init__(self, ratio=0.1): self.ratio = ratio
    def fit(self, features, augmented_features=None):
        x = features.reshape(-1, features.shape[-1]).detach().cpu()
        if augmented_features is None: raise ValueError("StabilitySelector requires augmented features")
        aug = augmented_features.reshape(-1, augmented_features.shape[-1]).detach().cpu()
        stability = F.cosine_similarity(x, aug, dim=1)
        n = max(1, int(len(x) * self.ratio))
        return x[torch.topk(stability, min(n, len(x)), largest=True).indices]
