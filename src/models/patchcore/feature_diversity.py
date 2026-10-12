import torch
import torch.nn.functional as F
from .coreset import MemorySelector

class FeatureDiversityStabilitySelector(MemorySelector):
    """Hash feature projections into buckets, then select stable representatives."""
    requires_augmented = True

    def __init__(self, ratio=0.1, num_projections=8, seed=0):
        self.ratio, self.num_projections, self.seed = ratio, num_projections, seed

    def fit(self, features, augmented_features=None):
        if augmented_features is None:
            raise ValueError("FeatureDiversityStabilitySelector requires augmented features")
        x = features.detach().cpu().reshape(-1, features.shape[-1])
        aug = augmented_features.detach().cpu().reshape_as(x)
        stability = F.cosine_similarity(x, aug, dim=1)
        n = max(1, min(len(x), int(len(x) * self.ratio)))
        g = torch.Generator().manual_seed(self.seed)
        proj = torch.randn(x.shape[1], self.num_projections, generator=g)
        codes = (x @ proj > 0).to(torch.int64)
        weights = (2 ** torch.arange(self.num_projections, dtype=torch.int64))
        buckets = (codes * weights).sum(1)
        selected = []
        for bucket in torch.unique(buckets).tolist():
            ids = torch.where(buckets == bucket)[0]
            selected.append(int(ids[torch.argmax(stability[ids])]))
        selected = list(dict.fromkeys(selected))
        if len(selected) < n:
            selected_set = set(selected)
            for idx in torch.argsort(stability, descending=True).tolist():
                if idx not in selected_set:
                    selected.append(idx); selected_set.add(idx)
                    if len(selected) == n: break
        return x[selected[:n]]
