import torch
import torch.nn.functional as F
from .coreset import MemorySelector

class DiversityStabilitySelector(MemorySelector):
    """Select stable patches while preserving image/position coverage."""
    requires_augmented = True

    def __init__(self, ratio=0.1, grid_bins=8):
        self.ratio, self.grid_bins = ratio, grid_bins

    def fit(self, features, augmented_features=None):
        if augmented_features is None:
            raise ValueError("DiversityStabilitySelector requires augmented features")
        b, p, d = features.shape
        x = features.detach().cpu(); aug = augmented_features.detach().cpu()
        stability = F.cosine_similarity(x, aug, dim=-1)
        n = max(1, int(b * p * self.ratio))
        # Spatial bins provide a cheap diversity constraint without an O(N^2)
        # feature-distance matrix. One best patch per bin is selected first.
        side = max(1, int(p ** 0.5)); selected = []
        bins = {}
        for image_idx in range(b):
            for patch_idx in range(p):
                row, col = divmod(patch_idx, side)
                key = (min(self.grid_bins - 1, row * self.grid_bins // side), min(self.grid_bins - 1, col * self.grid_bins // side))
                score = float(stability[image_idx, patch_idx])
                if key not in bins or score > bins[key][0]: bins[key] = (score, image_idx * p + patch_idx)
        selected.extend(v[1] for v in bins.values())
        all_scores = stability.reshape(-1)
        order = torch.argsort(all_scores, descending=True).tolist()
        selected_set = set(selected)
        for idx in order:
            if len(selected) >= n: break
            if idx not in selected_set: selected.append(idx); selected_set.add(idx)
        return x.reshape(-1, d)[selected[:n]]
