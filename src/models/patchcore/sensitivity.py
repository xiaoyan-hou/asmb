import torch
from .coreset import CoresetSelector, GreedyCoresetSelector, MemorySelector

class CoresetSensitivitySelector(MemorySelector):
    """Mix coverage sampling with low-density (boundary) representatives."""
    def __init__(self, ratio=0.1, sensitivity_fraction=0.3, seed=0, reference_size=2048, knn_k=10):
        self.ratio, self.sensitivity_fraction, self.seed, self.reference_size, self.knn_k = ratio, sensitivity_fraction, seed, reference_size, knn_k

    def fit(self, features):
        x = features.detach().cpu().reshape(-1, features.shape[-1])
        total = max(1, min(len(x), int(len(x) * self.ratio)))
        sensitive_n = min(total, max(1, int(total * self.sensitivity_fraction)))
        coverage_n = total - sensitive_n
        g = torch.Generator().manual_seed(self.seed)
        # Estimate local density against a bounded random reference set.
        ref = x[torch.randperm(len(x), generator=g)[:min(self.reference_size, len(x))]]
        distances = torch.cdist(x, ref)
        k = min(self.knn_k, ref.shape[0])
        density_distance = distances.topk(k, largest=False).values.mean(1)
        sensitive_ids = torch.topk(density_distance, sensitive_n, largest=True).indices.tolist()
        chosen = set(sensitive_ids)
        if coverage_n:
            # Coverage pool is sampled independently to preserve distribution.
            perm = torch.randperm(len(x), generator=g).tolist()
            for idx in perm:
                if idx not in chosen:
                    chosen.add(idx)
                    if len(chosen) >= total: break
        self.selected_sensitivity = density_distance[list(chosen)]
        return x[list(chosen)[:total]]


class GreedySensitivitySelector(MemorySelector):
    """Equal-budget greedy coverage plus low-density boundary features."""
    def __init__(self, ratio=0.1, sensitivity_fraction=0.3, seed=0, reference_size=2048, knn_k=10):
        self.ratio, self.sensitivity_fraction = ratio, sensitivity_fraction
        self.seed, self.reference_size, self.knn_k = seed, reference_size, knn_k

    @torch.no_grad()
    def fit(self, features):
        x = features.detach().cpu().reshape(-1, features.shape[-1]).float()
        total = max(1, min(len(x), int(len(x) * self.ratio)))
        sensitive_n = min(total, max(1, int(total * self.sensitivity_fraction)))
        coverage_n = total - sensitive_n
        g = torch.Generator().manual_seed(self.seed)
        ref = x[torch.randperm(len(x), generator=g)[:min(self.reference_size, len(x))]]
        density = []
        for chunk in x.split(1024):
            d = torch.cdist(chunk, ref)
            density.append(d.topk(min(self.knn_k, len(ref)), largest=False).values.mean(1))
        density = torch.cat(density)
        sensitive_ids = torch.topk(density, sensitive_n, largest=True).indices
        greedy = GreedyCoresetSelector(self.ratio, self.seed)
        coverage_candidates = greedy.fit_indices(x)
        sensitive_set = set(sensitive_ids.tolist())
        coverage_ids = [i for i in coverage_candidates.tolist() if i not in sensitive_set][:coverage_n]
        if len(coverage_ids) < coverage_n:
            for i in torch.randperm(len(x), generator=g).tolist():
                if i not in sensitive_set and i not in coverage_ids:
                    coverage_ids.append(i)
                    if len(coverage_ids) == coverage_n: break
        chosen = torch.cat([torch.tensor(coverage_ids), sensitive_ids])
        self.selected_sensitivity = density[chosen]
        return x[chosen]
