import torch

from .coreset import MemorySelector


class AdaptiveCoverageSelector(MemorySelector):
    """Select prototypes until normal-feature coverage saturates."""

    def __init__(self, min_ratio=0.01, max_ratio=0.20, gain_threshold=0.01,
                 patience=3, step_ratio=0.005, projection_dim=64,
                 reference_size=4096, distance_threshold=0.30, seed=0):
        self.min_ratio, self.max_ratio = min_ratio, max_ratio
        self.gain_threshold, self.patience = gain_threshold, patience
        self.step_ratio, self.projection_dim = step_ratio, projection_dim
        self.reference_size, self.distance_threshold = reference_size, distance_threshold
        self.seed = seed
        self.selected_fraction, self.coverage_history = None, []

    @torch.no_grad()
    def fit(self, features):
        x = features.reshape(-1, features.shape[-1]).detach().cpu().float()
        total = len(x)
        min_n = max(1, int(total * self.min_ratio))
        max_n = max(min_n, min(total, int(total * self.max_ratio)))
        step_n = max(1, int(total * self.step_ratio))
        g = torch.Generator().manual_seed(self.seed)
        dim = min(self.projection_dim, x.shape[1])
        projection = torch.randn(x.shape[1], dim, generator=g) / dim ** 0.5
        projected = torch.nn.functional.normalize(x @ projection, dim=1)
        ref_idx = torch.randperm(total, generator=g)[:min(total, self.reference_size)]
        reference = projected[ref_idx]
        seed_idx = torch.randperm(total, generator=g)[:min_n]
        selected = seed_idx.tolist()
        selected_mask = torch.zeros(total, dtype=torch.bool); selected_mask[seed_idx] = True
        min_dist = self._min_distance(reference, projected[seed_idx])
        candidate_distance = self._min_distance(projected, projected[seed_idx])
        candidate_distance[selected_mask] = -1
        previous = float(min_dist.mean()); initial = max(previous, 1e-12); low_gain_steps = 0
        self.coverage_history = [{"size": len(selected), "ratio": len(selected) / total,
                                  "mean_distance": previous, "coverage": 0.0,
                                  "relative_gain": None}]
        while len(selected) < max_n:
            add_n = min(step_n, max_n - len(selected))
            new_idx = torch.topk(candidate_distance, add_n).indices
            selected.extend(new_idx.tolist()); selected_mask[new_idx] = True
            min_dist = torch.minimum(min_dist, self._min_distance(reference, projected[new_idx]))
            candidate_distance = torch.minimum(candidate_distance, self._min_distance(projected, projected[new_idx]))
            candidate_distance[selected_mask] = -1
            current = float(min_dist.mean())
            relative_gain = (previous - current) / max(previous, 1e-12)
            coverage = 1.0 - current / initial
            self.coverage_history.append({"size": len(selected), "ratio": len(selected) / total,
                                          "mean_distance": current, "coverage": coverage,
                                          "relative_gain": relative_gain})
            saturated = relative_gain < self.gain_threshold and current <= self.distance_threshold
            low_gain_steps = low_gain_steps + 1 if saturated else 0
            previous = current
            if low_gain_steps >= self.patience: break
        selected_idx = torch.tensor(selected, dtype=torch.long)
        self.selected_fraction = len(selected) / total
        return x[selected_idx]

    @staticmethod
    def _min_distance(query, bank, query_chunk=1024, bank_chunk=1024):
        outputs = []
        for q in query.split(query_chunk):
            best = torch.full((len(q),), float("inf"))
            for b in bank.split(bank_chunk):
                best = torch.minimum(best, torch.cdist(q, b).min(dim=1).values)
            outputs.append(best)
        return torch.cat(outputs)


class AdaptiveSensitivitySelector(AdaptiveCoverageSelector):
    """Adaptive budget with a fixed sensitivity share of the selected bank."""
    def __init__(self, *args, sensitivity_fraction=0.3, **kwargs):
        super().__init__(*args, **kwargs)
        self.sensitivity_fraction = sensitivity_fraction
        self.knn_k = kwargs.get("knn_k", 10)

    @torch.no_grad()
    def fit(self, features):
        from .sensitivity import GreedySensitivitySelector
        # First determine the category-specific budget from coverage.
        budget_selector = AdaptiveCoverageSelector(self.min_ratio, self.max_ratio,
            self.gain_threshold, self.patience, self.step_ratio, self.projection_dim,
            self.reference_size, self.distance_threshold, self.seed)
        budget_selector.fit(features)
        self.selected_fraction = budget_selector.selected_fraction
        self.coverage_history = budget_selector.coverage_history
        selector = GreedySensitivitySelector(self.selected_fraction, self.sensitivity_fraction, self.seed,
                                             knn_k=self.knn_k)
        return selector.fit(features)
