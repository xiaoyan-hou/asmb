import torch

class MemorySelector:
    def fit(self, features): raise NotImplementedError

class CoresetSelector(MemorySelector):
    def __init__(self, ratio=0.1, seed=0): self.ratio, self.seed = ratio, seed
    def fit(self, features):
        x = features.reshape(-1, features.shape[-1]).detach().cpu()
        n = max(1, int(len(x) * self.ratio))
        if n >= len(x):
            return x
        # Exact farthest-point sampling requires an O(N^2) distance matrix and
        # is impractical for dense patch features. Uniform sampling is bounded
        # in memory, deterministic, and serves as the baseline selector that
        # can later be replaced by a research-grade selector.
        g = torch.Generator().manual_seed(self.seed)
        indices = torch.randperm(len(x), generator=g)[:n]
        return x[indices]


class GreedyCoresetSelector(MemorySelector):
    """Projected, batched approximation of PatchCore greedy k-center."""
    def __init__(self, ratio=0.1, seed=0, candidate_size=8192,
                 candidate_multiplier=2.0, projection_dim=64, selection_batch=32):
        self.ratio, self.seed, self.candidate_size = ratio, seed, candidate_size
        self.candidate_multiplier = candidate_multiplier
        self.projection_dim, self.selection_batch = projection_dim, selection_batch

    @torch.no_grad()
    def fit(self, features):
        x = features.reshape(-1, features.shape[-1]).detach().cpu().float()
        return x[self.fit_indices(x)]

    def fit_indices(self, features):
        x = features.reshape(-1, features.shape[-1]).detach().cpu().float()
        n = max(1, int(len(x) * self.ratio))
        if n >= len(x): return torch.arange(len(x))
        g = torch.Generator().manual_seed(self.seed)
        pool_n = min(len(x), max(self.candidate_size, int(n * self.candidate_multiplier)))
        pool_indices = torch.randperm(len(x), generator=g)[:pool_n]
        pool = x[pool_indices]
        dim = min(self.projection_dim, x.shape[1])
        projection = torch.randn(x.shape[1], dim, generator=g) / dim ** 0.5
        device = "cuda" if torch.cuda.is_available() else "cpu"
        projected = torch.nn.functional.normalize(pool @ projection, dim=1).to(device)
        selected = [0]
        selected_mask = torch.zeros(pool_n, dtype=torch.bool, device=device)
        selected_mask[0] = True
        min_dist = torch.cdist(projected, projected[:1]).squeeze(1)
        while len(selected) < n:
            add_n = min(self.selection_batch, n - len(selected))
            scores = min_dist.masked_fill(selected_mask, -1)
            new_idx = torch.topk(scores, add_n).indices
            selected.extend(new_idx.cpu().tolist())
            selected_mask[new_idx] = True
            min_dist = torch.minimum(min_dist, torch.cdist(projected, projected[new_idx]).min(1).values)
        return pool_indices[torch.tensor(selected, dtype=torch.long)]
