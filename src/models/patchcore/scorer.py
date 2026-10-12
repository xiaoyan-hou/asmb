import torch

class Scorer:
    def __init__(self, k=1, query_chunk_size=512, stability_weight=0.0):
        self.k, self.query_chunk_size, self.stability_weight = k, query_chunk_size, stability_weight
    @torch.no_grad()
    def score(self, query_features, memory_bank):
        q = query_features.reshape(-1, query_features.shape[-1])
        target = 'cuda' if torch.cuda.is_available() else 'cpu'
        q = q.to(target)
        bank = memory_bank.features.to(target)
        bank_stability = getattr(memory_bank, "stability", None)
        outputs = []
        for chunk in q.split(self.query_chunk_size):
            d = torch.cdist(chunk, bank)
            if bank_stability is not None and self.stability_weight:
                # Stable normal prototypes are trusted slightly more during
                # nearest-neighbor matching; alpha=0 recovers vanilla kNN.
                d = d * (1.0 - self.stability_weight * bank_stability.clamp(0, 1).unsqueeze(0))
            outputs.append(d.topk(self.k, largest=False).values.mean(1))
        return torch.cat(outputs).reshape(query_features.shape[:2]).cpu()
