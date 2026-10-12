import torch

class PatchCore:
    def __init__(self, extractor, selector, memory_bank, scorer, anomaly_map):
        self.extractor, self.selector, self.memory_bank, self.scorer, self.anomaly_map = extractor, selector, memory_bank, scorer, anomaly_map
    def fit(self, loader):
        chunks, augmented = [], []
        for batch in loader:
            images = batch["image"]
            chunks.append(self.extractor.extract(images))
            if getattr(self.selector, "requires_augmented", False):
                af = self.extractor.extract(torch.flip(images, dims=[3]))
                side = int(af.shape[1] ** 0.5)
                af = af.reshape(af.shape[0], side, side, -1).flip(dims=[2]).reshape_as(af)
                augmented.append(af)
        features = torch.cat(chunks)
        aug = torch.cat(augmented) if augmented else None
        validation = None
        if self.selector.__class__.__name__ == "AdaptiveGateSelector" and len(features) > 1:
            split = max(1, int(features.shape[0] * 0.8))
            validation = features[split:]
            features = features[:split]
            if aug is not None: aug = aug[:split]
            self.selector.ratio = min(1.0, self.selector.ratio / 0.8)
        if aug is not None:
            selected = self.selector.fit(features, aug, validation) if validation is not None else self.selector.fit(features, aug)
        else:
            selected = self.selector.fit(features)
        self.memory_bank.build(selected)
        if hasattr(self.selector, "selected_stability"):
            self.memory_bank.stability = self.selector.selected_stability.float()
        return self
    @torch.no_grad()
    def predict(self, loader):
        scores, maps, labels, masks, images = [], [], [], [], []
        for batch in loader:
            f = self.extractor.extract(batch["image"]); ps = self.scorer.score(f, self.memory_bank)
            side = int(ps.shape[1] ** 0.5); ps = ps[:, :side * side].reshape(ps.shape[0], side, side)
            maps.append(self.anomaly_map.generate(ps, (side, side), batch["mask"].shape[-2:]))
            scores.extend(ps.flatten(1).max(1).values.tolist()); labels.extend(batch["label"].tolist()); masks.extend(batch["mask"])
            images.extend(batch["image"].cpu())
        return scores, torch.cat(maps), labels, torch.stack(masks), images
