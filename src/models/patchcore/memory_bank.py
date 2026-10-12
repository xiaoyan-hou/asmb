import torch

class MemoryBank:
    def __init__(self, features=None): self.features = features
    def build(self, features): self.features = features.detach().cpu(); return self
    def save(self, path): torch.save(self.features, path)
    @classmethod
    def load(cls, path): return cls(torch.load(path, map_location="cpu"))
