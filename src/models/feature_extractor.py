import torch
import torch.nn.functional as F

class FeatureExtractor:
    """Extract multi-scale patch descriptors from a CNN backbone.

    Args:
        backbone: torchvision-style backbone.
        layers: layer names whose outputs are concatenated.
        device: computation device.
        pool: optional per-layer smoothing applied *before* concatenation.
            Anomalib's PatchCore uses ``pool="avg3x3"`` (``AvgPool2d(3, 1, 1)``),
            which is what the original PatchCore method specifies. The default
            (``None``) preserves the behaviour used by the published ASMB runs, so
            existing results are unaffected unless the caller opts in.
    """
    def __init__(self, backbone, layers=("layer2", "layer3"), device="cpu", pool=None):
        self.model, self.layers, self.device = backbone.to(device).eval(), layers, device
        if pool not in (None, "avg3x3"):
            raise ValueError(f"Unsupported pool mode: {pool!r} (expected None or 'avg3x3')")
        self.pool = pool
        self.features = {}
        for name in layers:
            getattr(self.model, name).register_forward_hook(lambda _, __, out, n=name: self.features.__setitem__(n, out))

    @torch.no_grad()
    def extract(self, images):
        self.features.clear(); self.model(images.to(self.device))
        maps = [self.features[n] for n in self.layers]
        if self.pool == "avg3x3":
            maps = [F.avg_pool2d(x, 3, 1, 1) for x in maps]
        target = maps[0].shape[-2:]
        maps = [F.interpolate(x, size=target, mode="bilinear", align_corners=False) for x in maps]
        return torch.cat(maps, dim=1).permute(0, 2, 3, 1).reshape(images.shape[0], -1, sum(x.shape[1] for x in maps))
