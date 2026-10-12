import torch
import torch.nn.functional as F

class AnomalyMap:
    def generate(self, patch_scores, feature_grid, output_size):
        b, h, w = patch_scores.shape
        return F.interpolate(patch_scores.reshape(b, 1, h, w), size=output_size, mode="bilinear", align_corners=False)[:, 0]
