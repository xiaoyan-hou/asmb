import torch
from torchvision.models import wide_resnet50_2, Wide_ResNet50_2_Weights

def build_backbone(pretrained=True):
    weights = Wide_ResNet50_2_Weights.DEFAULT if pretrained else None
    return wide_resnet50_2(weights=weights)
