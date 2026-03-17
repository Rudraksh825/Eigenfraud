"""2D CNN detector: modified ResNet-50 on full log-power spectrum maps.

Corvi et al. (ICASSP 2023) modification: remove the first downsampling
in layer1 so the network retains fine high-frequency forensic detail.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torchvision.models as tvm

from eigenfraud.config import ModelConfig


class SpectralResNet2D(nn.Module):
    """ResNet-50 with first-layer downsampling removed, operating on 2D spectra.

    Input:  (B, 3, H, W) log-spectrum tensor in [0, 1], typically 224×224.
    Output: (B, 1) raw logit.
    """

    def __init__(self, pretrained: bool = False):
        super().__init__()
        weights = tvm.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = tvm.resnet50(weights=weights)

        # Corvi et al.: remove stride in layer1's first bottleneck block
        # so the spatial resolution is preserved through the first residual stage.
        layer1_block = backbone.layer1[0]
        layer1_block.conv1.stride = (1, 1)
        if layer1_block.downsample is not None:
            layer1_block.downsample[0].stride = (1, 1)

        # Replace final FC with a single-logit head
        in_features = backbone.fc.in_features
        backbone.fc = nn.Linear(in_features, 1)

        self.backbone = backbone

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)  # (B, 1)


def build_2d_model(cfg: ModelConfig) -> SpectralResNet2D:
    return SpectralResNet2D(pretrained=cfg.pretrained)
