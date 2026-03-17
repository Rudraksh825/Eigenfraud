"""1D CNN detector operating on azimuthally averaged log-power spectra."""
from __future__ import annotations

import torch
import torch.nn as nn

from eigenfraud.config import ModelConfig


class SpectralCNN1D(nn.Module):
    """Lightweight 1D CNN for azimuthal spectrum classification.

    Input:  (B, 1, L) where L = image_size // 2 (e.g. 128 for 256px images).
    Output: (B, 1) raw logit.
    """

    def __init__(self, input_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            # Block 1
            nn.Conv1d(1, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
            # Block 2
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
            # Block 3
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
            # Block 4
            nn.Conv1d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1),
        )
        self.head = nn.Linear(128, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 1, L)
        features = self.net(x).squeeze(-1)  # (B, 128)
        return self.head(features)           # (B, 1)


def build_1d_model(cfg: ModelConfig) -> SpectralCNN1D:
    return SpectralCNN1D(input_dim=cfg.input_dim)
