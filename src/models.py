"""
Eigenfraud detector models.

CNN1D — operates on the 1D azimuthally averaged radial power spectrum.
         ~500k parameters. Captures only isotropic spectral structure.

CNN2D — operates on the full 2D log-power spectrum heatmap.
         ~2M parameters. Captures both isotropic and anisotropic structure.

Both output raw logits for 2-class (real / fake) classification.
"""

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# 1D CNN
# ---------------------------------------------------------------------------

class CNN1D(nn.Module):
    """
    Input:  (B, 1, L)  — L = radial profile length (typically 112)
    Output: (B, 2)     — logits
    """

    def __init__(self, input_length: int = 112, num_classes: int = 2):
        super().__init__()
        self.input_length = input_length

        def conv_block(in_ch, out_ch, k=3):
            return nn.Sequential(
                nn.Conv1d(in_ch, out_ch, kernel_size=k, padding=k // 2, bias=False),
                nn.BatchNorm1d(out_ch),
                nn.ReLU(inplace=True),
            )

        self.features = nn.Sequential(
            conv_block(1, 32),
            conv_block(32, 64),
            nn.MaxPool1d(2),         # L/2
            conv_block(64, 128),
            conv_block(128, 128),
            nn.MaxPool1d(2),         # L/4
            conv_block(128, 256),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(256, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L) → add channel dim
        if x.dim() == 2:
            x = x.unsqueeze(1)           # (B, 1, L)
        x = self.features(x)
        x = self.pool(x).squeeze(-1)     # (B, 256)
        return self.head(x)


# ---------------------------------------------------------------------------
# 2D CNN
# ---------------------------------------------------------------------------

class CNN2D(nn.Module):
    """
    Input:  (B, 1, H, W)  — centered log-power spectrum (recommended 224×224)
    Output: (B, 2)         — logits
    """

    def __init__(self, num_classes: int = 2):
        super().__init__()

        def conv_block(in_ch, out_ch, k=3):
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=k, padding=k // 2, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
            )

        self.features = nn.Sequential(
            conv_block(1, 32),
            conv_block(32, 64),
            nn.MaxPool2d(2),          # 112×112
            conv_block(64, 128),
            conv_block(128, 128),
            nn.MaxPool2d(2),          # 56×56
            conv_block(128, 256),
            nn.MaxPool2d(2),          # 28×28
            conv_block(256, 512),
            conv_block(512, 512),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(512, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 1, H, W)
        x = self.features(x)
        x = self.pool(x).flatten(1)   # (B, 512)
        return self.head(x)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_model(model_type: str, **kwargs) -> nn.Module:
    """
    Args:
        model_type: '1d' or '2d'
        **kwargs:   forwarded to the model constructor

    Returns:
        Instantiated model
    """
    if model_type == "1d":
        return CNN1D(**kwargs)
    elif model_type == "2d":
        return CNN2D(**kwargs)
    else:
        raise ValueError(f"Unknown model_type '{model_type}'. Choose '1d' or '2d'.")


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
