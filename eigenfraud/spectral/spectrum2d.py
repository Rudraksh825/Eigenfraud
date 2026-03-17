"""2D log-power spectrum preprocessing for the ResNet2D detector."""
from __future__ import annotations

import torch
import torch.nn.functional as F


def compute_log_spectrum_2d(
    image: torch.Tensor,
    output_size: int = 224,
) -> torch.Tensor:
    """Compute per-channel 2D log-power spectrum, normalized to [0, 1].

    Args:
        image: Float tensor (C, H, W) in [0, 1].
        output_size: Spatial size to resize the spectrum to.

    Returns:
        Float tensor (C, output_size, output_size) in [0, 1].
    """
    C, H, W = image.shape
    # Subtract per-channel mean to suppress DC spike
    mean = image.mean(dim=(-2, -1), keepdim=True)
    x = image - mean

    # FFT per channel — shape (C, H, W) complex
    F_c = torch.fft.fftshift(torch.fft.fft2(x))
    psd = torch.abs(F_c) ** 2
    log_psd = torch.log10(1.0 + psd)

    # Min-max normalize per channel
    lo = log_psd.flatten(1).min(dim=1).values[:, None, None]
    hi = log_psd.flatten(1).max(dim=1).values[:, None, None]
    log_psd = (log_psd - lo) / (hi - lo + 1e-8)

    # Resize to target size
    log_psd = F.interpolate(
        log_psd.unsqueeze(0),
        size=(output_size, output_size),
        mode="bilinear",
        align_corners=False,
    ).squeeze(0)

    return log_psd
