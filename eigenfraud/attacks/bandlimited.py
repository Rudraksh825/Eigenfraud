"""Band-limited frequency mask for targeted spectral attacks."""
from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class BandMask:
    """Annular frequency mask selecting coefficients in [low_cutoff, high_cutoff].

    Cutoffs are in normalized frequency units (0 = DC, 0.5 = Nyquist).
    Examples:
        BandMask(0.0, 0.2)  — low-frequency band (DC + low detail)
        BandMask(0.2, 0.5)  — mid-to-high frequencies (where LDM artifacts live)
        BandMask(0.1, 0.5)  — full artifact range for SD v1.4 (peaks at 0.125–0.5)
    """

    low_cutoff: float = 0.0
    high_cutoff: float = 1.0

    def make_mask(self, H: int, W: int, device: torch.device | str = "cpu") -> torch.Tensor:
        """Return a (H, W) float mask with 1s inside the annulus.

        The mask is broadcast-compatible with (B, C, H, W) delta_freq tensors.
        """
        freq_y = torch.fft.fftfreq(H, device=device).unsqueeze(1)  # (H, 1)
        freq_x = torch.fft.fftfreq(W, device=device).unsqueeze(0)  # (1, W)
        radius = torch.sqrt(freq_y ** 2 + freq_x ** 2)             # (H, W)
        mask = (radius >= self.low_cutoff) & (radius <= self.high_cutoff)
        return mask.float()


def pgd_freq_bandlimited(
    model,
    X: torch.Tensor,
    y: torch.Tensor,
    band_mask: BandMask,
    eps_freq: float = 0.05,
    alpha_freq: float = 0.005,
    num_iter: int = 20,
    spatial_eps: float | None = None,
) -> torch.Tensor:
    """Frequency-domain PGD restricted to a specific frequency band.

    Wraps pgd_freq with a pre-computed BandMask.
    """
    from eigenfraud.attacks.pgd_freq import pgd_freq

    _, _, H, W = X.shape
    mask = band_mask.make_mask(H, W, device=X.device)

    return pgd_freq(
        model=model,
        X=X,
        y=y,
        eps_freq=eps_freq,
        alpha_freq=alpha_freq,
        num_iter=num_iter,
        spatial_eps=spatial_eps,
        freq_mask=mask,
    )
