"""Frequency-domain PGD attack — the novel contribution.

Perturbation is parameterized directly in Fourier space. Gradients flow
through ifft2 automatically via Wirtinger calculus (PyTorch >= 1.8).

Uses fft2/ifft2 (not rfft2) to avoid Hermitian symmetry constraints.
The L∞ analog for complex coefficients applies sign() independently to
real and imaginary parts.
"""
from __future__ import annotations

import torch
import torch.nn as nn


def pgd_freq(
    model: nn.Module,
    X: torch.Tensor,
    y: torch.Tensor,
    eps_freq: float = 0.05,
    alpha_freq: float = 0.005,
    num_iter: int = 20,
    spatial_eps: float | None = None,
    freq_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Untargeted PGD with perturbation parameterized in Fourier space.

    Args:
        model: Classifier returning (B, 1) logits.
        X: Clean images (B, C, H, W) in [0, 1].
        y: Binary labels (B,).
        eps_freq: Per-coefficient L∞ budget in Fourier space (real & imag).
        alpha_freq: Step size per iteration in Fourier space.
        num_iter: Number of PGD steps.
        spatial_eps: If set, additionally clip the spatial perturbation
            to [-spatial_eps, spatial_eps] after each IFFT and re-FFT.
        freq_mask: Optional (H, W) float tensor in [0, 1] applied to
            delta_freq before IFFT each iteration (for band-limited attacks).

    Returns:
        Adversarial images (B, C, H, W) clamped to [0, 1].
    """
    model.eval()
    loss_fn = nn.BCEWithLogitsLoss()
    B, C, H, W = X.shape

    delta_freq = torch.zeros(B, C, H, W, dtype=torch.cfloat, device=X.device)

    for _ in range(num_iter):
        delta_freq = delta_freq.detach().requires_grad_(True)

        df = delta_freq
        if freq_mask is not None:
            df = df * freq_mask.to(X.device)

        delta_spatial = torch.fft.ifft2(df).real

        if spatial_eps is not None:
            delta_spatial = delta_spatial.clamp(-spatial_eps, spatial_eps)

        logits = model((X + delta_spatial).clamp(0.0, 1.0)).squeeze(1)
        loss = loss_fn(logits, y.float())
        loss.backward()

        with torch.no_grad():
            grad = delta_freq.grad
            step = alpha_freq * (torch.sign(grad.real) + 1j * torch.sign(grad.imag))
            delta_freq = delta_freq + step
            delta_freq = torch.complex(
                delta_freq.real.clamp(-eps_freq, eps_freq),
                delta_freq.imag.clamp(-eps_freq, eps_freq),
            )

    with torch.no_grad():
        df = delta_freq
        if freq_mask is not None:
            df = df * freq_mask.to(X.device)
        delta_spatial = torch.fft.ifft2(df).real
        if spatial_eps is not None:
            delta_spatial = delta_spatial.clamp(-spatial_eps, spatial_eps)

    return (X + delta_spatial).clamp(0.0, 1.0).detach()
