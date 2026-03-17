"""Spatial-domain PGD attack (L∞ threat model) — baseline."""
from __future__ import annotations

import torch
import torch.nn as nn


def pgd_spatial(
    model: nn.Module,
    X: torch.Tensor,
    y: torch.Tensor,
    eps: float = 8 / 255,
    alpha: float = 2 / 255,
    num_iter: int = 20,
) -> torch.Tensor:
    """Untargeted PGD in pixel space.

    Args:
        model: Classifier returning (B, 1) logits.
        X: Clean images (B, C, H, W) in [0, 1].
        y: Binary labels (B,) — 0=real, 1=fake.
        eps: L∞ perturbation budget.
        alpha: Step size per iteration.
        num_iter: Number of PGD steps.

    Returns:
        Adversarial images (B, C, H, W) clamped to [0, 1].
    """
    model.eval()
    loss_fn = nn.BCEWithLogitsLoss()
    delta = torch.zeros_like(X).uniform_(-eps, eps)
    delta = delta.clamp(-eps, eps)
    delta.requires_grad_(False)

    for _ in range(num_iter):
        delta = delta.detach().requires_grad_(True)
        logits = model(X + delta).squeeze(1)
        loss = loss_fn(logits, y.float())
        loss.backward()
        with torch.no_grad():
            delta = delta + alpha * delta.grad.sign()
            delta = delta.clamp(-eps, eps)

    return (X + delta).clamp(0.0, 1.0).detach()
