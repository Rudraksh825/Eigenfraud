"""1D azimuthally averaged power spectrum (Corvi et al., Dzanic et al.)."""
from __future__ import annotations

import numpy as np


def azimuthal_average(image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """1D azimuthally averaged power spectrum of a 2D grayscale image.

    Args:
        image: 2D float array, shape (H, W).

    Returns:
        bins: integer radial bin indices, shape (r_max+1,).
        psd_1d: mean power per radial bin, shape (r_max+1,).
    """
    F = np.fft.fftshift(np.fft.fft2(image - image.mean()))
    psd_2d = np.abs(F) ** 2

    H, W = image.shape
    cy, cx = H // 2, W // 2
    y, x = np.indices(psd_2d.shape)
    r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2).astype(int)
    r_max = min(H, W) // 2

    mask = r.ravel() <= r_max
    tbin = np.bincount(r.ravel()[mask], weights=psd_2d.ravel()[mask])
    nr = np.bincount(r.ravel()[mask])
    psd_1d = tbin / np.maximum(nr, 1)

    return np.arange(len(psd_1d)), psd_1d


def spectrum_to_1d_input(image: np.ndarray) -> np.ndarray:
    """Log-scaled, normalized 1D spectral vector ready for the CNN1D input.

    Applies azimuthal averaging then log10(1 + psd) normalization.

    Args:
        image: 2D float array (H, W) or 3D (H, W, C) — averaged over channels.

    Returns:
        1D float32 array of length min(H, W) // 2 + 1, values in [0, 1].
    """
    if image.ndim == 3:
        # average spectrum across channels
        _, psd = azimuthal_average(image.mean(axis=2))
    else:
        _, psd = azimuthal_average(image)

    log_psd = np.log10(1.0 + psd)
    # min-max normalize
    lo, hi = log_psd.min(), log_psd.max()
    if hi > lo:
        log_psd = (log_psd - lo) / (hi - lo)
    return log_psd.astype(np.float32)


def azimuthal_average_batch(images: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Mean azimuthal spectrum over a batch of images.

    Args:
        images: float array (N, H, W) — grayscale batch.

    Returns:
        bins, mean_psd_1d — averaged over N images.
    """
    psds = []
    for img in images:
        _, psd = azimuthal_average(img)
        psds.append(psd)
    psds = np.stack(psds, axis=0)
    bins = np.arange(psds.shape[1])
    return bins, psds.mean(axis=0)
