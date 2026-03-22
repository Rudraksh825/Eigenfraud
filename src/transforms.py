"""
Core frequency-domain transforms for Eigenfraud.

All functions operate on numpy arrays (H, W) grayscale images
or (H, W, C) RGB images (converted to grayscale internally).

Pipeline per image:
    raw image → grayscale → resize 224×224 → log_power_spectrum_2d → azimuthal_average
"""

import numpy as np
from PIL import Image


def to_grayscale_array(img: Image.Image, size: int = 224) -> np.ndarray:
    """Resize PIL image to (size, size) and convert to float32 grayscale array."""
    img = img.convert("L").resize((size, size), Image.LANCZOS)
    return np.array(img, dtype=np.float32)


def log_power_spectrum_2d(gray: np.ndarray) -> np.ndarray:
    """
    Compute the centered 2D log-power spectrum of a grayscale image.

    S(u,v) = log(1 + |F{I}(u,v)|^2)

    Args:
        gray: float32 array of shape (H, W)

    Returns:
        spectrum: float32 array of shape (H, W), centered (DC at center)
    """
    F = np.fft.fft2(gray)
    F_shifted = np.fft.fftshift(F)
    power = np.abs(F_shifted) ** 2
    log_power = np.log1p(power).astype(np.float32)
    return log_power


def azimuthal_average(spectrum: np.ndarray) -> np.ndarray:
    """
    Compute the 1D azimuthally averaged radial power spectrum.

    A(r) = mean of S(u,v) over all (u,v) with ||( u,v)|| ≈ r

    Args:
        spectrum: float32 array of shape (H, W), centered log-power spectrum

    Returns:
        profile: float32 array of shape (r_max,) where r_max = min(H,W)//2
    """
    H, W = spectrum.shape
    cy, cx = H // 2, W // 2

    # Build radial distance map
    y = np.arange(H) - cy
    x = np.arange(W) - cx
    xx, yy = np.meshgrid(x, y)
    r = np.sqrt(xx ** 2 + yy ** 2).astype(np.int32)

    r_max = min(H, W) // 2
    profile = np.zeros(r_max, dtype=np.float32)

    for radius in range(r_max):
        mask = r == radius
        if mask.any():
            profile[radius] = spectrum[mask].mean()

    return profile


def azimuthal_average_fast(spectrum: np.ndarray) -> np.ndarray:
    """
    Vectorized azimuthal average using np.bincount — faster than the loop version.
    Equivalent output to azimuthal_average().
    """
    H, W = spectrum.shape
    cy, cx = H // 2, W // 2

    y = np.arange(H) - cy
    x = np.arange(W) - cx
    xx, yy = np.meshgrid(x, y)
    r = np.round(np.sqrt(xx ** 2 + yy ** 2)).astype(np.int32).ravel()

    r_max = min(H, W) // 2
    flat = spectrum.ravel()

    # Only include pixels within r_max
    mask = r < r_max
    counts = np.bincount(r[mask], minlength=r_max)
    sums = np.bincount(r[mask], weights=flat[mask], minlength=r_max)

    # Avoid divide-by-zero for empty bins (shouldn't occur for r < r_max)
    with np.errstate(invalid="ignore"):
        profile = np.where(counts > 0, sums / counts, 0.0).astype(np.float32)

    return profile


def spectral_residual(mean_gen: np.ndarray, mean_real: np.ndarray) -> np.ndarray:
    """
    dS_gen(u,v) = S_gen_mean(u,v) - S_real_mean(u,v)

    Args:
        mean_gen:  averaged log-power spectrum for a generator class
        mean_real: averaged log-power spectrum for real images

    Returns:
        residual: same shape as inputs
    """
    return (mean_gen - mean_real).astype(np.float32)


def compute_mean_spectrum(image_paths: list, size: int = 224) -> np.ndarray:
    """
    Compute mean log-power spectrum over a list of image paths.
    Used for EDA / Figure 1.
    """
    accum = None
    count = 0
    for path in image_paths:
        try:
            img = Image.open(path)
            gray = to_grayscale_array(img, size=size)
            spec = log_power_spectrum_2d(gray)
            if accum is None:
                accum = spec.astype(np.float64)
            else:
                accum += spec
            count += 1
        except Exception:
            continue
    if accum is None or count == 0:
        raise ValueError("No images successfully processed.")
    return (accum / count).astype(np.float32)
