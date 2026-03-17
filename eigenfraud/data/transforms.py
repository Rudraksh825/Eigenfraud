"""Data augmentation and preprocessing transforms."""
from __future__ import annotations

import io
import random

import numpy as np
from PIL import Image, ImageFilter
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF

from eigenfraud.config import AugConfig
from eigenfraud.spectral.azimuthal import spectrum_to_1d_input
from eigenfraud.spectral.spectrum2d import compute_log_spectrum_2d


class SpectralAugmentation:
    """Wang et al. CVPR 2020 augmentation: Gaussian blur + JPEG compression.

    Each applied independently with the configured probability.
    """

    def __init__(self, cfg: AugConfig):
        self.blur_prob = cfg.blur_prob
        self.jpeg_prob = cfg.jpeg_prob
        self.jpeg_quality_range = cfg.jpeg_quality_range
        self.blur_sigma_range = cfg.blur_sigma_range

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() < self.blur_prob:
            sigma = random.uniform(*self.blur_sigma_range)
            img = img.filter(ImageFilter.GaussianBlur(radius=sigma))
        if random.random() < self.jpeg_prob:
            quality = random.randint(*self.jpeg_quality_range)
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=quality)
            buf.seek(0)
            img = Image.open(buf).convert("RGB")
        return img


class ToLogSpectrum2D:
    """Convert a PIL Image to a 2D log-spectrum tensor for ResNet2D."""

    def __init__(self, output_size: int = 224):
        self.output_size = output_size

    def __call__(self, img: Image.Image) -> torch.Tensor:
        tensor = TF.to_tensor(img)  # (C, H, W) in [0, 1]
        return compute_log_spectrum_2d(tensor, self.output_size)


class ToAzimuthalSpectrum:
    """Convert a PIL Image to a 1D azimuthal spectrum vector for CNN1D."""

    def __call__(self, img: Image.Image) -> torch.Tensor:
        arr = np.array(img).astype(np.float32) / 255.0
        vec = spectrum_to_1d_input(arr)  # (L,)
        return torch.from_numpy(vec).unsqueeze(0)  # (1, L) for Conv1d


def build_transforms(
    cfg: AugConfig,
    mode: str,
    model_type: str,
    image_size: int = 256,
    spectrum_size: int = 224,
) -> T.Compose:
    """Factory returning a composed transform pipeline.

    Args:
        cfg: Augmentation config.
        mode: "train" or "eval".
        model_type: "resnet2d" or "cnn1d".
        image_size: Spatial size for center-crop / resize before FFT.
        spectrum_size: Target size for 2D spectrum (resnet2d only).
    """
    spatial = [T.Resize(image_size), T.CenterCrop(image_size)]

    if mode == "train":
        spatial.append(SpectralAugmentation(cfg))

    if model_type == "resnet2d":
        spectral = [ToLogSpectrum2D(spectrum_size)]
    else:
        spectral = [ToAzimuthalSpectrum()]

    return T.Compose(spatial + spectral)
