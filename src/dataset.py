"""
FrequencyDataset — loads images and returns frequency-domain representations.

Expected directory layout (same as CIFAKE and GenImage):

    root/
        real/
            img001.jpg
            ...
        fake/          ← or generator name (e.g. "sdv14", "dalle")
            img001.jpg
            ...

For multi-generator GenImage cross-eval, instantiate separate datasets per
generator directory and concatenate / evaluate independently.
"""

import os
from pathlib import Path
from typing import Callable, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from .transforms import (
    azimuthal_average_fast,
    log_power_spectrum_2d,
    to_grayscale_array,
)

# Label convention: 0 = real, 1 = fake
LABEL_MAP = {"real": 0}  # anything not "real" is treated as fake (1)


def _find_images(directory: str) -> list:
    exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    paths = []
    for root, _, files in os.walk(directory):
        for f in files:
            if Path(f).suffix.lower() in exts:
                paths.append(os.path.join(root, f))
    return sorted(paths)


class FrequencyDataset(Dataset):
    """
    Returns (spectrum_2d, profile_1d, label) for each image.

    spectrum_2d : torch.FloatTensor  shape (1, H, W)   — for 2D CNN
    profile_1d  : torch.FloatTensor  shape (r_max,)    — for 1D CNN
    label       : int  (0 = real, 1 = fake)

    Args:
        root:        Path with 'real/' and one or more 'fake/' (or generator-named) subdirs.
        size:        Resize target before FFT (default 224).
        transform:   Optional callable applied to the PIL image before FFT.
        fake_dirs:   Explicit list of subdirectory names to treat as fake.
                     If None, every subdir that isn't 'real' is treated as fake.
    """

    def __init__(
        self,
        root: str,
        size: int = 224,
        transform: Optional[Callable] = None,
        fake_dirs: Optional[list] = None,
    ):
        self.size = size
        self.transform = transform
        self.samples: list[Tuple[str, int]] = []

        root = Path(root)
        subdirs = [d for d in root.iterdir() if d.is_dir()]

        for subdir in subdirs:
            if subdir.name.lower() == "real":
                label = 0
            elif fake_dirs is not None and subdir.name not in fake_dirs:
                continue
            else:
                label = 1

            for path in _find_images(str(subdir)):
                self.samples.append((path, label))

        if not self.samples:
            raise ValueError(f"No images found under {root}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]

        img = Image.open(path)
        if self.transform is not None:
            img = self.transform(img)

        gray = to_grayscale_array(img, size=self.size)
        spectrum_2d = log_power_spectrum_2d(gray)          # (H, W)
        profile_1d = azimuthal_average_fast(spectrum_2d)   # (r_max,)

        # 2D spectrum: add channel dim for Conv2d
        spectrum_2d_t = torch.from_numpy(spectrum_2d).unsqueeze(0)  # (1, H, W)
        profile_1d_t = torch.from_numpy(profile_1d)                 # (r_max,)

        return spectrum_2d_t, profile_1d_t, label

    def label_counts(self) -> dict:
        labels = [s[1] for s in self.samples]
        return {"real": labels.count(0), "fake": labels.count(1)}


def make_splits(
    dataset: FrequencyDataset,
    val_frac: float = 0.1,
    test_frac: float = 0.1,
    seed: int = 42,
) -> Tuple[Dataset, Dataset, Dataset]:
    """
    Stratified random split into train / val / test subsets.
    Returns three torch Subset objects.
    """
    from torch.utils.data import Subset
    from sklearn.model_selection import train_test_split

    indices = list(range(len(dataset)))
    labels = [dataset.samples[i][1] for i in indices]

    # Split off test first, then val from remaining
    train_val_idx, test_idx = train_test_split(
        indices, test_size=test_frac, stratify=labels, random_state=seed
    )
    tv_labels = [labels[i] for i in train_val_idx]
    val_size = val_frac / (1.0 - test_frac)
    train_idx, val_idx = train_test_split(
        train_val_idx, test_size=val_size, stratify=tv_labels, random_state=seed
    )

    return Subset(dataset, train_idx), Subset(dataset, val_idx), Subset(dataset, test_idx)
