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

import csv
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
            if subdir.name.lower() in ("real", "nature"):
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
        for attempt in range(len(self.samples)):
            path, label = self.samples[(idx + attempt) % len(self.samples)]
            try:
                img = Image.open(path)
                img.verify()          # catch truncated/corrupt files
                img = Image.open(path)  # re-open; verify() exhausts the file handle
            except Exception:
                continue              # skip corrupt image silently

            if self.transform is not None:
                img = self.transform(img)

            gray = to_grayscale_array(img, size=self.size)
            spectrum_2d = log_power_spectrum_2d(gray)          # (H, W)
            profile_1d = azimuthal_average_fast(spectrum_2d)   # (r_max,)

            # 2D spectrum: add channel dim for Conv2d
            spectrum_2d_t = torch.from_numpy(spectrum_2d).unsqueeze(0)  # (1, H, W)
            profile_1d_t = torch.from_numpy(profile_1d)                 # (r_max,)

            return spectrum_2d_t, profile_1d_t, label

        raise RuntimeError(f"No valid images found starting at index {idx}")

    def label_counts(self) -> dict:
        labels = [s[1] for s in self.samples]
        return {"real": labels.count(0), "fake": labels.count(1)}


class CachedFrequencyDataset(Dataset):
    """
    Fast drop-in for FrequencyDataset that reads pre-computed .npz files
    written by scripts/precompute.py instead of running FFT on-the-fly.

    Args:
        manifest: Path to manifest.csv produced by precompute.py
    """

    def __init__(self, manifest: str):
        self.samples: list[Tuple[str, int, str]] = []  # (path, label, cache_file)
        with open(manifest) as f:
            for row in csv.DictReader(f):
                if row["cache_file"]:
                    self.samples.append((row["path"], int(row["label"]), row["cache_file"]))
        if not self.samples:
            raise ValueError(f"No cached samples found in {manifest}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        _, label, cache_file = self.samples[idx]
        data = np.load(cache_file)
        spectrum_2d_t = torch.from_numpy(data["s2d"].astype(np.float32))  # (1,H,W)
        profile_1d_t  = torch.from_numpy(data["p1d"])                     # (r_max,)
        return spectrum_2d_t, profile_1d_t, label

    def label_counts(self) -> dict:
        labels = [s[1] for s in self.samples]
        return {"real": labels.count(0), "fake": labels.count(1)}


class ParquetFrequencyDataset(Dataset):
    """
    Loads images from HuggingFace-style parquet files (e.g. defactify_dataset/data/).

    Expects parquet files with columns:
        Image    — dict with 'bytes' key (HF Image feature)
        Label_A  — int, 0=real / 1=fake  (binary veracity label)

    Args:
        parquet_dir:  Directory containing *.parquet files for one split
                      (e.g. defactify_dataset/data/ filtered to train-*.parquet).
        split:        "train", "validation", or "test" — used to glob the right files.
        size:         Resize target before FFT (default 224).
    """

    def __init__(self, parquet_dir: str, split: str = "train", size: int = 224):
        import datasets as hf_datasets
        self.size = size
        parquet_dir = Path(parquet_dir)
        files = sorted(parquet_dir.glob(f"{split}-*.parquet"))
        if not files:
            raise ValueError(f"No parquet files matching '{split}-*.parquet' in {parquet_dir}")
        ds = hf_datasets.load_dataset(
            "parquet",
            data_files={"data": [str(f) for f in files]},
            split="data",
        )
        # Keep only Image and Label_A; convert to list of (pil_image, label)
        self.samples: list[Tuple] = []
        for row in ds:
            img_field = row["Image"]
            if isinstance(img_field, dict) and "bytes" in img_field:
                import io
                img = Image.open(io.BytesIO(img_field["bytes"]))
            else:
                img = img_field  # already a PIL image
            self.samples.append((img, int(row["Label_A"])))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        img, label = self.samples[idx]
        gray = to_grayscale_array(img, size=self.size)
        spectrum_2d = log_power_spectrum_2d(gray)
        profile_1d  = azimuthal_average_fast(spectrum_2d)
        spectrum_2d_t = torch.from_numpy(spectrum_2d).unsqueeze(0)
        profile_1d_t  = torch.from_numpy(profile_1d)
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
