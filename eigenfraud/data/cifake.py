"""CIFAKE dataset (Bird & Lotfi, IEEE Access 2024).

NOTE: 32×32 resolution limits spectral analysis to Nyquist = 16 cycles/image.
Use only as a sanity-check or for rapid prototyping, not primary evaluation.

Download:
    kaggle datasets download -d birdy654/cifake-real-and-ai-generated-synthetic-images

Directory layout:
    <root>/
        train/
            REAL/
            FAKE/
        test/
            REAL/
            FAKE/
"""
from __future__ import annotations

from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset


class CIFAKEDataset(Dataset):
    """CIFAKE dataset.

    Args:
        root: Path to the CIFAKE root directory.
        split: "train" or "test".
        transform: Transform applied to PIL Images.
    """

    def __init__(self, root: str | Path, split: str = "train", transform=None):
        self.root = Path(root) / split
        self.transform = transform

        self.samples: list[tuple[Path, int]] = []
        for label, subdir in ((0, "REAL"), (1, "FAKE")):
            d = self.root / subdir
            if not d.exists():
                raise FileNotFoundError(f"Expected directory not found: {d}")
            for ext in ("*.jpg", "*.jpeg", "*.png"):
                for p in sorted(d.glob(ext)):
                    self.samples.append((p, label))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, label
