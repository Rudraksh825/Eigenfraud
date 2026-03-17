"""GenImage dataset (Zhu et al., NeurIPS 2023).

Directory layout expected:
    <root>/
        <generator>/
            train/
                ai/       # AI-generated images
                nature/   # Real ImageNet images
            val/
                ai/
                nature/

Training protocol: train on a single generator (default: "Stable Diffusion V1.4"),
evaluate cross-generator on all eight.
"""
from __future__ import annotations

from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset

GENERATORS = [
    "Stable Diffusion V1.4",
    "Stable Diffusion V1.5",
    "Midjourney",
    "ADM",
    "GLIDE",
    "Wukong",
    "VQDM",
    "BigGAN",
]


class GenImageDataset(Dataset):
    """Single-generator or multi-generator GenImage dataset.

    Args:
        root: Path to the top-level GenImage directory.
        generators: Generator name(s) to include. Use a single string for
            standard single-source training; pass a list for cross-generator
            evaluation.
        split: "train" or "val".
        transform: Transform applied to PIL Images.
    """

    def __init__(
        self,
        root: str | Path,
        generators: str | list[str] = "Stable Diffusion V1.4",
        split: str = "train",
        transform=None,
    ):
        self.root = Path(root)
        self.split = split
        self.transform = transform

        if isinstance(generators, str):
            generators = [generators]
        self.generators = generators

        self.samples: list[tuple[Path, int, str]] = []  # (path, label, generator)
        for gen in self.generators:
            gen_dir = self.root / gen / split
            for label, subdir in ((1, "ai"), (0, "nature")):
                d = gen_dir / subdir
                if not d.exists():
                    raise FileNotFoundError(f"Expected directory not found: {d}")
                for ext in ("*.jpg", "*.jpeg", "*.png", "*.JPEG", "*.PNG"):
                    for p in sorted(d.glob(ext)):
                        self.samples.append((p, label, gen))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label, generator = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, label, generator
