"""FaceForensics++ dataset (Rössler et al., ICCV 2019).

Expects pre-extracted frames. Extract with:
    ffmpeg -i <video> -q:v 2 <output_dir>/%05d.png

Directory layout:
    <root>/
        original_sequences/youtube/<compression>/videos/<id>/
        manipulated_sequences/<method>/<compression>/videos/<id>/

Compression: "c0" (lossless), "c23" (recommended), "c40" (social-media stress).
"""
from __future__ import annotations

from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset

METHODS = ["Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures"]

# Standard 720/140/140 video ID split
_SPLIT_SIZES = {"train": 720, "val": 140, "test": 140}


class FFPlusPlusDataset(Dataset):
    """FaceForensics++ frame dataset.

    Args:
        root: Path to the FaceForensics++ root directory.
        split: "train", "val", or "test" (uses first N video IDs).
        compression: "c0", "c23", or "c40".
        methods: Manipulation methods to include as fake samples.
        transform: Transform applied to PIL Images.
    """

    def __init__(
        self,
        root: str | Path,
        split: str = "train",
        compression: str = "c23",
        methods: list[str] | None = None,
        transform=None,
    ):
        self.root = Path(root)
        self.split = split
        self.compression = compression
        self.methods = methods or METHODS
        self.transform = transform

        # Determine video ID range for this split
        sizes = list(_SPLIT_SIZES.values())
        offsets = [0, sizes[0], sizes[0] + sizes[1]]
        split_idx = list(_SPLIT_SIZES.keys()).index(split)
        start = offsets[split_idx]
        count = sizes[split_idx]
        self._video_ids = set(f"{i:03d}" for i in range(start, start + count))

        self.samples: list[tuple[Path, int]] = []
        self._collect_real()
        for method in self.methods:
            self._collect_fake(method)

    def _collect_real(self):
        real_dir = self.root / "original_sequences" / "youtube" / self.compression / "videos"
        if not real_dir.exists():
            raise FileNotFoundError(f"Real frames directory not found: {real_dir}")
        for vid_dir in sorted(real_dir.iterdir()):
            if vid_dir.name not in self._video_ids:
                continue
            for p in sorted(vid_dir.glob("*.png")) + sorted(vid_dir.glob("*.jpg")):
                self.samples.append((p, 0))

    def _collect_fake(self, method: str):
        fake_dir = (
            self.root / "manipulated_sequences" / method / self.compression / "videos"
        )
        if not fake_dir.exists():
            raise FileNotFoundError(f"Fake frames directory not found: {fake_dir}")
        for vid_dir in sorted(fake_dir.iterdir()):
            if vid_dir.name not in self._video_ids:
                continue
            for p in sorted(vid_dir.glob("*.png")) + sorted(vid_dir.glob("*.jpg")):
                self.samples.append((p, 1))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, label
