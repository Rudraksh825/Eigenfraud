"""
Step 0.3: Audit dataset construction artifacts.

For each split (real train, fake train, real test, fake test), tabulates:
  - image count
  - format distribution (JPEG / PNG / other)
  - resolution distribution (min, max, mode, unique count)
  - file size distribution (min, max, mean KB)

Usage:
  # CIFAKE
  python scripts/audit_dataset.py \
      --train-real data/processed/birdy654/cifake-real-and-ai-generated-synthetic-images/versions/3/train/REAL \
      --train-fake data/processed/birdy654/cifake-real-and-ai-generated-synthetic-images/versions/3/train/FAKE \
      --test-real  data/processed/birdy654/cifake-real-and-ai-generated-synthetic-images/versions/3/test/REAL \
      --test-fake  data/processed/birdy654/cifake-real-and-ai-generated-synthetic-images/versions/3/test/FAKE \
      --dataset CIFAKE
"""

import argparse
import os
import random
from collections import Counter
from pathlib import Path

from PIL import Image


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff", ".gif"}


def fmt_label(suffix: str) -> str:
    s = suffix.lower()
    if s in (".jpg", ".jpeg"):
        return "JPEG"
    if s == ".png":
        return "PNG"
    return suffix.upper().lstrip(".")


def audit_dir(path: Path, sample: int = 1000) -> dict:
    files = [f for f in path.iterdir() if f.suffix.lower() in IMAGE_EXTS]
    if not files:
        return {"count": 0}

    total = len(files)
    # Format and size: read from all files via stat (no PIL needed)
    formats = Counter()
    sizes_bytes = []
    for f in files:
        formats[fmt_label(f.suffix)] += 1
        sizes_bytes.append(f.stat().st_size)

    # Resolution: sample only
    sample_files = random.sample(files, min(sample, total))
    resolutions = Counter()
    for f in sample_files:
        try:
            with Image.open(f) as img:
                resolutions[img.size] += 1
        except Exception:
            pass

    sizes_kb = [s / 1024 for s in sizes_bytes]
    most_common_res, most_common_res_count = resolutions.most_common(1)[0]

    return {
        "count": total,
        "formats": dict(formats.most_common()),
        "resolution_unique": len(resolutions),
        "resolution_mode": f"{most_common_res[0]}×{most_common_res[1]} ({most_common_res_count} imgs, sampled {len(sample_files)})",
        "resolution_all": [f"{w}×{h}" for (w, h), _ in resolutions.most_common(5)],
        "size_kb_min": min(sizes_kb),
        "size_kb_max": max(sizes_kb),
        "size_kb_mean": sum(sizes_kb) / total,
    }


def print_table(dataset: str, splits: dict[str, dict]) -> None:
    print(f"\n{'='*60}")
    print(f"  {dataset} — Dataset Audit")
    print(f"{'='*60}")

    for split_name, stats in splits.items():
        print(f"\n  [{split_name}]")
        if stats["count"] == 0:
            print("    (empty)")
            continue
        print(f"    Count       : {stats['count']:,}")
        fmt_str = ", ".join(f"{k} {v:,} ({100*v/stats['count']:.1f}%)"
                            for k, v in stats["formats"].items())
        print(f"    Formats     : {fmt_str}")
        print(f"    Resolutions : {stats['resolution_unique']} unique; mode = {stats['resolution_mode']}")
        if len(stats["resolution_all"]) > 1:
            print(f"                  top: {', '.join(stats['resolution_all'])}")
        print(f"    Size (KB)   : min={stats['size_kb_min']:.1f}  "
              f"max={stats['size_kb_max']:.1f}  "
              f"mean={stats['size_kb_mean']:.1f}")

    print()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-real", required=True, type=Path)
    parser.add_argument("--train-fake", required=True, type=Path)
    parser.add_argument("--test-real",  required=True, type=Path)
    parser.add_argument("--test-fake",  required=True, type=Path)
    parser.add_argument("--dataset", default="Dataset")
    parser.add_argument("--sample", type=int, default=1000,
                        help="Number of images to sample per split for resolution check")
    args = parser.parse_args()

    splits = {
        "train / REAL": audit_dir(args.train_real, args.sample),
        "train / FAKE": audit_dir(args.train_fake, args.sample),
        "test  / REAL": audit_dir(args.test_real,  args.sample),
        "test  / FAKE": audit_dir(args.test_fake,  args.sample),
    }

    print_table(args.dataset, splits)


if __name__ == "__main__":
    main()
