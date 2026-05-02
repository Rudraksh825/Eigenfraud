"""
Audit Defactify dataset construction artifacts from parquet shards.

For each split × label (real/fake), tabulates:
  - image count
  - format distribution (JPEG / PNG / other, from magic bytes)
  - resolution distribution (mode, unique count)
  - file size distribution (min, max, mean KB)

Usage:
  python scripts/audit_defactify.py --data defactify_dataset/data
"""

import argparse
import io
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq
from PIL import Image


LABEL_NAMES = {0: "REAL", 1: "FAKE"}


def detect_format(data: bytes) -> str:
    if data[:2] == b'\xff\xd8':
        return "JPEG"
    if data[:4] == b'\x89PNG':
        return "PNG"
    return "OTHER"


def audit_rows(image_col, label_col, target_label: int) -> dict:
    formats = Counter()
    resolutions = Counter()
    sizes_kb = []

    for img_struct, label in zip(image_col, label_col):
        if label != target_label:
            continue
        data = img_struct["bytes"]
        if data is None:
            continue
        sizes_kb.append(len(data) / 1024)
        formats[detect_format(data)] += 1
        try:
            with Image.open(io.BytesIO(data)) as img:
                resolutions[img.size] += 1
        except Exception:
            pass

    count = len(sizes_kb)
    if count == 0:
        return {"count": 0}

    most_common_res, most_common_res_count = resolutions.most_common(1)[0]
    return {
        "count": count,
        "formats": dict(formats.most_common()),
        "resolution_unique": len(resolutions),
        "resolution_mode": f"{most_common_res[0]}×{most_common_res[1]} ({most_common_res_count} imgs)",
        "resolution_all": [f"{w}×{h}" for (w, h), _ in resolutions.most_common(5)],
        "size_kb_min": min(sizes_kb),
        "size_kb_max": max(sizes_kb),
        "size_kb_mean": sum(sizes_kb) / count,
    }


def print_table(dataset: str, splits: dict) -> None:
    print(f"\n{'='*60}")
    print(f"  {dataset} — Dataset Audit")
    print(f"{'='*60}")
    for split_name, stats in splits.items():
        print(f"\n  [{split_name}]")
        if stats["count"] == 0:
            print("    (empty)")
            continue
        print(f"    Count       : {stats['count']:,}")
        fmt_str = ", ".join(
            f"{k} {v:,} ({100*v/stats['count']:.1f}%)"
            for k, v in stats["formats"].items()
        )
        print(f"    Formats     : {fmt_str}")
        print(f"    Resolutions : {stats['resolution_unique']} unique; mode = {stats['resolution_mode']}")
        if len(stats["resolution_all"]) > 1:
            print(f"                  top: {', '.join(stats['resolution_all'])}")
        print(
            f"    Size (KB)   : min={stats['size_kb_min']:.1f}  "
            f"max={stats['size_kb_max']:.1f}  "
            f"mean={stats['size_kb_mean']:.1f}"
        )
    print()


def audit_split(parquet_dir: Path, glob_pattern: str, split_name: str) -> dict[str, dict]:
    files = sorted(parquet_dir.glob(glob_pattern))
    print(f"  Reading {split_name} ({len(files)} shards)...", flush=True)

    accum = {0: {"formats": Counter(), "resolutions": Counter(), "sizes_kb": []},
             1: {"formats": Counter(), "resolutions": Counter(), "sizes_kb": []}}

    for f in files:
        table = pq.read_table(f, columns=["Image", "Label_A"])
        image_col = table["Image"].to_pylist()
        label_col = table["Label_A"].to_pylist()
        for img_struct, label in zip(image_col, label_col):
            if label not in accum or img_struct is None:
                continue
            data = img_struct.get("bytes") if isinstance(img_struct, dict) else None
            if not data:
                continue
            a = accum[label]
            a["sizes_kb"].append(len(data) / 1024)
            a["formats"][detect_format(data)] += 1
            try:
                with Image.open(io.BytesIO(data)) as img:
                    a["resolutions"][img.size] += 1
            except Exception:
                pass

    results = {}
    for label, label_name in LABEL_NAMES.items():
        a = accum[label]
        count = len(a["sizes_kb"])
        key = f"{split_name} / {label_name}"
        if count == 0:
            results[key] = {"count": 0}
            continue
        most_common_res, most_common_res_count = a["resolutions"].most_common(1)[0]
        results[key] = {
            "count": count,
            "formats": dict(a["formats"].most_common()),
            "resolution_unique": len(a["resolutions"]),
            "resolution_mode": f"{most_common_res[0]}×{most_common_res[1]} ({most_common_res_count} imgs)",
            "resolution_all": [f"{w}×{h}" for (w, h), _ in a["resolutions"].most_common(5)],
            "size_kb_min": min(a["sizes_kb"]),
            "size_kb_max": max(a["sizes_kb"]),
            "size_kb_mean": sum(a["sizes_kb"]) / count,
        }
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, type=Path,
                        help="Path to defactify_dataset/data directory")
    args = parser.parse_args()

    splits_config = [
        ("train", "train-*.parquet"),
        ("test",  "test-*.parquet"),
        ("validation", "validation-*.parquet"),
    ]

    all_stats = {}
    for split_name, pattern in splits_config:
        all_stats.update(audit_split(args.data, pattern, split_name))

    print_table("Defactify", all_stats)


if __name__ == "__main__":
    main()
