"""
Pre-compute frequency-domain representations for a dataset and save to disk.

Reads raw images from a FrequencyDataset-compatible directory, computes
spectrum_2d (float16, 224×224) and profile_1d (float32, 112), and saves each
as a single .npz file alongside a manifest CSV.

Output layout:
    <cache-dir>/
        manifest.csv          — columns: path,label,cache_file
        <sha1-of-path>.npz    — keys: s2d (1,H,W float16), p1d (L, float32), label (int)

Usage:
    python scripts/precompute.py --data data/raw/genimage_all --cache-dir data/cache/genimage_all
    python scripts/precompute.py --data data/raw/genimage_all --cache-dir data/cache/genimage_all --workers 16
"""

import argparse
import csv
import hashlib
import os
import sys
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.dataset import FrequencyDataset
from src.transforms import azimuthal_average_fast, log_power_spectrum_2d, to_grayscale_array


def _cache_key(path: str) -> str:
    return hashlib.sha1(path.encode()).hexdigest()


def _process_one(args):
    path, label, cache_dir, size = args
    key = _cache_key(path)
    out = os.path.join(cache_dir, key + ".npz")

    if os.path.exists(out):
        return path, label, out, True  # already cached

    try:
        img = Image.open(path)
        img.verify()
        img = Image.open(path)
        gray = to_grayscale_array(img, size=size)
        s2d = log_power_spectrum_2d(gray)          # (H, W) float32
        p1d = azimuthal_average_fast(s2d)          # (r_max,) float32
        np.savez_compressed(
            out,
            s2d=s2d[np.newaxis].astype(np.float16),  # (1,H,W) — halve disk use
            p1d=p1d,
            label=np.int8(label),
        )
        return path, label, out, False
    except Exception as e:
        return path, label, None, e


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True, help="Root dir for FrequencyDataset")
    p.add_argument("--cache-dir", required=True, help="Where to write .npz files")
    p.add_argument("--size", type=int, default=224)
    p.add_argument("--workers", type=int, default=8)
    args = p.parse_args()

    os.makedirs(args.cache_dir, exist_ok=True)

    print(f"Scanning {args.data} ...")
    ds = FrequencyDataset(root=args.data, size=args.size)
    print(f"  {ds.label_counts()}")
    samples = ds.samples  # list of (path, label)

    manifest_path = os.path.join(args.cache_dir, "manifest.csv")
    # Load existing manifest so we can resume
    done = {}
    if os.path.exists(manifest_path):
        with open(manifest_path) as f:
            for row in csv.DictReader(f):
                done[row["path"]] = row["cache_file"]
        print(f"  Resuming — {len(done)} already in manifest")

    todo = [(p, l, args.cache_dir, args.size) for p, l in samples if p not in done]
    print(f"  Processing {len(todo)} new images with {args.workers} workers ...")

    errors = 0
    with open(manifest_path, "a", newline="") as csvf:
        writer = csv.writer(csvf)
        if os.path.getsize(manifest_path) == 0:
            writer.writerow(["path", "label", "cache_file"])

        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futures = {ex.submit(_process_one, t): t for t in todo}
            pbar = tqdm(as_completed(futures), total=len(todo), unit="img")
            for fut in pbar:
                path, label, out, status = fut.result()
                if out is None:
                    errors += 1
                    pbar.set_postfix(errors=errors)
                else:
                    writer.writerow([path, label, out])

    print(f"\nDone. Errors: {errors}  Manifest: {manifest_path}")
    print(f"Cache size: {sum(os.path.getsize(os.path.join(args.cache_dir, f)) for f in os.listdir(args.cache_dir) if f.endswith('.npz')) / 1e9:.1f} GB")


if __name__ == "__main__":
    main()
