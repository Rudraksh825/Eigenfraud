"""
Pre-compute frequency-domain representations and save as sharded .npz files.

Each shard holds --shard-size images (default 1000), so ~354 shards for the full
GenImage dataset instead of 354k individual files.  This stays well within the
Modal volume's 500k-inode cap.

Output layout:
    <cache-dir>/
        manifest.csv              — columns: path, label, shard_file, shard_idx
        shard_00000.npz           — keys: s2d (N,1,H,W float16), p1d (N,L float32)
        shard_00001.npz
        ...

Resumable: paths already in manifest.csv are skipped on re-run.

Usage:
    python scripts/precompute.py --data data/raw/genimage_all \\
        --cache-dir data/cache/genimage_sharded --workers 16
"""

import argparse
import csv
import os
import random
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.dataset import FrequencyDataset
from src.transforms import azimuthal_average_fast, log_power_spectrum_2d, to_grayscale_array


# ---------------------------------------------------------------------------
# Per-image worker (no disk I/O — returns arrays)
# ---------------------------------------------------------------------------

def _compute_one(args):
    path, label, size = args
    try:
        img = Image.open(path)
        img.verify()
        img = Image.open(path)
        gray = to_grayscale_array(img, size=size)
        s2d = log_power_spectrum_2d(gray)
        p1d = azimuthal_average_fast(s2d)
        return path, label, s2d[np.newaxis].astype(np.float16), p1d, None
    except Exception as e:
        return path, label, None, None, e


# ---------------------------------------------------------------------------
# Shard writer
# ---------------------------------------------------------------------------

def _write_shard(shard_path, s2d_list, p1d_list):
    np.savez_compressed(
        shard_path,
        s2d=np.stack(s2d_list),   # (N, 1, H, W) float16
        p1d=np.stack(p1d_list),   # (N, L) float32
    )


def _chunks(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data",       required=True, help="Root dir for FrequencyDataset")
    p.add_argument("--cache-dir",  required=True, help="Where to write shard files")
    p.add_argument("--size",       type=int, default=224)
    p.add_argument("--workers",    type=int, default=8)
    p.add_argument("--shard-size", type=int, default=1000, help="Images per shard")
    p.add_argument("--seed",       type=int, default=42)
    args = p.parse_args()

    os.makedirs(args.cache_dir, exist_ok=True)
    manifest_path = os.path.join(args.cache_dir, "manifest.csv")

    # Load already-done paths from existing manifest
    done_paths: set[str] = set()
    next_shard_idx = 0
    if os.path.exists(manifest_path):
        with open(manifest_path) as f:
            for row in csv.DictReader(f):
                done_paths.add(row["path"])
                # Track highest shard number so new shards don't collide
                shard_name = Path(row["shard_file"]).stem  # e.g. "shard_00042"
                idx = int(shard_name.split("_")[1])
                next_shard_idx = max(next_shard_idx, idx + 1)
        print(f"Resuming — {len(done_paths)} images already in manifest, "
              f"next shard index: {next_shard_idx}")

    # Scan dataset
    print(f"Scanning {args.data} ...")
    ds = FrequencyDataset(root=args.data, size=args.size)
    print(f"  {ds.label_counts()}")

    todo = [(path, label, args.size)
            for path, label in ds.samples
            if path not in done_paths]
    print(f"  {len(todo)} images to process ({len(done_paths)} already cached)")

    if not todo:
        print("Nothing to do.")
        return

    # Shuffle so each shard gets a mix of real/fake
    rng = random.Random(args.seed)
    rng.shuffle(todo)

    errors = 0
    shard_counter = next_shard_idx

    with open(manifest_path, "a", newline="") as csvf:
        writer = csv.writer(csvf)
        if os.path.getsize(manifest_path) == 0:
            writer.writerow(["path", "label", "shard_file", "shard_idx"])

        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            chunks = list(_chunks(todo, args.shard_size))
            pbar = tqdm(chunks, unit="shard", desc="Shards")
            for chunk in pbar:
                # Compute all images in this chunk in parallel
                futures = [ex.submit(_compute_one, item) for item in chunk]
                results = [f.result() for f in futures]

                good_paths, good_labels, good_s2d, good_p1d = [], [], [], []
                for path, label, s2d, p1d, err in results:
                    if err is not None:
                        errors += 1
                    else:
                        good_paths.append(path)
                        good_labels.append(label)
                        good_s2d.append(s2d)
                        good_p1d.append(p1d)

                if not good_s2d:
                    pbar.set_postfix(errors=errors)
                    continue

                shard_name = f"shard_{shard_counter:05d}.npz"
                shard_path = os.path.join(args.cache_dir, shard_name)
                _write_shard(shard_path, good_s2d, good_p1d)

                for within_idx, (path, label) in enumerate(zip(good_paths, good_labels)):
                    writer.writerow([path, label, shard_path, within_idx])
                csvf.flush()

                shard_counter += 1
                pbar.set_postfix(shards=shard_counter, errors=errors)

    n_shards = shard_counter - next_shard_idx
    shard_bytes = sum(
        os.path.getsize(os.path.join(args.cache_dir, f))
        for f in os.listdir(args.cache_dir)
        if f.startswith("shard_") and f.endswith(".npz")
    )
    print(f"\nDone. New shards: {n_shards}  Errors: {errors}  "
          f"Total cache size: {shard_bytes / 1e9:.1f} GB  "
          f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
