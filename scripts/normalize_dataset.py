"""
Normalize a real/fake image dataset to canonical format:
  Load → RGB (strip alpha/EXIF) → resize N×N → save as PNG

File-based usage (CIFAKE, GenImage):
    python scripts/normalize_dataset.py \
        --input data/raw/cifake/test \
        --output data/normalized/cifake/test

    python scripts/normalize_dataset.py \
        --input-real data/raw/imagenet_nature/val \
        --input-fake data/raw/ADM/imagenet_ai_0508_adm/train/ai \
        --output data/normalized/genimage

Defactify parquet usage:
    python scripts/normalize_dataset.py \
        --parquet defactify_dataset/data \
        --split test \
        --output data/normalized/defactify/test

Output layout: <output>/real/  and  <output>/fake/
"""

import argparse
import io
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from PIL import Image
from tqdm import tqdm

IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tiff', '.gif'}
RESIZE_METHODS = {
    'bilinear': Image.BILINEAR,
    'bicubic':  Image.BICUBIC,
    'lanczos':  Image.LANCZOS,
    'nearest':  Image.NEAREST,
}


def normalize_image(img: Image.Image, size: int, method: int) -> Image.Image:
    return img.convert('RGB').resize((size, size), method)


def _normalize_one(args):
    src_path, dst_path, size, method = args
    try:
        out = normalize_image(Image.open(src_path), size, method)
        out.save(dst_path, format='PNG')
        return True
    except Exception:
        return False


def process_file_dir(src: Path, dst: Path, size: int, method: int, label: str,
                     workers: int = 1) -> int:
    dst.mkdir(parents=True, exist_ok=True)
    paths = [p for p in src.rglob('*') if p.suffix.lower() in IMAGE_EXTS]
    tasks = [(p, dst / (p.stem + '.png'), size, method) for p in paths]
    ok = skip = 0
    if workers > 1:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = {ex.submit(_normalize_one, t): t for t in tasks}
            for f in tqdm(as_completed(futs), total=len(tasks), desc=f"  {label}", ncols=80):
                if f.result():
                    ok += 1
                else:
                    skip += 1
    else:
        for t in tqdm(tasks, desc=f"  {label}", ncols=80):
            if _normalize_one(t):
                ok += 1
            else:
                skip += 1
    print(f"  {label}: {ok} saved, {skip} skipped")
    return ok


def process_files_by_subdir(input_dir: Path, output_dir: Path, size: int, method: int,
                            workers: int = 1):
    """Auto-detect real/fake subdirs from naming convention."""
    found = False
    for subdir in sorted(input_dir.iterdir()):
        if not subdir.is_dir():
            continue
        name = subdir.name.lower()
        if 'real' in name:
            process_file_dir(subdir, output_dir / 'real', size, method, 'real', workers)
            found = True
        elif 'fake' in name:
            process_file_dir(subdir, output_dir / 'fake', size, method, 'fake', workers)
            found = True
    if not found:
        raise ValueError(
            f"No subdirectories containing 'real' or 'fake' found in {input_dir}"
        )


def process_parquet(parquet_dir: Path, split: str, output_dir: Path, size: int, method: int):
    import pyarrow.parquet as pq
    shards = sorted(parquet_dir.glob(f"{split}-*.parquet"))
    if not shards:
        raise FileNotFoundError(f"No {split}-*.parquet files in {parquet_dir}")

    real_dir = output_dir / 'real'
    fake_dir = output_dir / 'fake'
    real_dir.mkdir(parents=True, exist_ok=True)
    fake_dir.mkdir(parents=True, exist_ok=True)

    counts = {0: 0, 1: 0}
    skip = 0

    for shard in tqdm(shards, desc=f"  {split} shards", ncols=80):
        table = pq.read_table(shard, columns=["Image", "Label_A"])
        for img_s, lbl in zip(table["Image"].to_pylist(), table["Label_A"].to_pylist()):
            raw = (img_s.get("bytes") if isinstance(img_s, dict) else None) if img_s else None
            if not raw:
                skip += 1
                continue
            try:
                out = normalize_image(Image.open(io.BytesIO(raw)), size, method)
                label = 0 if lbl == 0 else 1
                dst = real_dir if label == 0 else fake_dir
                out.save(dst / f"{split}_{counts[label]:07d}.png", format='PNG')
                counts[label] += 1
            except Exception:
                skip += 1

    print(f"  real: {counts[0]} saved, fake: {counts[1]} saved, skipped: {skip}")


def main():
    parser = argparse.ArgumentParser(
        description='Normalize real/fake image dataset to canonical PNG format.'
    )
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument('--input',      type=Path, help='Dir with real/fake subdirs')
    src.add_argument('--input-real', type=Path, help='Real images dir (use with --input-fake)')
    src.add_argument('--parquet',    type=Path, help='Defactify-style parquet dir')

    parser.add_argument('--input-fake', type=Path,
                        help='Fake images dir (required with --input-real)')
    parser.add_argument('--split',   default='test',
                        help='Parquet split prefix (default: test)')
    parser.add_argument('--output',  type=Path, required=True,
                        help='Output root; creates real/ and fake/ subdirs')
    parser.add_argument('--size',    type=int, default=256,
                        help='Resize to size×size (default: 256)')
    parser.add_argument('--method',  default='bilinear', choices=list(RESIZE_METHODS.keys()),
                        help='Resize method (default: bilinear)')
    parser.add_argument('--workers', type=int, default=1,
                        help='Parallel worker threads (default: 1; use 8-16 for faster I/O)')
    args = parser.parse_args()

    if args.input_real and not args.input_fake:
        parser.error('--input-real requires --input-fake')

    method = RESIZE_METHODS[args.method]
    print(f"Output : {args.output}")
    print(f"Size   : {args.size}×{args.size}  method={args.method}")

    if args.input:
        process_files_by_subdir(args.input, args.output, args.size, method, args.workers)
    elif args.input_real:
        process_file_dir(args.input_real, args.output / 'real', args.size, method, 'real', args.workers)
        process_file_dir(args.input_fake, args.output / 'fake', args.size, method, 'fake', args.workers)
    else:
        process_parquet(args.parquet, args.split, args.output, args.size, method)


if __name__ == '__main__':
    main()
