"""
Re-encode images with swapped formats to test causal role of file format.

Real images (JPEG) → saved as PNG  (lossless)
Fake images (PNG)  → saved as JPEG at configurable quality (default 85)

Output layout: <output>/real/ and <output>/fake/

Usage:
    python scripts/format_swap.py \\
        --real data/raw/imagenet_nature/val \\
        --fake data/raw/ADM/imagenet_ai_0508_adm/train/ai \\
        --output /root/swapped/genimage

    # Then evaluate:
    python scripts/eval_external.py --detector cnndetection \\
        --data /root/swapped/genimage \\
        --out results/cnndetection_genimage_format_swapped.csv
"""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from PIL import Image
from tqdm import tqdm

IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tiff'}


def find_images(directory: Path) -> list:
    return sorted(p for p in directory.rglob('*') if p.suffix.lower() in IMAGE_EXTS)


def _convert_one(args):
    src_path, dst_path, out_format, jpeg_quality = args
    try:
        img = Image.open(src_path).convert('RGB')
        if out_format == 'PNG':
            img.save(dst_path, format='PNG')
        else:
            img.save(dst_path, format='JPEG', quality=jpeg_quality)
        return True
    except Exception:
        return False


def process_split(src: Path, dst: Path, out_format: str, jpeg_quality: int, workers: int):
    """Re-encode all images in src and save to dst in out_format."""
    dst.mkdir(parents=True, exist_ok=True)
    paths = find_images(src)
    ext = '.png' if out_format == 'PNG' else '.jpg'
    tasks = [(p, dst / (p.stem + ext), out_format, jpeg_quality) for p in paths]
    ok = skip = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(_convert_one, t): t for t in tasks}
        for f in tqdm(as_completed(futs), total=len(tasks),
                      desc=f'  {src.name} → {out_format}', ncols=80):
            if f.result():
                ok += 1
            else:
                skip += 1
    print(f'  {src.name}: {ok} saved, {skip} skipped')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--real', required=True, help='Real images directory (will be saved as PNG)')
    ap.add_argument('--fake', required=True, help='Fake images directory (will be saved as JPEG)')
    ap.add_argument('--output', required=True, help='Output root; creates real/ and fake/ subdirs')
    ap.add_argument('--jpeg-quality', type=int, default=85,
                    help='JPEG quality for re-encoding fake images (default: 85)')
    ap.add_argument('--workers', type=int, default=16,
                    help='Parallel worker threads (default: 16)')
    args = ap.parse_args()

    src_real = Path(args.real)
    src_fake = Path(args.fake)
    dst = Path(args.output)

    print(f'Format-swap experiment')
    print(f'  Real source : {src_real}  →  PNG (lossless)')
    print(f'  Fake source : {src_fake}  →  JPEG quality={args.jpeg_quality}')
    print(f'  Output      : {dst}')
    print()

    process_split(src_real, dst / 'real', out_format='PNG',  jpeg_quality=args.jpeg_quality, workers=args.workers)
    process_split(src_fake, dst / 'fake', out_format='JPEG', jpeg_quality=args.jpeg_quality, workers=args.workers)

    print()
    print('Done. Verify before running detectors:')
    print(f'  ls {dst}/real | head -3')
    print(f'  ls {dst}/fake | head -3')


if __name__ == '__main__':
    main()
