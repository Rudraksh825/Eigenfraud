"""
Zero out a radial frequency band in an image directory and save reconstructed images.

Band definitions (as fraction of r_max = H/2):
  low   r < 0.2 * r_max
  mid   0.2 * r_max <= r < 0.6 * r_max
  high  r >= 0.6 * r_max

Procedure per image: RGB → each channel FFT2 → zero band → iFFT2 → clip [0,255] → save PNG

Usage:
    python scripts/band_ablation.py \\
        --input /root/normalized/cifake/test \\
        --band high \\
        --output /root/ablated/cifake/high

    # Then evaluate with eval_external.py as normal:
    python scripts/eval_external.py --detector cnndetection \\
        --data /root/ablated/cifake/high \\
        --out results/cnndetection_cifake_ablated_high.csv
"""

import argparse
import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm

IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tiff'}


def make_band_mask(h: int, w: int, band: str) -> np.ndarray:
    cy, cx = h // 2, w // 2
    ys, xs = np.ogrid[:h, :w]
    r = np.sqrt((ys - cy) ** 2 + (xs - cx) ** 2)
    r_max = min(cy, cx)
    if band == 'low':
        mask = r < 0.2 * r_max
    elif band == 'mid':
        mask = (r >= 0.2 * r_max) & (r < 0.6 * r_max)
    elif band == 'high':
        mask = r >= 0.6 * r_max
    else:
        raise ValueError(f"Unknown band: {band!r}. Choose 'low', 'mid', or 'high'.")
    return mask


def ablate_image(img: Image.Image, band: str) -> Image.Image:
    arr = np.array(img.convert('RGB'), dtype=np.float32)
    h, w = arr.shape[:2]
    mask = make_band_mask(h, w, band)
    out = np.zeros_like(arr)
    for c in range(3):
        f = np.fft.fftshift(np.fft.fft2(arr[:, :, c]))
        f[mask] = 0.0
        ch = np.fft.ifft2(np.fft.ifftshift(f)).real
        out[:, :, c] = ch
    out = np.clip(out, 0, 255).astype(np.uint8)
    return Image.fromarray(out)


def process_dir(src: Path, dst: Path, band: str):
    dst.mkdir(parents=True, exist_ok=True)
    paths = [p for p in src.rglob('*') if p.suffix.lower() in IMAGE_EXTS]
    ok = skip = 0
    for p in tqdm(paths, desc=f"  {src.name}", ncols=80):
        try:
            img = Image.open(p)
            out = ablate_image(img, band)
            out.save(dst / (p.stem + '.png'), format='PNG')
            ok += 1
        except Exception as e:
            skip += 1
    print(f"  {src.name}: {ok} saved, {skip} skipped")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', required=True,
                    help='Directory with real/ and fake/ subdirs (normalized dataset)')
    ap.add_argument('--band', required=True, choices=['low', 'mid', 'high'])
    ap.add_argument('--output', required=True,
                    help='Output directory; will contain real/ and fake/ subdirs')
    args = ap.parse_args()

    src = Path(args.input)
    dst = Path(args.output)

    print(f"Band ablation: zeroing '{args.band}' frequencies")
    print(f"  Input : {src}")
    print(f"  Output: {dst}")

    for subdir in ['real', 'fake']:
        s = src / subdir
        if s.exists():
            process_dir(s, dst / subdir, args.band)
        else:
            print(f"  Warning: {s} not found, skipping")

    print("Done.")


if __name__ == '__main__':
    main()
