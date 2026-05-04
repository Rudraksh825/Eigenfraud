"""
Trivial format baseline: predict fake/real from file extension alone.
PNG → score 1.0 (fake), JPEG/JPG → score 0.0 (real).

Usage:
    # Single dir with real/ and fake/ subdirs:
    python scripts/trivial_baseline.py \\
        --data data/raw/cifake/test \\
        --dataset cifake --condition original \\
        --out results/format_baseline_cifake_original.csv

    # Separate real/fake dirs (GenImage raw layout):
    python scripts/trivial_baseline.py \\
        --real data/raw/imagenet_nature/val \\
        --fake data/raw/ADM/imagenet_ai_0508_adm/train/ai \\
        --dataset genimage --condition original \\
        --out results/format_baseline_genimage_original.csv
"""

import argparse
import csv
import os
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score, matthews_corrcoef

IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tiff', '.gif'}
PNG_EXTS   = {'.png', '.webp'}   # treated as fake
JPEG_EXTS  = {'.jpg', '.jpeg', '.bmp', '.tiff', '.gif'}  # treated as real


def find_images(directory: Path) -> list:
    return [p for p in directory.rglob('*') if p.suffix.lower() in IMAGE_EXTS]


def collect_from_dir(data_dir: Path):
    """Collect (path, label) from a dir with real/ and fake/ subdirs."""
    samples = []
    for subdir in sorted(data_dir.iterdir()):
        if not subdir.is_dir():
            continue
        name = subdir.name.lower()
        if 'real' in name:
            label = 0
        elif 'fake' in name:
            label = 1
        else:
            continue
        for p in find_images(subdir):
            samples.append((p, label))
    return samples


def collect_from_split(real_dir: Path, fake_dir: Path):
    """Collect (path, label) from separate real/fake directories."""
    samples = []
    for p in find_images(real_dir):
        samples.append((p, 0))
    for p in find_images(fake_dir):
        samples.append((p, 1))
    return samples


def format_score(path: Path) -> float:
    """1.0 if PNG-like (predict fake), 0.0 if JPEG-like (predict real)."""
    ext = path.suffix.lower()
    if ext in PNG_EXTS:
        return 1.0
    return 0.0


def main():
    ap = argparse.ArgumentParser()
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument('--data', help='Dir with real/ and fake/ subdirs')
    group.add_argument('--real', help='Real images directory')
    ap.add_argument('--fake', help='Fake images directory (required with --real)')
    ap.add_argument('--dataset', required=True, help='Dataset name (for metrics row)')
    ap.add_argument('--condition', default='original', help='Condition name (for metrics row)')
    ap.add_argument('--out', required=True, help='Output CSV path (path,label,score)')
    args = ap.parse_args()

    if args.real and not args.fake:
        ap.error('--fake is required when using --real')

    if args.data:
        samples = collect_from_dir(Path(args.data))
    else:
        samples = collect_from_split(Path(args.real), Path(args.fake))

    if not samples:
        raise ValueError('No images found.')

    labels  = [s[1] for s in samples]
    scores  = [format_score(s[0]) for s in samples]
    paths   = [str(s[0]) for s in samples]

    auc = roc_auc_score(labels, scores)
    acc = accuracy_score(labels, [s > 0.5 for s in scores])
    mcc = matthews_corrcoef(labels, [s > 0.5 for s in scores])
    n_real = labels.count(0)
    n_fake = labels.count(1)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['path', 'label', 'score'])
        for p, lbl, sc in zip(paths, labels, scores):
            w.writerow([p, lbl, f'{sc:.1f}'])

    print(f'Dataset   : {args.dataset}  ({args.condition})')
    print(f'Images    : {len(samples)}  (real={n_real}, fake={n_fake})')
    print(f'AUC       : {auc:.4f}')
    print(f'Accuracy  : {acc:.4f}')
    print(f'MCC       : {mcc:.4f}')
    print()
    print('# metrics.csv row:')
    print(f'format_baseline,{args.dataset},{args.condition},{auc:.4f},{acc:.4f},{mcc:.4f},{n_real},{n_fake}')
    print(f'\nSaved per-image CSV: {out}')


if __name__ == '__main__':
    main()
