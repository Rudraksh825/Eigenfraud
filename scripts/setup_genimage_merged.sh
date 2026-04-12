#!/bin/bash
# Create data/raw/genimage_all/ — a single merged directory for all GenImage training.
# Uses make_splits() (--data mode) since we don't have dedicated val splits for fake images.
#
# Layout:
#   data/raw/genimage_all/
#       nature/   → data/raw/imagenet_nature/val/   (50k real ImageNet images)
#       ADM/      → data/raw/ADM/.../train/ai
#       BigGAN/   → data/raw/BigGAN/.../train/ai
#       VQDM/     → data/raw/VQDM/.../train/ai
#       glide/    → data/raw/glide/.../train/ai
#       sdv4/     → data/raw/stable_diffusion_v_1_4/.../train/ai  (after extraction)
#       sdv5/     → data/raw/stable_diffusion_v_1_5/.../train/ai  (after extraction)
#       wukong/   → data/raw/wukong/.../train/ai                  (after extraction)
#
# Usage: bash scripts/setup_genimage_merged.sh
# Run after: imagenet_nature/val/ is populated, wave 2 extraction is complete.

set -euo pipefail

BASE="/__modal/volumes/vo-WzpOG7GaLWKcTLwBAnypIi/Eigenfraud/data/raw"
OUT="$BASE/genimage_all"

mkdir -p "$OUT"

# Real images from ImageNet val
NATURE_DIR="$BASE/imagenet_nature/val"
if [ ! -d "$NATURE_DIR" ]; then
    echo "ERROR: $NATURE_DIR not found. Run /tmp/download_imagenet_nature.py first."
    exit 1
fi
ln -sfn "$NATURE_DIR" "$OUT/nature"
echo "nature → imagenet_nature/val"

# Fake images: one symlink per generator
declare -A FAKE_DIRS=(
    ["ADM"]="$BASE/ADM/imagenet_ai_0508_adm/train/ai"
    ["BigGAN"]="$BASE/BigGAN/imagenet_ai_0419_biggan/train/ai"
    ["VQDM"]="$BASE/VQDM/imagenet_ai_0419_vqdm/train/ai"
    ["glide"]="$BASE/glide/imagenet_glide/train/ai"
    ["sdv4"]="$BASE/stable_diffusion_v_1_4/imagenet_ai_0419_sdv4/train/ai"
    ["sdv5"]="$BASE/stable_diffusion_v_1_5/imagenet_ai_0424_sdv5/train/ai"
    ["wukong"]="$BASE/wukong/imagenet_ai_0424_wukong/train/ai"
)

for gen in ADM BigGAN VQDM glide sdv4 sdv5 wukong; do
    dir="${FAKE_DIRS[$gen]}"
    if [ -d "$dir" ]; then
        ln -sfn "$dir" "$OUT/$gen"
        count=$(ls "$dir" | wc -l)
        echo "$gen → $dir ($count images)"
    else
        echo "WARNING: $dir not found, skipping $gen"
    fi
done

echo ""
echo "=== Setup complete. Verify with: ==="
echo "python -c \"from src.dataset import FrequencyDataset; d = FrequencyDataset('$OUT'); print(d.label_counts())\""
echo ""
echo "=== Train with: ==="
echo "python scripts/train.py --model 2d --data $OUT --epochs 30 --out-dir results/genimage --wandb"
