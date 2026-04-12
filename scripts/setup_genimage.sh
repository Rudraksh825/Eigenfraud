#!/bin/bash
# Extract GenImage archives and create a unified training directory.
#
# Archives use split-zip format: {archive}.zip + {archive}.z01 ... {archive}.zNN
# Directory structure inside each archive:
#   {archive}/train/ai/      ← fake (label 1)
#   {archive}/train/nature/  ← real (label 0)
#   {archive}/val/ai/
#   {archive}/val/nature/
#
# Unified layout created under data/raw/genimage/:
#   train/real -> ADM's train/nature/   (shared ImageNet pool; one copy to avoid duplicates)
#   train/ADM  -> ADM's train/ai/
#   train/BigGAN -> BigGAN's train/ai/
#   ... etc.
#   val/real -> ADM's val/nature/
#   val/ADM  -> ADM's val/ai/
#   ... etc.

set -euo pipefail

BASE="/__modal/volumes/vo-WzpOG7GaLWKcTLwBAnypIi/Eigenfraud/data/raw"
GENIMAGE="$BASE/genimage"

# Maps: short name → (source subdir, archive basename)
declare -A SRCDIR=(
    ["ADM"]="ADM"
    ["BigGAN"]="BigGAN"
    ["VQDM"]="VQDM"
    ["glide"]="glide"
    ["Midjourney"]="Midjourney"
    ["sdv4"]="stable_diffusion_v_1_4"
    ["sdv5"]="stable_diffusion_v_1_5"
    ["wukong"]="wukong"
)

declare -A ARCHIVE=(
    ["ADM"]="imagenet_ai_0508_adm"
    ["BigGAN"]="imagenet_ai_0419_biggan"
    ["VQDM"]="imagenet_ai_0419_vqdm"
    ["glide"]="imagenet_glide"
    ["Midjourney"]="imagenet_midjourney"
    ["sdv4"]="imagenet_ai_0419_sdv4"
    ["sdv5"]="imagenet_ai_0424_sdv5"
    ["wukong"]="imagenet_ai_0424_wukong"
)

# Extract in order from smallest to largest (saves largest for last when most space is freed)
EXTRACTION_ORDER="ADM BigGAN VQDM glide sdv4 sdv5 wukong Midjourney"

for gen in $EXTRACTION_ORDER; do
    archive="${ARCHIVE[$gen]}"
    srcdir="${SRCDIR[$gen]}"
    zip_file="$BASE/$srcdir/$archive.zip"
    inner_dir="$BASE/$srcdir/$archive"

    if [ -d "$inner_dir" ]; then
        echo "[$gen] Already extracted, skipping."
        continue
    fi

    if [ ! -f "$zip_file" ]; then
        echo "[$gen] WARNING: $zip_file not found, skipping."
        continue
    fi

    echo ""
    echo "[$gen] Extracting $archive.zip ..."
    echo "  Free space before: $(df -h "$BASE" | awk 'NR==2{print $4}')"

    7z x "$zip_file" -o"$BASE/$srcdir/"

    echo "[$gen] Extraction done. Removing archive parts to free space..."
    rm -f "$BASE/$srcdir/$archive.zip"
    # Remove all split parts (.z01 through .z99)
    rm -f "$BASE/$srcdir/$archive".z[0-9][0-9]

    echo "  Free space after: $(df -h "$BASE" | awk 'NR==2{print $4}')"
    echo "[$gen] Done."
done

echo ""
echo "=== All extractions complete. Creating unified training directory... ==="

mkdir -p "$GENIMAGE/train" "$GENIMAGE/val"

# Real images: use ADM's nature/ (all generators share the same ImageNet pool)
ln -sfn "$BASE/ADM/imagenet_ai_0508_adm/train/nature" "$GENIMAGE/train/real"
ln -sfn "$BASE/ADM/imagenet_ai_0508_adm/val/nature"   "$GENIMAGE/val/real"
echo "real/ symlinks created (pointing to ADM/nature/)"

# Fake images: one symlink per generator
for gen in $EXTRACTION_ORDER; do
    archive="${ARCHIVE[$gen]}"
    srcdir="${SRCDIR[$gen]}"
    inner_dir="$BASE/$srcdir/$archive"

    if [ ! -d "$inner_dir/train/ai" ]; then
        echo "WARNING: $inner_dir/train/ai not found, skipping $gen fake symlink"
        continue
    fi

    ln -sfn "$inner_dir/train/ai" "$GENIMAGE/train/$gen"
    ln -sfn "$inner_dir/val/ai"   "$GENIMAGE/val/$gen"
    echo "$gen symlinks created"
done

echo ""
echo "=== Setup complete. Verify with: ==="
echo "cd /__modal/volumes/vo-WzpOG7GaLWKcTLwBAnypIi/Eigenfraud"
echo "python -c \"from src.dataset import FrequencyDataset; d = FrequencyDataset('$GENIMAGE/train'); print(d.label_counts())\""
