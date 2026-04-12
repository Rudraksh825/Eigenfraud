#!/bin/bash
# Extract GenImage archives in two parallel waves, then create unified training dir.
# Skips Midjourney (212G). Extracts 7 generators total.
#
# Wave 1 (small, ~127G): ADM + BigGAN + VQDM + glide  — run in parallel
# Wave 2 (large, ~273G): sdv4 + sdv5 + wukong         — run in parallel after wave 1 archives deleted
#
# NOTE: rm -rf does not work on this volume. Archives are deleted with rm -f (files only).
#       Partial extraction dirs should be renamed before running (not deleted).
#
# Usage: bash scripts/setup_genimage_parallel.sh

set -euo pipefail

BASE="/__modal/volumes/vo-WzpOG7GaLWKcTLwBAnypIi/Eigenfraud/data/raw"
GENIMAGE="$BASE/genimage"
LOG_DIR="/tmp/genimage_logs"
mkdir -p "$LOG_DIR"

extract_and_cleanup() {
    local gen="$1"
    local srcdir="$2"
    local archive="$3"
    local logfile="$LOG_DIR/${gen}.log"

    local zip_file="$BASE/$srcdir/$archive.zip"
    local inner_dir="$BASE/$srcdir/$archive"

    if [ -d "$inner_dir" ]; then
        echo "[$gen] Already extracted, skipping." | tee "$logfile"
        return 0
    fi

    echo "[$gen] Starting extraction..." | tee "$logfile"
    7z x "$zip_file" -o"$BASE/$srcdir/" >> "$logfile" 2>&1
    echo "[$gen] Extraction done. Deleting archive parts..." | tee -a "$logfile"
    rm -f "$BASE/$srcdir/$archive.zip" "$BASE/$srcdir/$archive".z[0-9][0-9]
    echo "[$gen] Done." | tee -a "$logfile"
}

free_space() {
    df -h "$BASE" | awk 'NR==2{print $4}'
}

# ---------------------------------------------------------------------------
# Wave 1: ADM (37G) + BigGAN (23G) + VQDM (36G) + glide (31G) = ~127G
# ---------------------------------------------------------------------------
echo "=== Wave 1: ADM + BigGAN + VQDM + glide (parallel) ==="
echo "Free space before wave 1: $(free_space)"

extract_and_cleanup ADM     ADM                    imagenet_ai_0508_adm &
extract_and_cleanup BigGAN  BigGAN                 imagenet_ai_0419_biggan &
extract_and_cleanup VQDM    VQDM                   imagenet_ai_0419_vqdm &
extract_and_cleanup glide   glide                  imagenet_glide &

wait
echo ""
echo "=== Wave 1 complete. Free space: $(free_space) ==="

# ---------------------------------------------------------------------------
# Wave 2: sdv4 (90G) + sdv5 (92G) + wukong (91G) = ~273G
# ---------------------------------------------------------------------------
echo ""
echo "=== Wave 2: sdv4 + sdv5 + wukong (parallel) ==="
echo "Free space before wave 2: $(free_space)"

extract_and_cleanup sdv4    stable_diffusion_v_1_4 imagenet_ai_0419_sdv4 &
extract_and_cleanup sdv5    stable_diffusion_v_1_5 imagenet_ai_0424_sdv5 &
extract_and_cleanup wukong  wukong                 imagenet_ai_0424_wukong &

wait
echo ""
echo "=== Wave 2 complete. Free space: $(free_space) ==="

# ---------------------------------------------------------------------------
# Create unified training directory
# ---------------------------------------------------------------------------
echo ""
echo "=== Creating unified training directory ==="

mkdir -p "$GENIMAGE/train" "$GENIMAGE/val"

# Real images: ADM's nature/ (shared ImageNet pool, one copy to avoid duplication)
ln -sfn "$BASE/ADM/imagenet_ai_0508_adm/train/nature" "$GENIMAGE/train/real"
ln -sfn "$BASE/ADM/imagenet_ai_0508_adm/val/nature"   "$GENIMAGE/val/real"

# Fake images: one symlink per generator
declare -A INNER=(
    ["ADM"]="ADM/imagenet_ai_0508_adm"
    ["BigGAN"]="BigGAN/imagenet_ai_0419_biggan"
    ["VQDM"]="VQDM/imagenet_ai_0419_vqdm"
    ["glide"]="glide/imagenet_glide"
    ["sdv4"]="stable_diffusion_v_1_4/imagenet_ai_0419_sdv4"
    ["sdv5"]="stable_diffusion_v_1_5/imagenet_ai_0424_sdv5"
    ["wukong"]="wukong/imagenet_ai_0424_wukong"
)

for gen in ADM BigGAN VQDM glide sdv4 sdv5 wukong; do
    ln -sfn "$BASE/${INNER[$gen]}/train/ai" "$GENIMAGE/train/$gen"
    ln -sfn "$BASE/${INNER[$gen]}/val/ai"   "$GENIMAGE/val/$gen"
    echo "  $gen symlinks created"
done

echo ""
echo "=== All done! Verify with: ==="
echo "cd /__modal/volumes/vo-WzpOG7GaLWKcTLwBAnypIi/Eigenfraud"
echo "python -c \"from src.dataset import FrequencyDataset; d = FrequencyDataset('data/raw/genimage/train'); print(d.label_counts())\""
