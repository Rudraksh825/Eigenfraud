#!/usr/bin/env bash
# Download datasets for Eigenfraud.
# Usage: bash setup_data.sh [--ff-url <faceforensics_server_url>]
#
# Prerequisites:
#   - CIFAKE: set KAGGLE_USERNAME and KAGGLE_KEY env vars (or place ~/.kaggle/kaggle.json)
#   - FaceForensics++: pass the server URL you received via the approval form with --ff-url

set -e

FF_URL=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --ff-url) FF_URL="$2"; shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

mkdir -p data/raw data/processed

# ── CIFAKE ────────────────────────────────────────────────────────────────────
echo "==> Downloading CIFAKE from Kaggle..."
kaggle datasets download -d birdy654/cifake-real-and-ai-generated-synthetic-images \
    -p data/raw/cifake --unzip
echo "    CIFAKE done."

# ── FaceForensics++ ───────────────────────────────────────────────────────────
if [[ -n "$FF_URL" ]]; then
    echo "==> Downloading FaceForensics++ (c23 videos, original + Deepfakes)..."
    echo "" | python faceforensics.txt data/raw/faceforensics \
        --dataset original --compression c23 --type videos --server EU
    echo "" | python faceforensics.txt data/raw/faceforensics \
        --dataset Deepfakes --compression c23 --type videos --server EU
    echo "    FaceForensics++ done."
else
    echo "==> Skipping FaceForensics++ (no --ff-url provided)."
    echo "    After form approval, run:"
    echo "      bash setup_data.sh --ff-url <server_url>"
fi

echo ""
echo "All done. Data is in data/raw/."
