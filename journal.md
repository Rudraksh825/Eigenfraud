# Eigenfraud — Project Journal

This journal logs every decision, implementation, result, discussion, and conclusion in this project — including things that are later reverted or abandoned. The goal is a complete, honest record of how the project evolved.

---

## Project Overview

**Eigenfraud** is an AI-generated image detector that works entirely in the frequency domain. Instead of looking at pixel-space features, it converts images to their 2D log-power spectra (via FFT) and trains CNNs on that representation. The hypothesis is that generative models leave characteristic spectral fingerprints — periodic artifacts, unusual frequency distributions — that are detectable even when pixel-space content looks convincing.

**Two model variants:**
- `CNN1D` — operates on the 1D azimuthally averaged radial power spectrum (shape: 112,). ~500k params. Captures only isotropic spectral structure.
- `CNN2D` — operates on the full 2D log-power spectrum heatmap (shape: 1×224×224). ~2M params. Captures both isotropic and anisotropic structure (e.g., grid artifacts at specific angles).

**Output:** Binary classification logits — 0 = real, 1 = fake.

---

## Repository Structure (as of 2026-03-27)

```
src/
  transforms.py   — math layer: image → grayscale 224×224 → 2D log-power spectrum → 1D azimuthal average
  dataset.py      — FrequencyDataset: wraps transforms into a PyTorch Dataset
  models.py       — CNN1D and CNN2D definitions + build_model factory
  __init__.py

scripts/
  train.py        — training loop (AdamW + cosine LR, WandB optional)
  eval.py         — evaluation: AUC, accuracy, EER

notebooks/
  verifying.ipynb       — early sanity check notebook
  ManualInspection.ipynb — manual inspection of model outputs / spectra

figures/
  fig1_prototype.png
  fig2_prototype.png
  mean_profiles.png
  mean_spectra_2d.png
  pipeline_per_image.png
  sanity_spectra.png

results/
  best_1d.pt      — best 1D CNN checkpoint (saved by val AUC)
  best_2d.pt      — best 2D CNN checkpoint (saved by val AUC)

faceforensics.txt — FaceForensics++ download script (from official repo)
setup_data.sh     — data download script (CIFAKE via Kaggle, FF++ optional)
setup.txt         — SSH setup instructions
requirements.txt  — torch, numpy, scipy, matplotlib, sklearn, tqdm, Pillow, h5py, wandb, kaggle
notes.txt         — informal module-level notes
running.txt       — training commands / run log
AllowClaude.txt   — bash path fix for Claude's shell
```

---

## Pipeline Details

**Per-image transform (from `src/transforms.py`):**
1. Load PIL image → convert to grayscale → resize to 224×224 float32
2. 2D FFT → `fftshift` (center DC) → compute `log(1 + |F|²)` → 2D log-power spectrum (224×224)
3. Azimuthal average: for each integer radius r, average all spectrum values at that distance from center → 1D profile of length 112 (= 224//2)

Two azimuthal average implementations:
- `azimuthal_average()` — loop-based, reference implementation
- `azimuthal_average_fast()` — vectorized with `np.bincount` (used in dataset)

**Dataset (`src/dataset.py`):**
- `FrequencyDataset`: expects `root/real/` and `root/fake/` (or any non-`real` subdir = fake)
- Returns `(spectrum_2d, profile_1d, label)` per item
- `make_splits()`: stratified train/val/test split using sklearn

**Training (`scripts/train.py`):**
- AdamW optimizer, lr=3e-4, weight_decay=1e-4
- CosineAnnealingLR scheduler
- Metrics: cross-entropy loss, AUC (via sklearn), accuracy
- Best checkpoint saved by val AUC → `results/best_{model}.pt`
- Optional WandB logging (`--wandb`)

**Evaluation (`scripts/eval.py`):**
- Loads checkpoint, runs on test split (or all data)
- Reports: AUC, Accuracy, EER (Equal Error Rate)

---

## Data

**CIFAKE** (primary dataset):
- Real images: CIFAR-10 real photos
- Fake images: Stable Diffusion v1.4 generated equivalents
- Source: Kaggle (`birdy654/cifake-real-and-ai-generated-synthetic-images`)
- Layout: `data/raw/cifake/train/` and `data/raw/cifake/test/`

**FaceForensics++** (planned/optional):
- Video-based face manipulation dataset
- Download script: `faceforensics.txt` (official FF++ downloader)
- Layout: `data/raw/faceforensics/`
- Datasets: original, Deepfakes, Face2Face, FaceShifter, FaceSwap, NeuralTextures
- Requires form approval for access

---

## Chronological Log

### 2026-03-27 — Project Setup & Initial Training

**Commits:**
- `idk` — initial state
- `reset` — reset
- `Data setup and init` — data download + initial code
- `transfer` — current state (code as described above, both checkpoints present)

**What was built:**
- Complete spectral transform pipeline (`transforms.py`)
- FrequencyDataset with both 1D and 2D output
- CNN1D (~500k params) and CNN2D (~2M params)
- Training script with AdamW + cosine LR
- Eval script with AUC + EER metrics

**Training runs logged in `running.txt`:**
```
# 1D model on processed CIFAKE
python scripts/train.py --model 1d \
  --train-dir data/processed/birdy654/cifake-real-and-ai-generated-synthetic-images/versions/3/train \
  --val-dir data/processed/birdy654/cifake-real-and-ai-generated-synthetic-images/versions/3/test

# 2D model on processed CIFAKE
python scripts/train.py --model 2d \
  --train-dir data/processed/birdy654/cifake-real-and-ai-generated-synthetic-images/versions/3/train \
  --val-dir data/processed/birdy654/cifake-real-and-ai-generated-synthetic-images/versions/3/test

# 2D model on raw CIFAKE
python scripts/train.py --model 2d --train-dir data/raw/cifake/train --val-dir data/raw/cifake/test
```

**Checkpoints saved:**
- `results/best_1d.pt` — 1D CNN best checkpoint
- `results/best_2d.pt` — 2D CNN best checkpoint

**Figures generated (EDA / sanity checks):**
- `figures/sanity_spectra.png` — verify spectral transform looks correct
- `figures/mean_spectra_2d.png` — mean 2D spectrum: real vs fake
- `figures/mean_profiles.png` — mean 1D profile: real vs fake
- `figures/pipeline_per_image.png` — per-image pipeline visualization
- `figures/fig1_prototype.png`, `fig2_prototype.png` — prototype figures

**Notable design decisions:**
- Use `azimuthal_average_fast()` (bincount-based) in the dataset for speed; kept the loop version as a reference
- Label convention: 0 = real, 1 = fake (anything not in a "real" subdir)
- Save checkpoint only when val AUC improves (not val loss) — AUC is more meaningful for imbalanced detection tasks

### 2026-03-28 — Journal Initialization

**Decision:** Start maintaining `journal.md` for complete project history.
**Reason:** To have a full record of decisions, results, and reasoning across sessions so context is never lost.

**Also created:** `CLAUDE.md` — instructs Claude to update the journal at every step.

### 2026-03-28 — CLAUDE.md Improved

**What:** Rewrote `CLAUDE.md` to include concrete training/eval/data commands, architecture data-flow description, and checkpoint format. Removed redundant project structure listing already discoverable from code.
**Why:** `/init` requested a more useful CLAUDE.md for future Claude Code sessions. The prior version lacked commands and cross-file architectural context.
**Result / Status:** Done.

### 2026-03-29 — GenImage Dataset Audit

**What:** Inspected `data/raw/GenImage/` to assess fitness for training.
**Why:** User plans to train 1D/2D CNNs on GenImage for cross-generator robustness.
**Result / Status:** Dataset is NOT usable for training. Four separate blockers found — see detailed findings below.

**Findings:**

1. **Wrong dataset — this is GenImage++, not GenImage.**
   The downloaded repo is `Lunahera/genimagepp` (a NeurIPS 2025 submission). This is a *test-only* evaluation benchmark, not the original GenImage training dataset. It has no training splits by design.

2. **Download massively incomplete — 13 of 21 blobs truncated at exactly 4 GiB.**
   Only 3 image archives are fully intact: `flux` (6k images), `flux_amateur` (6k), `flux_krea_amateur` (6k). The remaining subsets (sd3, flux_realistic, sd3_realistic, flux_multistyle, sdxl_multistyle, sd1.5_multistyle, flux_photo, plus all real-image blobs) are cut off. The 4 GiB cutoff strongly suggests a 32-bit file size limit in the download tool used.

3. **No real images present in working blobs.**
   All three intact archives contain only `1_fake` images. The `0_real` directories exist but are empty — the ImageNet real images appear to be in the truncated (broken) blobs.

4. **Data is not extracted; directory naming incompatible.**
   Images are packed inside `.tar.zstd` archives. FrequencyDataset cannot read them as-is. Additionally, the archive uses `0_real`/`1_fake` subdirectory names; FrequencyDataset checks for the exact name `"real"` so `0_real` would be mislabeled as fake.

**What to do:** Download the *original* GenImage dataset (`feifeiobama/GenImage` on HuggingFace, or the official repo at github.com/GenImage-Dataset/GenImage). It has 1.35M images from 8 generators (SD v1.4, SD v1.5, VQDM, Wukong, GLIDE, ADM, Midjourney, DALL-E 2) with proper train/val splits. GenImage++ can then be used as a hard generalization test after training.

---

### 2026-03-29 — CLAUDE.md Minor Improvements

**What:** Updated `CLAUDE.md` via `/init` — added `--weight-decay` to key flags, documented `fake_dirs` param for multi-generator datasets (e.g. GenImage), noted `spectral_residual`/`compute_mean_spectrum` EDA helpers in transforms.py, added `notebooks/` to key files, noted WandB project name is `"specter"`.
**Why:** Code inspection revealed these details were accurate but missing from the CLAUDE.md.
**Result / Status:** Done.

---

### 2026-03-30 — CLAUDE.md Minor Improvements

**What:** Updated `CLAUDE.md` via `/init` — clarified `--split all` vs `--split test` distinction for eval.py (wrong to use `test` on CIFAKE's pre-split dirs), noted `azimuthal_average_fast` is the production path and `azimuthal_average` is reference-only, clarified `spectral_residual`/`compute_mean_spectrum` are EDA-only.
**Why:** Code inspection revealed these nuances were missing and could cause subtle mistakes (e.g., accidentally re-splitting a pre-split test set).
**Result / Status:** Done.

---

## Open Questions / Future Work

- What are the actual AUC/accuracy numbers for the trained checkpoints? (need to run eval)
- How does the 2D model compare to 1D on CIFAKE?
- Does the spectral approach generalize to other generators (not just SD v1.4)?
- FaceForensics++ integration: frame extraction from video needed before FFT pipeline
- Consider: normalization of spectra per-image (subtract mean, divide by std) before feeding to CNN
- Consider: data augmentation in frequency domain (e.g., random rotation of spectrum)
