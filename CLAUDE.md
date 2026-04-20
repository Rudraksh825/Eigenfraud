# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Journal Requirement (MANDATORY)

**`journal.md` must be updated at every step.**

Before finishing any response that involves a decision, code change, experiment, observed result, or noteworthy finding — append an entry to `journal.md`.

### Journal entry format

```markdown
### YYYY-MM-DD — <short title>

**What:** <what was done or decided>
**Why:** <the reasoning or motivation>
**Result / Status:** <outcome, or "in progress">
```

Include numbers for experimental results. Say why for reverted changes. Do not defer — write the entry in the same response as the work.

---

## Project: Eigenfraud

**The project has pivoted.** We are no longer building a detector. The new goal is to *audit* existing AI-generated image detectors — our own (CNN1D/CNN2D) and three external ones — to show that their reported performance is inflated by dataset construction artifacts (format and resolution biases) rather than genuine generative signatures.

**Thesis:** Existing detectors learn frequency shortcuts from JPEG-vs-PNG and resolution mismatches between real and fake splits, not from the generative process itself.

**Three contributions (see `PIVOT.md` for full plan):**
1. Measure spectral bias in CIFAKE, Defactify, GenImage datasets
2. Show format normalization reduces reported AUC (quantify the AUC delta)
3. Band ablation framework: zero out low/mid/high frequency bands and re-evaluate — shortcut-reliant detectors collapse on "high removed"

**Five detectors under evaluation:**
- CNN2D (ours, `results/best_2d.pt` or `results/genimage/best_2d.pt`)
- CNN1D (ours, `results/best_1d.pt` or `results/genimage/best_1d.pt`)
- CNNDetection — `PeterWang512/CNNDetection`
- FreqNet — `chuangchuangtan/FreqNet-DeepfakeDetection`
- UnivFD — `Yuheng-Li/UniversalFakeDetect`

**Scripts to be written (per PIVOT.md):**
- `scripts/normalize_dataset.py` — load → RGB → resize 256×256 bilinear → save as PNG (strip EXIF/alpha, no spatial augmentation)
- `scripts/eval_external.py` — unified wrapper: `--detector {cnndetection,freqnet,univfd}`, `--data`, `--weights`; outputs CSV `(path, label, score)`
- `scripts/band_ablation.py` — FFT → zero band (low/mid/high) → iFFT → clip [0,255] → save ablated test set

**Key cross-dataset finding motivating the pivot:** GenImage-trained CNN2D scores AUC 0.53 / MCC 0.05 on CIFAKE test — chance level. Spectral features do not generalize across generator families, consistent with shortcut learning.

**Label convention:** 0 = real, 1 = fake.

**Existing checkpoints (reference, not the focus of new work):** `results/best_{1d,2d}.pt` (CIFAKE-trained), `results/genimage/best_{1d,2d}.pt` (GenImage-trained).

---

## Commands

### Install dependencies
```bash
pip install -r requirements.txt
```

### Evaluate our own models (for baseline / pivot comparison)
```bash
# Evaluate on CIFAKE test set
python scripts/eval.py --checkpoint results/best_2d.pt --data data/raw/cifake/test --split all

# Compare both models
python scripts/eval.py --checkpoint results/best_1d.pt results/best_2d.pt --data data/raw/cifake/test --split all
```

`--split test` invokes `make_splits()` internally — only use when `--data` points to a single unsplit directory. For CIFAKE pre-split dirs, always use `--split all`. **`eval.py` has no `--cache` flag** — always pass `--data` pointing to raw image directory.

### Dataset sanity check
```bash
python -c "from src.dataset import FrequencyDataset; d = FrequencyDataset('data/raw/cifake/test'); print(d.label_counts())"
```

### Legacy: train our own models (no longer the primary workflow)
```bash
python scripts/train.py --model 2d --train-dir data/raw/cifake/train --val-dir data/raw/cifake/test
python scripts/train.py --model 1d --train-dir data/raw/cifake/train --val-dir data/raw/cifake/test
```
See the full training reference (multi-GPU, GenImage, cached spectra) in git history or `H100_TRAINING.md`.

---

## Architecture

The spectral pipeline (unchanged, still used by `eval.py` and the characterization notebooks):

```
PIL image
  └─ src/transforms.py: to_grayscale_array()       → float32 224×224
  └─ src/transforms.py: log_power_spectrum_2d()    → 2D log-power spectrum (224×224)
  └─ src/transforms.py: azimuthal_average_fast()   → 1D radial profile (112,)
  └─ src/dataset.py: FrequencyDataset.__getitem__  → (spectrum_2d [1,224,224], profile_1d [112], label)
```

**Dataset classes in `dataset.py`:**
- `FrequencyDataset` — raw images, FFT on-the-fly. Data layout: `root/real/` (label 0) + any other subdir (label 1, "nature" also treated as real).
- `CachedFrequencyDataset` — reads sharded `.npz` files via `manifest.csv`; LRU-cached per worker.
- `ParquetFrequencyDataset` — HuggingFace parquet (defactify_dataset); reads `Image` bytes + `Label_A`.

**Models (`src/models.py`):** `CNN1D` (~180k params, 1D radial profile input), `CNN2D` (~4M params, 2D spectrum input). Vanilla CNNs — no pretrained weights, no residual connections, no dropout. `build_model(model_type)` factory.

**Checkpoint format:** `{"epoch", "model_type", "model_state", "val_auc", "args"}`.

**`src/transforms.py` EDA utilities** (not in training path, useful for characterization notebooks): `spectral_residual()`, `compute_mean_spectrum()`.

**Corrupt-image retry:** `FrequencyDataset.__getitem__` silently skips corrupt files by cycling forward — index semantics are non-deterministic when corrupt files exist.

**Duplicate `collate_fn`:** `train.py` and `eval.py` each define their own identical `collate_fn`. If `FrequencyDataset.__getitem__` return signature changes, both must be updated.

---

## Key files
- `src/transforms.py` — all FFT/spectral math
- `src/dataset.py` — dataset classes and `make_splits`
- `src/models.py` — CNN1D, CNN2D, build_model
- `scripts/eval.py` — evaluation (AUC, accuracy, EER, MCC)
- `scripts/train.py` — training loop (legacy, AdamW + cosine LR, DDP-capable); WandB project `"specter"`
- `scripts/precompute.py` — pre-compute spectra to sharded `.npz` / `manifest.csv`
- `notebooks/` — `verifying.ipynb`, `ManualInspection.ipynb`, `infer_images.ipynb`, `fft_roundtrip.ipynb`
- `PIVOT.md` — full phased research plan with deliverables
- `journal.md` — complete project history

## Code Style
- No unnecessary abstractions or speculative features.
- No docstrings or comments added to unchanged code.
- Prefer editing existing files over creating new ones.

## Intentional omissions in CNN1D/CNN2D (do not add unless asked)
No data augmentation, no spectrum normalization, no dropout, no mixed precision, no gradient clipping.
