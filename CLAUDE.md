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

AI-generated image detector operating entirely in the frequency domain. Images are converted to 2D log-power spectra via FFT, then classified with CNNs. The hypothesis: generative models leave characteristic spectral fingerprints detectable even when pixel content looks realistic.

**Label convention:** 0 = real, 1 = fake.

**Checkpoints:** `results/best_1d.pt` and `results/best_2d.pt` — saved by best val AUC.

---

## Commands

### Install dependencies
```bash
pip install -r requirements.txt
```

### Download data
```bash
# CIFAKE (requires ~/.kaggle/kaggle.json or KAGGLE_USERNAME/KAGGLE_KEY env vars)
bash setup_data.sh

# FaceForensics++ (requires form approval)
bash setup_data.sh --ff-url <server_url>
```

### Train
```bash
# Pre-split data (CIFAKE layout with train/ and test/ dirs)
python scripts/train.py --model 1d --train-dir data/raw/cifake/train --val-dir data/raw/cifake/test
python scripts/train.py --model 2d --train-dir data/raw/cifake/train --val-dir data/raw/cifake/test

# Single directory (auto-split 80/10/10)
python scripts/train.py --model 1d --data data/raw/mydata

# With WandB logging
python scripts/train.py --model 2d --train-dir ... --val-dir ... --wandb
```

Key flags: `--epochs` (default 30), `--lr` (default 3e-4), `--batch-size` (default 64), `--size` (image resize before FFT, default 224), `--weight-decay` (default 1e-4).

### Evaluate
```bash
# Evaluate on a pre-split test directory (use --split all; --split test re-runs make_splits which is wrong for CIFAKE)
python scripts/eval.py --checkpoint results/best_2d.pt --data data/raw/cifake/test --split all

# Compare both models
python scripts/eval.py --checkpoint results/best_1d.pt results/best_2d.pt --data data/raw/cifake/test --split all
```

`--split test` invokes `make_splits()` internally — only use it when `--data` points to a single unsplit directory. For CIFAKE's pre-split dirs, always use `--split all`.

---

## Architecture

The pipeline across files:

```
PIL image
  └─ src/transforms.py: to_grayscale_array()       → float32 224×224
  └─ src/transforms.py: log_power_spectrum_2d()    → 2D log-power spectrum (224×224)
  └─ src/transforms.py: azimuthal_average_fast()   → 1D radial profile (112,)
  └─ src/dataset.py: FrequencyDataset.__getitem__  → (spectrum_2d [1,224,224], profile_1d [112], label)
```

`FrequencyDataset` always computes **both** representations per item. The training and eval scripts select which one to feed based on `model_type`:
- `"1d"` → feeds `profile_1d` to `CNN1D`
- `"2d"` → feeds `spectrum_2d` to `CNN2D`

**`make_splits()`** in `dataset.py` does a stratified 80/10/10 train/val/test split. Only used when `--data` is passed (single-dir mode); CIFAKE's own train/test dirs are used directly otherwise.

**Checkpoint format** (`.pt` files): `{"epoch", "model_type", "model_state", "val_auc", "args"}`. The `model_type` field drives model construction at eval time.

**Data layout expected by `FrequencyDataset`:**
```
root/
  real/   ← label 0
  fake/   ← label 1 (any non-"real" subdir name is treated as fake)
```

For multi-generator datasets (e.g. GenImage with subdirs like `sdv14/`, `dalle/`), pass `fake_dirs=["sdv14", "dalle"]` to `FrequencyDataset` to select specific generators, or omit it to treat all non-"real" subdirs as fake.

---

## Key files
- `src/transforms.py` — math layer: all FFT/spectral logic. `azimuthal_average_fast()` is used in dataset; `azimuthal_average()` is a loop-based reference. `spectral_residual()` and `compute_mean_spectrum()` are EDA-only utilities not in the training path.
- `src/dataset.py` — `FrequencyDataset`, `make_splits`
- `src/models.py` — `CNN1D`, `CNN2D`, `build_model` factory
- `scripts/train.py` — training loop (AdamW + cosine LR); WandB project name is `"specter"`
- `scripts/eval.py` — evaluation (AUC, accuracy, EER)
- `notebooks/` — exploratory analysis and sanity checks
- `journal.md` — full project history

## Code Style
- No unnecessary abstractions or speculative features.
- No docstrings or comments added to unchanged code.
- Prefer editing existing files over creating new ones.
