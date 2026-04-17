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

**Checkpoints:** Saved by best val AUC to `--out-dir` (default `results/`). CIFAKE checkpoints land at `results/best_{1d,2d}.pt`; GenImage at `results/genimage/best_{1d,2d}.pt`. Per-dataset result dirs: `results/cifake/`, `results/defactify/`, `results/genimage/`.

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

### Set up GenImage dataset
```bash
# Extract individual generators and create merged symlink directory
bash scripts/setup_genimage.sh          # sequential, single generator
bash scripts/setup_genimage_parallel.sh # parallel extraction (faster)
bash scripts/setup_genimage_merged.sh   # create data/raw/genimage_all/ with symlinks across generators
```
`setup_genimage_merged.sh` is re-runnable — safe to call again after new generators are extracted to add symlinks.

See `H100_TRAINING.md` for the full multi-GPU GenImage training runbook (extract → precompute → DDP train).

### Pre-compute spectra (recommended for large datasets)
```bash
# Compute and cache sharded .npz files (resumable — skips already-cached files)
python scripts/precompute.py --data data/raw/genimage_all \
    --cache-dir data/cache/genimage_sharded --workers 16
```
Output: `<cache-dir>/manifest.csv` + `shard_NNNNN.npz` files (~354 shards of 1000 images each). Pass the manifest to `--cache` during training to skip on-the-fly FFT. Shard size is configurable via `--shard-size` (default 1000).

**Shard format:** each `.npz` contains `s2d` (N,1,H,W float16) and `p1d` (N,L float32). Manifest columns: `path, label, shard_file, shard_idx`. `CachedFrequencyDataset` loads shards on demand with an 8-shard LRU cache per DataLoader worker.

**Inode cap note:** the Modal volume at `/__modal/volumes/vo-WzpOG7GaLWKcTLwBAnypIi` has a 500k-inode limit. The sharded layout uses ~354 files vs the old 354k, staying well within the cap. Always verify with `df -i /__modal/volumes/vo-WzpOG7GaLWKcTLwBAnypIi` before running.

### GenImage training (current path — on-the-fly FFT, no cache)
```bash
# 2D (main detector)
nohup python scripts/train.py --model 2d --data data/raw/genimage_all \
    --epochs 30 --out-dir results/genimage --class-weight --wandb \
    > /tmp/genimage_2d.log 2>&1 & echo $!

# 1D (for 1D-vs-2D comparison)
nohup python scripts/train.py --model 1d --data data/raw/genimage_all \
    --epochs 30 --out-dir results/genimage --class-weight --wandb \
    > /tmp/genimage_1d.log 2>&1 & echo $!
```
Dataset at time of writing: ~50k real (ImageNet val) + ~304k fake across 4 generators (ADM, BigGAN, VQDM, glide). `sdv4` symlink is currently broken; `sdv5`/`wukong` archives are unextracted (Wave 2 never finished). `--class-weight` handles the ~1:6 real/fake imbalance. Monitor with `tr '\r' '\n' < /tmp/genimage_2d.log | grep "^Epoch" | tail`.

### Train
```bash
# Pre-split data (CIFAKE layout with train/ and test/ dirs)
python scripts/train.py --model 1d --train-dir data/raw/cifake/train --val-dir data/raw/cifake/test
python scripts/train.py --model 2d --train-dir data/raw/cifake/train --val-dir data/raw/cifake/test

# Single directory (auto-split 80/10/10)
python scripts/train.py --model 1d --data data/raw/mydata

# Pre-computed cache (fastest — no FFT at training time)
python scripts/train.py --model 2d --cache data/cache/genimage_all/manifest.csv

# HuggingFace parquet dataset (defactify_dataset layout)
python scripts/train.py --model 2d --parquet-dir defactify_dataset/data/

# Multi-GPU DDP (8 GPUs)
torchrun --nproc_per_node=8 scripts/train.py --model 2d --cache data/cache/genimage_all/manifest.csv

# With WandB logging
python scripts/train.py --model 2d --train-dir ... --val-dir ... --wandb
```

Key flags: `--epochs` (default 30), `--lr` (default 3e-4), `--batch-size` (default 64), `--size` (image resize before FFT, default 224), `--weight-decay` (default 1e-4), `--out-dir` (checkpoint output dir, default `results`), `--workers` (default 4), `--seed` (default 42), `--class-weight` (inverse-frequency loss weighting for imbalanced data).

### Evaluate
```bash
# Evaluate on a pre-split test directory (use --split all; --split test re-runs make_splits which is wrong for CIFAKE)
python scripts/eval.py --checkpoint results/best_2d.pt --data data/raw/cifake/test --split all

# Compare both models
python scripts/eval.py --checkpoint results/best_1d.pt results/best_2d.pt --data data/raw/cifake/test --split all

# GenImage checkpoints (--out-dir was results/genimage, so checkpoint path differs)
python scripts/eval.py --checkpoint results/genimage/best_2d.pt results/genimage/best_1d.pt \
    --data data/raw/genimage_all --split all
```

`--split test` invokes `make_splits()` internally — only use it when `--data` points to a single unsplit directory. For CIFAKE's pre-split dirs, always use `--split all`.

**`eval.py` has no `--cache` flag.** It always uses `FrequencyDataset` via `--data`. To evaluate a model trained with `--cache`, pass `--data` pointing to the raw image directory instead.

### Monitor background training logs
tqdm uses `\r` which makes raw `cat` unreadable. Filter to epoch summaries:
```bash
cat /tmp/train_2d.log | tr '\r' '\n' | grep "^Epoch"
```

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

**Three dataset classes in `dataset.py`:**
- `FrequencyDataset` — raw images, FFT computed on-the-fly
- `CachedFrequencyDataset` — reads sharded `.npz` files via `manifest.csv` from `precompute.py`; manifest columns: `path, label, shard_file, shard_idx`; shards LRU-cached per worker (512-entry `functools.lru_cache` on `_load_shard`)
- `ParquetFrequencyDataset` — HuggingFace-style parquet files (defactify_dataset); reads `Image` bytes + `Label_A` columns

**`make_splits()`** in `dataset.py` does a stratified 80/10/10 train/val/test split. Only used when `--data` or `--cache` is passed (single-dir/cache mode); CIFAKE's own train/test dirs are used directly otherwise. Works with both `FrequencyDataset` and `CachedFrequencyDataset` (both have `.samples` with `(path, label, ...)` tuples). Type hint says `FrequencyDataset` but is intentionally duck-typed.

**Checkpoint format** (`.pt` files): `{"epoch", "model_type", "model_state", "val_auc", "args"}`. The `model_type` field drives model construction at eval time.

**Data layout expected by `FrequencyDataset`:**
```
root/
  real/   ← label 0  (also "nature" — used by some GenImage splits)
  fake/   ← label 1 (any subdir not named "real" or "nature" is treated as fake)
```

For multi-generator datasets (e.g. GenImage with subdirs like `sdv14/`, `dalle/`), pass `fake_dirs=["sdv14", "dalle"]` to `FrequencyDataset` to select specific generators, or omit it to treat all non-real subdirs as fake.

**GenImage++ naming caveat:** GenImage++ uses `0_real`/`1_fake` dirs. `0_real` is not currently recognized as real and would be mislabeled. If working with GenImage++, add `"0_real"` to the `("real", "nature")` tuple at `dataset.py:80`.

**Corrupt-image retry:** `FrequencyDataset.__getitem__` silently skips corrupt files by cycling forward through `.samples`. This means index semantics are non-deterministic when corrupt files exist — items returned may not correspond to the requested index.

**`--val-dir` is optional with `--train-dir`:** if omitted, `train.py` silently falls back to using `--train-dir` as val dir (line 100). Always pass `--val-dir` explicitly.

**Duplicate `collate_fn`:** `train.py` and `eval.py` each define their own identical `collate_fn`. If `FrequencyDataset.__getitem__` return signature changes (e.g. adding a field), both must be updated.

**Dataset sanity check:**
```bash
python -c "from src.dataset import FrequencyDataset; d = FrequencyDataset('data/raw/genimage/train'); print(d.label_counts())"
```

---

## Key files
- `src/transforms.py` — math layer: all FFT/spectral logic. `azimuthal_average_fast()` is used in dataset; `azimuthal_average()` is a loop-based reference. `spectral_residual()` and `compute_mean_spectrum()` are EDA-only utilities not in the training path.
- `src/dataset.py` — `FrequencyDataset`, `CachedFrequencyDataset`, `ParquetFrequencyDataset`, `make_splits`
- `src/models.py` — `CNN1D` (~180k params), `CNN2D` (~4M params), `build_model` factory. Both are custom vanilla CNNs: no pretrained weights, no residual/skip connections, no dropout.
- `scripts/precompute.py` — pre-compute and cache spectra to `.npz`; produces `manifest.csv` for `--cache` training mode
- `scripts/train.py` — training loop (AdamW + cosine LR, DDP-capable); WandB project name is `"specter"`
- `scripts/eval.py` — evaluation (AUC, accuracy, EER)
- `notebooks/` — `verifying.ipynb` (spectral sanity checks), `ManualInspection.ipynb` (model output inspection), `infer_images.ipynb` (inference on individual images)
- `journal.md` — full project history

## Training pipeline — intentional omissions

These are deliberate design choices, not gaps to fill:
- **No data augmentation** — `transform` is always `None` in all training scripts.
- **No spectrum normalization** — raw log-power values feed directly into the CNN (no per-image mean/std normalization).
- **No dropout** — neither CNN1D nor CNN2D has dropout layers.
- **No mixed precision** — no `torch.cuda.amp` usage.
- **No gradient clipping** — `optimizer.step()` is called without clipping.

Do not add any of these unless explicitly asked.

## Code Style
- No unnecessary abstractions or speculative features.
- No docstrings or comments added to unchanged code.
- Prefer editing existing files over creating new ones.
