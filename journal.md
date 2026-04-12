# Eigenfraud — Project Journal

This journal logs every decision, implementation, result, discussion, and conclusion in this project — including things that are later reverted or abandoned. The goal is a complete, honest record of how the project evolved.

---

### 2026-04-12 — Checkpoint reorganization + CIFAKE training

**What:** Moved defactify checkpoints from `results/best_1d.pt` / `results/best_2d.pt` into `results/defactify/` subdir. Created `results/cifake/` subdir. Launched 1D and 2D training runs on CIFAKE (50k real / 50k fake, balanced, 30 epochs) with PIDs 11232/11233. CIFAKE downloaded fresh via kagglehub (symlink `/data/raw/cifake` had pointed to stale `/root/.cache` path from prior environment).
**Why:** Default `--out-dir results` would have overwritten defactify checkpoints when CIFAKE training completed. Subdirectory-per-dataset pattern (consistent with existing `results/genimage/`) keeps runs isolated and comparable.
**Result / Status:** In progress — logs at `/tmp/cifake_1d.log` and `/tmp/cifake_2d.log`. Defactify checkpoints safely archived: 1D val_auc=0.8439 (epoch 29), 2D val_auc=0.9049 (epoch 11).

---

### 2026-04-12 — CLAUDE.md improvements

**What:** Updated CLAUDE.md to document previously undocumented features: `scripts/precompute.py` (cache workflow), multi-GPU `torchrun` training, `CachedFrequencyDataset`/`ParquetFrequencyDataset`, `--cache`/`--parquet-dir`/`--class-weight`/`--seed` flags, and corrected the GenImage++ fix location (tuple at `dataset.py:79`, not `LABEL_MAP`).
**Why:** `/init` review revealed these omissions; future Claude instances would not know about the precompute path or DDP support.
**Result / Status:** Complete.

---

### 2026-04-11 — defactify_dataset parquet support

**What:** Added `ParquetFrequencyDataset` to `src/dataset.py` and `--parquet-dir` flag to `scripts/train.py` to support the defactify dataset (HuggingFace parquet format). Added `datasets` and `pyarrow` to `requirements.txt`.
**Why:** defactify_dataset stores images embedded in parquet files (HF Image feature with `bytes` key) rather than as files in `real/`/`fake/` subdirs. Uses `Label_A` (0=real, 1=fake) matching project convention. Pre-defined train/validation splits are used directly.
**Result / Status:** In progress — not yet trained.

---

### 2026-04-11 — Single-image inference notebook

**What:** Created `notebooks/infer_images.ipynb` to run `REAL.png` and `FAKE.png` through both checkpoints interactively.
**Why:** `eval.py` requires a labeled directory and only produces aggregate metrics — no per-image probabilities. Notebook gives visual output and per-image P(fake) scores.
**Result / Status:** Notebook has 4 sections: (1) display raw images, (2) show 2D log-power spectra + 1D radial profiles overlaid, (3) load both checkpoints and print an inference table with P(real)/P(fake)/pred/correct columns, (4) bar chart of P(fake) per image for 1D vs 2D CNN.

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

---

### 2026-04-02 — Comprehensive Progress Summary Generated

**What:** Created `specter_progress_summary.md` — a full reference document covering project structure, data pipeline (exact preprocessing steps, azimuthal average math, bin sizes), model architectures (exact layer counts, actual param counts), training setup, results from checkpoints, active plan for GenImage, and all known TODOs.
**Why:** User requested a handoff document detailed enough for a new Claude instance to immediately help write a paper or debug code.
**Result / Status:** Done. Key numbers surfaced: CNN1D actual param count is 180,002 (not ~500k as noted in journal); CNN2D is 4,078,050 (not ~2M). Val AUC: 1D = 0.9449 (epoch 29), 2D = 0.9650 (epoch 12). No test-set eval has been run. No cross-generator eval. No adversarial attacks implemented.

---

### 2026-04-10 — Wave 2 extraction restarted sequentially (PID 62093)

**What:** Killed parallel 7z extraction (PIDs 12818/12819/12820) after it stalled — only ~200 files added across 2 hours despite 93% CPU. Renamed partial dirs to `_partial3`. Restarted as sequential: sdv4 → sdv5 → wukong, one at a time, no `-mmt` flag. Log at `/tmp/genimage_wave2.log`.
**Why:** 3 parallel processes caused severe I/O contention on the Modal volume — CPU-bound decompression couldn't flush to disk. Sequential removes contention.
**Result / Status:** In progress. sdv4 extracting first.

---

### 2026-04-10 — Wave 2 GenImage extraction launched

**What:** Deleted leftover Wave 1 archives (ADM, BigGAN, VQDM, glide — ~127G) that were never cleaned up from prior run. Installed `p7zip-full` (7z wasn't on PATH). Launched Wave 2 extraction (sdv4, sdv5, wukong) in parallel background (PID 5630); logs at `/tmp/genimage_logs/{sdv4,sdv5,wukong}.log`. The unified `data/raw/genimage/` dir with symlinks was already created by the earlier (partially failed) script run — symlinks will resolve correctly once extraction completes.
**Why:** Wave 1 (ADM, BigGAN, VQDM, glide) was already extracted from prior session. Wave 2 hadn't been done. Midjourney excluded (212G, too large). p7zip-full needed because `unzip` doesn't handle split ZIPs.
**Result / Status:** In progress. After completion: verify with dataset sanity check, then train 2D and 1D models on GenImage with `--out-dir results/genimage`.

---

### 2026-04-10 — CLAUDE.md improvements

**What:** Updated `CLAUDE.md` with three additions: (1) documented `--out-dir` and `--workers` flags for `train.py`; (2) clarified that `FrequencyDataset` treats both `"real"` and `"nature"` as label 0 (the prior docs only mentioned `"real"`); (3) added a GenImage++ naming caveat (`0_real`/`1_fake` dirs not currently handled) and a dataset sanity-check one-liner.
**Why:** The `/init` slash command requested a review of CLAUDE.md. These were gaps between the docs and the actual code.
**Result / Status:** Done.

---

### 2026-04-02 — GenImage Extraction and Training Setup

**What:** Started extraction of 8 GenImage generator archives (ADM, BigGAN, VQDM, glide, sdv4, sdv5, wukong, Midjourney) and prepared unified training directory. Two changes made:
1. Patched `src/dataset.py`: added `"nature"` to the real-label check (GenImage uses `ai/` for fake and `nature/` for real, not `fake/`/`real/`).
2. Wrote `scripts/setup_genimage.sh`: extracts each generator archive via `7z x` (installed `p7zip-full`; `unzip` does not support multi-part ZIPs), deletes archive parts after extraction to reclaim space, then creates `data/raw/genimage/train/` and `data/raw/genimage/val/` with symlinks (ADM's `nature/` as shared `real/`; each generator's `ai/` as its own fake subdir).
**Why:** CIFAKE training done; next goal is GenImage cross-generator training. `unzip` failed on split ZIPs (`zipfile claims to be last disk of a multi-part archive`). Space strategy: extract-then-delete-archives keeps free space roughly constant (~382G free throughout). Real images: using only ADM's `nature/` to avoid 8× duplication (same ImageNet pool across all generators). This creates 1:8 real:fake imbalance — acceptable for now; can add class weights later.
**Result / Status:** Extraction running in background (PID 11485, log at `/tmp/genimage_setup.log`). ADM extracting first (37G, ~331k files). Training will start after all 8 generators are extracted.

---

### 2026-04-10 — GenImage real images missing; Wave 2 extraction restarted

**What:** Audited data state after VM restart. Found: (1) Wave 1 generators (ADM, BigGAN, VQDM, glide) extracted only `train/ai/` — archives deleted, cannot recover `val/` or `nature/`; (2) Wave 2 (sdv4, sdv5, wukong) archives still present, extraction never completed; (3) GenImage HuggingFace archives contain ONLY fake images per generator — real/nature images are a separate download; (4) No Kaggle/HF credentials present for re-downloading.
**Why:** Previous extraction (PID 62093) died when VM restarted. Inspecting the sdv4 archive confirmed the structural issue: each generator zip only has `{archive}/train/ai/`, not `train/nature/` or `val/`.
**Result / Status:** Wave 2 extraction restarted sequentially (PID 7279), log at `/tmp/genimage_wave2.log`. **Blocker resolved:** user logged into HuggingFace. Downloading imagenet-1k val (50k images) as real/nature images (PID 10040, log `/tmp/imagenet_download.log`).

---

### 2026-04-11 — ImageNet real images download + GenImage merged setup

**What:** Three changes to unblock GenImage training:
1. Downloading ImageNet-1k validation split (50k images, ~6.7GB) to `data/raw/imagenet_nature/val/` as the real/nature class — the GenImage HuggingFace archives only contain fake images, so ImageNet must be sourced separately.
2. Wrote `scripts/setup_genimage_merged.sh` to create `data/raw/genimage_all/` with symlinks: `nature/ → imagenet_nature/val/`, plus each generator's `train/ai/`.
3. Added `--class-weight` flag to `scripts/train.py` (computes loss weights as n/(2×count) per class) to handle the ~1:7 real:fake imbalance in the merged dataset.
**Why:** GenImage fake archives on HuggingFace are fake-images-only (no nature/ subdirectory). No dedicated val split exists for fake images, so training uses `--data` + `make_splits()`. Class imbalance (~50k real vs 300k–1M fake) requires weighted loss.
**Result / Status:** Complete. ImageNet download finished (50,000 images). `bash scripts/setup_genimage_merged.sh` run successfully: 50k real + 370k fake (ADM/BigGAN/VQDM/glide + sdv4-partial). Training launched: `python scripts/train.py --model 2d --data data/raw/genimage_all --epochs 30 --out-dir results/genimage --class-weight --wandb` (PID 25084, log `/tmp/train_2d.log`, running on CUDA). Wave 2 (sdv5/wukong) still extracting in background — will retrain or fine-tune once complete.


---

### 2026-04-11 — GenImage training path confirmed

**What:** Audited actual data state before starting training. `genimage/train/real` symlink is broken (points to non-existent `ADM/train/nature/`). All `genimage/val/` symlinks are broken (only `train/` was extracted from each archive, no `val/ai/` exists). Real images: 27,784 at `imagenet_nature/val/`. Pre-split `--train-dir`/`--val-dir` mode is not viable.
**Why:** Confirming what's actually present vs what the old symlink structure assumed.
**Result / Status:** Confirmed plan: (1) `bash scripts/setup_genimage_merged.sh` to build `genimage_all/` with working `nature/` symlink + all fake generator symlinks, (2) train with `--data data/raw/genimage_all --epochs 30 --out-dir results/genimage --class-weight --wandb`. No code changes needed.

---

### 2026-04-11 — Machine killed; handoff state for H100 node

**What:** Current machine being killed. Summary of state for the next machine:

- **ImageNet val**: ✅ Complete — 50,000 images at `data/raw/imagenet_nature/val/`
- **Wave 1 generators**: ✅ Fully extracted — ADM, BigGAN, VQDM, glide all at `data/raw/<gen>/.../train/ai/`
- **sdv4**: ⚠️ Partially extracted — 111,542 / ~135,000 images at `data/raw/stable_diffusion_v_1_4/imagenet_ai_0419_sdv4/train/ai/`. Stalled at 82% due to "No space left on device" errors (likely inode exhaustion on the volume). The `.zip` file is still present at `data/raw/stable_diffusion_v_1_4/imagenet_ai_0419_sdv4.zip` (2.86 GB).
- **sdv5, wukong**: ❌ Not started — zips present at `data/raw/stable_diffusion_v_1_5/` and `data/raw/wukong/`
- **genimage_all/**: Symlinks set up for nature + ADM + BigGAN + VQDM + glide + sdv4 (partial)
- **Training**: Not running. Was killed before H100 migration.
- **Code**: DDP + precompute pipeline complete (`scripts/precompute.py`, `scripts/train.py`, `src/dataset.py:CachedFrequencyDataset`)

**On the new H100 machine, follow `H100_TRAINING.md` in order:**
1. `bash scripts/setup_genimage_merged.sh` — refresh symlinks
2. Run `nohup bash /tmp/extract_wave2.sh > /tmp/genimage_wave2.log 2>&1 &` — skips already-extracted, continues sdv4 + sdv5 + wukong
3. `python scripts/precompute.py --data data/raw/genimage_all --cache-dir data/cache/genimage_all --workers 16`
4. `torchrun --nproc_per_node=8 scripts/train.py --model 2d --cache data/cache/genimage_all/manifest.csv --epochs 30 --out-dir results/genimage --class-weight`

**Why:** Volume appears to have hit inode limit (df shows 382G free but writes failing). New machine/volume should not have this issue.
**Result / Status:** Machine killed. Resuming on H100 node.

---

### 2026-04-11 — DDP + pre-computed spectra for 8× H100

**What:** Two changes to support full 8-GPU utilization:
1. `scripts/precompute.py` — scans a FrequencyDataset dir, computes spectrum_2d (float16) + profile_1d (float32) per image, saves `.npz` files + `manifest.csv`. Resumable, parallel, skips corrupt images. Usage: `python scripts/precompute.py --data data/raw/genimage_all --cache-dir data/cache/genimage_all --workers 16`
2. `CachedFrequencyDataset` added to `src/dataset.py` — reads from manifest.csv, zero FFT overhead at train time.
3. `scripts/train.py` upgraded to DDP: detects `RANK` env var (set by `torchrun`), wraps model in `DistributedDataParallel`, uses `DistributedSampler`, aggregates AUC/loss across ranks, saves checkpoints on rank 0 only.
**Why:** On-the-fly FFT on 420k images is ~5 hrs/epoch bottleneck. With 8 H100s the GPU would be idle waiting for CPU. Pre-compute pays ~1-2 hrs once, then each epoch takes minutes.
**Result / Status:** Code complete. Launch sequence: (1) run precompute.py on the data node, (2) `torchrun --nproc_per_node=8 scripts/train.py --model 2d --cache data/cache/genimage_all/manifest.csv --epochs 30 --out-dir results/genimage --class-weight --wandb`

---

### 2026-04-11 — Corrupt image crash + skip-on-error fix

**What:** Training (PID 26141) crashed mid-epoch-1 with `PIL.UnidentifiedImageError` on `BigGAN/116_biggan_00094.png`. Fixed `src/dataset.py` `__getitem__` to wrap image open + `img.verify()` in a try/except loop that advances to the next sample on any PIL error. Killed crashed processes and relaunched (PID 48222).
**Why:** GenImage archives contain at least one corrupt/truncated PNG. Without the fix any corrupt file kills the entire training run.
**Result / Status:** Fix applied; training restarted cleanly on CUDA.
