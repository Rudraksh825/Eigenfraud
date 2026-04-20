# Eigenfraud — Project Journal

This journal logs every decision, implementation, result, discussion, and conclusion in this project — including things that are later reverted or abandoned. The goal is a complete, honest record of how the project evolved.

---

### 2026-04-19 — Step 0.3: CIFAKE dataset audit

**What:** Wrote `scripts/audit_dataset.py` and ran it on all four CIFAKE splits (train/REAL, train/FAKE, test/REAL, test/FAKE). Results:

| Split | Count | Format | Resolution | Size (KB) mean |
|-------|-------|--------|------------|----------------|
| train / REAL | 50,000 | JPEG 100% | 32×32 (uniform) | 0.9 |
| train / FAKE | 50,000 | JPEG 100% | 32×32 (uniform) | 0.9 |
| test / REAL | 10,000 | JPEG 100% | 32×32 (uniform) | 0.9 |
| test / FAKE | 10,000 | JPEG 100% | 32×32 (uniform) | 0.9 |

**Why:** PIVOT.md step 0.3 — tabulate format, resolution, file size per split to identify construction artifacts.
**Result / Status:** CIFAKE shows **no format or resolution bias** between real and fake splits — both are uniform JPEG 32×32 with nearly identical file sizes. If our detector learned shortcuts on CIFAKE, they are not from format differences. Defactify not yet audited (no local raw data). GenImage audit requires remote access.

---

### 2026-04-19 — Full project pivot confirmed; CLAUDE.md rewritten

**What:** Confirmed complete pivot away from building a detector. CLAUDE.md was rewritten from scratch to reflect the new mission: auditing existing detectors (CNN1D, CNN2D, CNNDetection, FreqNet, UnivFD) for frequency shortcuts induced by dataset construction artifacts. Old training-centric commands demoted to "legacy" section. New primary work: `normalize_dataset.py`, `eval_external.py`, `band_ablation.py` (all to be written per PIVOT.md phases).
**Why:** User confirmed the pivot is complete — the cross-dataset collapse (AUC 0.53, MCC 0.05 for GenImage→CIFAKE) motivates the shortcut audit thesis. Detector building is no longer the goal.
**Result / Status:** CLAUDE.md updated. No code written yet — PIVOT.md phases begin from Phase 0.

---

### 2026-04-17 — FFT round-trip explainer notebook

**What:** Created `notebooks/fft_roundtrip.ipynb` using `1.png` and `2.png` (same building, different colors). Four sections: (1) FFT decomposition into magnitude + phase + log-power spectrum, (2) full round-trip showing original → grayscale → spectrum → phase → reconstructed, (3) reconstruction error verification (should be ~1e-10), (4) phase vs magnitude experiment — magnitude-only (zero phase), phase-only (unit magnitude), and phase-swapped reconstructions. Demonstrates that phase carries spatial structure while magnitude carries texture/energy, and that our pipeline discards phase.
**Why:** User wants a visual explanation for peers showing the FFT is invertible and what the log-power spectrum represents.
**Result / Status:** Notebook created, not yet run.

---

### 2026-04-17 — Spectral residual inverse-FFT notebook

**What:** Created `notebooks/spectral_residual_spatial.ipynb`. Computes mean log-power spectra for real and fake (CIFAKE, N=500 each), takes the residual (Fake − Real), then inverse-FFTs it to pixel space. The inverse uses `|ΔF| = sqrt(expm1(|ΔS|))` with zero phase. Also includes a per-image version that computes a single fake image's spectral deviation from the real mean and overlays it as a heatmap on the original.
**Why:** User wanted to see what the frequency-domain residual looks like mapped back to image/pixel space — i.e. the spatial pattern of generator artifacts.
**Result / Status:** Notebook created, not yet run.

---

### 2026-04-17 — Cross-dataset and in-distribution eval results with MCC

**What:** Ran eval.py (now with MCC) on CIFAKE test set (10k real / 10k fake) using both GenImage-trained and CIFAKE-trained checkpoints.

**Results:**

| Train set | Model | AUC | Acc | EER | MCC |
|-----------|-------|-----|-----|-----|-----|
| CIFAKE (in-dist) | 2D CNN | 0.9525 | 0.8633 | 0.1150 | 0.7405 |
| CIFAKE (in-dist) | 1D CNN | 0.9399 | 0.7740 | 0.1325 | 0.6027 |
| GenImage (cross) | 2D CNN | 0.5303 | 0.5242 | 0.4753 | 0.0483 |
| GenImage (cross) | 1D CNN | 0.4615 | 0.4672 | 0.5180 | -0.0890 |

**Why:** Assess whether spectral fingerprints generalize across generators. GenImage was trained on ADM/BigGAN/VQDM/glide; CIFAKE test contains SD v1.4 fakes.
**Result / Status:** Complete. Cross-dataset models are at chance level — spectral features learned from one generator family do not transfer to another. In-distribution MCC of 0.74 (2D) is solid but below the comparison paper's 0.87+ (which uses pretrained features). The 2D→1D gap (0.74 vs 0.60 MCC) confirms anisotropic spectral structure contributes meaningfully.

---

### 2026-04-17 — Added MCC metric to eval.py

**What:** Added Matthews Correlation Coefficient (MCC) to `scripts/eval.py` output alongside AUC, accuracy, and EER. Uses `sklearn.metrics.matthews_corrcoef` on the thresholded predictions (p >= 0.5).
**Why:** The comparison paper (Table 1) reports MCC as its primary metric. Adding it to eval output enables direct comparison.
**Result / Status:** Done. Both single-checkpoint and multi-checkpoint display formats updated.

---

### 2026-04-17 — Full paper revision: tone, GenImage results, citations, image placeholders

**What:** Revised all six paper sections plus main.bib:
- **Tone/confidence:** Removed grand "we present/introduce/demonstrate" framing throughout; replaced with hedged exploratory language ("we find", "results suggest", "preliminary observations"). Added explicit class-project scope disclaimer in intro. Softened contributions from formal "three contributions" to plain "what this project does."
- **GenImage results added:** Sec 4 now reports completed GenImage experiment (CNN2D test AUC=0.9990, acc=98.18%, EER=1.02%; CNN1D val AUC=0.9375). Four-generator subset caveat (ADM/BigGAN/VQDM/glide only) and JPEG-bias warning (citing Grommelt et al. 2024) made prominent.
- **Results table updated:** Added GenImage column; added dagger footnotes about val-vs-test distinction and JPEG confound.
- **Planned experiments removed:** The E1 LOGO/E2 JPEG/E3 PGD sections were cut; GenImage is now a completed result section. Adversarial robustness mentioned briefly in limitations only.
- **Image placeholders added:** Four `\TODO{}` figure placeholders with generation instructions: (1) pipeline figure (use existing `figures/pipeline_per_image.png`), (2) EDA spectra (use `figures/mean_spectra_2d.png` + `figures/mean_profiles.png`), (3) results bar chart (matplotlib code snippet in comments), (4) Grad-CAM heatmap (generation instructions in comments).
- **Bibliography:** Replaced placeholder `main.bib` with 30 real entries covering all cited works with correct authors/venues/years: Wang 2020, Durall 2020, Ho 2020, Rombach 2022, Corvi 2023, Ricker 2024, Frank 2020, Zhang 2019, Ojha 2023, Cozzolino 2024, Radford 2021, Oquab 2023, Grommelt 2024, Carlini 2020, Bird/Lotfi 2024 (CIFAKE), GenImage NeurIPS 2023, GenImage++, Defactify 2025, FF++ 2019, van der Schaaf 1996, Odena 2016, Loshchilov AdamW 2019, Loshchilov SGDR 2017, Madry 2018, Selvaraju 2017, Synthbuster 2023, SPAI 2025.
- **Author updated:** main.tex author field set to Rudraksh Awasthi.
**Why:** User requested: less formal/confident tone, honesty about class-project scope, updated GenImage results, image placeholders with descriptions, and proper complete bibliography.
**Result / Status:** All six .tex files rewritten; main.bib fully populated; preamble.tex has graphicx added. Paper should compile (with \TODO{} placeholders visible). Figures need to be copied to the paper directory or \graphicspath set.

---

### 2026-04-16 — Paper: remove all em-dashes from prose

**What:** Replaced every `---` em-dash in the six content `.tex` files with natural prose alternatives (commas, colons, parentheses, or restructured sentences). Remaining `---` occurrences are all inside LaTeX comments and do not appear in the rendered PDF.
**Why:** User preference for less formal punctuation style.
**Result / Status:** Complete. Zero em-dashes in rendered text.

---

### 2026-04-16 — Paper: remove CVPR formatting, add SynthID section

**What:** Three changes to the LaTeX paper:
1. Removed CVPR submission formatting: switched `\usepackage[review]{cvpr}` to `\usepackage{cvpr}` in `main.tex`, and removed `\paperID`, `\confName`, `\confYear` definitions. This eliminates the line numbers on the sides and the CVPR header/footer — the paper now renders as a clean standalone document.
2. Added `\subsection{SynthID Watermark Detection (Exploratory)}` in `sec/4_experiments.tex`, placed after the GenImage section and before Interpretability. Describes the finding that SynthID's frequency-domain perturbation is detectable as a spectral anomaly using the existing CNN2D pipeline, without access to the watermarking key.
3. Added a SynthID paragraph to Related Work (`sec/2_formatting.tex`) contextualising the watermarking literature, and updated the conclusion (`sec/5_conclusion.tex`) to mention SynthID detection as an additional finding.
4. Added `synthid2023` and `cox2007watermarking` entries to `main.bib`.

**Why:** User: (a) is using the CVPR template for its formatting, not for submission — CVPR headers/line numbers should be removed; (b) has been testing whether Eigenfraud's spectral pipeline can detect SynthID watermarks in Gemini-generated images without the key, and it is working well — this result should be included in the paper.
**Result / Status:** Complete. All .tex and .bib files updated.

---

### 2026-04-17 — CLAUDE.md corrections via /init

**What:** Fixed three inaccuracies in CLAUDE.md: (1) LRU cache size "8-shard" → "512-entry `functools.lru_cache`" (matching actual code); (2) GenImage++ line reference `dataset.py:81` → `dataset.py:80`; (3) clarified checkpoint locations — added per-dataset dirs (`results/cifake/`, `results/defactify/`, `results/genimage/`).
**Why:** `/init` skill review of live source revealed mismatches between documentation and code.
**Result / Status:** CLAUDE.md updated, no code changed.

---

### 2026-04-14 — GenImage 1D training result

**What:** Ran `train.py --model 1d` on GenImage (cache `/tmp/genimage_shards/manifest.csv`, 30 epochs, batch 128, `--class-weight`). SSH session closed; log lost but checkpoint survived on Modal volume.
**Why:** 1D vs 2D ablation for the paper — quantifies the value of anisotropic spectral structure.
**Result / Status:** Best val AUC = **0.9375** at epoch 29. Compare: 2D CNN = 0.9990. Gap of ~6 AUC points. The 1D model loses directional/grid-artifact structure by collapsing the spectrum to a radial average — this gap is the ablation evidence that anisotropic features matter.

---

### 2026-04-14 — CLAUDE.md improvements

**What:** Added three missing documentation items to CLAUDE.md: (1) `eval.py` has no `--cache` flag — must use `--data` pointing to raw images for eval even when training used `--cache`; (2) added GenImage eval command example with correct checkpoint path (`results/genimage/best_*.pt`); (3) noted `precompute.py --shard-size` flag (default 1000).
**Why:** These were genuine gaps that could cause confusion — especially the eval/cache limitation.
**Result / Status:** Done.

---

### 2026-04-14 — GenImage 2D test eval results

**What:** Ran `eval.py` on the GenImage test split (35,406 samples, held-out 10% via `make_splits`). Dataset: 50k real + 304k fake across 4 generators (ADM, BigGAN, VQDM, glide).
**Why:** Get formal test metrics for the paper — val AUC from training is not sufficient.
**Result / Status:** AUC=0.9990, Accuracy=98.18%, EER=1.02%. Near-perfect detection on in-distribution GenImage generators. Cross-dataset eval (GenImage↔CIFAKE) still pending.

---

### 2026-04-14 — Fixed shard loading performance (decompression + LRU cache)

**What:** Training on local `/tmp` shards was still 5.74s/batch. Root cause: (1) `_load_shard` LRU cache was only 8 slots — with 355 shards and random access, nearly every access was a cache miss; (2) shards were written with `savez_compressed`, so every cold load required decompression. Fix: increased `maxsize` from 8 to 512 in `src/dataset.py:126`, and converted all 355 local `/tmp` shards from compressed to uncompressed in-place (26 GB → 34 GB) using a 16-worker parallel script. Volume copy remains compressed/intact.
**Why:** After the first cold load of each shard, all subsequent accesses are memory hits. With maxsize=512 and 355 shards, nothing is ever evicted.
**Result / Status:** Shards decompressed. Training not yet restarted. Recommend `--workers 2 --batch-size 128`.

---

### 2026-04-14 — Copied shards to local disk to fix training I/O bottleneck

**What:** Training on `CachedFrequencyDataset` from the Modal volume was running at 15s/batch (4426 batches/epoch × 30 epochs = weeks). Copied all 355 shard `.npz` files (26 GB) from `data/cache/genimage_sharded/` on the Modal network volume to `/tmp/genimage_shards/` on the instance's local SSD. Rewrote manifest at `/tmp/genimage_shards/manifest.csv` with updated absolute shard paths.
**Why:** The Modal volume is network-attached storage — random shard reads with an 8-slot LRU cache cause constant cache misses and network fetches. Local SSD eliminates that latency.
**Result / Status:** Copy complete. Training not yet restarted. Command: `python scripts/train.py --model 2d --cache /tmp/genimage_shards/manifest.csv --epochs 30 --out-dir results/genimage --class-weight --workers 8 --batch-size 128`. Note: `/tmp` is ephemeral — if the instance restarts, re-copy from the volume.

---

### 2026-04-14 — CLAUDE.md improvements via /init

**What:** Made three targeted edits to `CLAUDE.md`: (1) fixed wrong line number in GenImage++ naming caveat (`dataset.py:79` → `dataset.py:81`); (2) added explicit note that CNN1D and CNN2D are custom vanilla CNNs with no pretrained weights, no skip connections, and no dropout; (3) added a "Training pipeline — intentional omissions" section documenting that no augmentation, normalization, dropout, mixed precision, or gradient clipping are used, with an explicit instruction not to add them without being asked.
**Why:** `/init` command triggered codebase review. The line number was wrong after verifying against the actual source. The omissions section prevents future Claude instances from "helpfully" adding these.
**Result / Status:** Done.

---

### 2026-04-13 — Wrote explainer.md

**What:** Created `explainer.md` — a comprehensive document covering the full pipeline (grayscale conversion, FFT, log-power spectrum, azimuthal averaging), the math behind each CNN layer type (conv, BN, ReLU, pooling, cross-entropy), both model architectures with parameter counts, what the models learn, four interpretability/visualisation methods (spectral residual, Grad-CAM, first-layer filters, feature PCA), the AdamW + cosine annealing optimiser math, and a full hyperparameter reference.
**Why:** User requested a thorough explanation covering the intuition, math, interpretability, and hyperparameters in one place.
**Result / Status:** Done — file at `explainer.md`.

---

### 2026-04-13 — CIFAKE retrain results

**What:** Retrained 1D and 2D models on CIFAKE from scratch (same default hyperparams: lr=3e-4, batch=64, wd=1e-4, 30 epochs, no class-weight). Previous run had been lost from disk; CIFAKE was re-downloaded before retraining.
**Why:** User was unhappy with the prior results; wanted a fresh run.
**Result / Status:** Both models are worse than the previous run. 1D: 0.9399 (epoch 25) vs 0.9405 (epoch 16). 2D: 0.9525 (epoch 13) vs 0.9648 (epoch 13). 2D regression of −0.012 AUC is notable; the identical best epoch suggests the model hit its ceiling earlier. Previous checkpoints should be preferred if they can be recovered.

---

### 2026-04-13 — CLAUDE.md corrections and gap-fills via /init

**What:** Ran `/init` to audit CLAUDE.md against actual source. Fixed: (1) CNN1D param count was wrong (~500k → ~180k, verified by summing conv/BN/linear layers); (2) CNN2D param count was wrong (~2M → ~4M, actual 4,078,050). Added: (3) explicit notebook listing (verifying, ManualInspection, infer_images); (4) note that `make_splits` is duck-typed and works with both `FrequencyDataset` and `CachedFrequencyDataset`.
**Why:** The param counts in CLAUDE.md contradicted the verified counts in `specter_progress_summary.md` and manual calculation; incorrect counts mislead future instances making architecture decisions.
**Result / Status:** Done — CLAUDE.md now reflects correct values.

---

### 2026-04-12 — CLAUDE.md improvements via /init

**What:** Ran `/init` to review and improve CLAUDE.md. Added: (1) GenImage setup scripts section (`setup_genimage.sh`, `setup_genimage_parallel.sh`, `setup_genimage_merged.sh`) with note that merged script is re-runnable; (2) reference to `H100_TRAINING.md` for multi-GPU runbook; (3) log monitoring tip for reading tqdm-polluted background training logs.
**Why:** These scripts existed in the repo but were undocumented in CLAUDE.md; the log-reading pattern is non-obvious.
**Result / Status:** Complete.

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

---

### 2026-04-13 — CVPR paper draft written

**What:** Filled in the CVPR 2026 author-kit template in `Paper_template/author-kit-CVPR2026-v1-latex-/`. Wrote abstract (`sec/0_abstract.tex`), introduction (`sec/1_intro.tex`), related work (`sec/2_formatting.tex`), method (`sec/3_finalcopy.tex`), experiments (`sec/4_experiments.tex`), and discussion/conclusion (`sec/5_conclusion.tex`). Updated `main.tex` title to "Eigenfraud: Frequency-Only Detection of AI-Generated Images via Radial and 2D Spectral CNNs" and added the two new section includes. Added `booktabs` and `enumitem` to `preamble.tex` for the results table and intro contribution list. Reports completed CIFAKE (2D 0.9525 / 1D 0.9399) and Defactify (2D 0.9049 / 1D 0.8439) runs with an explicit table, and describes three planned experiments (E1 GenImage LOGO, E2 JPEG robustness, E3 Fourier-space PGD) with expected outcomes.
**Why:** Abstract submission needed. User asked for a full pass over the template using existing results, with unfinished work framed as plan + expected outcomes. Uses `\cite{}` placeholders throughout for the user to fill in later.
**Result / Status:** Draft complete, not yet compiled. No bibliography keys are yet populated in `main.bib` — all citations will currently show as `?` until the user adds bib entries.

---

### 2026-04-13 — GenImage precompute blocked by inode cap; plan: on-the-fly training

**What:** Attempted to precompute spectra for `data/raw/genimage_all` (50k real + 304k fake across ADM/BigGAN/VQDM/glide + broken sdv4 symlink on a fresh volume). Run reported 321,062 errors / 354,058 total, only 32,996 `.npz` files written (all ADM). Root cause: the Modal volume at `/__modal/volumes/vo-WzpOG7GaLWKcTLwBAnypIi` has a **500,000 inode hard limit**, and `df -i` shows `IUse%=100%` — we hit the inode ceiling after ~33k cache files, and every subsequent `np.savez_compressed` returned `[Errno 28] No space left on device` despite 382 GB of free *bytes*. Same failure mode as the 2026-04-11 sdv4 partial-extract incident; this is a volume-level limit, not a precompute bug. Manifest/code are fine — `_process_one` succeeds on every tested file when writing to `/tmp`.
**Why:** Confirmed by running `_process_one` against the real cache dir in isolation: returns `FileNotFoundError`/`ENOSPC` with "No space left on device" on the `.npz` write. `df -h` shows 0% bytes used; `df -i` shows 100% inodes used. Packing spectra into one `.npz` per image is the wrong data layout for this filesystem.
**Result / Status:** Precompute path abandoned for now. **Decision: train directly on raw images with on-the-fly FFT** via `--data data/raw/genimage_all` (no `.npz` cache). No new inodes, one training process, slower per epoch but acceptable on 1× H100. Sharded cache (~354 files instead of 354k) is a later fallback if on-the-fly proves too slow.

**Next-session runbook (fresh chat pick-up point):**
1. `python -c "from src.dataset import FrequencyDataset; print(FrequencyDataset('data/raw/genimage_all').label_counts())"` — confirm real/fake counts unchanged (~50k/304k).
2. `df -i /__modal/volumes/vo-WzpOG7GaLWKcTLwBAnypIi` — confirm inode cap still binding before doing anything cache-related.
3. Launch training in the background (2D first, 1D second):
   ```
   nohup python scripts/train.py --model 2d --data data/raw/genimage_all \
       --epochs 30 --out-dir results/genimage --class-weight --wandb \
       > /tmp/genimage_2d.log 2>&1 & echo $!
   ```
   Then 1D with `--model 1d` and `/tmp/genimage_1d.log`.
4. Monitor with: `tr '\r' '\n' < /tmp/genimage_2d.log | grep "^Epoch" | tail`.
5. Do **not** rerun `scripts/precompute.py` against the Modal volume — it will hit the same inode wall.
6. If per-epoch time is unacceptable, refactor `precompute.py` to write sharded `.h5`/`.npz` files (target: ~1000 images/shard, ~354 files total) before retrying the cached path.

**Also note:** `data/raw/genimage_all/sdv4` symlink is currently broken (points to `stable_diffusion_v_1_4/...` path that does not exist on this volume); `label_counts` correctly excludes it. sdv5 and wukong archives are still unextracted — Wave 2 from the earlier H100 runbook never finished. Training above uses 4 generators (ADM, BigGAN, VQDM, glide) plus ImageNet nature, not the full 8.

---

### 2026-04-14 — CLAUDE.md review and fixes

**What:** Ran `/init` to review CLAUDE.md against the current codebase. Found and fixed one bug (wrong param counts in `src/models.py` docstring — said ~500k/~2M, actual is ~180k/~4M, verified with `count_parameters()`). Added three missing gotchas to CLAUDE.md: (1) corrupt-image retry semantics in `FrequencyDataset.__getitem__`, (2) silent `--val-dir` fallback to `--train-dir` in `train.py`, (3) duplicate `collate_fn` definitions in `train.py` and `eval.py`.
**Why:** CLAUDE.md is the primary onboarding document for future Claude sessions; accurate param counts and documented gotchas reduce debugging time.
**Result / Status:** `src/models.py` docstring corrected. CLAUDE.md updated with 4 new notes under Architecture section.

---

### 2026-04-14 — GenImage training launched

**What:** Launched both 2D and 1D GenImage training runs (on-the-fly FFT, no cache).
- Pre-flight: dataset = 50k real + 304k fake; inodes at 78% (111k free — no longer at wall); GPU = H100 80GB.
- 2D PID 5809: `python scripts/train.py --model 2d --data data/raw/genimage_all --epochs 30 --out-dir results/genimage --class-weight --wandb > /tmp/genimage_2d.log`
- 1D PID 5872: `python scripts/train.py --model 1d --data data/raw/genimage_all --epochs 30 --out-dir results/genimage --class-weight --wandb > /tmp/genimage_1d.log`
**Why:** Main GenImage training run. 4 generators active (ADM, BigGAN, VQDM, glide + ImageNet real). sdv4 symlink broken, sdv5/wukong not extracted.
**Result / Status:** In progress. Monitor: `tr '\r' '\n' < /tmp/genimage_2d.log | grep "^Epoch" | tail`

---

### 2026-04-14 — Sharded precompute + CachedFrequencyDataset rewrite

**What:** Killed the on-the-fly GenImage training runs (~7% through epoch 1 at ~3 sec/batch = ~3 hrs/epoch). Rewrote `scripts/precompute.py` and `src/dataset.py:CachedFrequencyDataset` to use sharded `.npz` files.
- `precompute.py`: computes spectra in parallel (ProcessPoolExecutor), buffers into chunks of `--shard-size` (default 1000), writes each chunk as one `shard_NNNNN.npz` with keys `s2d` (N,1,H,W float16) and `p1d` (N,L float32). Manifest columns: `path, label, shard_file, shard_idx`. Samples are pre-shuffled so each shard has a real/fake mix. Resumable.
- `CachedFrequencyDataset`: reads 4-column manifest, loads shards via module-level `lru_cache(maxsize=8)` per worker process. Tuple format changed from `(path, label, cache_file)` to `(path, label, shard_file, shard_idx)`.
**Why:** On-the-fly FFT was ~3 sec/batch → ~3 hrs/epoch → 90 hrs for 30 epochs. The old per-image precompute hit the Modal volume's 500k inode cap after ~33k files. Sharded layout uses ~354 files for 354k images.
**Result / Status:** Code written and imports verified. Precompute not yet run — user will run in their own tmux session.

**Commands to run:**
```bash
# Step 1: precompute (run in tmux, takes ~1-2 hrs with 16 workers)
python scripts/precompute.py --data data/raw/genimage_all \
    --cache-dir data/cache/genimage_sharded --workers 16

# Step 2: train from cache
python scripts/train.py --model 2d --cache data/cache/genimage_sharded/manifest.csv \
    --epochs 30 --out-dir results/genimage --class-weight

python scripts/train.py --model 1d --cache data/cache/genimage_sharded/manifest.csv \
    --epochs 30 --out-dir results/genimage --class-weight
```

### 2026-04-17 — Add GenImage checkpoints to inference notebook

**What:** Updated `notebooks/infer_images.ipynb` to test images against all available checkpoints — CIFAKE (1D+2D) and GenImage (1D+2D). Also fixed checkpoint paths from `results/best_*.pt` to `results/cifake/best_*.pt` to match actual file locations.
**Why:** Previously only CIFAKE checkpoints were tested. Adding GenImage checkpoints lets us compare how models trained on different datasets classify the same images.
**Result / Status:** Notebook updated, needs re-run.

### 2026-04-19 — Reformatted PIVOT.md as checklist

**What:** Restructured `PIVOT.md` from prose into a markdown checklist with `- [ ]` items per step. Condensed the narrative framing into a compact header (thesis, contributions, arc table) and kept all task details inline with their checkboxes.
**Why:** Easier to track progress through the 6-phase plan at a glance. Checklist format makes it clear what's done vs pending.
**Result / Status:** Complete. All content preserved, just reorganized.
