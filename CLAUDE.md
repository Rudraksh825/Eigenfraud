# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Journal Requirement (MANDATORY)

**`Rud MDs/journal.md` must be updated at every step.**

Before finishing any response that involves a decision, code change, experiment, observed result, or noteworthy finding — append an entry to `Rud MDs/journal.md`.

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

**Detectors under evaluation (all cloned to `detectors/`):**
- CNN2D (ours, `results/best_2d.pt` or `results/genimage/best_2d.pt`)
- CNN1D (ours, `results/best_1d.pt` or `results/genimage/best_1d.pt`)
- CNNDetection — `detectors/CNNDetection/` (`PeterWang512/CNNDetection`)
- FreqNet — `detectors/FreqNet-DeepfakeDetection/` (`chuangchuangtan/FreqNet-DeepfakeDetection`)
- UnivFD — `detectors/UniversalFakeDetect/` (`Yuheng-Li/UniversalFakeDetect`)
- NPR — `detectors/NPR-DeepfakeDetection/`
- FatFormer — `detectors/FatFormer/`
- B-Free — `detectors/B-Free/`

**Phase status (as of 2026-04-22):**

| Phase | Status | Key output |
|-------|--------|------------|
| 0 — Audit + detector setup | ✅ Complete | `status.md` Table 1; all 6 detectors verified |
| 1 — Dataset characterization | ✅ Complete | `figures/fig1_*.png`, `results/spectra/*.npz` |
| 2 — Normalization pipeline | 🔶 Scripts done; outputs ephemeral | `/root/normalized/` — must re-run each machine |
| 3.2 — Baseline eval (original) | 🔶 CIFAKE done; Defactify partial | `results/metrics.csv` |
| 3.3 — Eval (normalized) | ⬜ Not started | Requires Phase 2 normalization first |
| 4 — Band ablation | ⬜ Not started | `scripts/band_ablation.py` not yet written |
| 5–6 — Synthesis + writing | ⬜ Not started | |

**Scripts written:**
- `scripts/audit_dataset.py` — audits file-based datasets (CIFAKE, GenImage); reports count, format, resolution, file size per split
- `scripts/audit_defactify.py` — same audit for Defactify parquet shards; reads `Image` bytes + `Label_A`
- `scripts/eval_external.py` — unified wrapper for all 6 external detectors; `--detector {cnndetection,freqnet,npr,univfd,fatformer,bfree}`, `--data`, `--weights`; outputs CSV `(path, label, score)`. Prints AUC + Acc@0.5 to stdout; does **not** compute MCC and does **not** write to `results/metrics.csv` — those must be done manually (see Results Logging below)
- `scripts/characterize_datasets.py` — computes mean radial/2D spectra per dataset × split; produces `figures/fig1_*.png` and `results/spectra/*.npz`
- `scripts/normalize_dataset.py` — load → RGB → resize N×N bilinear → save as PNG (strip EXIF/alpha, no spatial augmentation); supports file-based (`--input`, `--input-real/--input-fake`) and parquet (`--parquet`) inputs; output layout: `<out>/real/` + `<out>/fake/`

**Scripts still to write:**
- `scripts/band_ablation.py` — interface: `--input <dir> --band {low,mid,high} --output <dir>`; FFT → zero radial band → iFFT → clip [0,255] → save PNG. Band radii: low `r < 0.2·r_max`, mid `0.2–0.6·r_max`, high `r ≥ 0.6·r_max`

**Key experimental findings so far:**
- Cross-dataset collapse: GenImage-trained CNN2D scores AUC 0.53 / MCC 0.05 on CIFAKE — chance level. Motivates the pivot.
- Phase 3 baseline (CIFAKE, original): **all 6 external detectors score AUC < 0.5** (0.29–0.50), indicating inverted decision boundaries when applied to CIFAKE out-of-distribution. CNNDetection=0.375, FreqNet=0.473, NPR=0.435, UnivFD=0.300, FatFormer=0.290, B-Free=0.497. Full data in `results/metrics.csv`.

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

### Dataset construction audit (Phase 0.3)
```bash
# CIFAKE / GenImage (file-based)
python scripts/audit_dataset.py \
    --train-real data/raw/cifake/train/REAL --train-fake data/raw/cifake/train/FAKE \
    --test-real  data/raw/cifake/test/REAL  --test-fake  data/raw/cifake/test/FAKE \
    --dataset CIFAKE

# Defactify (parquet shards)
python scripts/audit_defactify.py --data defactify_dataset/data
```

### Evaluate external detectors (Phase 3)
```bash
# Weights default to their documented paths; override with --weights if needed
python scripts/eval_external.py --detector cnndetection \
    --data data/raw/cifake/test \
    --out results/cnndetection_cifake.csv

# B-Free requires a weights *directory* (contains config.yaml), not a .pth file
python scripts/eval_external.py --detector bfree \
    --data data/raw/cifake/test \
    --weights detectors/B-Free/code/weights/BFREE_dino2reg4 \
    --out results/bfree_cifake.csv
```

Note: `univfd` and `fatformer` auto-download CLIP ViT-L/14 (~890 MB) to `~/.cache/clip/` on first run. `fatformer` also requires `pip install pytorch_wavelets`.

### Dataset characterization (Phase 1 / Phase 2)
```bash
# Figure 1 — original data (defaults to CIFAKE test, Defactify test parquets, GenImage val+ADM)
python scripts/characterize_datasets.py

# Figure 2 — normalized data (pass explicit paths)
python scripts/characterize_datasets.py \
    --cifake-real  /root/normalized/cifake/test/real \
    --cifake-fake  /root/normalized/cifake/test/fake \
    --defy-real    /root/normalized/defactify/test/real \
    --defy-fake    /root/normalized/defactify/test/fake \
    --genimage-real /root/normalized/genimage/real \
    --genimage-fake /root/normalized/genimage/fake \
    --fig-prefix fig2 --spectra-tag norm
```

Without explicit path flags the script uses hardcoded raw-data paths (CIFAKE test, Defactify test parquets, GenImage imagenet_nature/val + ADM).

### Dataset sanity check
```bash
python -c "from src.dataset import FrequencyDataset; d = FrequencyDataset('data/raw/cifake/test'); print(d.label_counts())"
```

### Normalize a dataset (Phase 2)

**Storage constraint:** The Modal volume has ~4 inodes free — normalized outputs cannot be stored there. Always write to `/root/normalized/` (root filesystem). This is ephemeral and must be re-run on each new machine.

```bash
# CIFAKE (~5 min)
python scripts/normalize_dataset.py \
    --input data/raw/cifake/test \
    --output /root/normalized/cifake/test

# Defactify parquet (~20 min)
python scripts/normalize_dataset.py \
    --parquet defactify_dataset/data --split test \
    --output /root/normalized/defactify/test

# GenImage (~2 hours) — run with nohup
nohup python scripts/normalize_dataset.py \
    --input-real data/raw/imagenet_nature/val \
    --input-fake data/raw/ADM/imagenet_ai_0508_adm/train/ai \
    --output /root/normalized/genimage \
    > /root/normalize_genimage.log 2>&1 &
```

Default: `--size 256 --method bilinear`. Override with `--size` and `--method {bilinear,bicubic,lanczos,nearest}`.

### Legacy: train our own models (no longer the primary workflow)
```bash
python scripts/train.py --model 2d --train-dir data/raw/cifake/train --val-dir data/raw/cifake/test
python scripts/train.py --model 1d --train-dir data/raw/cifake/train --val-dir data/raw/cifake/test
```
See full training reference (multi-GPU, GenImage, cached spectra) in git history.

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
- `scripts/audit_dataset.py` — Phase 0.3 artifact audit for file-based datasets
- `scripts/audit_defactify.py` — Phase 0.3 artifact audit for Defactify parquet shards
- `scripts/eval_external.py` — unified inference wrapper for all 6 external detectors; outputs CSV `(path, label, score)`
- `scripts/characterize_datasets.py` — computes mean radial/2D spectra, HF L2 divergence, JPEG quality stats; produces Figures 1 and 1b
- `scripts/normalize_dataset.py` — Phase 2 normalization; strips format/resolution bias; outputs `real/` + `fake/` PNG dirs
- `notebooks/` — `verifying.ipynb`, `ManualInspection.ipynb`, `infer_images.ipynb`, `fft_roundtrip.ipynb`
- `detectors/` — cloned external detector repos (CNNDetection, FreqNet-DeepfakeDetection, UniversalFakeDetect, NPR-DeepfakeDetection, FatFormer, B-Free)
- `PIVOT.md` — full phased research plan with deliverables
- `status.md` — post-pivot progress tracker (completed findings, up-next tasks)
- `Rud MDs/journal.md` — complete project history

## Results Logging (MANDATORY)

**All experiment outputs must be saved to disk in full — not just summarized in status.md.**

### What to save and where

| Output type | Save location | Notes |
|-------------|---------------|-------|
| Script stdout (audit, characterize, eval) | `results/logs/<script>_<dataset>.txt` | Use `tee`: `python script.py ... \| tee results/logs/foo.txt` |
| Per-image scores (eval_external, eval.py) | `results/<detector>_<dataset>.csv` | Already handled by `--out`; never skip this flag |
| Spectral arrays | `results/spectra/<dataset>_<split>_<condition>.npz` | `characterize_datasets.py` does this; replicate for normalized/ablated runs |
| Aggregate metrics (AUC, MCC, Acc) | `results/metrics.csv` — one row per (detector, dataset, condition) | Append after every eval run |
| Figures | `figures/fig<N>_<description>.png` | Already followed |

### `results/metrics.csv` schema

```
detector,dataset,condition,auc,accuracy,mcc,n_real,n_fake
cnndetection,cifake,original,0.8321,0.7910,0.583,10000,10000
```

`condition` values: `original`, `normalized`, `ablated_low`, `ablated_mid`, `ablated_high`

### metrics.csv population workflow

`eval_external.py` does NOT write to `metrics.csv`. After each eval run, manually append a row:

```bash
# Compute MCC from a per-image CSV (requires threshold tuning or use 0.5)
python - <<'EOF'
import pandas as pd, numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score, matthews_corrcoef
df = pd.read_csv("results/cnndetection_cifake_original.csv")
auc = roc_auc_score(df.label, df.score)
acc = accuracy_score(df.label, df.score > 0.5)
mcc = matthews_corrcoef(df.label, df.score > 0.5)
n_real, n_fake = (df.label==0).sum(), (df.label==1).sum()
print(f"cnndetection,cifake,original,{auc:.4f},{acc:.4f},{mcc:.4f},{n_real},{n_fake}")
EOF
# Then paste the printed line into results/metrics.csv
```

### Rule

If a script produces a number that could go in a paper table, it must be saved to a file — not just printed to stdout and read off the screen. `status.md` holds summaries; `results/` holds the actual data.

---

## Code Style
- No unnecessary abstractions or speculative features.
- No docstrings or comments added to unchanged code.
- Prefer editing existing files over creating new ones.

## Intentional omissions in CNN1D/CNN2D (do not add unless asked)
No data augmentation, no spectrum normalization, no dropout, no mixed precision, no gradient clipping.
