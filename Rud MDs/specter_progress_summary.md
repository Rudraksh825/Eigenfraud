# Specter / Eigenfraud — Progress Summary
*Generated 2026-04-02. Written for a Claude instance with zero prior context.*

---

## 1. Project Overview

**Specter** (repo name: **Eigenfraud**) is an AI-generated image detector that operates entirely in the frequency domain. Instead of pixel-space features, every image is converted to its 2D log-power spectrum via FFT, and a CNN is trained on that representation. The hypothesis: generative models (diffusion, GAN) leave characteristic spectral fingerprints — grid artifacts, unusual frequency distributions, directional biases — detectable even when pixel content looks realistic.

**Label convention:** 0 = real, 1 = fake (used everywhere in code and checkpoints).

**WandB project name:** `"specter"

---

## 2. Project Structure

```
/__modal/volumes/vo-WzpOG7GaLWKcTLwBAnypIi/Eigenfraud/
│
├── src/
│   ├── __init__.py
│   ├── transforms.py       — math layer: all FFT / spectral logic
│   ├── dataset.py          — FrequencyDataset + make_splits()
│   └── models.py           — CNN1D, CNN2D, build_model factory
│
├── scripts/
│   ├── train.py            — training loop (AdamW + cosine LR, WandB optional)
│   └── eval.py             — evaluation: AUC, accuracy, EER
│
├── notebooks/
│   ├── verifying.ipynb         — early sanity-check notebook
│   └── ManualInspection.ipynb  — manual inspection of model outputs / spectra
│
├── figures/
│   ├── fig1_prototype.png
│   ├── fig2_prototype.png
│   ├── mean_profiles.png       — mean 1D radial profile: real vs fake
│   ├── mean_spectra_2d.png     — mean 2D spectrum: real vs fake
│   ├── pipeline_per_image.png  — per-image pipeline visualization
│   └── sanity_spectra.png      — spectral transform sanity check
│
├── results/
│   ├── best_1d.pt      — best 1D CNN checkpoint (by val AUC)
│   └── best_2d.pt      — best 2D CNN checkpoint (by val AUC)
│
├── data/
│   ├── raw/
│   │   ├── cifake/
│   │   │   ├── train/      — CIFAKE training set (real/ + fake/)
│   │   │   └── test/       — CIFAKE test set (real/ + fake/)
│   │   ├── ADM/            — GenImage split zip archives (not yet extracted)
│   │   ├── BigGAN/
│   │   ├── VQDM/
│   │   ├── glide/
│   │   ├── Midjourney/
│   │   ├── stable_diffusion_v_1_4/
│   │   ├── stable_diffusion_v_1_5/
│   │   ├── wukong/
│   │   └── GenImage/       — GenImage++ blobs (Git LFS, NOT the original GenImage)
│   └── processed/          — early CIFAKE Kaggle download path (legacy)
│
├── CLAUDE.md               — Claude Code instructions + project overview
├── journal.md              — full chronological project log
├── currentplan.txt         — active plan: GenImage extraction + training
├── requirements.txt        — Python dependencies
├── setup_data.sh           — CIFAKE download via Kaggle; FF++ optional
├── faceforensics.txt       — FF++ official download script
├── running.txt             — training commands run log
├── notes.txt               — informal module notes
└── AllowClaude.txt         — bash PATH fix for Claude shell
```

---

## 3. Data Pipeline

### 3.1 Datasets Used

**CIFAKE** (primary, currently used for all trained checkpoints):
- Real images: CIFAR-10 real photos
- Fake images: Stable Diffusion v1.4 equivalents
- Source: Kaggle `birdy654/cifake-real-and-ai-generated-synthetic-images`
- Pre-split layout: `data/raw/cifake/train/` and `data/raw/cifake/test/`
- Both dirs contain `real/` and `fake/` subdirectories

**GenImage** (original, 8 generators — planned but NOT yet trained):
- Generators: ADM, BigGAN, VQDM, GLIDE, Midjourney, SD v1.4, SD v1.5, Wukong
- ~1.35M images total, ImageNet-scale
- Raw data present as split zip archives in `data/raw/{generator}/`
- **Status: not yet extracted** — see Section 7 (TODO)

**GenImage++** (holdout benchmark — NOT the same as GenImage):
- Stored as Git LFS blobs in `data/raw/GenImage/`
- This is `Lunahera/genimagepp` (NeurIPS 2025), test-only, no training splits
- Most blobs truncated at 4 GiB (32-bit download limit bug during acquisition)
- Only 3 intact archives: `flux` (6k), `flux_amateur` (6k), `flux_krea_amateur` (6k)
- All intact archives contain only `1_fake/`; real images are in broken blobs
- **Status: unusable until re-downloaded**

**FaceForensics++** (optional, not yet used):
- Video-based face manipulation dataset
- Layout: `data/raw/faceforensics/`
- Requires form approval; download script in `faceforensics.txt`

### 3.2 `FrequencyDataset` (`src/dataset.py`)

**Expected directory layout:**
```
root/
    real/       ← label 0
    fake/       ← label 1 (ANY non-"real" subdir name is treated as fake)
```

**Instantiation:**
```python
from src.dataset import FrequencyDataset
ds = FrequencyDataset(root="data/raw/cifake/train", size=224)
# returns (spectrum_2d, profile_1d, label) per item
```

**Key parameters:**
- `size` (int, default 224): resize target before FFT
- `transform` (Optional[Callable]): applied to PIL image before FFT
- `fake_dirs` (Optional[list]): explicit subdirs to treat as fake; if None, every non-`real` subdir is fake

**Image discovery:** `_find_images()` walks subdirectories recursively, includes `.jpg`, `.jpeg`, `.png`, `.webp`, `.bmp`.

**Per-item return type:**
```
spectrum_2d : torch.FloatTensor  shape (1, 224, 224)  — for CNN2D
profile_1d  : torch.FloatTensor  shape (112,)          — for CNN1D
label       : int  (0 = real, 1 = fake)
```

**Multi-generator usage (GenImage):** Pass `fake_dirs=["ADM", "stable_diffusion_v_1_5"]` etc. to select specific generator subfolders. Omit to treat all non-real subdirs as fake.

### 3.3 `make_splits()` (`src/dataset.py`)

Stratified 80/10/10 train/val/test split using `sklearn.model_selection.train_test_split`. Returns three `torch.utils.data.Subset` objects.

**Only use this when `--data` (single directory) is passed.** Never use it on CIFAKE — CIFAKE has its own pre-split `train/` and `test/` dirs. Using `make_splits` on CIFAKE's `test/` dir would re-split the test set, making eval results wrong.

```python
train_ds, val_ds, test_ds = make_splits(dataset, val_frac=0.1, test_frac=0.1, seed=42)
```

### 3.4 Per-Image Preprocessing (`src/transforms.py`)

The full transform pipeline, applied in `FrequencyDataset.__getitem__`:

**Step 1: `to_grayscale_array(img, size=224)`**
```python
img = img.convert("L").resize((size, size), Image.LANCZOS)
return np.array(img, dtype=np.float32)  # shape: (224, 224)
```
- Converts RGB/RGBA/any mode to single-channel grayscale via PIL `"L"` mode
- Resizes to 224×224 using LANCZOS resampling
- Output range: [0.0, 255.0] float32
- **No normalization is applied** (neither per-image zero-mean nor global standardization)

**Step 2: `log_power_spectrum_2d(gray)`**
```python
F = np.fft.fft2(gray)           # 2D DFT, shape (224, 224), complex
F_shifted = np.fft.fftshift(F)  # shift DC component to center
power = np.abs(F_shifted) ** 2  # power spectrum
log_power = np.log1p(power).astype(np.float32)  # log(1 + |F|^2)
```
- Output: float32 array of shape (224, 224)
- DC (zero frequency) at center (112, 112) after fftshift
- `log1p` compresses dynamic range; avoids log(0)
- Output range: [0, ~log(255² × 224²)] ≈ [0, ~20]

**Step 3: `azimuthal_average_fast(spectrum)`** (production path; used in dataset)
```python
H, W = spectrum.shape       # 224, 224
cy, cx = H // 2, W // 2     # 112, 112
y = np.arange(H) - cy
x = np.arange(W) - cx
xx, yy = np.meshgrid(x, y)
r = np.round(np.sqrt(xx**2 + yy**2)).astype(np.int32).ravel()

r_max = min(H, W) // 2      # = 112
flat = spectrum.ravel()

mask = r < r_max             # only pixels within radius 112
counts = np.bincount(r[mask], minlength=r_max)
sums   = np.bincount(r[mask], weights=flat[mask], minlength=r_max)
profile = np.where(counts > 0, sums / counts, 0.0).astype(np.float32)
```
- Output: float32 array of shape **(112,)**
- Bins are integer radii 0, 1, 2, …, 111
- Each bin is the mean log-power over all pixels at that integer distance from center
- Captures only **isotropic** spectral structure (rotationally averaged)
- `azimuthal_average()` is a loop-based reference implementation kept for validation; not used in training

### 3.5 Augmentation / JPEG Re-compression

**None currently applied.** The `transform` parameter of `FrequencyDataset` is always `None` in all training scripts. No random crops, flips, JPEG re-compression, or frequency-domain augmentations.

### 3.6 Normalization

No per-image or dataset-level normalization of the spectra before feeding to the CNN. The raw log-power values go directly into the model. This is an open question — per-image normalization (subtract mean, divide std) might improve generalization.

---

## 4. Model Architectures

### 4.1 Branch A: `CNN1D` (`src/models.py`)

**Input:** `(B, 112)` → unsqueezed to `(B, 1, 112)` inside `forward()`  
**Output:** `(B, 2)` raw logits  
**Parameters:** **180,002**

```
Layer             Channels    Kernel  Padding  Output shape (L=112)
────────────────────────────────────────────────────────────────────
Conv1d + BN + ReLU   1 → 32    k=3     p=1     (B, 32, 112)
Conv1d + BN + ReLU  32 → 64    k=3     p=1     (B, 64, 112)
MaxPool1d(2)                                    (B, 64,  56)
Conv1d + BN + ReLU  64 → 128   k=3     p=1     (B, 128, 56)
Conv1d + BN + ReLU 128 → 128   k=3     p=1     (B, 128, 56)
MaxPool1d(2)                                    (B, 128, 28)
Conv1d + BN + ReLU 128 → 256   k=3     p=1     (B, 256, 28)
AdaptiveAvgPool1d(1)                            (B, 256,  1) → (B, 256)
Linear(256, 2)                                  (B, 2)
```

All conv layers use `bias=False` (BN handles bias). Activation is ReLU (inplace). No Dropout.

### 4.2 Branch B: `CNN2D` (`src/models.py`)

**This is a custom CNN, NOT ResNet-18.**  
**Input:** `(B, 1, 224, 224)` — single-channel log-power spectrum  
**Output:** `(B, 2)` raw logits  
**Parameters:** **4,078,050**

```
Layer              Channels    Kernel  Padding  Output shape (H=W=224)
──────────────────────────────────────────────────────────────────────
Conv2d + BN + ReLU   1 → 32    k=3     p=1     (B, 32, 224, 224)
Conv2d + BN + ReLU  32 → 64    k=3     p=1     (B, 64, 224, 224)
MaxPool2d(2)                                    (B, 64, 112, 112)
Conv2d + BN + ReLU  64 → 128   k=3     p=1     (B, 128, 112, 112)
Conv2d + BN + ReLU 128 → 128   k=3     p=1     (B, 128, 112, 112)
MaxPool2d(2)                                    (B, 128,  56,  56)
Conv2d + BN + ReLU 128 → 256   k=3     p=1     (B, 256,  56,  56)
MaxPool2d(2)                                    (B, 256,  28,  28)
Conv2d + BN + ReLU 256 → 512   k=3     p=1     (B, 512,  28,  28)
Conv2d + BN + ReLU 512 → 512   k=3     p=1     (B, 512,  28,  28)
AdaptiveAvgPool2d(1)                            (B, 512,   1,   1) → (B, 512)
Linear(512, 2)                                  (B, 2)
```

All conv layers use `bias=False`. Activation is ReLU (inplace). No Dropout. No skip connections (not ResNet). No pretrained weights.

### 4.3 `build_model` Factory

```python
from src.models import build_model
model = build_model("1d", input_length=112)  # returns CNN1D
model = build_model("2d")                    # returns CNN2D
```

### 4.4 Checkpoint Format

```python
{
    "epoch":       int,           # epoch at which this checkpoint was saved
    "model_type":  "1d" or "2d",  # drives model construction at eval time
    "model_state": OrderedDict,   # state_dict
    "val_auc":     float,         # val AUC at save time
    "args":        dict,          # vars(args) from training run
}
```

---

## 5. Training Setup

### 5.1 Command-line interface (`scripts/train.py`)

```
--model        required   "1d" or "2d"
--train-dir    optional   directory with real/ + fake/ subdirs (pre-split mode)
--val-dir      optional   validation directory (pre-split mode)
--data         optional   single root dir → auto 80/10/10 split
--epochs       default 30
--lr           default 3e-4
--batch-size   default 64
--weight-decay default 1e-4
--size         default 224 (image resize before FFT)
--workers      default 4
--out-dir      default "results"
--wandb        flag, enables W&B logging
--seed         default 42
```

### 5.2 Loss, Optimizer, Scheduler

```python
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
```

- **Loss:** CrossEntropyLoss over 2-class logits (not BCEWithLogitsLoss)
- **Optimizer:** AdamW with weight decay 1e-4
- **LR schedule:** Cosine annealing from lr=3e-4 to 0 over `T_max=epochs` steps (no warmup)
- **Checkpoint strategy:** save only when val AUC improves (not val loss)

### 5.3 Training Tricks

- No gradient clipping
- No dropout
- No label smoothing
- No mixed precision (no `torch.cuda.amp`)
- No data augmentation
- No per-image spectrum normalization
- `pin_memory=True` when using CUDA, False otherwise

### 5.4 Metrics Logged Per Epoch

- `train/loss`, `train/auc`, `train/acc`
- `val/loss`, `val/auc`, `val/acc`

AUC is computed via `sklearn.metrics.roc_auc_score`. Accuracy uses 0.5 threshold on softmax probability for class 1.

### 5.5 Hardware

Trained on a Modal Labs GPU instance (exact GPU model not recorded in checkpoints). Device selection logic:
```python
device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps"  if torch.backends.mps.is_available() else
    "cpu"
)
```

---

## 6. Results

### 6.1 Trained Checkpoints

Both checkpoints were trained on **CIFAKE only** (SD v1.4 vs real CIFAR-10):

| Model  | Checkpoint       | Best Val AUC | Epoch Saved | Epochs Run | Training Data Path |
|--------|-----------------|--------------|-------------|------------|--------------------|
| CNN1D  | `results/best_1d.pt` | **0.9449** | 29/30 | 30 | `data/processed/…/cifake/…/train` |
| CNN2D  | `results/best_2d.pt` | **0.9650** | 12/30 | 30 | `data/raw/cifake/train` |

Note: the 1D model was trained on a `data/processed/` path (Kaggle download artifact), while the 2D model was trained on `data/raw/cifake/` — these should be the same images, but the paths differ.

The 2D checkpoint was saved at epoch 12, meaning val AUC did not improve in the final 18 epochs — possible overfitting or plateau.

### 6.2 Cross-Generator / Cross-Dataset Eval

**Not yet run.** No cross-generator evaluation has been performed. The following are planned (see `currentplan.txt`):
- CIFAKE-trained models → GenImage val set
- GenImage-trained models → CIFAKE test set
- GenImage-trained models → GenImage++ holdout

### 6.3 EER and Test Set Accuracy

No formal test set evaluation has been run via `eval.py`. The numbers above are validation-set AUC from training logs, stored in checkpoints. To get test metrics:
```bash
python scripts/eval.py \
  --checkpoint results/best_1d.pt results/best_2d.pt \
  --data data/raw/cifake/test \
  --split all
```

### 6.4 Known Findings from EDA

From figures in `figures/` and notebooks:
- The mean 2D spectrum shows clear structural differences between real and SD v1.4 images (`mean_spectra_2d.png`)
- The mean 1D radial profile also differs between classes (`mean_profiles.png`), confirming the spectral fingerprint hypothesis for this generator
- These EDA results motivated the model choice but no quantitative analysis from them has been formalized

---

## 7. Adversarial Attacks

**Not implemented.** No PGD or other adversarial attack has been coded, run, or tested. This is entirely future work.

---

## 8. Evaluation Script (`scripts/eval.py`)

```bash
# Correct usage for CIFAKE pre-split test dir:
python scripts/eval.py \
  --checkpoint results/best_1d.pt results/best_2d.pt \
  --data data/raw/cifake/test \
  --split all

# WRONG for CIFAKE: --split test calls make_splits() which re-splits the test dir
```

Metrics reported: AUC, Accuracy (threshold=0.5), EER (equal error rate via linear interpolation on ROC curve).

Multi-checkpoint mode prints a comparison table. Single-checkpoint mode prints a detailed report.

---

## 9. Active Plan (from `currentplan.txt`)

The next major milestone is **training on GenImage** (8 generators) for cross-generator robustness, then evaluating on GenImage++ as a hard holdout.

**Steps:**

1. **Extract GenImage archives** — each generator has split zip archives in `data/raw/{generator}/`:
   ```bash
   cd data/raw/ADM && zip -FF imagenet_adm.zip --out full.zip && unzip full.zip
   # or: 7z x imagenet_adm.zip
   ```

2. **Patch `src/dataset.py`** if GenImage uses `0_real` subdirectory naming (instead of `real`):
   ```python
   # In FrequencyDataset.__init__, change:
   if subdir.name.lower() == "real":
   # to:
   if subdir.name.lower() in ("real", "0_real"):
   ```

3. **Organize into unified train/val dirs:**
   ```
   data/raw/genimage/train/
       real/              ← ImageNet real (shared across generators)
       ADM/
       BigGAN/
       ...
   data/raw/genimage/val/
       real/
       ...
   ```

4. **Train:**
   ```bash
   python scripts/train.py --model 1d \
     --train-dir data/raw/genimage/train \
     --val-dir data/raw/genimage/val \
     --epochs 30 --wandb --out-dir results/genimage
   python scripts/train.py --model 2d \
     --train-dir data/raw/genimage/train \
     --val-dir data/raw/genimage/val \
     --epochs 30 --wandb --out-dir results/genimage
   ```

5. **Cross-dataset eval** (CIFAKE ↔ GenImage)

6. **GenImage++ holdout** (after training is complete)

---

## 10. Known Issues / TODO

### Data
- [ ] GenImage original dataset not yet extracted (split zip archives present but unextracted)
- [ ] GenImage++ is nearly entirely broken (13/21 blobs truncated at 4 GiB) — needs fresh download
- [ ] No real images are present in the intact GenImage++ blobs
- [ ] `data/processed/` path used for 1D training is a Kaggle artifact path — document or clean up

### Model / Training
- [ ] No per-image normalization of spectra (open question: would it help?)
- [ ] No data augmentation in frequency domain
- [ ] 2D checkpoint saved at epoch 12 — val AUC plateaued/regressed for 18 epochs; should investigate
- [ ] Param count mismatch from journal: CNN1D is **180k** (not ~500k), CNN2D is **4.08M** (not ~2M)
- [ ] No dropout anywhere — likely causes overfitting at scale
- [ ] No mixed precision training (slow on larger datasets)

### Evaluation
- [ ] No test-set eval has been run — val AUC from training is all we have
- [ ] No cross-generator eval run
- [ ] No per-generator breakdown
- [ ] FaceForensics++ not integrated (requires frame extraction from video)

### Future Work
- [ ] Adversarial PGD attack in Fourier space (epsilon/steps TBD)
- [ ] Spectrum normalization (per-image: subtract mean, divide std)
- [ ] Frequency-domain augmentation (e.g., random spectrum rotation)
- [ ] Consider ResNet-18 or ViT backbone for 2D branch
- [ ] Consider ensemble of 1D and 2D outputs
- [ ] Cross-dataset generalization (the core scientific question)

---

## 11. Quick-Reference: Key Code Locations

| What | Where |
|------|-------|
| Grayscale + resize | `src/transforms.py:15` `to_grayscale_array()` |
| 2D FFT + log-power | `src/transforms.py:21` `log_power_spectrum_2d()` |
| Azimuthal average (fast) | `src/transforms.py:72` `azimuthal_average_fast()` |
| Dataset class | `src/dataset.py:47` `FrequencyDataset` |
| Train/val/test split | `src/dataset.py:116` `make_splits()` |
| CNN1D definition | `src/models.py:21` |
| CNN2D definition | `src/models.py:63` |
| Model factory | `src/models.py:105` `build_model()` |
| Training loop | `scripts/train.py:58` `run_epoch()` |
| Eval + EER | `scripts/eval.py:36` `compute_eer()`, `scripts/eval.py:50` `evaluate_checkpoint()` |
