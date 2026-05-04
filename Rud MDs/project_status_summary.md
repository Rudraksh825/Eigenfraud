# Eigenfraud — Project Status Summary

### What This Project Is

**Eigenfraud** started as a deepfake detector built on spectral analysis (FFT of images → CNN classifier). After early experiments showed the model failing badly out-of-distribution — GenImage-trained CNN2D scored AUC 0.53 (chance) on CIFAKE — the project pivoted entirely.

The new goal is an **audit paper**, not a detector paper. The thesis: existing AI-image detectors don't detect AI generation. They detect dataset construction artifacts — specifically JPEG-vs-PNG format mismatches and resolution mismatches between real and fake splits — that happen to correlate with "fake" in their training data. Their reported near-perfect benchmark numbers are inflated by shortcuts, not forensic capability.

This is being written up as a CVPR 2026 submission.

---

### What Is Done

#### Phase 0 — Dataset Audit
Three benchmarks were audited for construction artifacts:

| Dataset | Format bias | Resolution bias | Notes |
|---------|------------|----------------|-------|
| CIFAKE | None | None | Both classes 32×32 JPEG — no shortcuts possible |
| Defactify | None | **Severe** | Fake has 5–6 unique resolutions vs 180+ for real |
| GenImage | **Severe** | **Severe** | Real = 100% JPEG, Fake = 100% PNG. A 1-line file-extension classifier achieves AUC = **1.000** |

All 6 external detectors were cloned and verified:
- CNNDetection, FreqNet, NPR, UnivFD, FatFormer, B-Free — all wrapped by `scripts/eval_external.py`

#### Phase 1 — Spectral Characterization
Mean radial power spectra computed for all three datasets. High-frequency L2 divergence between real and fake:
- CIFAKE: **0.34** (near-zero — no artifacts)
- Defactify: **4.23** (12× CIFAKE)
- GenImage: **6.16** (18× CIFAKE)

Divergence tracks construction bias severity perfectly. Figures saved to `figures/fig1_radial_spectra.png` and `figures/fig1b_2d_heatmaps.png`. Raw spectra in `results/spectra/*.npz`.

#### Phase 2 — Normalization Pipeline
`scripts/normalize_dataset.py` written: loads images → RGB → bilinear resize to 256×256 → save as PNG (strips EXIF, alpha, format metadata). This is the "deconfounding" operation — it removes both the format shortcut (JPEG vs PNG) and the resolution shortcut by making all images identical format and size.

Outputs go to `/root/normalized/` which is **ephemeral** (lost on machine restart, must be re-run each session).

#### Phase 3 — Baseline Evaluations
All 7 detectors (6 external + CNN2D ours) evaluated across all 3 datasets in both original and normalized conditions. **45 rows in `results/metrics.csv`**. Key results:

**CIFAKE (no artifacts):**
| Detector | Original AUC | Normalized AUC | Δ |
|----------|------------|--------------|---|
| CNNDetection | 0.375 | 0.375 | 0.000 |
| FreqNet | 0.473 | 0.473 | 0.000 |
| NPR | 0.435 | 0.435 | 0.000 |
| UnivFD | 0.300 | 0.371 | +0.071 |
| FatFormer | 0.290 | 0.290 | 0.000 |
| B-Free | 0.497 | **0.637** | **+0.140** |
| CNN2D (ours) | 0.530 | 0.472 | −0.058 |

Every external detector is **below chance** on CIFAKE. This is not random OOD failure — it's systematic inversion. The detectors learned "JPEG = real, smooth = fake." On CIFAKE, the real CIFAR-10 photos are heavily compressed blocky 32×32 JPEGs and the SD-generated fakes are smooth — so the rule fires backwards with confidence.

**GenImage (severe format + resolution bias):**
| Detector | CIFAKE AUC | GenImage orig | GenImage norm | Swing (CIF→GI) |
|----------|-----------|--------------|--------------|---------------|
| CNNDetection | 0.375 | 0.658 | 0.652 | +0.283 |
| FreqNet | 0.473 | **0.977** | **0.978** | +0.504 |
| NPR | 0.435 | **0.951** | **0.940** | +0.516 |
| UnivFD | 0.300 | **0.961** | **0.940** | +0.661 |
| FatFormer | 0.290 | **0.975** | **0.968** | +0.685 |
| B-Free | 0.497 | **0.919** | **0.919** | +0.422 |

The average swing is **+0.51 AUC** between CIFAKE and GenImage — the same detectors, same pipeline, only differing in whether the JPEG=real/PNG=fake shortcut is accessible.

**Critical nuance — GenImage normalized barely changes (max Δ = 0.02):** This means the high GenImage performance isn't *purely* format shortcuts — ADM genuinely leaves spectral artifacts that these detectors also pick up. The CIFAKE inversion is about detectors trained on full-resolution JPEGs firing backward on heavily-compressed 32×32 CIFAR-10, not symmetric format confusion.

**Defactify (resolution bias only, no format bias):** All detectors near chance (0.49–0.61), both original and normalized. Resolution bias alone doesn't trigger the JPEG/PNG shortcut. B-Free reaches 0.611 via genuine visual quality signals.

#### Trivial Format Baseline
`scripts/trivial_baseline.py` implemented. Predicts fake=PNG, real=JPEG using file extension only:
- CIFAKE: AUC = 0.500 (both classes JPEG — confirmed no format bias)
- Defactify: AUC = 0.500 (both classes JPEG — confirmed no format bias)
- GenImage: AUC = **1.000** — perfect separation from file extension alone

This is the single most devastating number in the paper for Act 1.

---

### What We Have

**Persisted on Modal volume (permanent):**
- `results/metrics.csv` — 45 rows, all detectors × datasets × conditions
- `results/<detector>_<dataset>_<condition>.csv` — per-image scores for every completed eval
- `results/logs/` — full stdout from every script run
- `results/spectra/*.npz` — raw spectral arrays
- `figures/` — all generated figures
- `detectors/` — all 6 external detector repos with weights
- `scripts/` — all pipeline scripts
- `results/cifake/best_{1d,2d}.pt`, `results/genimage/best_{1d,2d}.pt` — our trained checkpoints
- Paper draft: `Paper_template/author-kit-CVPR2026-v1-latex-/` — complete LaTeX source

**Ephemeral (lost on machine restart, must re-run):**
- `/root/normalized/{cifake,defactify,genimage}/` — normalized image dirs
- `/root/ablated/{cifake,defactify}/{low,mid,high}/` — CIFAKE + Defactify ablated dirs
- `/root/swapped/genimage/` — format-swapped GenImage

**Paper status:** Full draft written. Abstract, intro, related work, methods, all three experimental sections (Acts 1–3), conclusion — all present with real numbers. Currently readable and complete as a draft. One line in the conclusion says band ablation "will appear in the final version."

---

### What Is Planned

#### The One Remaining Experiment: Band Ablation

The conclusion of the paper explicitly promises this. The `scripts/band_ablation.py` is already written and parallelized.

**What needs to happen:**
1. Re-generate ablated image dirs (all ephemeral, all lost):
   - CIFAKE: ~5 min on CPU
   - Defactify: ~20 min on CPU
   - GenImage: ~30 min on H100 (or ~3 hrs on CPU — needs GPU)
2. Run 7 detectors × 3 bands × 3 datasets = **63 eval runs**
   - On CPU: ~2 days total
   - On H100: ~3–4 hours total

**Why GPU is needed:** The eval step is the bottleneck. `eval_external.py` already auto-detects CUDA (`default='cuda' if torch.cuda.is_available() else 'cpu'`). On GPU, ResNet50 and ViT-L/14 models process images ~15× faster than CPU. GenImage alone (127k images × 7 detectors × 3 bands = 21 eval runs) would take 10+ hours on CPU, ~1 hour on H100.

**After band ablation:** Add ablation results to `results/metrics.csv`, write a new subsection in `sec/4_experiments.tex`, and remove the "will appear in final version" hedge from the conclusion.

---

## The Band Ablation Experiment

### What It Is

Band ablation is a controlled frequency knockout experiment. For each image, we:
1. Compute the 2D FFT (decompose the image into its frequency components)
2. Zero out all frequency components in a specific radial band
3. Apply the inverse FFT to reconstruct a modified image
4. Evaluate detectors on the modified images

The three bands are defined as fractions of the maximum spatial frequency (r_max = half the image width for a 256×256 image):
- **Low band:** r < 0.2 × r_max — large-scale structure, coarse textures, global brightness gradients
- **Mid band:** 0.2 ≤ r < 0.6 × r_max — medium textures, edges, most semantic content
- **High band:** r ≥ 0.6 × r_max — fine detail, noise patterns, compression artifacts (JPEG DCT ringing, PNG filtering)

### What It Tests

Each detector produces three new AUC scores per dataset: one with low frequencies zeroed, one with mid zeroed, one with high zeroed.

**The expected pattern for a shortcut-reliant detector:**
- Remove **low**: AUC roughly unchanged (semantic content stays, shortcut may stay)
- Remove **mid**: AUC roughly unchanged (the shortcut is in fine detail, not semantics)
- Remove **high**: AUC **collapses toward chance** — the JPEG/PNG artifacts, compression ringing, and format-correlated fine detail are gone, leaving the detector with nothing to exploit

**The expected pattern for a genuine detector:**
- AUC remains relatively stable across all three conditions, because it's using mid-frequency generator-specific patterns (diffusion model spectral peaks, GAN fingerprints) that aren't concentrated in the high band alone

### Why It Matters for the Paper

The normalization experiment already showed the *correlation* — detectors score differently on biased vs. unbiased benchmarks. But band ablation is more surgical: it shows **which frequency bands each detector actually depends on**.

If FreqNet (AUC 0.977 on GenImage) completely collapses when the high band is zeroed but survives when the low or mid bands are zeroed, that is direct causal evidence that its near-perfect score comes from high-frequency format artifacts — not from learning what ADM-generated images look like.

It also answers a follow-up question the normalization results raised: if GenImage performance barely changes after normalization (max Δ = 0.02), are detectors using mid-frequency ADM artifacts or low-frequency content patterns? Band ablation breaks this apart cleanly.

### What Gets Added to the Paper

A fourth subsection in the experiments section, with a table like:

| Detector | Original | Low removed | Mid removed | High removed |
|----------|----------|------------|------------|-------------|
| CNNDetection | 0.658 | ? | ? | ? |
| FreqNet | 0.977 | ? | ? | ? |
| NPR | 0.951 | ? | ? | ? |
| UnivFD | 0.961 | ? | ? | ? |
| FatFormer | 0.975 | ? | ? | ? |
| B-Free | 0.919 | ? | ? | ? |

If the high-removed column shows large drops for the high-AUC detectors, it closes the causal argument: the frequency shortcuts are specifically in the high-frequency band, exactly where JPEG vs PNG differences live.
