# Research Plan: Diagnosing Frequency Shortcuts and Dataset Biases in AI-Generated Image Detectors

## Thesis

Existing AI-generated image detectors learn frequency shortcuts introduced by dataset construction artifacts rather than genuine generative signatures, and their reported performance substantially overstates real-world reliability.

## Three Contributions

1. A systematic audit showing CIFAKE, Defactify, and GenImage contain measurable format-induced spectral biases between real and fake splits
2. Empirical evidence that these biases inflate reported detector performance, quantified as AUC delta under format normalization
3. A frequency band ablation framework identifying which spectral regions each detector relies on, revealing shortcut-learning behavior

## Narrative Arc

| Act | Claim | Evidence |
|-----|-------|----------|
| 1 | The bias exists in the data | Spectral divergence between real/fake splits across 3 datasets |
| 2 | The bias propagates into detectors | Performance drops under format normalization |
| 3 | Detectors use the wrong frequency bands | Band ablation reveals high-freq shortcut reliance |

---

## Phase 0 — Orientation (Day 1)

- [ ] **0.1** Read Grommelt et al. "Fake or JPEG?" (arXiv 2403.17608) — note methodology, controlled experiments, metrics
- [ ] **0.2** Read Wang et al. "What do neural networks learn in image classification? A frequency shortcut perspective" (arXiv 2307.09829) — note framing and narrative structure
- [ ] **0.3** Audit CIFAKE and Defactify: for each split (real train, fake train, real test, fake test), tabulate image count, format distribution (JPEG vs PNG), resolution distribution, file size distribution → becomes **Table 1**
- [ ] **0.4** Clone and verify single-image inference for all three external detectors:

| Detector | Repo |
|----------|------|
| CNNDetection | `PeterWang512/CNNDetection` |
| FreqNet | `chuangchuangtan/FreqNet-DeepfakeDetection` |
| UnivFD | `Yuheng-Li/UniversalFakeDetect` |

---

## Phase 1 — Dataset Characterization (Days 2–5)

*Act 1 of the paper. No training — pure data analysis.*

- [ ] **1.1** Create `notebooks/dataset_characterization.ipynb`. For each dataset × split, compute:
  - [ ] Mean radial power spectrum (via `transforms.py`)
  - [ ] Variance of radial power spectrum
  - [ ] Mean 2D log-power spectrum heatmap
  - [ ] JPEG quality factor distribution (Pillow `_getexif()`)
  - [ ] Image resolution distribution
- [ ] **1.2** Plot mean radial spectrum: real vs fake per dataset. Compute L2 distance between curves in the high-frequency band (>80% of Nyquist) — this is the bias signal number
- [ ] **1.3** Write a Modal script to run the same characterization on GenImage remotely. Save spectra as `.npz`, pull locally
- [ ] **1.4** Produce **Figure 1**: 3-panel plot (CIFAKE / Defactify / GenImage), each panel = blue (real) vs red (fake) mean radial spectrum. High-freq divergence should be visually obvious

---

## Phase 2 — Normalization Pipeline (Days 6–8)

- [ ] **2.1** Write `scripts/normalize_dataset.py` (~60 lines):
  - Load → RGB (strip alpha/EXIF) → resize 256×256 bilinear → save as PNG
  - Identical processing for real and fake splits
  - No spatial augmentation (no crop/flip) — resize and re-encode only
  - Parameterize resize method for paper reporting
- [ ] **2.2** Run `normalize_dataset.py` on all datasets (CIFAKE, Defactify, GenImage)
- [ ] **2.3** Re-run characterization notebook on normalized data. Produce **Figure 2** (same layout as Figure 1, normalized data). Check whether bias signal weakens/disappears
- [ ] **2.4** Compute and tabulate L2 spectral divergence before vs after normalization per dataset → **"Spectral Divergence Before and After Normalization"** table

---

## Phase 3 — Detector Evaluation Harness (Days 9–14)

*Highest-risk phase. Budget extra 2–3 days for environment issues.*

- [ ] **3.1** Write `scripts/eval_external.py`: unified wrapper that takes `--detector {cnndetection,freqnet,univfd}`, `--data`, and `--weights`, outputs CSV of `(path, label, score)`. Match each repo's exact preprocessing
- [ ] **3.2** Evaluate all 5 detectors (3 external + CNN1D + CNN2D) on original test sets of CIFAKE and Defactify. Report AUC, Accuracy, MCC → **"Performance on Original Benchmarks"** baseline
- [ ] **3.3** Evaluate all 5 detectors on normalized test sets. Compute delta per detector → **Table 2**: AUC original, AUC normalized, delta. Bold the largest drops

---

## Phase 4 — Band Ablation (Days 15–20)

*Act 3 — most novel contribution.*

- [ ] **4.1** Write `scripts/band_ablation.py`: given an image + band spec, zero out that band in FFT space and reconstruct. Band definitions:
  - Low: \( r < 0.2 \cdot r_{\max} \)
  - Mid: \( 0.2 \cdot r_{\max} \leq r < 0.6 \cdot r_{\max} \)
  - High: \( r \geq 0.6 \cdot r_{\max} \)
  - Procedure: FFT → zero band → iFFT → clip [0, 255]
- [ ] **4.2** Generate ablated test sets for each band × each normalized dataset. Save to disk (`test_ablated_low/`, `test_ablated_mid/`, `test_ablated_high/`)
- [ ] **4.3** Evaluate all 5 detectors on all ablated sets. Fill the matrix:

| Detector | Full | Low Removed | Mid Removed | High Removed |
|----------|------|-------------|-------------|--------------|
| CNNDetection | | | | |
| FreqNet | | | | |
| UnivFD | | | | |
| CNN2D | | | | |
| CNN1D | | | | |

- [ ] **4.4** Produce **Figure 3**: AUC vs removed band, one line per detector. Shortcut-reliant detectors will drop steeply at "high removed"

---

## Phase 5 — Synthesis (Days 21–23)

- [ ] **5.1** Build **Figure 4** — 2×2 summary matrix. Classify each detector:

| | Not Shortcut-Reliant | Shortcut-Reliant |
|---|---|---|
| **Bias-Robust** | Ideal | Learned a real but meaningless frequency feature |
| **Bias-Sensitive** | Learning real signals but contaminated by construction | Dangerous — benchmarks look good, learning nothing real |

- [ ] **5.2** Place CNN1D and CNN2D in the matrix. Given cross-dataset failure (GenImage→CIFAKE, MCC ≈ 0.05), expect bias-sensitive + shortcut-reliant quadrant. Frame the narrative: built a detector → hit the wall → now have the diagnostic framework to explain why

---

## Phase 6 — Writing (Days 24–30)

- [ ] **6.1** Draft paper (4–6 pages, workshop format):
  - [ ] Abstract (150 words max)
  - [ ] Introduction — problem motivation, 3 contributions as bullet points
  - [ ] Related Work (0.5–1 page) — Grommelt, Wang ICCV 2023, evaluated detectors
  - [ ] Methodology — normalization pipeline, band ablation procedure
  - [ ] Experiments — Tables 1–2, Figures 1–4
  - [ ] Discussion — implications, limitations
  - [ ] Conclusion (2 paragraphs)
- [ ] **6.2** Finalize figures:

| Figure | Content |
|--------|---------|
| 1 | Mean radial spectra real vs fake, original data, 3 panels |
| 2 | Same as 1, normalized data — shows bias removal |
| 3 | AUC vs removed frequency band, one line per detector |
| 4 | 2×2 summary: bias-sensitive vs robust, shortcut-reliant vs not |

- [ ] **6.3** Submit to workshop target:

| Workshop | Fit |
|----------|-----|
| Workshop on Media Forensics (WMF) | Exact scope; key authors (Corvi et al.) publish here |
| Workshop on Responsible Generative AI | Broader audience; diagnostic work in scope |

---

## Critical Path

| Days | Phase | Key Deliverable |
|------|-------|-----------------|
| 1 | Orientation | Papers read, 3 detectors running locally |
| 2–5 | Characterization | `dataset_characterization.ipynb`, Figure 1 |
| 6–8 | Normalization | `normalize_dataset.py`, normalized datasets, spectral divergence table |
| 9–14 | Evaluation | `eval_external.py`, Table 2 (baseline + normalized) |
| 15–20 | Ablation | `band_ablation.py`, ablated test sets, Figure 3 |
| 21–23 | Synthesis | Figure 4, own-model cross-check |
| 24–30 | Writing | Full draft |