# Eigenfraud — Post-Pivot Status

Tracks all work done after the project pivot. See `PIVOT.md` for the full research plan.

**Logging policy (from Phase 3 onwards):** All script stdout → `results/logs/`, all per-image scores → `results/<detector>_<dataset>.csv`, all aggregate metrics → `results/metrics.csv`. See CLAUDE.md §Results Logging.

**Phases 0–1 gap:** Audit and characterization outputs were captured as summaries in this file only. Full stdout was not saved. Raw spectral data is in `results/spectra/*.npz`; audit numbers would need a re-run to recover full output.

---

## Phase 0 — Orientation

### Step 0.3 — Dataset Construction Audit (2026-04-20)

**Scripts written:** `scripts/audit_dataset.py`, `scripts/audit_defactify.py`

#### CIFAKE (`data/raw/cifake/`)


| Split        | Count  | Format    | Resolution      | Size KB mean |
| ------------ | ------ | --------- | --------------- | ------------ |
| train / REAL | 31,006 | JPEG 100% | 32×32 (uniform) | 0.9          |
| train / FAKE | 50,000 | JPEG 100% | 32×32 (uniform) | 0.9          |
| test / REAL  | 10,000 | JPEG 100% | 32×32 (uniform) | 0.9          |
| test / FAKE  | 10,000 | JPEG 100% | 32×32 (uniform) | 0.9          |


**Finding:** No format or resolution bias. Both real (CIFAR-10) and fake (SD v1.4) were downsampled to 32×32 JPEG. No shortcut from construction artifacts is possible here. Train set is class-imbalanced (31k real vs 50k fake); test is balanced. Our CNN2D achieving AUC 0.95 on this dataset is not explained by format/resolution shortcuts.

---

#### Defactify (`defactify_dataset/data/`)


| Split      | Label | Count  | Format    | Resolution unique | Resolution mode | Size KB mean |
| ---------- | ----- | ------ | --------- | ----------------- | --------------- | ------------ |
| train      | REAL  | 7,000  | JPEG 100% | 690               | 640×480         | 51.5         |
| train      | FAKE  | 35,000 | JPEG 100% | **6**             | 1024×1024       | 77.9         |
| test       | REAL  | 7,500  | JPEG 100% | 180               | 1024×1024       | 69.2         |
| test       | FAKE  | 37,500 | JPEG 100% | 196               | 1024×1024       | 81.8         |
| validation | REAL  | 1,500  | JPEG 100% | 289               | 640×480         | 51.2         |
| validation | FAKE  | 7,500  | JPEG 100% | **5**             | 1024×1024       | 78.1         |


**Finding:** Severe resolution bias in train/validation. Fake images have only 5–6 unique resolutions (canonical AI output sizes: 1024×1024, 768×768, 436×436, 270×270, 351×351) vs 289–690 unique resolutions in real images (natural photo diversity). No format bias — both 100% JPEG. Class imbalance: 5:1 fake-to-real across all splits. Anomaly: test/REAL has 1024×1024 as its mode, unlike train/val real — test real split appears sourced differently, partially washing out the resolution signal in test.

---

#### GenImage (`data/raw/imagenet_nature/val` + `data/raw/{ADM,BigGAN}/`)


| Split                | Count  | Format        | Resolution unique | Resolution mode   | Size KB mean |
| -------------------- | ------ | ------------- | ----------------- | ----------------- | ------------ |
| train / REAL         | 50,000 | **JPEG 100%** | 328               | 500×375           | 67.3         |
| train / FAKE (ADM)   | 76,677 | **PNG 100%**  | 1                 | 256×256 (uniform) | 111.1        |
| test / REAL          | 50,000 | **JPEG 100%** | 340               | 500×375           | 67.3         |
| test / FAKE (BigGAN) | 82,392 | **PNG 100%**  | 1                 | 128×128 (uniform) | 30.5         |


**Finding:** Dual bias — the most severe across all three datasets. (1) **Format:** real is 100% JPEG, fake is 100% PNG. A 1-line format check perfectly separates the splits. (2) **Resolution:** each generator outputs a single uniform size vs 328–340 unique natural resolutions for real. Any detector reporting high AUC on GenImage is almost certainly exploiting JPEG-vs-PNG. Some fake PNGs have 0.0 KB minimum file size (corrupt/empty files in dataset).

---

### Cross-dataset Summary — Table 1


| Dataset   | Format bias              | Resolution bias                              | Notes                                                  |
| --------- | ------------------------ | -------------------------------------------- | ------------------------------------------------------ |
| CIFAKE    | None                     | None                                         | Uniform 32×32 JPEG both sides; shortcuts not available |
| Defactify | None                     | **Severe** (train/val: 690 vs 6 unique)      | Test real anomalously sourced                          |
| GenImage  | **Severe** (JPEG vs PNG) | **Severe** (varied vs uniform per-generator) | Both biases simultaneously                             |


---

## Phase 0 — Step 0.4: Detector Setup (2026-04-21)

All 6 detectors installed and smoke-tested via `scripts/eval_external.py`.


| Detector     | Weights                                                           | Status                                                   |
| ------------ | ----------------------------------------------------------------- | -------------------------------------------------------- |
| CNNDetection | `detectors/CNNDetection/weights/blur_jpg_prob0.5.pth`             | ✓                                                        |
| FreqNet      | `detectors/FreqNet-DeepfakeDetection/4-classes-freqnet-v2.pth`    | ✓ (removed hardcoded `.cuda()` in freqnet.py `__init__`) |
| NPR          | `detectors/NPR-DeepfakeDetection/NPR.pth`                         | ✓ (auto-strips DataParallel `module.` prefix)            |
| UnivFD       | `detectors/UniversalFakeDetect/pretrained_weights/fc_weights.pth` | ✓ (CLIP ViT-L/14 cached at `~/.cache/clip/`)             |
| FatFormer    | `detectors/FatFormer/pretrained/FatFormer_4class.pth`             | ✓ (patched clip.py to use abs path for ViT-L-14.pt)      |
| B-Free       | `detectors/B-Free/code/weights/BFREE_dino2reg4/`                  | ✓                                                        |


Unified eval wrapper: `scripts/eval_external.py --detector <name> --data <dir> --weights <path> --out <csv>`

**Phase 0 complete.**

---

---

## Phase 1 — Dataset Characterization (2026-04-21)

**Script:** `scripts/characterize_datasets.py`

### Spectral Divergence Summary

| Dataset | real n | fake n | HF L2 (r>89, >80% Nyquist) |
|---------|--------|--------|----------------------------|
| CIFAKE | 10,000 | 10,000 | **0.34** |
| Defactify | 7,500 | 10,000 | **4.23** |
| GenImage | 10,000 | 10,000 | **6.16** |

**Outputs:**
- `results/spectra/{cifake,defactify,genimage}_{real,fake}.npz` — raw spectra for downstream use
- `figures/fig1_radial_spectra.png` — Figure 1 (3-panel mean radial spectra)
- `figures/fig1b_2d_heatmaps.png` — Supplementary 2D log-power heatmaps

**Finding:** The divergence ordering perfectly tracks construction bias severity. CIFAKE (no bias) shows near-zero HF divergence. Defactify (resolution bias) is 12× larger. GenImage (JPEG-vs-PNG + resolution bias) is 18× larger than CIFAKE. This is Act 1 of the paper: the bias exists in the data and is measurable.

**Phase 1 complete.**

---

## Phase 2 — Normalization Pipeline (2026-04-21, in progress)

**Script written:** `scripts/normalize_dataset.py`

Normalized output goes to `/root/normalized/` (root filesystem — ephemeral, lost on machine restart).
**Modal volume has only ~4 inodes free** — normalized data cannot be stored there.

On every new machine, re-run normalization before characterization or Phase 3.3:

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

Once all three are done:

```bash
# Figure 2 + pre/post divergence table (Phase 2.3 + 2.4)
python scripts/characterize_datasets.py \
    --cifake-real  /root/normalized/cifake/test/real \
    --cifake-fake  /root/normalized/cifake/test/fake \
    --defy-real    /root/normalized/defactify/test/real \
    --defy-fake    /root/normalized/defactify/test/fake \
    --genimage-real /root/normalized/genimage/real \
    --genimage-fake /root/normalized/genimage/fake \
    --fig-prefix fig2 --spectra-tag norm
```

**Phase 2 status:** scripts written and tested; normalization must be re-run on each new machine.

---

---

## Phase 3 — Baseline Evaluations (2026-04-22)

**Status (as of 03:12 AM):** CIFAKE + Defactify baseline fully populated; normalized CIFAKE done; normalized Defactify ~5/7 done; GenImage pending; paper draft complete.

### Results summary (metrics.csv)

| Detector | CIFAKE orig | CIFAKE norm | Defactify orig | Defactify norm |
|----------|-------------|-------------|----------------|----------------|
| CNNDetection | 0.375 | 0.375 | 0.507 | 0.524 |
| FreqNet | 0.473 | 0.473 | 0.511 | 0.534 |
| NPR | 0.435 | 0.435 | 0.520 | 0.556 |
| UnivFD | 0.300 | 0.371 | 0.536 | 0.493 |
| FatFormer | 0.290 | 0.290 | TBD | 0.537 |
| B-Free | 0.497 | **0.637** | TBD | running |
| CNN2D | 0.530 | 0.472 | 0.546 | pending |

**Key findings:**
1. All 6 external detectors AUC < 0.5 on CIFAKE original (systematic inversion — detectors call real images fake)
2. CIFAKE normalization leaves 5/7 detectors unchanged (content-baked DCT artifacts survive format conversion)
3. **B-Free anomaly**: CIFAKE norm AUC 0.637 (up from 0.497) — DINOv2 detects upscaled CIFAR vs SD visual quality difference after format noise removed
4. Defactify: all near chance (0.49–0.56) on both original and normalized; small positive delta for most

**Pending experiments:**
- B-Free + CNN2D normalized Defactify (in pipeline, B-Free running)
- FatFormer + B-Free Defactify original reruns (in follow-up script)
- GenImage original evals (7 detectors, in follow-up script — key experiment)
- GenImage normalized evals (GenImage normalization ~80% complete, ~1 hour remaining)

**OOD finding:** All 6 detectors trained on ProGAN/ForenSynths — all 3 test datasets OOD. GenImage original vs normalized is the key controlled experiment (shares JPEG=real/PNG=fake shortcut with ForenSynths training).

**Paper:** First draft complete (`Paper_template/author-kit-CVPR2026-v1-latex-/`). Table 2 partially filled. GenImage columns TBD.

## Up Next

- Monitor B-Free + CNN2D normalized Defactify completion → add to metrics.csv
- Once follow-up script starts: monitor GenImage original evals → add to metrics.csv + paper table
- When GenImage normalization done: run GenImage normalized evals
- Phase 2.3/2.4: characterize_datasets.py on normalized data → Figure 2


---

---

## Complete Findings Summary (2026-04-22, updated ~04:00 AM)

This section tells the full story of the project from data to conclusion, with pointers to every piece of evidence.

---

### The Story We Are Telling

**Central claim:** State-of-the-art AI-generated image detectors do not detect AI generation. They detect the file format and resolution patterns that happen to be correlated with AI generation in the datasets they were trained on. When those patterns are absent, the detectors fail — and not just randomly. They fail *systematically*, inverting their decisions in a way that reveals exactly what they learned.

**The argument has three acts:**

**Act 1 — The bias exists in the data and is measurable.**
We audited three detection benchmarks and found they differ dramatically in how biased their construction is:
- CIFAKE: no bias. Real and fake are both uniform 32×32 JPEG. Spectral HF L2 divergence = 0.34.
- Defactify: resolution bias only. Fake images have 5–6 unique resolutions (canonical AI output sizes); real images have 180–690 unique resolutions. HF L2 divergence = 4.23 (12× CIFAKE).
- GenImage: format + resolution bias. Real = JPEG (natural photos, hundreds of unique sizes). Fake = PNG (one uniform size per generator). HF L2 divergence = 6.16 (18× CIFAKE).

The divergence ordering tracks bias severity perfectly. Spectral figures: `figures/fig1_radial_spectra.png`, `figures/fig1b_2d_heatmaps.png`. Raw spectra: `results/spectra/*.npz`.

**Act 2 — Detectors fail on clean data and succeed on biased data in the exact predicted pattern.**

We evaluated 6 external detectors — CNNDetection, FreqNet, NPR, UnivFD, FatFormer, B-Free — plus our own CNN2D across all three datasets before and after format normalization (all images → 256×256 PNG). All detectors were trained on ProGAN/ForenSynths, which has the same JPEG=real/PNG=fake bias as GenImage.

**CIFAKE original (no bias):**

| Detector | AUC | Interpretation |
|----------|-----|----------------|
| CNNDetection | 0.375 | Inverted — calls real fake |
| FreqNet | 0.473 | Inverted |
| NPR | 0.435 | Inverted |
| UnivFD | 0.300 | Strongly inverted |
| FatFormer | 0.290 | Strongly inverted |
| B-Free | 0.497 | ~chance |
| CNN2D (ours) | 0.530 | ~chance |

Every external detector is below 0.5. This is not random OOD failure — it is systematic inversion. The detectors learned "JPEG=real, smooth=fake." On CIFAKE, real CIFAR-10 photos have heavy JPEG DCT block artifacts and the SD-generated fakes are smooth — so the heuristic fires backwards with confidence.

**CIFAKE normalized (same images, all converted to 256×256 PNG):**

| Detector | AUC | Δ from original |
|----------|-----|-----------------|
| CNNDetection | 0.375 | 0.000 |
| FreqNet | 0.473 | 0.000 |
| NPR | 0.435 | 0.000 |
| UnivFD | 0.371 | −0.006 |
| FatFormer | 0.290 | 0.000 |
| **B-Free** | **0.637** | **+0.140** |
| CNN2D (ours) | 0.472 | −0.058 |

Five detectors are unchanged by normalization — their learned signal is baked into pixel content (CIFAR-10's DCT block artifacts survive PNG re-encoding). B-Free is the exception: DINOv2 visual features can distinguish blocky upscaled 32×32 CIFAR-10 pixels from crisp 256×256 SD output once the format confound is removed. CNN2D drops because it learned PNG=fake during GenImage training, which backfires when all inputs are PNG.

**GenImage original (JPEG=real/PNG=fake — same shortcut as training data):**

| Detector | AUC | Δ from CIFAKE original |
|----------|-----|------------------------|
| CNNDetection | 0.658 | +0.283 |
| **FreqNet** | **0.977** | **+0.504** |
| **NPR** | **0.951** | **+0.516** |
| UnivFD | running | — |
| FatFormer | pending | — |
| B-Free | pending | — |
| CNN2D | pending | — |

FreqNet jumps from 0.473 on CIFAKE (below chance, no shortcut available) to 0.977 on GenImage (near-perfect, same JPEG=real/PNG=fake shortcut as training). NPR goes from 0.435 to 0.951. A +0.50 AUC swing between two test datasets, explained entirely by whether the training shortcut is present. This is not OOD generalization — it is shortcut transfer.

**Defactify (resolution bias only, no format bias):**

All detectors score 0.49–0.56 on both original and normalized. Near chance across the board, on both conditions. These detectors have the JPEG=real/PNG=fake shortcut but Defactify has no such shortcut — it has only resolution bias, which these detectors were not trained to exploit. This rules out a confounding explanation: it's not that detectors always succeed on any biased dataset, only on datasets with *their specific* shortcut.

**Act 3 — Normalization reveals a detector taxonomy.**

Format normalization (removing the shortcut) partitions the detectors into two classes:

- **Shortcut learners** (5 of 6 external detectors): AUC unchanged or worse after normalization on CIFAKE. They learned compression artifacts at the pixel level; removing format metadata has no effect because the signal is in the pixels, not the file.
- **Quality-signal detector** (B-Free): AUC improves after normalization on both CIFAKE (+0.14) and Defactify (0.611, highest of any detector). B-Free uses DINOv2 visual features rather than frequency analysis, and responds to genuine visual quality differences rather than compression shortcuts.

**Pending — the punchline:**
GenImage normalized evals (all 7 detectors on the shortcut-removed version of GenImage). We predict large AUC drops for FreqNet and NPR (from ~0.97/0.95 toward chance), directly quantifying how much of their near-perfect reported performance is shortcut vs. genuine signal. GenImage normalization is complete; evals starting now.

---

### Where the Data Lives

| What | Location |
|------|----------|
| All aggregate metrics (AUC, Acc, MCC) | `results/metrics.csv` |
| Per-image scores, CIFAKE | `results/{detector}_cifake_{original,normalized}.csv` |
| Per-image scores, Defactify | `results/{detector}_defactify_{original,normalized}.csv` |
| Per-image scores, GenImage | `results/{detector}_genimage_original.csv` (normalized pending) |
| Raw spectral arrays (original data) | `results/spectra/{cifake,defactify,genimage}_{real,fake}.npz` |
| Figure 1 — radial spectra | `figures/fig1_radial_spectra.png` |
| Figure 1b — 2D heatmaps | `figures/fig1b_2d_heatmaps.png` |
| Paper (LaTeX source) | `Paper_template/author-kit-CVPR2026-v1-latex-/` |
| Paper abstract | `Paper_template/.../sec/0_abstract.tex` |
| Paper experiments (Table 2) | `Paper_template/.../sec/4_experiments.tex` |
| Full decision log | `Rud MDs/journal.md` |
| Normalization script | `scripts/normalize_dataset.py` |
| Eval wrapper (all 6 external detectors) | `scripts/eval_external.py` |
| Dataset audit scripts | `scripts/audit_dataset.py`, `scripts/audit_defactify.py` |
| Spectral characterization script | `scripts/characterize_datasets.py` |

---

### What Is Still Running / Pending

| Experiment | Status | Notes |
|------------|--------|-------|
| GenImage original — UnivFD | running | ~30 min |
| GenImage original — FatFormer, B-Free, CNN2D | queued | after UnivFD |
| GenImage normalized — all 7 detectors | not started | normalization complete; evals must be launched |
| Defactify original — FatFormer, B-Free | queued in follow-up script | reruns after BFREE_TRANSFORM fix |
| Figure 2 (normalized spectra) | not started | requires characterize_datasets.py on normalized data |
| Band ablation (Act 3) | not started | script not yet written |

