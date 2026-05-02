# Specter — Project Plan
**CS280 | Frequency-Domain Detection of AI-Generated Images**

---

## Project Overview

Specter is the frequency-domain analog of Wang et al. (CVPR 2020), extended to the diffusion era. The core question: do diffusion-generated images share universal spectral artifacts detectable by a frequency-only CNN — the same way GAN images did?

$$\text{Wang et al. 2020 (GANs)} \xrightarrow{\text{Specter}} \text{Diffusion Models}$$

**Two detector variants trained in parallel:**
- **1D CNN** — operates on the azimuthally averaged radial power spectrum
- **2D CNN** — operates on the full log-power spectrum heatmap

**No semantic features. No pretrained encoders. Frequency domain only.**

---

## Overall Arc

$$\text{Setup} \xrightarrow{\text{wk 1}} \text{Data Pipeline} \xrightarrow{\text{wk 2}} \text{EDA} \xrightarrow{\text{wk 3–4}} \text{Training} \xrightarrow{\text{wk 5}} \text{Experiments} \xrightarrow{\text{wk 6–7}} \text{Write-up}$$

---

## Phase 0 — Environment & Codebase Setup
**Days 1–3 · ~5 hours**

### Tasks

**01. Submit FaceForensics++ access request** ⚑ CRITICAL  
Do this before anything else. The form submission requires manual approval and can take 2–5 days. Missing this on Day 1 delays Phase 4.

**02. Create conda environment**
```bash
conda create -n specter python=3.10
conda activate specter
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install numpy scipy matplotlib scikit-learn jupyter tqdm Pillow h5py wandb
```

**03. Set up repo structure**
```
specter/
├── notebooks/        # EDA and exploration
├── src/
│   ├── dataset.py    # FrequencyDataset class
│   ├── models.py     # 1D and 2D CNN definitions
│   └── transforms.py # FFT, azimuthal averaging
├── scripts/
│   ├── train.py
│   └── eval.py
├── data/
│   ├── raw/
│   └── processed/
├── results/
├── figures/
└── CLAUDE.md
```

**04. Download CIFAKE from HuggingFace**  
No approval needed. ~1.7 GB. Use this to get the full pipeline working before GenImage arrives.

**05. Write CLAUDE.md**  
Project context file. Cover: architecture decisions, dataset quirks, novelty claims, must-cite papers.

---

## Phase 1 — Data Pipeline
**Days 3–10 · ~12 hours**

### Core transform applied to every image

$$S(u, v) = \log\!\left(1 + \left|\mathcal{F}\{I\}(u, v)\right|^2\right)$$

### Tasks

**01. Implement `FrequencyDataset` class**  
Loads image → 2D FFT log-magnitude spectrum → 1D azimuthal average → returns both representations + label.

**02. Implement 2D log-power spectrum**  
Center-shifted, normalized. Handle variable image sizes with fixed crop/resize (224×224 recommended).

**03. Implement 1D azimuthal averaging**

$$A(r) = \frac{1}{2\pi r} \oint_{\|(u,v)\|=r} S(u,v)\, d\theta$$

Bin by radial distance from DC component. Returns 1D vector of length $\lfloor H/2 \rfloor$. Use vectorized NumPy.

**04. Write DataLoader + train/val/test splits**  
Stratified splits. **Cross-generator split:** train on SD v1.4 only, hold out DALL-E and Midjourney for zero-shot evaluation.

**05. Download GenImage subset**  
Start with SD v1.4 + SD v1.5 + real images. Add DALL-E and Midjourney generators once the pipeline works.

**06. Write sanity checks**  
Plot a few 2D spectra and radial profiles. Confirm label balance, spectrum range, no NaNs.

---

## Phase 2 — EDA & Spectral Characterization
**Days 8–16 · ~10 hours**

This phase is not setup — it is Contribution 1. The figures produced here go directly into the paper.

### Tasks

**01. Compute mean log-power spectra per generator**  
Average over $N \geq 500$ images per generator. This is **Figure 1** of the paper — the visual analog of Wang et al. Figure 3.

$$\bar{S}_{\text{gen}}(u,v) = \frac{1}{N}\sum_{i=1}^{N} \log\!\left(1 + \left|\mathcal{F}\{I_i\}(u,v)\right|^2\right)$$

**02. Compute spectral residuals**

$$\Delta S_{\text{gen}}(u,v) = \bar{S}_{\text{gen}}(u,v) - \bar{S}_{\text{real}}(u,v)$$

Visualize as heatmaps. Identify shared peaks vs generator-specific patterns.

**03. Compute 1D radial profiles** — **Figure 2**

$$A_{\text{gen}}(r) = \frac{1}{2\pi r} \oint_{\|(u,v)\|=r} \Delta S_{\text{gen}}(u,v)\, d\theta$$

If diffusion models share excess energy at radius $r^*$, this is your central empirical claim.

**04. Statistical tests on spectral differences**  
KS test or Welch t-test on radial bins (real vs. fake). Quantifies significance.

**05. Export publication-quality figures**  
EDA notebook → figures. Invest in clean matplotlib/seaborn styling. These figures go directly into the paper.

**06. Cross-check against "Fake or JPEG?" findings**  
GenImage has JPEG compression bias. Flag any spectrum distortions that could be JPEG artifacts vs genuine generator artifacts before drawing conclusions.

---

## Phase 3 — Model Training
**Days 14–25 · ~15 hours**

### Architecture specs

**1D CNN**
- Input: radial profile vector, length ~112–256
- Architecture: 4–5 `Conv1D` layers + BatchNorm + ReLU, global average pool, 2-class head
- ~500k parameters

**2D CNN**
- Input: log-power spectrum, shape $(H \times W, 1)$
- Architecture: 5 `Conv2D` layers + BatchNorm + ReLU, adaptive average pool, 2-class head
- ~2M parameters

**Why 1D vs 2D is scientifically motivated:**  
The 1D model captures only what is radially universal (isotropic structure). The 2D model additionally exploits anisotropic, generator-specific orientation patterns. This is a test of whether diffusion artifacts are isotropic or anisotropic — not an arbitrary engineering comparison.

### Tasks

**01. Implement 1D CNN in PyTorch**

**02. Implement 2D CNN in PyTorch**

**03. Training script with logging**  
WandB or TensorBoard. Log train/val loss, AUC, accuracy per epoch. Checkpoint best val-AUC model.

**04. Train both models on CIFAKE first**  
Fastest iteration cycle. Debug architectures completely before touching GenImage.

**05. Hyperparameter sweep**  
LR $\in \{10^{-3}, 3 \times 10^{-4}, 10^{-4}\}$, batch size $\in \{64, 128\}$, weight decay. Keep it simple — 2–3 combos per model.

**06. Train both models on GenImage (SD v1.4 subset)**  
Full run. In-distribution benchmark. Record AUC, accuracy, EER.

---

## Phase 4 — Experiments & Robustness
**Days 22–35 · ~15 hours**

### Tasks

**01. Cross-generator generalization** — **Table 1** ⚑ CENTRAL EXPERIMENT  
Train on SD v1.4 → test on DALL-E, Midjourney, ADM, GLIDE, SD v1.5. Direct analog of Wang et al.'s ProGAN → everything else experiment.

**02. JPEG compression robustness**  
Degrade test images at quality $\in \{90, 75, 50, 30\}$. Plot AUC degradation curve for both models.

**03. Upscaling robustness**  
Bicubic upscale test images at $1.5\times$, $2\times$. Upscaling smears frequency artifacts — quantify how much.

**04. Frequency-space PGD adversarial attack** ⚑ CONTRIBUTION 2  
Add perturbation in the Fourier domain:

$$x_{\text{adv}} = \mathcal{F}^{-1}\!\left(\mathcal{F}(x) + \varepsilon \cdot \text{sign}\!\left(\nabla_{\mathcal{F}(x)}\,\mathcal{L}\right)\right)$$

This is the interpretability contribution. The attack reveals which spectral frequencies drive the classifier's decision.

**05. Visualize adversarial frequency perturbations** — **Figure 3**  
Show which spectral bins PGD modifies. Overlay on radial profile from Figure 2. If attacked frequencies align with the EDA spectral peaks, that is the interpretability payoff.

**06. FaceForensics++ evaluation** *(if access arrives)*  
Inference only. Measures generalization to the facial forgery domain — demonstrates breadth beyond GenImage.

---

## Phase 5 — Analysis & Comparison
**Days 30–38 · ~8 hours**

### Tasks

**01. 1D vs 2D head-to-head comparison table** — **Contribution 1**

| Metric | 1D CNN | 2D CNN |
|---|---|---|
| In-distribution AUC | | |
| Cross-generator AUC | | |
| JPEG robustness (q=50) | | |
| Parameters | ~500k | ~2M |
| Inference speed | | |

**02. Ablation: spectrum resolution**  
Vary radial bins $\in \{64, 128, 256\}$ for 1D. Vary input resolution $\in \{64^2, 128^2, 224^2\}$ for 2D.

**03. Failure mode analysis**  
Which generators fool each model? Cross-reference failures with EDA spectral plots — do generators without the shared spectral peak also evade detection?

**04. SPAI comparison**  
Reproduce or cite SPAI CVPR 2025 numbers. Honest comparison of where Specter matches, exceeds, or falls short.

---

## Phase 6 — Paper Write-up
**Days 35–45 · ~20 hours**

### Structure

**Abstract + Introduction**  
Lead with the Wang et al. parallel:
> "Wang et al. showed GAN images share detectable spectral artifacts. We ask: does this hold for diffusion models?"  

State contributions explicitly. Do not bury them.

**Related Work** (~1.5 pages)  
Organize as: (1) frequency-only detectors, (2) semantic/VLM detectors, (3) robustness work. Must cite:
- Durall et al., CVPR 2020
- Wang et al., CVPR 2020
- Synthbuster
- SPAI, CVPR 2025
- UGAD
- "Fake or JPEG?", ECCV 2024

**Method**  
Formalize: 2D spectrum computation, azimuthal averaging, both CNN architectures. Include the math with full notation.

**Experiments**  
- Table 1: cross-generator generalization
- Table 2: JPEG and upscaling robustness
- Figure 1: mean spectra + residuals per generator
- Figure 2: radial profiles stacked comparison
- Figure 3: adversarial frequency perturbations

**Discussion + Conclusion**  
Commit to an interpretation: do diffusion models share a universal spectral signature (like GANs), or not? Both outcomes are valid scientific claims. Do not hedge.

---

## Novelty Claims (hardened)

These are your answers when the professor asks "what's new here?":

**Claim 1 — Systematic 1D vs 2D head-to-head**  
No existing paper directly compares a 1D radial-profile CNN against a 2D full-spectrum CNN under controlled conditions with theoretical motivation. The comparison is grounded in the isotropic vs anisotropic spectral structure question, not arbitrary engineering.

**Claim 2 — Fourier-space adversarial attacks for interpretability**  
No existing paper applies frequency-space PGD specifically to a frequency-only detector for the purpose of identifying which spectral bands drive detection. This combination is absent from prior work.

---

## The Three Load-Bearing Figures

Your paper lives or dies on these:

**Figure 1** — Mean log-power spectrum heatmaps + $\Delta S_{\text{gen}}(u,v)$ residuals. Visual proof that artifacts exist and differ across generators. Template: Wang et al. Figure 3.

**Figure 2** — Radial profiles $A_{\text{gen}}(r)$ stacked on one plot with the real-image baseline. If diffusion models share a bump at a common radius $r^*$, this is where it appears.

**Figure 3** — Adversarial Fourier perturbation visualization overlaid on Figure 2 profiles. If attacked frequencies align with the spectral peaks, that is the interpretability result.

---

## Must-Cite Prior Art

| Paper | Venue | Why |
|---|---|---|
| Wang et al. | CVPR 2020 | Primary structural parallel (GAN frequency artifacts) |
| Durall et al. | CVPR 2020 | Foundational frequency-domain detection, azimuthal averaging |
| Synthbuster | — | Diffusion-era frequency artifact baseline |
| SPAI | CVPR 2025 | Current SOTA frequency-only detector |
| UGAD | — | Radial integral operation comparison |
| "Fake or JPEG?" | ECCV 2024 | Reveals dataset bias in GenImage — critical for evaluation validity |

---

## Constraints (never violate)

- No semantic features anywhere in the detector pipeline
- No pretrained visual encoders (CLIP, DINOv2, ViT, ResNet, etc.)
- All CNN inputs derived from the Fourier transform of images only
- The 1D/2D comparison is a sub-contribution within the Wang-to-Specter narrative, not the top-level framing
