# Eigenfraud: A Complete Technical Explainer

This document walks through everything that happens in this project from raw pixel to classification decision — the math, the intuition, what the model learns, how you can see what it learned, and what the hyperparameters mean.

---

## 1. Why the Frequency Domain?

Before touching any code, it is worth understanding why we are doing this at all instead of just training a CNN directly on pixel values.

Generative models — diffusion models, GANs — produce images by running a learned process that operates largely in a compressed latent space and then decodes back to pixels. That decoding process has structure. Specifically:

- **Upsampling artifacts.** Nearly every generator uses some form of learned upsampling (transposed convolutions, bilinear upsampling followed by convolution, or attention mechanisms). These operations repeat patterns at regular spatial intervals, which means they inject energy at specific *frequencies*.
- **Spectral bias of neural networks.** Neural networks tend to learn low-frequency functions first and struggle to reproduce fine high-frequency texture faithfully. Real photographs contain a characteristic distribution of high-frequency energy (from grain, fine texture, noise). Fake images often have too little or anomalously structured high-frequency content.
- **The 1/f law of natural images.** Real photographs obey a statistical law: the power spectrum falls off roughly as 1/f² — lower frequencies have more energy, and it drops predictably as you go toward higher frequencies. Any deviation from this is a signal that the image may not be natural.

The hypothesis of this project is that these spectral signatures are *consistent enough across generators* that a model trained on them generalises better than one trained on pixels alone. Pixel-space detectors tend to overfit to specific generator artifacts (JPEG compression artifacts, specific color profiles) that change when the generator is updated. The frequency spectrum is a more fundamental representation.

---

## 2. The Preprocessing Pipeline, Step by Step

### Step 1: Load and Convert to Grayscale

```
src/transforms.py — to_grayscale_array(img, size=224)
```

```python
img = img.convert("L").resize((size, size), Image.LANCZOS)
return np.array(img, dtype=np.float32)
```

**What happens:**  
A colour image (RGB, three channels of values 0–255) is collapsed to a single luminance channel using PIL's `"L"` mode, which computes:

```
L = 0.299·R + 0.587·G + 0.114·B
```

This is the standard human-perception-weighted luminance formula. We then resize to 224×224 using LANCZOS resampling (a high-quality filter that avoids aliasing).

**Why grayscale?**  
Spectral artifacts from generators show up in luminance. Doing a separate FFT per channel would triple the computation for minimal gain — the fingerprints we care about are spatial-structural, not colour-specific. Also, grayscale gives us a single 2D array to FFT cleanly.

**Why 224×224?**  
Conventional. ImageNet-pretrained models use this size, and it gives a spectrum with r_max = 112 frequency bins after azimuthal averaging — large enough to capture meaningful structure, small enough to be fast.

**Output:** float32 array of shape `(224, 224)`, values in `[0.0, 255.0]`.

---

### Step 2: The 2D Discrete Fourier Transform

```
src/transforms.py — log_power_spectrum_2d(gray)
```

```python
F = np.fft.fft2(gray)
F_shifted = np.fft.fftshift(F)
power = np.abs(F_shifted) ** 2
log_power = np.log1p(power).astype(np.float32)
```

**What the FFT does:**  
The Discrete Fourier Transform decomposes a 2D signal into a sum of 2D sinusoids. For an image of size H×W, it produces a complex-valued array of the same size H×W. Each entry at position `(u, v)` is a complex number whose:
- **magnitude** tells you how much of that particular frequency is present in the image
- **phase** tells you where in space that frequency pattern is aligned

The formula for the 2D DFT is:

```
F(u, v) = Σ_x Σ_y  f(x, y) · exp(-2πi · (ux/H + vy/W))
```

where `f(x, y)` is the pixel value at position `(x, y)`.

`u` and `v` are the spatial frequencies in the horizontal and vertical directions respectively. `u=0, v=0` is the DC component — the average brightness of the image. As `u` and `v` increase, you are describing finer and finer spatial patterns.

**fftshift:**  
By default, numpy's FFT output puts the DC component (zero frequency) in the top-left corner, with frequencies arranged in a non-intuitive order. `fftshift` rearranges this so the DC component is in the *centre* of the array, with low frequencies surrounding it and high frequencies at the edges. This centred layout is much more natural to look at and to learn from.

**Power spectrum:**  
We take `|F(u,v)|²` — the squared magnitude of each complex number. This discards phase and gives us the *power* at each frequency. Phase carries information about where in the image a particular pattern occurs, but we want to know *whether* certain frequencies are present, not where. For a statistical fingerprint of a generator, power is what matters.

**Log compression:**  
The power spectrum has enormous dynamic range — the DC component can be millions of times larger than high-frequency components. `log1p(x) = log(1 + x)` compresses this range into something manageable for a neural network to learn from. The `+1` avoids `log(0)` when power is zero.

**Output:** float32 array of shape `(224, 224)`, centred (DC in middle), values roughly in `[0, 20]`.

---

### Step 3: Azimuthal Averaging — Getting the 1D Profile

```
src/transforms.py — azimuthal_average_fast(spectrum)
```

```python
r = np.round(np.sqrt(xx**2 + yy**2)).astype(np.int32).ravel()
r_max = min(H, W) // 2   # = 112
# bin pixels by their integer radius, average power within each bin
profile = sums / counts   # shape: (112,)
```

**What this does:**  
The 2D spectrum is symmetric around the centre (because the image is real-valued, not complex). We exploit this by computing the average power at each integer distance r from the centre. The result is a 1D curve: `A(r) = mean of all pixels at radius r`.

This is called an *azimuthal average* — you are averaging around the circle at each radius, collapsing the 2D directional information into a single radial profile.

**What the profile looks like:**  
For natural images it falls roughly as a power law — high near the centre (low frequencies, high power) and dropping as you go outward. If you plot it on a log-log scale it is approximately a straight line with negative slope (close to -2, from the 1/f² law).

**What you lose:**  
Directional information. If a GAN produces a horizontal banding artifact (strong energy along the u-axis), it will appear in the 2D spectrum as spikes along a specific direction. The azimuthal average collapses that into the radial bin at that radius — the information is partially preserved (those bins will have elevated power) but you cannot tell which *direction* the artifact came from.

**Why keep it:**  
It is a much simpler signal (112 numbers instead of 224×224) that is rotation-invariant. For generators whose artifacts are isotropic (present in all directions equally), the 1D profile is sufficient. It also trains faster.

**Output:** float32 array of shape `(112,)`.

---

## 3. What the Two Models Receive

Every image goes through the full pipeline and produces *both* representations. The training script then selects one based on `--model`:

| Flag | Input tensor | Shape |
|------|-------------|-------|
| `--model 1d` | azimuthal profile | `(112,)` |
| `--model 2d` | log-power spectrum | `(1, 224, 224)` |

The `1` in `(1, 224, 224)` is the channel dimension — it is a single-channel "image" of the spectrum, analogous to a grayscale image, fed into a 2D CNN.

---

## 4. The Math of a Convolutional Neural Network

Before describing our specific architectures, here is what each layer type does mathematically.

### 4.1 Convolution

A 1D convolution with kernel of size k applied to input `x` of shape `(C_in, L)`:

```
output(c_out, i) = Σ_{c_in} Σ_{j=0}^{k-1}  weight(c_out, c_in, j) · x(c_in, i + j - k//2)
```

For 2D convolution operating on input `(C_in, H, W)`:

```
output(c_out, i, j) = Σ_{c_in} Σ_{p} Σ_{q}  weight(c_out, c_in, p, q) · x(c_in, i+p, j+q)
```

The weights are learned. Each output channel `c_out` has its own filter — a small 2D template — that slides across the input looking for a specific local pattern. In the frequency domain, these learned templates correspond to things like "elevated power at radius r" or "a spike at this specific (u,v) coordinate."

With `padding = k//2`, the spatial dimensions are preserved. With `bias=False` (our setting), the bias term is dropped — this works because Batch Normalisation (next) adds a learnable shift anyway.

### 4.2 Batch Normalisation

After each convolution, before the activation:

```
x_norm = (x - μ_batch) / sqrt(σ²_batch + ε)
output = γ · x_norm + β
```

where `μ_batch` and `σ²_batch` are the mean and variance computed over the current batch, and `γ` and `β` are learnable parameters (one pair per channel).

**What it does:**  
Forces each channel's activations to have roughly zero mean and unit variance during training. This makes gradients flow more reliably through deep networks and allows higher learning rates. The learnable `γ` and `β` let the network restore any scale and shift it needs after normalisation.

**Why `bias=False` in conv layers:**  
Because Batch Norm's `β` parameter serves the same role as a bias. Keeping a separate bias in the conv layer would be redundant — it would just be absorbed into `β` anyway.

### 4.3 ReLU

```
ReLU(x) = max(0, x)
```

Applied element-wise after Batch Norm. Introduces non-linearity — without it, the whole network would just be a linear function no matter how many layers you stack. `inplace=True` in our code means the operation modifies the tensor in memory rather than allocating a new one, saving memory.

### 4.4 Max Pooling

```
MaxPool(x, kernel=2) : output(i) = max(x[2i], x[2i+1])   (1D)
MaxPool(x, kernel=2) : output(i,j) = max over 2×2 window  (2D)
```

Halves spatial dimensions by taking the maximum in each non-overlapping window. This makes the features progressively more translation-invariant (the exact position of a pattern matters less) and reduces computation for subsequent layers.

### 4.5 Adaptive Average Pooling

```
AdaptiveAvgPool(output_size=1) → global average
```

Collapses the entire spatial extent to a single value per channel by averaging. Applied at the end of the feature extractor to get a fixed-size vector regardless of input resolution. For 1D: `(B, 256, L) → (B, 256)`. For 2D: `(B, 512, H, W) → (B, 512)`.

### 4.6 Linear (Fully Connected) Layer

```
output = x · Wᵀ + b
```

Maps the pooled feature vector to 2 output logits (one for real, one for fake). The logit for fake being higher than the logit for real means the model thinks the image is fake.

### 4.7 Cross-Entropy Loss

The training objective. Given logits `z = [z_real, z_fake]`, the predicted probability of fake is:

```
p_fake = exp(z_fake) / (exp(z_real) + exp(z_fake))    (softmax)
```

Cross-entropy loss for a batch:

```
L = - (1/N) Σ_i [ y_i · log(p_fake_i) + (1 - y_i) · log(1 - p_fake_i) ]
```

where `y_i = 1` if the image is fake, `0` if real. The loss is zero when the model is perfectly confident and correct, and gets arbitrarily large when it is confident but wrong.

---

## 5. The CNN1D Architecture

**Input:** azimuthal profile, shape `(B, 112)` → unsqueezed to `(B, 1, 112)` inside `forward()`.

```
Layer                   Channels       Output shape
─────────────────────────────────────────────────────
Conv1d(k=3) + BN + ReLU    1 → 32     (B, 32,  112)
Conv1d(k=3) + BN + ReLU   32 → 64     (B, 64,  112)
MaxPool1d(2)                           (B, 64,   56)
Conv1d(k=3) + BN + ReLU   64 → 128    (B, 128,  56)
Conv1d(k=3) + BN + ReLU  128 → 128    (B, 128,  56)
MaxPool1d(2)                           (B, 128,  28)
Conv1d(k=3) + BN + ReLU  128 → 256    (B, 256,  28)
AdaptiveAvgPool1d(1)                   (B, 256)
Linear(256 → 2)                        (B, 2)
─────────────────────────────────────────────────────
Total trainable parameters: ~180,000
```

**What each layer group is doing:**

- **First two conv layers (1→32→64):** Learn to detect local patterns in the radial profile. With kernel size 3, each filter looks at 3 adjacent radial bins at a time. These early filters might learn things like "sharp drop at this radius" or "unusually elevated power at mid-frequencies."

- **MaxPool (→56 bins):** Halves resolution. After this, each position represents 2 original bins. The model becomes less sensitive to exact frequency location and more sensitive to regions.

- **Next two conv layers (64→128→128):** Combine the local patterns from the first stage. Now looking at broader frequency regions. Might learn "the mid-frequency range has a flat profile instead of the expected fall-off."

- **MaxPool (→28 bins):** Further aggregates.

- **Final conv (128→256):** Builds a rich representation of the full radial profile.

- **Global average pool → Linear:** Collapses the spatial extent and maps to 2 logits.

**What it is learning:** Deviations from the natural 1/f² power law at specific frequency ranges. If SD v1.4 consistently produces images with too much energy at, say, bins 30–50 (mid-to-high spatial frequencies), the CNN will learn a filter that responds to elevated values in that range and associates it with fake.

---

## 6. The CNN2D Architecture

**Input:** log-power spectrum, shape `(B, 1, 224, 224)`.

```
Layer                    Channels        Output shape
──────────────────────────────────────────────────────
Conv2d(k=3) + BN + ReLU    1 → 32      (B, 32,  224, 224)
Conv2d(k=3) + BN + ReLU   32 → 64      (B, 64,  224, 224)
MaxPool2d(2)                             (B, 64,  112, 112)
Conv2d(k=3) + BN + ReLU   64 → 128     (B, 128, 112, 112)
Conv2d(k=3) + BN + ReLU  128 → 128     (B, 128, 112, 112)
MaxPool2d(2)                             (B, 128,  56,  56)
Conv2d(k=3) + BN + ReLU  128 → 256     (B, 256,  56,  56)
MaxPool2d(2)                             (B, 256,  28,  28)
Conv2d(k=3) + BN + ReLU  256 → 512     (B, 512,  28,  28)
Conv2d(k=3) + BN + ReLU  512 → 512     (B, 512,  28,  28)
AdaptiveAvgPool2d(1)                     (B, 512)
Linear(512 → 2)                          (B, 2)
──────────────────────────────────────────────────────
Total trainable parameters: ~4,080,000
```

This is a custom CNN — not ResNet, not VGG, not pretrained on anything. No skip connections. No dropout. No pretrained weights.

**What the 2D CNN can see that the 1D cannot:**

The 2D spectrum retains directional information. Key examples of directional artifacts:

- **Grid artifacts from transposed convolutions.** Many GAN generators use transposed convolutions (sometimes called "deconvolutions") to upsample. If the stride is 2, this creates a checkerboard pattern in the output — which in the frequency domain appears as spikes at multiples of the Nyquist frequency. In the 2D spectrum, these show up as a regular grid of bright spots.

- **Horizontal/vertical axis bias.** Some generators produce slightly different statistics along the main frequency axes (u=0 or v=0 lines) compared to diagonal directions. This shows up as a cross shape or asymmetry in the 2D spectrum that disappears when you azimuthally average.

- **Directional textures.** Some diffusion models show more energy in one direction because of how their attention mechanisms or U-Net architecture propagates information directionally.

The 2D CNN's early filters (3×3 on a 224×224 input) learn local templates in frequency space — detecting local bright spots, edges, and gradients in the spectrum. Later layers combine these into larger-scale patterns.

---

## 7. What Properties Is the CNN Actually Learning?

This is the core question. The model is a black box after training, but we can reason about it and, more importantly, inspect it.

**Conceptually, the model learns decision boundaries in the space of spectral signatures.** For CIFAKE (SD v1.4 vs CIFAR-10 real), these boundaries likely correspond to:

1. **The overall spectral slope.** Real CIFAR-10 images have a characteristic rate at which power falls off from low to high frequencies. SD v1.4 images may have a systematically different slope — perhaps steeper (too smooth) or shallower (too noisy at high frequencies). The 1D model's first layer will detect this as a change in the shape of the radial profile.

2. **Periodic spectral artifacts.** Stable Diffusion uses a VAE decoder with several upsampling stages. These can inject harmonic frequencies — spikes at specific radii in the spectrum. The model learns that "energy spike at radius r=47" means fake.

3. **Missing high-frequency texture.** Real photographs from cameras have random, unstructured high-frequency content (sensor noise, film grain, micro-texture). Generative models tend to produce smoother images that lack this. In the spectrum, this shows up as a deficit of power at high radii. The model learns "if the power at high frequencies is lower than it should be, it is fake."

4. **Anisotropy.** Real images are statistically isotropic — their spectra look roughly the same in all directions. Some generators break this symmetry. The 2D model can detect asymmetric energy distributions that the 1D model cannot.

---

## 8. How to Visualise What the Model Has Learned

There are several concrete methods, roughly ordered from simplest to most computationally involved.

### 8.1 Spectral Residual (Already in the Codebase)

```python
# src/transforms.py — spectral_residual() and compute_mean_spectrum()
residual = mean_fake_spectrum - mean_real_spectrum
```

This is not a model interpretability technique — it is dataset-level analysis. But it directly shows you *where in frequency space* fakes differ from reals, before any model is involved.

**How to run it:**
```python
from src.transforms import compute_mean_spectrum, spectral_residual
import matplotlib.pyplot as plt

real_paths = [...]   # list of real image paths
fake_paths = [...]   # list of fake image paths

mean_real = compute_mean_spectrum(real_paths)
mean_fake = compute_mean_spectrum(fake_paths)
residual = spectral_residual(mean_fake, mean_real)

plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1); plt.imshow(mean_real, cmap='viridis'); plt.title('Mean real spectrum')
plt.subplot(1, 3, 2); plt.imshow(mean_fake, cmap='viridis'); plt.title('Mean fake spectrum')
plt.subplot(1, 3, 3); plt.imshow(residual, cmap='RdBu_r', vmin=-residual.std()*3, vmax=residual.std()*3)
plt.title('Residual (fake − real)'); plt.colorbar()
plt.tight_layout(); plt.savefig('figures/spectral_residual.png')
```

Red regions in the residual are frequencies where fakes have *more* power. Blue regions are where they have *less*. This directly shows you the generator's fingerprint without any model involved.

### 8.2 Mean 1D Radial Profiles Side by Side

```python
import numpy as np
import matplotlib.pyplot as plt
from src.transforms import compute_mean_spectrum, azimuthal_average_fast

mean_real = compute_mean_spectrum(real_paths)
mean_fake = compute_mean_spectrum(fake_paths)

profile_real = azimuthal_average_fast(mean_real)
profile_fake = azimuthal_average_fast(mean_fake)

r = np.arange(len(profile_real))
plt.figure(figsize=(8, 4))
plt.plot(r, profile_real, label='Real')
plt.plot(r, profile_fake, label='Fake (SD v1.4)')
plt.xlabel('Radial frequency bin')
plt.ylabel('Mean log power')
plt.legend()
plt.title('Radial power profile: real vs fake')
plt.savefig('figures/mean_profiles.png')
```

The deviation between the two curves is exactly what the 1D CNN is learning to detect. Where the curves separate most is where the model has the most discriminative signal.

### 8.3 Grad-CAM on the 2D Spectrum

Gradient-weighted Class Activation Mapping (Grad-CAM) is the standard technique for asking "which part of the input was most important for this prediction?"

**The idea:** Run a forward pass, then backpropagate the gradient of the output logit (for class "fake") back to the final convolutional feature map. Average the gradients across spatial positions to get a weight per channel. Weighted-sum the feature maps with those weights and ReLU. The result is a heatmap over the input showing which spatial regions of the spectrum were most activated.

In our case, the "spatial positions" of the spectrum correspond to specific frequency coordinates. A Grad-CAM heatmap on the spectrum tells you: "the model paid attention to these frequency regions when deciding this image was fake."

**Implementation:**

```python
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from src.models import build_model
from src.transforms import to_grayscale_array, log_power_spectrum_2d
from PIL import Image

def gradcam_2d(model, spectrum_tensor, target_class=1):
    """
    spectrum_tensor: (1, 1, H, W) — single image, batched
    target_class: 1 = fake
    Returns: heatmap (H, W) numpy array
    """
    # Hook to capture the final conv layer's output and gradient
    activations = {}
    gradients = {}

    def forward_hook(module, input, output):
        activations['value'] = output

    def backward_hook(module, grad_input, grad_output):
        gradients['value'] = grad_output[0]

    # Register hooks on the last conv block (index -1 of model.features)
    last_conv = model.features[-1]   # last Conv2d + BN + ReLU block
    fh = last_conv.register_forward_hook(forward_hook)
    bh = last_conv.register_full_backward_hook(backward_hook)

    model.eval()
    spectrum_tensor.requires_grad_(True)
    logits = model(spectrum_tensor)

    model.zero_grad()
    logits[0, target_class].backward()

    fh.remove()
    bh.remove()

    # Grad-CAM
    grads = gradients['value'][0]        # (C, H', W')
    acts  = activations['value'][0]      # (C, H', W')
    weights = grads.mean(dim=(1, 2))     # (C,) — global average pooling of gradients
    cam = (weights[:, None, None] * acts).sum(dim=0)   # (H', W')
    cam = F.relu(cam)
    cam = cam.detach().cpu().numpy()
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

    # Upsample to original spectrum size
    from PIL import Image as PILImage
    cam_img = PILImage.fromarray((cam * 255).astype(np.uint8)).resize(
        (spectrum_tensor.shape[-1], spectrum_tensor.shape[-2]), PILImage.BILINEAR)
    return np.array(cam_img) / 255.0


# Usage:
ckpt = torch.load('results/cifake/best_2d.pt', map_location='cpu', weights_only=False)
model = build_model('2d')
model.load_state_dict(ckpt['model_state'])

img = Image.open('path/to/image.jpg')
gray = to_grayscale_array(img)
spec = log_power_spectrum_2d(gray)
spec_t = torch.from_numpy(spec).unsqueeze(0).unsqueeze(0)  # (1,1,224,224)

cam = gradcam_2d(model, spec_t, target_class=1)

fig, axes = plt.subplots(1, 3, figsize=(14, 4))
axes[0].imshow(spec, cmap='viridis'); axes[0].set_title('Log-power spectrum')
axes[1].imshow(cam, cmap='jet');      axes[1].set_title('Grad-CAM (fake logit)')
axes[2].imshow(spec, cmap='viridis', alpha=0.6)
axes[2].imshow(cam, cmap='jet', alpha=0.4)
axes[2].set_title('Overlay')
plt.tight_layout(); plt.savefig('figures/gradcam_example.png')
```

**What you will see:** Bright regions in the Grad-CAM overlay are the frequency coordinates the model focuses on when classifying. If the generator has a grid artifact at a specific (u,v) location, you should see a hotspot there. If the model is keying off the overall high-frequency region, you will see the outer ring of the spectrum lit up.

### 8.4 First-Layer Filter Visualisation

The first convolutional layer's filters are the most directly interpretable because they operate on the raw input. For the 2D model, they are 32 filters of shape (1, 3, 3) — 32 small 3×3 templates that look for local patterns in the spectrum.

```python
first_conv = model.features[0][0]   # first Conv2d
filters = first_conv.weight.detach().cpu().numpy()  # (32, 1, 3, 3)
filters = filters[:, 0, :, :]   # (32, 3, 3)

# Normalise each filter independently
filters_norm = (filters - filters.min(axis=(1,2), keepdims=True))
filters_norm /= (filters_norm.max(axis=(1,2), keepdims=True) + 1e-8)

fig, axes = plt.subplots(4, 8, figsize=(12, 6))
for i, ax in enumerate(axes.flat):
    ax.imshow(filters_norm[i], cmap='RdBu_r', vmin=0, vmax=1)
    ax.axis('off')
plt.suptitle('CNN2D first-layer filters (3×3)')
plt.tight_layout(); plt.savefig('figures/first_layer_filters.png')
```

These filters are edge detectors, gradient detectors, and texture detectors — applied to the spectrum. They tell you what local spectral patterns the model is sensitive to.

### 8.5 Activation Histograms

For a set of real and fake images, extract the features at the global average pool layer (the 256- or 512-dimensional vector just before the final linear) and compare them. If the model is working, the distributions of these vectors should be separable.

```python
from torch.utils.data import DataLoader
from src.dataset import FrequencyDataset

model.eval()
real_features, fake_features = [], []

with torch.no_grad():
    for spec2d, prof1d, label in loader:
        x = spec2d   # for 2d model
        # Extract features up to the global avg pool
        feat = model.pool(model.features(x)).flatten(1)  # (B, 512)
        for i, lbl in enumerate(label):
            if lbl == 0:
                real_features.append(feat[i].numpy())
            else:
                fake_features.append(feat[i].numpy())

real_features = np.array(real_features)
fake_features = np.array(fake_features)

# PCA or t-SNE to 2D for visualisation
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
all_feat = np.vstack([real_features, fake_features])
all_2d = pca.fit_transform(all_feat)
n_real = len(real_features)

plt.figure(figsize=(8, 6))
plt.scatter(all_2d[:n_real, 0], all_2d[:n_real, 1], alpha=0.3, label='Real', s=5)
plt.scatter(all_2d[n_real:, 0], all_2d[n_real:, 1], alpha=0.3, label='Fake', s=5)
plt.legend(); plt.title('PCA of CNN2D features before classification head')
plt.savefig('figures/feature_pca.png')
```

If you see two distinct clusters, the model has learned a representation in which real and fake are linearly separable — and you can inspect which principal components are most separable to understand what the model is keying off.

---

## 9. The Optimiser and Learning Rate Schedule

### AdamW

The parameters are updated using AdamW (Adam with decoupled weight decay):

```
m_t = β₁ · m_{t-1} + (1 - β₁) · g_t           (first moment — exponential avg of gradients)
v_t = β₂ · v_{t-1} + (1 - β₂) · g_t²          (second moment — exponential avg of squared gradients)

m̂_t = m_t / (1 - β₁ᵗ)                         (bias-corrected estimates)
v̂_t = v_t / (1 - β₂ᵗ)

θ_t = θ_{t-1} - α · m̂_t / (√v̂_t + ε)  -  α · λ · θ_{t-1}
```

Where:
- `α` = learning rate = `3e-4`
- `β₁ = 0.9` (default) — controls how much the current gradient vs history influences the step
- `β₂ = 0.999` (default) — controls how much the current squared gradient vs history influences the step size
- `ε = 1e-8` (default) — prevents division by zero
- `λ` = weight decay = `1e-4` — this term shrinks all weights toward zero at each step, penalising large weights

**Why AdamW over standard Adam?** In standard Adam, weight decay is entangled with the adaptive learning rate — large-gradient parameters effectively have smaller weight decay. AdamW decouples them, applying weight decay uniformly regardless of gradient magnitude. This is generally better for generalisation.

### Cosine Annealing

```python
scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
```

The learning rate follows a cosine curve from `lr` down to `0` over `T_max` epochs:

```
lr_t = lr_min + 0.5 · (lr_max - lr_min) · (1 + cos(π · t / T_max))
```

With `lr_min = 0` (default), this means the learning rate starts at `3e-4`, gradually decreases, and reaches approximately `0` at epoch 30. This avoids oscillating around a minimum at the end of training — as the learning rate shrinks, the model makes finer and finer adjustments, settling into a good optimum.

There is **no warmup** — the learning rate starts at full value from epoch 1. For small models like ours this is generally fine.

---

## 10. Hyperparameter Reference

These are all the hyperparameters in play, what they do, and what values we use.

| Hyperparameter | Value | What it controls |
|----------------|-------|-----------------|
| `--lr` | `3e-4` | Initial learning rate for AdamW. Higher = faster learning but risk of instability. Lower = more stable but slower. |
| `--weight-decay` | `1e-4` | L2 regularisation strength. Penalises large weights, helps generalisation. Too high and the model underlearns. |
| `--batch-size` | `64` | Number of images per gradient update. Larger batches give more stable gradient estimates but require more memory. |
| `--epochs` | `30` | Total training duration. Checkpoint is saved only when val AUC improves, so more epochs does not mean overwriting good checkpoints. |
| `--size` | `224` | Image resize target before FFT. Larger = more frequency resolution (more bins), slower. Smaller = coarser spectrum, faster. |
| `--seed` | `42` | Random seed for reproducibility (model init, data split, sampler shuffle). |
| `--class-weight` | off | If enabled, weights the loss inversely by class frequency: `w_c = N / (2 · N_c)`. Use when the dataset has many more reals than fakes or vice versa. |
| `β₁` | `0.9` | AdamW gradient momentum (not exposed as a flag, hardcoded in torch default). |
| `β₂` | `0.999` | AdamW squared-gradient momentum (not exposed as a flag). |
| `T_max` | `=epochs` | Cosine annealing period. Equals total epochs, so LR decays to ~0 by the end. |

**Current CIFAKE results under these hyperparameters:**
- CNN1D: val AUC = 0.9399 (best at epoch 25/30)
- CNN2D: val AUC = 0.9525 (best at epoch 13/30)

The 2D model peaking at epoch 13 and not improving for 17 more epochs is a sign the model is either overfitting to the training set or has saturated the representational capacity of its architecture for this dataset size. Options to investigate: reduce learning rate, add dropout before the linear head, use a larger model, or add data augmentation.

---

## 11. The Full Picture: End to End

```
Raw image (JPEG/PNG, any resolution, RGB)
    │
    ▼  to_grayscale_array(size=224)
float32 grayscale (224×224), values [0, 255]
    │
    ▼  np.fft.fft2 → fftshift → |·|² → log1p
float32 log-power spectrum (224×224), values ~[0, 20]
    │
    ├──────────────────────────────────────────────►  2D CNN path
    │                                                  (1, 224, 224) tensor
    │                                                  5 conv stages + GAP
    │                                                  512-dim feature vector
    │                                                  Linear(512→2) logits
    │
    ▼  azimuthal_average_fast
float32 radial profile (112,)
    │
    └──────────────────────────────────────────────►  1D CNN path
                                                       (112,) tensor
                                                       3 conv stages + GAP
                                                       256-dim feature vector
                                                       Linear(256→2) logits
                                                               │
                                                               ▼
                                                       softmax → P(fake) ∈ [0,1]
                                                       threshold 0.5 → binary label
                                                       AUC computed over full validation set
```

The model never sees the original pixel values. It sees only a mathematical transformation of spatial frequency content. Every decision it makes is based purely on whether the spectral fingerprint of the image looks like something a generator would produce.

---

## 12. What This Cannot Tell You

- **It cannot detect every generator.** A model trained only on SD v1.4 (CIFAKE) may fail on generators with different spectral signatures. Cross-generator generalisation is the unresolved core challenge.
- **It cannot localise fakes spatially.** The spectrum of a full image is global — it contains no information about *where* in the image a manipulation occurred. This is a fundamental limitation of the approach.
- **It is not robust to JPEG recompression.** JPEG compression modifies the high-frequency content of an image in a way that can wash out generator-specific fingerprints. A robust detector would need to account for this.
- **It cannot explain itself in semantic terms.** Grad-CAM can show you *which frequencies* the model focuses on, but it cannot tell you *what visual content* those frequencies correspond to in the original image.
