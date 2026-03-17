"""Visualize averaged power spectra per generator.

Produces Figure 1 of a typical frequency forensics paper:
- 1D azimuthal average per generator (log scale)
- 2D log-spectrum averaged over N images per generator

Usage:
    python scripts/visualize_spectra.py configs/base.yaml
    python scripts/visualize_spectra.py configs/base.yaml data.root=~/data n_per_gen=50
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image

from eigenfraud.config import Config
from eigenfraud.data.genimage import GENERATORS
from eigenfraud.spectral.azimuthal import azimuthal_average_batch
from eigenfraud.spectral.spectrum2d import compute_log_spectrum_2d

N_PER_GEN_DEFAULT = 100


def load_images(gen_dir: Path, n: int, size: int) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Load n AI-generated and n real images from a generator directory."""
    tf = T.Compose([T.Resize(size), T.CenterCrop(size)])
    reals, fakes = [], []
    for label, subdir, store in [("nature", reals), ("ai", fakes)]:
        d = gen_dir / "val" / label
        if not d.exists():
            return [], []
        paths = sorted(d.glob("*.jpg")) + sorted(d.glob("*.jpeg")) + sorted(d.glob("*.png"))
        for p in paths[:n]:
            img = tf(Image.open(p).convert("RGB"))
            store.append(np.array(img).astype(np.float32) / 255.0)
    return reals, fakes


def main():
    cfg = Config.from_cli()
    n = N_PER_GEN_DEFAULT
    # Allow override: n_per_gen=50 on CLI
    for arg in sys.argv[2:]:
        if arg.startswith("n_per_gen="):
            n = int(arg.split("=")[1])

    data_root = Path(cfg.data.root).expanduser() / "GenImage"
    out_dir = Path("outputs/figures")
    out_dir.mkdir(parents=True, exist_ok=True)

    fig_1d, ax_1d = plt.subplots(figsize=(10, 5))
    fig_2d, axes_2d = plt.subplots(2, len(GENERATORS), figsize=(3 * len(GENERATORS), 6))

    for col, gen in enumerate(GENERATORS):
        gen_dir = data_root / gen
        reals, fakes = load_images(gen_dir, n, cfg.data.image_size)
        if not fakes:
            print(f"  Skipping {gen}: no data found")
            continue

        # 1D azimuthal average (fakes only, grayscale)
        gray_fakes = np.array([img.mean(axis=2) for img in fakes])
        bins, psd = azimuthal_average_batch(gray_fakes)
        freqs = bins / cfg.data.image_size
        ax_1d.semilogy(freqs, psd, label=gen, alpha=0.8)

        # 2D spectrum (mean of fakes and reals separately)
        def mean_spectrum_2d(imgs):
            specs = []
            for img in imgs:
                t = torch.from_numpy(img.transpose(2, 0, 1))
                specs.append(compute_log_spectrum_2d(t).numpy())
            return np.mean(specs, axis=0).mean(0)  # average over channels

        axes_2d[0, col].imshow(mean_spectrum_2d(reals), cmap="inferno")
        axes_2d[0, col].set_title(f"{gen}\nReal", fontsize=7)
        axes_2d[0, col].axis("off")

        axes_2d[1, col].imshow(mean_spectrum_2d(fakes), cmap="inferno")
        axes_2d[1, col].set_title("Fake", fontsize=7)
        axes_2d[1, col].axis("off")

    # Mark LDM peak positions (0.125, 0.25, 0.375, 0.5)
    for peak in [0.125, 0.25, 0.375, 0.5]:
        ax_1d.axvline(peak, color="gray", linestyle="--", alpha=0.4, linewidth=0.8)

    ax_1d.set_xlabel("Normalized frequency (cycles/pixel)")
    ax_1d.set_ylabel("Mean power spectral density")
    ax_1d.set_title("Azimuthally averaged power spectra (fake images)")
    ax_1d.legend(fontsize=8, ncol=2)
    ax_1d.grid(True, alpha=0.3)

    fig_1d.tight_layout()
    fig_1d.savefig(out_dir / "spectra_1d.pdf")
    fig_2d.tight_layout()
    fig_2d.savefig(out_dir / "spectra_2d.pdf")
    print(f"Figures saved to {out_dir}/")


if __name__ == "__main__":
    main()
