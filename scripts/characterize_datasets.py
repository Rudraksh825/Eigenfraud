"""
Phase 1/2: Dataset Characterization

For each dataset × split (real / fake), computes:
  - Mean 2D log-power spectrum
  - Mean + std radial power spectrum (1D)
  - JPEG quality factor distribution

Default (raw data → Figure 1):
    python scripts/characterize_datasets.py

Normalized data → Figure 2:
    python scripts/characterize_datasets.py \
        --cifake-real  data/normalized/cifake/test/real \
        --cifake-fake  data/normalized/cifake/test/fake \
        --defy-real    data/normalized/defactify/test/real \
        --defy-fake    data/normalized/defactify/test/fake \
        --genimage-real data/normalized/genimage/real \
        --genimage-fake data/normalized/genimage/fake \
        --fig-prefix fig2 --spectra-tag norm
"""

import argparse
import sys, os, io, random
import numpy as np
from pathlib import Path
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm

REPO = Path(__file__).parent.parent
sys.path.insert(0, str(REPO / 'src'))
from transforms import to_grayscale_array, log_power_spectrum_2d, azimuthal_average_fast

random.seed(42)

MAX_SAMPLES = 10_000          # per split cap
SPECTRUM_SIZE = 224           # resize target for FFT
R_MAX = SPECTRUM_SIZE // 2    # = 112 radial bins
HIGH_FREQ_START = int(0.8 * R_MAX)  # index 89 — >80% of Nyquist


# ---------------------------------------------------------------------------
# Spectrum accumulator
# ---------------------------------------------------------------------------

class SpectrumAccum:
    """Online accumulator for mean and variance of 2D and 1D spectra."""

    def __init__(self):
        self.n = 0
        self.sum_2d = None
        self.sum_sq_2d = None
        self.sum_1d = None
        self.sum_sq_1d = None

    def add(self, spec2d: np.ndarray):
        spec1d = azimuthal_average_fast(spec2d)
        if self.n == 0:
            self.sum_2d    = spec2d.astype(np.float64)
            self.sum_sq_2d = spec2d.astype(np.float64) ** 2
            self.sum_1d    = spec1d.astype(np.float64)
            self.sum_sq_1d = spec1d.astype(np.float64) ** 2
        else:
            self.sum_2d    += spec2d
            self.sum_sq_2d += spec2d.astype(np.float64) ** 2
            self.sum_1d    += spec1d
            self.sum_sq_1d += spec1d.astype(np.float64) ** 2
        self.n += 1

    def result(self):
        assert self.n > 0
        mean_2d = (self.sum_2d / self.n).astype(np.float32)
        mean_1d = (self.sum_1d / self.n).astype(np.float32)
        std_1d  = np.sqrt(np.maximum(self.sum_sq_1d / self.n - (self.sum_1d / self.n) ** 2, 0)).astype(np.float32)
        return mean_2d, mean_1d, std_1d, self.n


# ---------------------------------------------------------------------------
# Image sources
# ---------------------------------------------------------------------------

def process_file_paths(paths, max_n=MAX_SAMPLES, desc=""):
    paths = list(paths)
    if len(paths) > max_n:
        paths = random.sample(paths, max_n)
    acc = SpectrumAccum()
    qualities = []
    for p in tqdm(paths, desc=desc, ncols=80):
        try:
            img = Image.open(p)
            _collect_jpeg_quality(img, qualities)
            gray = to_grayscale_array(img, size=SPECTRUM_SIZE)
            acc.add(log_power_spectrum_2d(gray))
        except Exception:
            continue
    return acc.result(), qualities


def process_parquet(parquet_files, target_label, max_n=MAX_SAMPLES, desc=""):
    import pyarrow.parquet as pq
    acc = SpectrumAccum()
    qualities = []
    for pf in tqdm(parquet_files, desc=desc, ncols=80):
        if acc.n >= max_n:
            break
        table = pq.read_table(pf, columns=["Image", "Label_A"])
        for img_s, lbl in zip(table["Image"].to_pylist(), table["Label_A"].to_pylist()):
            if acc.n >= max_n:
                break
            if lbl != target_label or img_s is None:
                continue
            data = img_s.get("bytes") if isinstance(img_s, dict) else None
            if not data:
                continue
            try:
                img = Image.open(io.BytesIO(data))
                _collect_jpeg_quality(img, qualities)
                gray = to_grayscale_array(img, size=SPECTRUM_SIZE)
                acc.add(log_power_spectrum_2d(gray))
            except Exception:
                continue
    return acc.result(), qualities


def _collect_jpeg_quality(img: Image.Image, out: list):
    try:
        info = img.info
        q = info.get("quality", None)
        if q is not None:
            out.append(int(q))
            return
        # Try quantization table heuristic
        if hasattr(img, "quantization") and img.quantization:
            q_table = img.quantization.get(0, [])
            if q_table:
                out.append(int(q_table[0]))
    except Exception:
        pass


# ---------------------------------------------------------------------------
# L2 spectral divergence in high-frequency band
# ---------------------------------------------------------------------------

def hf_l2(mean_real_1d: np.ndarray, mean_fake_1d: np.ndarray) -> float:
    return float(np.linalg.norm(mean_real_1d[HIGH_FREQ_START:] - mean_fake_1d[HIGH_FREQ_START:]))


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

COLORS = {"real": "#2196F3", "fake": "#F44336"}
FREQ_AXIS = np.arange(R_MAX) / R_MAX  # normalized 0-1 (1 = Nyquist)


def plot_fig1(datasets: dict, out_path: Path):
    """
    3-panel figure: mean radial spectrum real vs fake for CIFAKE / Defactify / GenImage.
    datasets = {name: {"real": (mean_1d, std_1d), "fake": (mean_1d, std_1d), "l2_hf": float}}
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=False)
    for ax, (dname, data) in zip(axes, datasets.items()):
        for split in ("real", "fake"):
            m, s = data[split]
            ax.plot(FREQ_AXIS, m, color=COLORS[split], label=split.capitalize(), linewidth=1.5)
            ax.fill_between(FREQ_AXIS, m - s, m + s, alpha=0.15, color=COLORS[split])
        ax.axvline(x=HIGH_FREQ_START / R_MAX, color="gray", linestyle="--",
                   linewidth=0.8, label=f">80% Nyquist (L2={data['l2_hf']:.2f})")
        ax.set_title(dname, fontsize=13, fontweight="bold")
        ax.set_xlabel("Normalized frequency (0=DC, 1=Nyquist)")
        ax.set_ylabel("Mean log-power")
        ax.legend(fontsize=9)
        ax.set_xlim(0, 1)
    fig.suptitle("Mean Radial Power Spectrum: Real vs Fake", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved Figure 1 → {out_path}")


def plot_fig1b(datasets_2d: dict, out_path: Path):
    """2×N grid of mean 2D log-power heatmaps."""
    names = list(datasets_2d.keys())
    fig, axes = plt.subplots(2, len(names), figsize=(5 * len(names), 9))
    for col, dname in enumerate(names):
        for row, split in enumerate(("real", "fake")):
            ax = axes[row][col]
            spec = datasets_2d[dname][split]
            im = ax.imshow(spec, cmap="inferno", origin="upper", aspect="equal")
            ax.set_title(f"{dname} — {split.capitalize()}", fontsize=10)
            ax.axis("off")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle("Mean 2D Log-Power Spectrum", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved Figure 1b → {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _file_paths(d: Path):
    return sorted(p for p in d.rglob('*') if p.suffix.lower() in
                  {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tiff'})


def main():
    parser = argparse.ArgumentParser()
    # CIFAKE overrides
    parser.add_argument('--cifake-real',    type=Path, default=None)
    parser.add_argument('--cifake-fake',    type=Path, default=None)
    # Defactify overrides — either parquet dir (raw) or file dirs (normalized)
    parser.add_argument('--defy-parquet',   type=Path, default=None,
                        help='Defactify parquet dir (default raw path)')
    parser.add_argument('--defy-real',      type=Path, default=None,
                        help='Defactify real files dir (overrides parquet)')
    parser.add_argument('--defy-fake',      type=Path, default=None,
                        help='Defactify fake files dir (overrides parquet)')
    # GenImage overrides
    parser.add_argument('--genimage-real',  type=Path, default=None)
    parser.add_argument('--genimage-fake',  type=Path, default=None)
    # Output naming
    parser.add_argument('--fig-prefix',     default='fig1',
                        help='Figure filename prefix (default: fig1)')
    parser.add_argument('--spectra-tag',    default='',
                        help='Suffix for spectra npz filenames (default: none)')
    args = parser.parse_args()

    out_spectra = REPO / "results" / "spectra"
    out_figures = REPO / "figures"
    out_spectra.mkdir(parents=True, exist_ok=True)
    out_figures.mkdir(parents=True, exist_ok=True)

    tag = ('_' + args.spectra_tag) if args.spectra_tag else ''
    fig_data   = {}
    fig_data_b = {}

    def _save_and_record(name, m2d_r, m1d_r, s1d_r, n_r, m2d_f, m1d_f, s1d_f, n_f):
        l2 = hf_l2(m1d_r, m1d_f)
        print(f"  HF L2 divergence (r>{HIGH_FREQ_START}): {l2:.4f}")
        key = name.lower()
        np.savez(out_spectra / f"{key}_real{tag}.npz",
                 mean_2d=m2d_r, mean_1d=m1d_r, std_1d=s1d_r, n=n_r)
        np.savez(out_spectra / f"{key}_fake{tag}.npz",
                 mean_2d=m2d_f, mean_1d=m1d_f, std_1d=s1d_f, n=n_f)
        fig_data[name]   = {"real": (m1d_r, s1d_r), "fake": (m1d_f, s1d_f), "l2_hf": l2}
        fig_data_b[name] = {"real": m2d_r, "fake": m2d_f}

    # ------------------------------------------------------------------
    # CIFAKE
    # ------------------------------------------------------------------
    print("\n=== CIFAKE ===")
    cifake_real = args.cifake_real or (REPO / "data/raw/cifake/test/REAL")
    cifake_fake = args.cifake_fake or (REPO / "data/raw/cifake/test/FAKE")

    (m2d_r, m1d_r, s1d_r, n_r), _ = process_file_paths(
        _file_paths(cifake_real), desc="  CIFAKE real ")
    print(f"  real:  n={n_r}")
    (m2d_f, m1d_f, s1d_f, n_f), _ = process_file_paths(
        _file_paths(cifake_fake), desc="  CIFAKE fake ")
    print(f"  fake:  n={n_f}")
    _save_and_record("cifake", m2d_r, m1d_r, s1d_r, n_r, m2d_f, m1d_f, s1d_f, n_f)

    # ------------------------------------------------------------------
    # Defactify — parquet (raw) or files (normalized)
    # ------------------------------------------------------------------
    print("\n=== Defactify ===")
    if args.defy_real and args.defy_fake:
        (m2d_r, m1d_r, s1d_r, n_r), _ = process_file_paths(
            _file_paths(args.defy_real), desc="  Defactify real")
        print(f"  real:  n={n_r}")
        (m2d_f, m1d_f, s1d_f, n_f), _ = process_file_paths(
            _file_paths(args.defy_fake), desc="  Defactify fake")
        print(f"  fake:  n={n_f}")
    else:
        defy_dir = args.defy_parquet or (REPO / "defactify_dataset" / "data")
        test_parquets = sorted(defy_dir.glob("test-*.parquet"))
        (m2d_r, m1d_r, s1d_r, n_r), _ = process_parquet(
            test_parquets, target_label=0, desc="  Defactify real")
        print(f"  real:  n={n_r}")
        (m2d_f, m1d_f, s1d_f, n_f), _ = process_parquet(
            test_parquets, target_label=1, desc="  Defactify fake")
        print(f"  fake:  n={n_f}")
    _save_and_record("defactify", m2d_r, m1d_r, s1d_r, n_r, m2d_f, m1d_f, s1d_f, n_f)

    # ------------------------------------------------------------------
    # GenImage (imagenet_nature real vs ADM fake)
    # ------------------------------------------------------------------
    print("\n=== GenImage ===")
    gi_real = args.genimage_real or (REPO / "data/raw/imagenet_nature/val")
    gi_fake = args.genimage_fake or (REPO / "data/raw/ADM/imagenet_ai_0508_adm/train/ai")

    (m2d_r, m1d_r, s1d_r, n_r), _ = process_file_paths(
        _file_paths(gi_real), desc="  GenImage real")
    print(f"  real:  n={n_r}")
    (m2d_f, m1d_f, s1d_f, n_f), _ = process_file_paths(
        _file_paths(gi_fake), desc="  GenImage fake(ADM)")
    print(f"  fake:  n={n_f}")
    _save_and_record("genimage", m2d_r, m1d_r, s1d_r, n_r, m2d_f, m1d_f, s1d_f, n_f)

    # ------------------------------------------------------------------
    # Figures
    # ------------------------------------------------------------------
    # Remap keys to display names
    display = {"cifake": "CIFAKE", "defactify": "Defactify", "genimage": "GenImage"}
    fig_display   = {display[k]: v for k, v in fig_data.items()}
    fig_display_b = {display[k]: v for k, v in fig_data_b.items()}

    print("\nGenerating figures...")
    plot_fig1(fig_display,   out_figures / f"{args.fig_prefix}_radial_spectra.png")
    plot_fig1b(fig_display_b, out_figures / f"{args.fig_prefix}b_2d_heatmaps.png")

    # ------------------------------------------------------------------
    # Summary table
    # ------------------------------------------------------------------
    print("\n=== Spectral Divergence Summary ===")
    print(f"{'Dataset':<12} {'HF L2 (r>'+str(HIGH_FREQ_START)+')':<20}")
    print("-" * 32)
    for name, data in fig_display.items():
        print(f"{name:<12} {data['l2_hf']:.4f}")


if __name__ == "__main__":
    main()
