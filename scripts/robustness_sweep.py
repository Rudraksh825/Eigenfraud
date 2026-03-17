"""Robustness sweep: AUC vs. JPEG QF and resize factor.

Usage:
    python scripts/robustness_sweep.py configs/robustness.yaml \
        robustness.checkpoint=outputs/checkpoints/run_best.pt
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import torch

from eigenfraud.config import Config
from eigenfraud.data.genimage import GenImageDataset
from eigenfraud.data.transforms import build_transforms
from eigenfraud.eval.robustness import run_robustness_sweep
from eigenfraud.models.cnn1d import build_1d_model
from eigenfraud.models.resnet2d import build_2d_model


def main():
    cfg = Config.from_cli()
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(cfg.robustness.checkpoint, map_location=device)
    if cfg.model.type == "resnet2d":
        model = build_2d_model(cfg.model)
    else:
        model = build_1d_model(cfg.model)
    model.load_state_dict(ckpt["model_state"])
    model.to(device)

    # Dataset with NO spectral transform — robustness.py applies PIL perturbations first
    data_root = Path(cfg.data.root).expanduser()
    val_ds = GenImageDataset(
        data_root / "GenImage", generators=cfg.data.genimage_generator,
        split="val", transform=None
    )
    # Attach the spectral transform so robustness.py can call it after perturbing
    val_ds.spectral_transform = build_transforms(
        cfg.aug, mode="eval", model_type=cfg.model.type, image_size=cfg.data.image_size
    )

    print("Running robustness sweep...")
    df = run_robustness_sweep(
        model, val_ds,
        jpeg_qualities=cfg.robustness.jpeg_qualities,
        resize_factors=cfg.robustness.resize_factors,
        device=device,
    )
    print(df.to_string(index=False, float_format="{:.4f}".format))

    out_dir = Path(cfg.robustness.output_csv).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(cfg.robustness.output_csv, index=False)
    print(f"\nSaved CSV to {cfg.robustness.output_csv}")

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    jpeg_df = df[df["perturbation_type"] == "jpeg"].sort_values("param_value", ascending=False)
    axes[0].plot(jpeg_df["param_value"], jpeg_df["auc"], marker="o")
    axes[0].set_xlabel("JPEG Quality Factor")
    axes[0].set_ylabel("AUC")
    axes[0].set_title("Robustness to JPEG Compression")
    axes[0].invert_xaxis()
    axes[0].grid(True)

    resize_df = df[df["perturbation_type"] == "resize"].sort_values("param_value")
    axes[1].plot(resize_df["param_value"], resize_df["auc"], marker="o")
    axes[1].set_xlabel("Resize Scale Factor")
    axes[1].set_ylabel("AUC")
    axes[1].set_title("Robustness to Downscaling")
    axes[1].grid(True)

    fig_path = out_dir / "robustness_curves.pdf"
    plt.tight_layout()
    plt.savefig(fig_path)
    print(f"Saved figure to {fig_path}")


if __name__ == "__main__":
    main()
