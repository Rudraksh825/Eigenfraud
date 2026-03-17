"""Adversarial attack script: spatial PGD vs. frequency PGD.

Usage:
    python scripts/attack.py configs/attack_freq.yaml attack.checkpoint=outputs/checkpoints/run_best.pt
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from eigenfraud.config import Config
from eigenfraud.data.genimage import GenImageDataset
from eigenfraud.data.transforms import build_transforms
from eigenfraud.attacks.pgd_spatial import pgd_spatial
from eigenfraud.attacks.pgd_freq import pgd_freq
from eigenfraud.attacks.bandlimited import BandMask, pgd_freq_bandlimited
from eigenfraud.eval.metrics import compute_metrics
from eigenfraud.models.cnn1d import build_1d_model
from eigenfraud.models.resnet2d import build_2d_model


def collate_drop_generator(batch):
    imgs, labels, _ = zip(*batch)
    return torch.stack(imgs), torch.tensor(labels)


def run_attack(model, loader, cfg, device):
    atk_cfg = cfg.attack
    all_scores_clean, all_scores_adv, all_labels = [], [], []

    for imgs, labels in tqdm(loader, desc=f"Attack ({atk_cfg.type})"):
        imgs, labels = imgs.to(device), labels.to(device)

        if atk_cfg.type == "pgd_spatial":
            adv = pgd_spatial(model, imgs, labels, atk_cfg.eps, atk_cfg.alpha, atk_cfg.num_iter)
        elif atk_cfg.type == "pgd_freq":
            adv = pgd_freq(model, imgs, labels, atk_cfg.eps_freq, atk_cfg.alpha_freq,
                           atk_cfg.num_iter, atk_cfg.spatial_eps)
        elif atk_cfg.type == "pgd_freq_band":
            band = BandMask(atk_cfg.band_low, atk_cfg.band_high)
            adv = pgd_freq_bandlimited(model, imgs, labels, band, atk_cfg.eps_freq,
                                       atk_cfg.alpha_freq, atk_cfg.num_iter, atk_cfg.spatial_eps)
        else:
            raise ValueError(f"Unknown attack type: {atk_cfg.type!r}")

        with torch.no_grad():
            s_clean = torch.sigmoid(model(imgs).squeeze(1)).cpu().numpy()
            s_adv = torch.sigmoid(model(adv).squeeze(1)).cpu().numpy()

        all_scores_clean.extend(s_clean)
        all_scores_adv.extend(s_adv)
        all_labels.extend(labels.cpu().numpy())

    return (
        np.array(all_scores_clean),
        np.array(all_scores_adv),
        np.array(all_labels),
    )


def main():
    cfg = Config.from_cli()
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(cfg.attack.checkpoint, map_location=device)
    if cfg.model.type == "resnet2d":
        model = build_2d_model(cfg.model)
    else:
        model = build_1d_model(cfg.model)
    model.load_state_dict(ckpt["model_state"])
    model.to(device)

    val_tf = build_transforms(
        cfg.aug, mode="eval", model_type=cfg.model.type, image_size=cfg.data.image_size
    )
    data_root = Path(cfg.data.root).expanduser()
    val_ds = GenImageDataset(
        data_root / "GenImage", generators=cfg.data.genimage_generator, split="val",
        transform=val_tf
    )
    # Subsample for tractable attack runtime
    n = min(cfg.attack.n_samples, len(val_ds))
    indices = torch.randperm(len(val_ds))[:n].tolist()
    subset = Subset(val_ds, indices)
    loader = DataLoader(subset, batch_size=16, shuffle=False,
                        num_workers=cfg.data.num_workers, collate_fn=collate_drop_generator)

    scores_clean, scores_adv, labels = run_attack(model, loader, cfg, device)
    m_clean = compute_metrics(scores_clean, labels)
    m_adv = compute_metrics(scores_adv, labels)

    print("\nClean:     ", {k: f"{v:.4f}" for k, v in m_clean.items()})
    print("Adversarial:", {k: f"{v:.4f}" for k, v in m_adv.items()})
    print("ΔAUC:      ", f"{m_adv['auc'] - m_clean['auc']:+.4f}")
    print("ΔAP:       ", f"{m_adv['ap'] - m_clean['ap']:+.4f}")


if __name__ == "__main__":
    main()
