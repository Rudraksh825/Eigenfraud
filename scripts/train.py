"""Training script for 1D and 2D spectral detectors.

Usage:
    python scripts/train.py configs/train_2d.yaml
    python scripts/train.py configs/train_1d.yaml data.batch_size=32 train.lr=5e-5
"""
from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from eigenfraud.config import Config
from eigenfraud.data.genimage import GenImageDataset
from eigenfraud.data.transforms import build_transforms
from eigenfraud.eval.metrics import compute_metrics
from eigenfraud.models.cnn1d import build_1d_model
from eigenfraud.models.resnet2d import build_2d_model


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_model(cfg: Config) -> nn.Module:
    if cfg.model.type == "resnet2d":
        return build_2d_model(cfg.model)
    return build_1d_model(cfg.model)


def collate_drop_generator(batch):
    """DataLoader collate that drops the generator string from GenImage items."""
    imgs, labels, _ = zip(*batch)
    return torch.stack(imgs), torch.tensor(labels, dtype=torch.long)


def main():
    cfg = Config.from_cli()
    set_seed(cfg.seed)
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")

    train_tf = build_transforms(
        cfg.aug, mode="train", model_type=cfg.model.type, image_size=cfg.data.image_size
    )
    val_tf = build_transforms(
        cfg.aug, mode="eval", model_type=cfg.model.type, image_size=cfg.data.image_size
    )

    data_root = Path(cfg.data.root).expanduser()
    train_ds = GenImageDataset(data_root / "GenImage", generators=cfg.data.genimage_generator,
                               split="train", transform=train_tf)
    val_ds = GenImageDataset(data_root / "GenImage", generators=cfg.data.genimage_generator,
                             split="val", transform=val_tf)

    train_loader = DataLoader(
        train_ds, batch_size=cfg.data.batch_size, shuffle=True,
        num_workers=cfg.data.num_workers, collate_fn=collate_drop_generator, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.data.batch_size, shuffle=False,
        num_workers=cfg.data.num_workers, collate_fn=collate_drop_generator, pin_memory=True
    )

    model = build_model(cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.train.lr,
                                  weight_decay=cfg.train.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.train.epochs)
    loss_fn = nn.BCEWithLogitsLoss()

    ckpt_dir = Path(cfg.train.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_auc = 0.0

    for epoch in range(1, cfg.train.epochs + 1):
        # --- Train ---
        model.train()
        train_loss = 0.0
        for imgs, labels in tqdm(train_loader, desc=f"Epoch {epoch}/{cfg.train.epochs} train"):
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(imgs).squeeze(1)
            loss = loss_fn(logits, labels.float())
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(imgs)
        train_loss /= len(train_ds)
        scheduler.step()

        # --- Validate ---
        model.eval()
        all_scores, all_labels = [], []
        with torch.no_grad():
            for imgs, labels in tqdm(val_loader, desc=f"Epoch {epoch}/{cfg.train.epochs} val"):
                imgs = imgs.to(device)
                logits = model(imgs).squeeze(1)
                scores = torch.sigmoid(logits).cpu().numpy()
                all_scores.extend(scores)
                all_labels.extend(labels.numpy())

        m = compute_metrics(np.array(all_scores), np.array(all_labels))
        print(
            f"Epoch {epoch:3d} | loss={train_loss:.4f} | "
            f"AUC={m['auc']:.4f} | AP={m['ap']:.4f} | ACC={m['acc']:.4f}"
        )

        if m["auc"] > best_auc:
            best_auc = m["auc"]
            torch.save(
                {"epoch": epoch, "model_state": model.state_dict(), "metrics": m, "config": cfg},
                ckpt_dir / f"{cfg.run_name}_best.pt",
            )
            print(f"  Saved best checkpoint (AUC={best_auc:.4f})")

    print(f"Training complete. Best val AUC: {best_auc:.4f}")


if __name__ == "__main__":
    main()
