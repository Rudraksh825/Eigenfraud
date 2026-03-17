"""Cross-generator evaluation script.

Usage:
    python scripts/evaluate.py configs/train_2d.yaml attack.checkpoint=outputs/checkpoints/run_best.pt
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from eigenfraud.config import Config
from eigenfraud.data.genimage import GENERATORS, GenImageDataset
from eigenfraud.data.transforms import build_transforms
from eigenfraud.eval.metrics import per_generator_metrics
from eigenfraud.models.cnn1d import build_1d_model
from eigenfraud.models.resnet2d import build_2d_model


def collate_with_generator(batch):
    imgs, labels, gens = zip(*batch)
    return torch.stack(imgs), torch.tensor(labels), list(gens)


def main():
    cfg = Config.from_cli()
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")

    # Load checkpoint
    ckpt_path = cfg.attack.checkpoint
    ckpt = torch.load(ckpt_path, map_location=device)
    if cfg.model.type == "resnet2d":
        model = build_2d_model(cfg.model)
    else:
        model = build_1d_model(cfg.model)
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()

    val_tf = build_transforms(
        cfg.aug, mode="eval", model_type=cfg.model.type, image_size=cfg.data.image_size
    )

    data_root = Path(cfg.data.root).expanduser()
    # Evaluate on all generators
    val_ds = GenImageDataset(
        data_root / "GenImage", generators=GENERATORS, split="val", transform=val_tf
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.data.batch_size, shuffle=False,
        num_workers=cfg.data.num_workers, collate_fn=collate_with_generator, pin_memory=True
    )

    all_scores, all_labels, all_gens = [], [], []
    with torch.no_grad():
        for imgs, labels, gens in tqdm(val_loader, desc="Evaluating"):
            imgs = imgs.to(device)
            logits = model(imgs).squeeze(1)
            scores = torch.sigmoid(logits).cpu().numpy()
            all_scores.extend(scores)
            all_labels.extend(labels.numpy())
            all_gens.extend(gens)

    df = per_generator_metrics(
        np.array(all_scores), np.array(all_labels), all_gens
    )

    print("\nCross-generator evaluation:")
    print(df.to_string(index=False, float_format="{:.4f}".format))

    out_dir = Path("outputs/results")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{cfg.run_name}_eval.json"
    df.to_json(out_path, orient="records", indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
