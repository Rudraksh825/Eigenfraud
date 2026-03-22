"""
Train a Specter detector (1D or 2D CNN) on a frequency-domain image dataset.

Usage (pre-split data, e.g. CIFAKE):
    python scripts/train.py --model 1d --train-dir data/processed/cifake/train --val-dir data/processed/cifake/test
    python scripts/train.py --model 2d --train-dir data/processed/cifake/train --val-dir data/processed/cifake/test --wandb

Usage (single directory, auto-split):
    python scripts/train.py --model 1d --data data/processed/mydata
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score
import numpy as np

from src.dataset import FrequencyDataset, make_splits
from src.models import build_model, count_parameters


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, choices=["1d", "2d"])
    # Pre-split mode (preferred for CIFAKE which has its own train/test dirs)
    p.add_argument("--train-dir", default=None, help="Directory with real/ and fake/ subdirs for training")
    p.add_argument("--val-dir",   default=None, help="Directory with real/ and fake/ subdirs for validation")
    # Single-dir auto-split mode (fallback)
    p.add_argument("--data", default=None, help="Single root dir — will be split into train/val/test automatically")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--size", type=int, default=224, help="Image resize target before FFT")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--out-dir", default="results")
    p.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    if args.train_dir is None and args.data is None:
        p.error("Provide either --train-dir + --val-dir, or --data")
    return args


def collate_fn_1d(batch):
    spec2d, prof1d, labels = zip(*batch)
    return torch.stack(spec2d), torch.stack(prof1d), torch.tensor(labels)


def run_epoch(model, loader, criterion, optimizer, device, model_type, train: bool):
    model.train() if train else model.eval()
    total_loss, all_labels, all_probs = 0.0, [], []

    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for spec2d, prof1d, labels in loader:
            labels = labels.to(device)
            x = prof1d.to(device) if model_type == "1d" else spec2d.to(device)

            logits = model(x)
            loss = criterion(logits, labels)

            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * labels.size(0)
            probs = torch.softmax(logits, dim=1)[:, 1].detach().cpu().numpy()
            all_probs.extend(probs)
            all_labels.extend(labels.cpu().numpy())

    n = len(all_labels)
    avg_loss = total_loss / n
    auc = roc_auc_score(all_labels, all_probs) if len(set(all_labels)) > 1 else float("nan")
    acc = np.mean(np.array(all_probs) >= 0.5 == np.array(all_labels))
    return avg_loss, auc, acc


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Data
    if args.train_dir is not None:
        train_ds = FrequencyDataset(root=args.train_dir, size=args.size)
        val_dir  = args.val_dir if args.val_dir else args.train_dir
        val_ds   = FrequencyDataset(root=val_dir, size=args.size)
        print(f"Train: {train_ds.label_counts()}  |  Val: {val_ds.label_counts()}")
        ref_dataset = train_ds
    else:
        dataset = FrequencyDataset(root=args.data, size=args.size)
        print(f"Dataset: {dataset.label_counts()}")
        train_ds, val_ds, _ = make_splits(dataset, seed=args.seed)
        ref_dataset = dataset

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, collate_fn=collate_fn_1d, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False,
                              num_workers=args.workers, collate_fn=collate_fn_1d, pin_memory=True)

    # Model
    r_max = ref_dataset[0][1].shape[0]  # radial profile length
    model_kwargs = {"input_length": r_max} if args.model == "1d" else {}
    model = build_model(args.model, **model_kwargs).to(device)
    print(f"Model: {args.model.upper()} CNN  |  params: {count_parameters(model):,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # WandB
    if args.wandb:
        import wandb
        wandb.init(project="specter", config=vars(args))
        wandb.watch(model, log_freq=100)

    best_val_auc = 0.0
    ckpt_path = os.path.join(args.out_dir, f"best_{args.model}.pt")

    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_auc, tr_acc = run_epoch(model, train_loader, criterion, optimizer,
                                             device, args.model, train=True)
        vl_loss, vl_auc, vl_acc = run_epoch(model, val_loader, criterion, optimizer,
                                             device, args.model, train=False)
        scheduler.step()

        print(f"Epoch {epoch:3d}/{args.epochs}  "
              f"train loss={tr_loss:.4f} auc={tr_auc:.4f} acc={tr_acc:.4f}  |  "
              f"val loss={vl_loss:.4f} auc={vl_auc:.4f} acc={vl_acc:.4f}")

        if args.wandb:
            wandb.log({"train/loss": tr_loss, "train/auc": tr_auc, "train/acc": tr_acc,
                       "val/loss": vl_loss, "val/auc": vl_auc, "val/acc": vl_acc,
                       "epoch": epoch})

        if vl_auc > best_val_auc:
            best_val_auc = vl_auc
            torch.save({"epoch": epoch, "model_type": args.model,
                        "model_state": model.state_dict(),
                        "val_auc": vl_auc, "args": vars(args)}, ckpt_path)
            print(f"  ✓ Saved checkpoint (val AUC {vl_auc:.4f})")

    print(f"\nBest val AUC: {best_val_auc:.4f}  →  {ckpt_path}")
    if args.wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
