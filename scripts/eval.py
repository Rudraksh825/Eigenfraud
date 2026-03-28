"""
Evaluate trained Eigenfraud checkpoint(s) on a test directory.

Usage:
    python scripts/eval.py --checkpoint results/best_2d.pt --data data/processed/cifake --split test
    python scripts/eval.py --checkpoint results/best_1d.pt results/best_2d.pt --data data/raw/cifake/test --split all
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import torch
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, accuracy_score

from src.dataset import FrequencyDataset, make_splits
from src.models import build_model


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", nargs="+", required=True)
    p.add_argument("--data", required=True, help="Directory with real/ and fake/ subdirs")
    p.add_argument("--split", choices=["test", "all"], default="test",
                   help="'test' uses the held-out split from make_splits; 'all' uses every image")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def compute_eer(labels, scores) -> float:
    """Equal Error Rate via linear interpolation on the ROC curve."""
    from sklearn.metrics import roc_curve
    fpr, tpr, _ = roc_curve(labels, scores)
    fnr = 1 - tpr
    eer_idx = np.argmin(np.abs(fpr - fnr))
    return float((fpr[eer_idx] + fnr[eer_idx]) / 2)


def collate_fn(batch):
    spec2d, prof1d, labels = zip(*batch)
    return torch.stack(spec2d), torch.stack(prof1d), torch.tensor(labels)


def evaluate_checkpoint(ckpt_path, args, device, loader, dataset) -> dict:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model_type = ckpt["model_type"]

    r_max = dataset[0][1].shape[0]
    model_kwargs = {"input_length": r_max} if model_type == "1d" else {}
    model = build_model(model_type, **model_kwargs).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    all_labels, all_probs = [], []
    with torch.no_grad():
        for spec2d, prof1d, labels in loader:
            x = prof1d.to(device) if model_type == "1d" else spec2d.to(device)
            logits = model(x)
            probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
            all_probs.extend(probs)
            all_labels.extend(labels.numpy())

    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    preds = (all_probs >= 0.5).astype(int)

    return {
        "name": model_type.upper() + " CNN",
        "ckpt": ckpt_path,
        "auc": roc_auc_score(all_labels, all_probs),
        "acc": accuracy_score(all_labels, preds),
        "eer": compute_eer(all_labels, all_probs),
    }


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

    # Determine image size from first checkpoint
    first_ckpt = torch.load(args.checkpoint[0], map_location=device, weights_only=False)
    size = first_ckpt.get("args", {}).get("size", 224)

    # Data
    dataset = FrequencyDataset(root=args.data, size=size)
    print(f"Dataset: {dataset.label_counts()}")

    if args.split == "test":
        _, _, eval_ds = make_splits(dataset, seed=args.seed)
        print(f"Using held-out test split: {len(eval_ds)} samples")
    else:
        eval_ds = dataset
        print(f"Using all samples: {len(eval_ds)}")

    loader = DataLoader(eval_ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.workers, collate_fn=collate_fn)

    results = [evaluate_checkpoint(ckpt, args, device, loader, eval_ds) for ckpt in args.checkpoint]

    if len(results) == 1:
        r = results[0]
        print(f"\n{'='*50}")
        print(f"Checkpoint : {r['ckpt']}")
        print(f"Test dir   : {args.data}")
        print(f"Model      : {r['name']}")
        print(f"{'='*50}")
        print(f"AUC        : {r['auc']:.4f}")
        print(f"Accuracy   : {r['acc']:.4f}")
        print(f"EER        : {r['eer']:.4f}")
        print(f"{'='*50}")
    else:
        W = 50
        print(f"\nTest dir : {args.data}  (split: {args.split})")
        print("=" * W)
        print(f"{'Model':<10}{'Checkpoint':<26}{'AUC':>7}{'Acc':>7}{'EER':>7}")
        print("-" * W)
        for r in results:
            print(f"{r['name']:<10}{r['ckpt']:<26}{r['auc']:>7.4f}{r['acc']:>7.4f}{r['eer']:>7.4f}")
        print("=" * W)


if __name__ == "__main__":
    main()
