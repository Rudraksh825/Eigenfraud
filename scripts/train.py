"""
Train a Specter detector (1D or 2D CNN) on a frequency-domain image dataset.

Single-GPU (original mode):
    python scripts/train.py --model 2d --train-dir data/processed/cifake/train --val-dir data/processed/cifake/test

Single-GPU with pre-computed cache:
    python scripts/train.py --model 2d --cache data/cache/genimage_all/manifest.csv

Multi-GPU DDP (8 GPUs):
    torchrun --nproc_per_node=8 scripts/train.py --model 2d --cache data/cache/genimage_all/manifest.csv

Auto-split from raw directory:
    python scripts/train.py --model 1d --data data/processed/mydata
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from sklearn.metrics import roc_auc_score
import numpy as np

from tqdm import tqdm

from src.dataset import FrequencyDataset, CachedFrequencyDataset, ParquetFrequencyDataset, make_splits
from src.models import build_model, count_parameters


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, choices=["1d", "2d"])
    # Data sources (pick one)
    p.add_argument("--train-dir", default=None, help="Pre-split train dir (real/ + fake/)")
    p.add_argument("--val-dir",   default=None, help="Pre-split val dir")
    p.add_argument("--data",      default=None, help="Single root dir — auto-split 80/10/10")
    p.add_argument("--cache",       default=None, help="manifest.csv from scripts/precompute.py")
    p.add_argument("--parquet-dir", default=None, help="defactify_dataset/data/ dir (parquet splits)")
    # Hyperparams
    p.add_argument("--epochs",       type=int,   default=30)
    p.add_argument("--lr",           type=float, default=3e-4)
    p.add_argument("--batch-size",   type=int,   default=64)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--size",         type=int,   default=224, help="Image resize target before FFT")
    p.add_argument("--workers",      type=int,   default=4)
    p.add_argument("--out-dir",      default="results")
    p.add_argument("--wandb",        action="store_true")
    p.add_argument("--seed",         type=int,   default=42)
    p.add_argument("--class-weight", action="store_true",
                   help="Weight loss inversely by class frequency")
    args = p.parse_args()
    if args.train_dir is None and args.data is None and args.cache is None and args.parquet_dir is None:
        p.error("Provide --train-dir/--val-dir, --data, --cache, or --parquet-dir")
    return args


# ── DDP helpers ───────────────────────────────────────────────────────────────

def setup_ddp():
    """Initialize process group if launched via torchrun. Returns (rank, world_size)."""
    if "RANK" in os.environ:
        dist.init_process_group(backend="nccl")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        torch.cuda.set_device(rank)
        return rank, world_size
    return 0, 1


def is_main(rank):
    return rank == 0


def teardown_ddp():
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


# ── Data ──────────────────────────────────────────────────────────────────────

def build_datasets(args):
    if args.parquet_dir is not None:
        train_ds = ParquetFrequencyDataset(args.parquet_dir, split="train",      size=args.size)
        val_ds   = ParquetFrequencyDataset(args.parquet_dir, split="validation", size=args.size)
        ref_ds   = train_ds
    elif args.cache is not None:
        full_ds = CachedFrequencyDataset(manifest=args.cache)
        train_ds, val_ds, _ = make_splits(full_ds, seed=args.seed)
        ref_ds = full_ds
    elif args.train_dir is not None:
        train_ds = FrequencyDataset(root=args.train_dir, size=args.size)
        val_ds   = FrequencyDataset(root=args.val_dir or args.train_dir, size=args.size)
        ref_ds   = train_ds
    else:
        full_ds  = FrequencyDataset(root=args.data, size=args.size)
        train_ds, val_ds, _ = make_splits(full_ds, seed=args.seed)
        ref_ds   = full_ds
    return train_ds, val_ds, ref_ds


def build_loaders(train_ds, val_ds, args, rank, world_size):
    pin = torch.cuda.is_available()
    if world_size > 1:
        train_sampler = DistributedSampler(train_ds, num_replicas=world_size,
                                           rank=rank, shuffle=True, seed=args.seed)
        val_sampler   = DistributedSampler(val_ds,   num_replicas=world_size,
                                           rank=rank, shuffle=False)
        train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                                  sampler=train_sampler, num_workers=args.workers,
                                  collate_fn=collate_fn, pin_memory=pin)
        val_loader   = DataLoader(val_ds,   batch_size=args.batch_size,
                                  sampler=val_sampler,   num_workers=args.workers,
                                  collate_fn=collate_fn, pin_memory=pin)
    else:
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                                  num_workers=args.workers, collate_fn=collate_fn,
                                  pin_memory=pin)
        val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False,
                                  num_workers=args.workers, collate_fn=collate_fn,
                                  pin_memory=pin)
    return train_loader, val_loader


def collate_fn(batch):
    spec2d, prof1d, labels = zip(*batch)
    return torch.stack(spec2d), torch.stack(prof1d), torch.tensor(labels)


# ── Train / eval ─────────────────────────────────────────────────────────────

def run_epoch(model, loader, criterion, optimizer, device, model_type, train: bool,
              desc: str = "", rank: int = 0, world_size: int = 1):
    model.train() if train else model.eval()
    total_loss, all_labels, all_probs = 0.0, [], []

    if hasattr(loader.sampler, "set_epoch") and train:
        # Set epoch for DistributedSampler so shuffling differs each epoch
        pass  # caller sets this

    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        pbar = tqdm(loader, desc=desc, leave=False, unit="batch", disable=(rank != 0))
        for spec2d, prof1d, labels in pbar:
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
            if rank == 0:
                pbar.set_postfix(loss=f"{loss.item():.4f}")

    # Aggregate across ranks
    if world_size > 1:
        # Gather all labels/probs to rank 0 for AUC calculation
        all_labels_t = torch.tensor(all_labels, dtype=torch.float32, device=device)
        all_probs_t  = torch.tensor(all_probs,  dtype=torch.float32, device=device)
        loss_t       = torch.tensor([total_loss, float(len(all_labels))], device=device)

        gathered_labels = [torch.zeros_like(all_labels_t) for _ in range(world_size)]
        gathered_probs  = [torch.zeros_like(all_probs_t)  for _ in range(world_size)]
        dist.all_gather(gathered_labels, all_labels_t)
        dist.all_gather(gathered_probs,  all_probs_t)
        dist.all_reduce(loss_t, op=dist.ReduceOp.SUM)

        all_labels = torch.cat(gathered_labels).cpu().numpy()
        all_probs  = torch.cat(gathered_probs).cpu().numpy()
        total_loss = loss_t[0].item()
        n = int(loss_t[1].item())
    else:
        n = len(all_labels)

    avg_loss = total_loss / n
    auc = roc_auc_score(all_labels, all_probs) if len(set(all_labels)) > 1 else float("nan")
    acc = np.mean((np.array(all_probs) >= 0.5) == np.array(all_labels))
    return avg_loss, auc, acc


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    rank, world_size = setup_ddp()
    torch.manual_seed(args.seed + rank)
    os.makedirs(args.out_dir, exist_ok=True)

    if world_size > 1:
        device = torch.device(f"cuda:{rank}")
    else:
        device = torch.device("cuda" if torch.cuda.is_available()
                              else "mps" if torch.backends.mps.is_available() else "cpu")

    if is_main(rank):
        print(f"Device: {device}  |  World size: {world_size}")

    # Data
    train_ds, val_ds, ref_ds = build_datasets(args)
    if is_main(rank):
        if hasattr(train_ds, "label_counts"):
            print(f"Train: {train_ds.label_counts()}  |  Val: {val_ds.label_counts()}")
        else:
            # Subset — compute manually
            tr_labels = [ref_ds.samples[i][1] for i in train_ds.indices]
            va_labels = [ref_ds.samples[i][1] for i in val_ds.indices]
            print(f"Train: real={tr_labels.count(0)} fake={tr_labels.count(1)}  |  "
                  f"Val: real={va_labels.count(0)} fake={va_labels.count(1)}")

    train_loader, val_loader = build_loaders(train_ds, val_ds, args, rank, world_size)

    # Model
    if args.cache is not None:
        r_max = ref_ds[0][1].shape[0]
    else:
        r_max = ref_ds[0][1].shape[0]
    model_kwargs = {"input_length": r_max} if args.model == "1d" else {}
    model = build_model(args.model, **model_kwargs).to(device)

    if world_size > 1:
        model = DDP(model, device_ids=[rank])

    if is_main(rank):
        print(f"Model: {args.model.upper()} CNN  |  params: {count_parameters(model):,}")

    # Loss
    if args.class_weight:
        if hasattr(train_ds, "dataset"):  # Subset from make_splits
            train_labels = [train_ds.dataset.samples[i][1] for i in train_ds.indices]
        else:
            train_labels = [s[1] for s in train_ds.samples]
        n = len(train_labels)
        n_real = train_labels.count(0)
        n_fake = train_labels.count(1)
        w = torch.tensor([n / (2 * n_real), n / (2 * n_fake)], dtype=torch.float32).to(device)
        if is_main(rank):
            print(f"Class weights: real={w[0]:.3f}  fake={w[1]:.3f}  (real={n_real}, fake={n_fake})")
        criterion = nn.CrossEntropyLoss(weight=w)
    else:
        criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # WandB (rank 0 only)
    if args.wandb and is_main(rank):
        import wandb
        wandb.init(project="specter", config=vars(args))
        wandb.watch(model, log_freq=100)

    best_val_auc = 0.0
    ckpt_path = os.path.join(args.out_dir, f"best_{args.model}.pt")

    for epoch in range(1, args.epochs + 1):
        if hasattr(train_loader.sampler, "set_epoch"):
            train_loader.sampler.set_epoch(epoch)

        tr_loss, tr_auc, tr_acc = run_epoch(
            model, train_loader, criterion, optimizer, device, args.model,
            train=True, desc=f"Epoch {epoch:3d}/{args.epochs} train",
            rank=rank, world_size=world_size)
        vl_loss, vl_auc, vl_acc = run_epoch(
            model, val_loader, criterion, optimizer, device, args.model,
            train=False, desc=f"Epoch {epoch:3d}/{args.epochs} val  ",
            rank=rank, world_size=world_size)
        scheduler.step()

        if is_main(rank):
            print(f"Epoch {epoch:3d}/{args.epochs}  "
                  f"train loss={tr_loss:.4f} auc={tr_auc:.4f} acc={tr_acc:.4f}  |  "
                  f"val loss={vl_loss:.4f} auc={vl_auc:.4f} acc={vl_acc:.4f}")

            if args.wandb:
                import wandb
                wandb.log({"train/loss": tr_loss, "train/auc": tr_auc, "train/acc": tr_acc,
                           "val/loss": vl_loss, "val/auc": vl_auc, "val/acc": vl_acc,
                           "epoch": epoch})

            if vl_auc > best_val_auc:
                best_val_auc = vl_auc
                state = model.module.state_dict() if world_size > 1 else model.state_dict()
                torch.save({"epoch": epoch, "model_type": args.model,
                            "model_state": state,
                            "val_auc": vl_auc, "args": vars(args)}, ckpt_path)
                print(f"  ✓ Saved checkpoint (val AUC {vl_auc:.4f})")

        if world_size > 1:
            dist.barrier()

    if is_main(rank):
        print(f"\nBest val AUC: {best_val_auc:.4f}  →  {ckpt_path}")
        if args.wandb:
            import wandb
            wandb.finish()

    teardown_ddp()


if __name__ == "__main__":
    main()
