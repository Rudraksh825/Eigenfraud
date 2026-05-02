# H100 Training Runbook

Data lives on the Modal volume — it will be present on the new machine automatically.
Run Steps 1 and 2 in parallel (extraction is CPU/disk only, training is GPU only).

---

## Step 1 — Resume sdv5 + wukong extraction (CPU, background)

sdv4 was already extracted on the previous machine. Pick up sdv5 and wukong:

```bash
nohup bash /tmp/extract_wave2.sh > /tmp/genimage_wave2.log 2>&1 &
echo "Extraction PID: $!"
```

> The script skips already-extracted generators, so re-running it is safe.
> It will go straight to sdv5 → wukong.

Monitor:
```bash
tail -f /tmp/genimage_wave2.log
```

---

## Step 2 — Train on available data (GPU, run immediately in parallel)

### 2a — Set up the merged dataset symlinks

```bash
bash scripts/setup_genimage_merged.sh
ls data/raw/genimage_all/
# Should show: ADM  BigGAN  VQDM  glide  nature  sdv4
# (sdv5 and wukong will be missing until Step 1 finishes — that's fine)
```

### 2b — Pre-compute spectra (CPU, run once, ~1–2 hrs)

```bash
nohup python -u scripts/precompute.py \
  --data data/raw/genimage_all \
  --cache-dir data/cache/genimage_all \
  --workers 16 \
> /tmp/precompute.log 2>&1 &
echo "Precompute PID: $!"
```

Monitor:
```bash
tail -f /tmp/precompute.log
```

Done when: `Done. Errors: N  Manifest: data/cache/genimage_all/manifest.csv`

### 2c — Train 2D model on 8 GPUs

```bash
nohup torchrun --nproc_per_node=8 scripts/train.py \
  --model 2d \
  --cache data/cache/genimage_all/manifest.csv \
  --epochs 30 \
  --out-dir results/genimage \
  --class-weight \
  --workers 4 \
> /tmp/train_2d.log 2>&1 &
echo "Train 2D PID: $!"
```

### 2d — Train 1D model on 8 GPUs

```bash
nohup torchrun --nproc_per_node=8 scripts/train.py \
  --model 1d \
  --cache data/cache/genimage_all/manifest.csv \
  --epochs 30 \
  --out-dir results/genimage \
  --class-weight \
  --workers 4 \
> /tmp/train_1d.log 2>&1 &
echo "Train 1D PID: $!"
```

---

## Step 3 — Add sdv5 + wukong once extraction finishes

When `tail /tmp/genimage_wave2.log` shows `Wave 2 complete`:

```bash
# Re-run merge to add sdv5/wukong symlinks
bash scripts/setup_genimage_merged.sh
ls data/raw/genimage_all/
# Now shows: ADM  BigGAN  VQDM  glide  nature  sdv4  sdv5  wukong

# Extend the cache (resumes, only processes new images)
python scripts/precompute.py \
  --data data/raw/genimage_all \
  --cache-dir data/cache/genimage_all \
  --workers 16

# Retrain from scratch on full dataset
torchrun --nproc_per_node=8 scripts/train.py \
  --model 2d \
  --cache data/cache/genimage_all/manifest.csv \
  --epochs 30 --out-dir results/genimage_full \
  --class-weight --wandb
```

---

## Monitoring

```bash
# Epoch summaries only (no tqdm noise)
cat /tmp/train_2d.log | tr '\r' '\n' | grep "^Epoch"
cat /tmp/train_1d.log | tr '\r' '\n' | grep "^Epoch"

# Check all processes are alive
ps aux | grep -E "train.py|precompute|extract_wave" | grep -v grep
```

---

## Checkpoints

- `results/genimage/best_2d.pt`
- `results/genimage/best_1d.pt`
- `results/genimage_full/best_2d.pt` — after full 7-generator retrain
