"""
Run the full band-ablation pipeline: normalize → ablate (3 bands) → eval all detectors
→ append to results/metrics.csv.

Usage:
    # Full run (all datasets, all bands, all detectors)
    python scripts/run_ablation_pipeline.py

    # Resume after a crash without redoing completed work
    python scripts/run_ablation_pipeline.py --skip-existing

    # Single dataset
    python scripts/run_ablation_pipeline.py --datasets cifake

    # Recommended launch (hours-long run):
    nohup python scripts/run_ablation_pipeline.py --skip-existing \\
        > /root/ablation_pipeline.log 2>&1 &
    tail -f /root/ablation_pipeline.log
"""

import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, matthews_corrcoef, roc_auc_score

REPO = Path(__file__).parent.parent
NORM_ROOT = Path("/root/normalized")
ABLATED_ROOT = Path("/root/ablated")
METRICS_CSV = REPO / "results" / "metrics.csv"
LOGS = REPO / "results" / "logs"

DATASETS = {
    "cifake": {
        "norm_args": ["--input", str(REPO / "data/raw/cifake/test")],
        "norm_out": NORM_ROOT / "cifake" / "test",
        "checkpoint": str(REPO / "results/cifake/best_2d.pt"),
    },
    "defactify": {
        "norm_args": [
            "--parquet", str(REPO / "defactify_dataset/data"),
            "--split", "test",
        ],
        "norm_out": NORM_ROOT / "defactify" / "test",
        "checkpoint": str(REPO / "results/defactify/best_2d.pt"),
    },
    "genimage": {
        "norm_args": [
            "--input-real", str(REPO / "data/raw/imagenet_nature/val"),
            "--input-fake", str(REPO / "data/raw/ADM/imagenet_ai_0508_adm/train/ai"),
        ],
        "norm_out": NORM_ROOT / "genimage",
        "checkpoint": str(REPO / "results/genimage/best_2d.pt"),
    },
}

BANDS = ["low", "mid", "high"]
EXTERNAL_DETECTORS = ["cnndetection", "freqnet", "npr", "univfd", "fatformer", "bfree"]


def run(cmd, log_path=None):
    """Run a subprocess, streaming output to stdout and optionally a log file."""
    cmd = [str(c) for c in cmd]
    print(f"\n>>> {' '.join(cmd)}", flush=True)
    if log_path:
        Path(log_path).parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, "w") as lf:
            proc = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
            )
            for line in proc.stdout:
                print(line, end="", flush=True)
                lf.write(line)
            proc.wait()
        if proc.returncode != 0:
            raise RuntimeError(f"Command failed (exit {proc.returncode}): {' '.join(cmd)}")
    else:
        subprocess.run(cmd, check=True)


def append_metrics(csv_path, detector, dataset, condition):
    csv_path = Path(csv_path)
    if not csv_path.exists() or csv_path.stat().st_size == 0:
        print(f"  WARNING: missing or empty CSV {csv_path}, skipping metrics", flush=True)
        return
    df = pd.read_csv(csv_path)
    auc = roc_auc_score(df.label, df.score)
    acc = accuracy_score(df.label, df.score > 0.5)
    mcc = matthews_corrcoef(df.label, df.score > 0.5)
    n_real = int((df.label == 0).sum())
    n_fake = int((df.label == 1).sum())
    row = f"{detector},{dataset},{condition},{auc:.4f},{acc:.4f},{mcc:.4f},{n_real},{n_fake}"
    print(f"  METRICS  {row}", flush=True)
    with open(METRICS_CSV, "a") as f:
        f.write(row + "\n")


def normalize(dataset_name, cfg, skip):
    out = cfg["norm_out"]
    if skip and (out / "real").exists():
        print(f"[{dataset_name}] Norm dir exists — skipping normalize", flush=True)
        return
    cmd = (
        [sys.executable, str(REPO / "scripts/normalize_dataset.py")]
        + cfg["norm_args"]
        + ["--output", str(out)]
    )
    run(cmd, log_path=LOGS / f"normalize_{dataset_name}.txt")


def ablate(dataset_name, norm_out, band, skip):
    out = ABLATED_ROOT / dataset_name / band
    if skip and (out / "real").exists():
        print(f"[{dataset_name}/{band}] Ablated dir exists — skipping ablation", flush=True)
        return out
    cmd = [
        sys.executable, str(REPO / "scripts/band_ablation.py"),
        "--input", str(norm_out),
        "--band", band,
        "--output", str(out),
    ]
    run(cmd, log_path=LOGS / f"ablate_{dataset_name}_{band}.txt")
    return out


def eval_cnn2d(checkpoint, data_dir, out_csv, skip):
    out_csv = Path(out_csv)
    if skip and out_csv.exists():
        print(f"  cnn2d: output CSV exists — skipping eval", flush=True)
        return
    cmd = [
        sys.executable, str(REPO / "scripts/eval.py"),
        "--checkpoint", checkpoint,
        "--data", str(data_dir),
        "--split", "all",
        "--out", str(out_csv),
    ]
    run(cmd, log_path=LOGS / f"eval_cnn2d_{out_csv.stem}.txt")


def eval_external(detector, data_dir, out_csv, skip):
    out_csv = Path(out_csv)
    if skip and out_csv.exists():
        print(f"  {detector}: output CSV exists — skipping eval", flush=True)
        return
    cmd = [
        sys.executable, str(REPO / "scripts/eval_external.py"),
        "--detector", detector,
        "--data", str(data_dir),
        "--out", str(out_csv),
    ]
    run(cmd, log_path=LOGS / f"eval_{detector}_{out_csv.stem}.txt")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--datasets", nargs="+", default=["cifake", "defactify", "genimage"],
        choices=list(DATASETS),
    )
    parser.add_argument("--bands", nargs="+", default=BANDS, choices=BANDS)
    parser.add_argument(
        "--skip-existing", action="store_true",
        help="Skip normalize/ablate/eval steps where output already exists",
    )
    args = parser.parse_args()

    LOGS.mkdir(parents=True, exist_ok=True)

    for dataset_name in args.datasets:
        cfg = DATASETS[dataset_name]
        skip = args.skip_existing
        print(f"\n{'='*60}", flush=True)
        print(f"DATASET: {dataset_name}", flush=True)
        print(f"{'='*60}", flush=True)

        normalize(dataset_name, cfg, skip)
        norm_out = cfg["norm_out"]

        for band in args.bands:
            print(f"\n--- {dataset_name} / ablated_{band} ---", flush=True)
            ablated_dir = ablate(dataset_name, norm_out, band, skip)
            condition = f"ablated_{band}"

            # CNN2D (dataset-matched checkpoint)
            out_csv = REPO / "results" / f"cnn2d_{dataset_name}_ablated_{band}.csv"
            eval_cnn2d(cfg["checkpoint"], ablated_dir, out_csv, skip)
            append_metrics(out_csv, "cnn2d", dataset_name, condition)

            # External detectors
            for det in EXTERNAL_DETECTORS:
                out_csv = REPO / "results" / f"{det}_{dataset_name}_ablated_{band}.csv"
                eval_external(det, ablated_dir, out_csv, skip)
                append_metrics(out_csv, det, dataset_name, condition)

        print(f"\n=== {dataset_name}: all bands complete ===", flush=True)

    print("\n=== Pipeline complete ===", flush=True)


if __name__ == "__main__":
    main()
