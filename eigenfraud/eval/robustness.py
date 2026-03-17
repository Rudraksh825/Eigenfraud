"""Robustness evaluation: JPEG QF and resize factor sweeps."""
from __future__ import annotations

import io

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm

from eigenfraud.eval.metrics import compute_metrics


# ---------------------------------------------------------------------------
# Perturbation functions
# ---------------------------------------------------------------------------

def jpeg_perturb(img: Image.Image, quality: int) -> Image.Image:
    """Re-encode a PIL image with JPEG at the given quality factor."""
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def resize_perturb(img: Image.Image, scale: float) -> Image.Image:
    """Downscale and optionally upscale back to original size."""
    orig_size = img.size  # (W, H)
    new_w = max(1, int(orig_size[0] * scale))
    new_h = max(1, int(orig_size[1] * scale))
    img = img.resize((new_w, new_h), Image.BILINEAR)
    if scale < 1.0:
        img = img.resize(orig_size, Image.BILINEAR)
    return img


# ---------------------------------------------------------------------------
# Sweep runner
# ---------------------------------------------------------------------------

@torch.no_grad()
def _eval_with_perturb(
    model: nn.Module,
    dataset,
    perturb_fn,
    device: torch.device,
    batch_size: int = 64,
) -> dict[str, float]:
    """Run inference on a dataset with an image-level perturbation applied."""
    model.eval()
    all_scores, all_labels = [], []

    # We iterate the underlying dataset manually to apply PIL-level perturbations
    for idx in tqdm(range(len(dataset)), leave=False):
        item = dataset.dataset[idx] if hasattr(dataset, "dataset") else dataset[idx]
        pil_img, label = item[0], item[1]

        # item[0] may already be a tensor if transform was applied — skip perturb
        # The dataset should be constructed WITHOUT spectral transforms for robustness eval
        if not isinstance(pil_img, Image.Image):
            raise TypeError(
                "Dataset items must be PIL Images for robustness evaluation. "
                "Construct the dataset with transform=None and pass spectral_transform separately."
            )

        perturbed = perturb_fn(pil_img)
        tensor = dataset.spectral_transform(perturbed).unsqueeze(0).to(device)
        logit = model(tensor).squeeze()
        score = torch.sigmoid(logit).item()
        all_scores.append(score)
        all_labels.append(int(label))

    return compute_metrics(np.array(all_scores), np.array(all_labels))


def run_robustness_sweep(
    model: nn.Module,
    dataset,
    jpeg_qualities: list[int],
    resize_factors: list[float],
    device: torch.device,
    batch_size: int = 64,
) -> pd.DataFrame:
    """Run full JPEG QF and resize factor robustness sweeps.

    Args:
        model: Trained detector.
        dataset: Dataset with no spectral transform applied; must expose a
            `.spectral_transform` attribute for the post-perturbation transform.
        jpeg_qualities: JPEG QF values to test (e.g. [100, 90, 75, 50, 30]).
        resize_factors: Scale factors to test (e.g. [1.0, 0.9, 0.8, 0.7, 0.5]).
        device: Torch device.
        batch_size: Unused (kept for API symmetry); evaluation is sample-by-sample.

    Returns:
        DataFrame with columns [perturbation_type, param_value, ap, acc, auc,
        pd_at_1fpr, pd_at_10fpr, delta_ap, delta_auc].
    """
    rows = []

    # JPEG sweep
    baseline_jpeg = None
    for qf in jpeg_qualities:
        print(f"  JPEG QF={qf}")
        m = _eval_with_perturb(
            model, dataset, lambda img, q=qf: jpeg_perturb(img, q), device
        )
        rows.append({"perturbation_type": "jpeg", "param_value": qf, **m})
        if qf == 100:
            baseline_jpeg = m

    # Resize sweep
    baseline_resize = None
    for sf in resize_factors:
        print(f"  Resize ×{sf}")
        m = _eval_with_perturb(
            model, dataset, lambda img, s=sf: resize_perturb(img, s), device
        )
        rows.append({"perturbation_type": "resize", "param_value": sf, **m})
        if sf == 1.0:
            baseline_resize = m

    df = pd.DataFrame(rows)

    # Add delta columns relative to clean baseline
    def _delta(row):
        if row["perturbation_type"] == "jpeg":
            base = baseline_jpeg
        else:
            base = baseline_resize
        if base is None:
            return pd.Series({"delta_ap": float("nan"), "delta_auc": float("nan")})
        return pd.Series({
            "delta_ap": row["ap"] - base["ap"],
            "delta_auc": row["auc"] - base["auc"],
        })

    df = pd.concat([df, df.apply(_delta, axis=1)], axis=1)
    return df
