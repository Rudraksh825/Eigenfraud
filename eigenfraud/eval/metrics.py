"""Detection metrics: AP, Accuracy, AUC, Pd@FPR."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    roc_curve,
)


def compute_metrics(scores: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    """Compute standard detection metrics.

    Args:
        scores: Predicted probabilities (sigmoid of logits), shape (N,).
        labels: Binary ground-truth labels (0=real, 1=fake), shape (N,).

    Returns:
        Dict with keys: ap, acc, auc, pd_at_1fpr, pd_at_10fpr.
    """
    ap = float(average_precision_score(labels, scores))
    auc = float(roc_auc_score(labels, scores))
    acc = float(((scores >= 0.5) == labels).mean())

    fpr, tpr, _ = roc_curve(labels, scores)
    pd_at_1 = float(np.interp(0.01, fpr, tpr))
    pd_at_10 = float(np.interp(0.10, fpr, tpr))

    return {
        "ap": ap,
        "acc": acc,
        "auc": auc,
        "pd_at_1fpr": pd_at_1,
        "pd_at_10fpr": pd_at_10,
    }


def per_generator_metrics(
    scores: np.ndarray,
    labels: np.ndarray,
    generator_ids: list[str],
) -> pd.DataFrame:
    """Compute metrics disaggregated by generator, plus an aggregate row.

    Args:
        scores: Predicted probabilities, shape (N,).
        labels: Binary labels, shape (N,).
        generator_ids: Generator name for each sample, length N.

    Returns:
        DataFrame with columns [generator, ap, acc, auc, pd_at_1fpr, pd_at_10fpr].
        Last row is "mean" (mean of per-generator values, not pooled).
    """
    generator_ids = np.array(generator_ids)
    rows = []
    for gen in sorted(set(generator_ids)):
        mask = generator_ids == gen
        m = compute_metrics(scores[mask], labels[mask])
        rows.append({"generator": gen, **m})

    # Aggregate
    agg = compute_metrics(scores, labels)
    rows.append({"generator": "ALL", **agg})

    df = pd.DataFrame(rows)
    return df
