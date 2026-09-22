from __future__ import annotations

import numpy as np
import pandas as pd

from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def evaluate_binary_classifier(
    y_true,
    y_probability,
    threshold: float = 0.50,
) -> dict:
    """
    Evaluate a binary classifier using probability predictions.

    Primary metric:
        PR-AUC / Average Precision

    Secondary metrics:
        ROC-AUC
        Precision
        Recall
        F1
        Brier score
    """

    y_true = np.asarray(y_true)
    y_probability = np.asarray(y_probability)

    if len(y_true) != len(y_probability):
        raise ValueError(
            "y_true and y_probability must have the same length"
        )

    if not np.all(np.isfinite(y_probability)):
        raise ValueError(
            "Predicted probabilities contain NaN or infinite values"
        )

    if np.any((y_probability < 0) | (y_probability > 1)):
        raise ValueError(
            "Predicted probabilities must be between 0 and 1"
        )

    y_pred = (y_probability >= threshold).astype(int)

    # PR-AUC is valid even when the positive class is rare.
    pr_auc = average_precision_score(
        y_true,
        y_probability,
    )

    # ROC-AUC requires both classes to be present.
    if len(np.unique(y_true)) == 2:
        roc_auc = roc_auc_score(
            y_true,
            y_probability,
        )
    else:
        roc_auc = np.nan

    precision = precision_score(
        y_true,
        y_pred,
        zero_division=0,
    )

    recall = recall_score(
        y_true,
        y_pred,
        zero_division=0,
    )

    f1 = f1_score(
        y_true,
        y_pred,
        zero_division=0,
    )

    brier = brier_score_loss(
        y_true,
        y_probability,
    )

    return {
        "pr_auc": float(pr_auc),
        "roc_auc": float(roc_auc),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "brier_score": float(brier),
        "threshold": float(threshold),
    }


def evaluate_multiple_thresholds(
    y_true,
    y_probability,
    thresholds: list[float] | None = None,
) -> pd.DataFrame:
    """
    Evaluate precision, recall and F1 across classification thresholds.

    Threshold selection should be performed on validation data,
    never on the final test set.
    """

    if thresholds is None:
        thresholds = [
            0.10,
            0.15,
            0.20,
            0.25,
            0.30,
            0.35,
            0.40,
            0.45,
            0.50,
            0.55,
            0.60,
            0.70,
            0.80,
            0.90,
        ]

    rows = []

    y_true = np.asarray(y_true)
    y_probability = np.asarray(y_probability)

    for threshold in thresholds:
        y_pred = (
            y_probability >= threshold
        ).astype(int)

        rows.append(
            {
                "threshold": threshold,
                "precision": precision_score(
                    y_true,
                    y_pred,
                    zero_division=0,
                ),
                "recall": recall_score(
                    y_true,
                    y_pred,
                    zero_division=0,
                ),
                "f1": f1_score(
                    y_true,
                    y_pred,
                    zero_division=0,
                ),
            }
        )

    return pd.DataFrame(rows)


def print_metrics(
    name: str,
    metrics: dict,
) -> None:
    """Print model evaluation metrics."""

    print()
    print("=" * 70)
    print(name.upper())
    print("=" * 70)

    print(f"PR-AUC:       {metrics['pr_auc']:.4f}")
    print(f"ROC-AUC:      {metrics['roc_auc']:.4f}")
    print(f"Precision:    {metrics['precision']:.4f}")
    print(f"Recall:       {metrics['recall']:.4f}")
    print(f"F1:           {metrics['f1']:.4f}")
    print(f"Brier score:  {metrics['brier_score']:.4f}")
    print(f"Threshold:    {metrics['threshold']:.2f}")


def compare_models(results: dict[str, dict]) -> pd.DataFrame:
    """
    Convert model metric dictionaries into a comparison table.
    """

    rows = []

    for model_name, metrics in results.items():
        row = {
            "model": model_name,
            **metrics,
        }

        rows.append(row)

    return (
        pd.DataFrame(rows)
        .sort_values(
            "pr_auc",
            ascending=False,
        )
        .reset_index(drop=True)
    )