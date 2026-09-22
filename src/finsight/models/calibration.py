from __future__ import annotations

import numpy as np
import pandas as pd

from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss


def calculate_calibration_metrics(
    y_true,
    probabilities,
    n_bins: int = 10,
) -> dict:
    """
    Calculate probability calibration metrics.

    Returns:
        brier_score
        mean_absolute_calibration_error
        max_calibration_error
        calibration_table
    """

    y_true = np.asarray(y_true)
    probabilities = np.asarray(probabilities)

    if len(y_true) != len(probabilities):
        raise ValueError(
            "y_true and probabilities must have "
            "the same length."
        )

    if not np.all(np.isfinite(probabilities)):
        raise ValueError(
            "Probabilities contain non-finite values."
        )

    if np.any(
        (probabilities < 0)
        | (probabilities > 1)
    ):
        raise ValueError(
            "Probabilities must be between 0 and 1."
        )

    brier = brier_score_loss(
        y_true,
        probabilities,
    )

    fraction_positive, mean_prediction = (
        calibration_curve(
            y_true,
            probabilities,
            n_bins=n_bins,
            strategy="quantile",
        )
    )

    calibration_table = pd.DataFrame(
        {
            "mean_predicted_probability":
                mean_prediction,
            "observed_positive_rate":
                fraction_positive,
        }
    )

    calibration_table[
        "absolute_calibration_error"
    ] = (
        calibration_table[
            "mean_predicted_probability"
        ]
        - calibration_table[
            "observed_positive_rate"
        ]
    ).abs()

    mean_absolute_error = (
        calibration_table[
            "absolute_calibration_error"
        ].mean()
    )

    max_error = (
        calibration_table[
            "absolute_calibration_error"
        ].max()
    )

    return {
        "brier_score": float(brier),
        "mean_absolute_calibration_error":
            float(mean_absolute_error),
        "max_calibration_error":
            float(max_error),
        "calibration_table":
            calibration_table,
    }


def print_calibration_report(
    model_name: str,
    metrics: dict,
) -> None:
    """
    Print a readable calibration report.
    """

    print()
    print("=" * 80)
    print(f"{model_name} CALIBRATION")
    print("=" * 80)

    print(
        f"Brier score: "
        f"{metrics['brier_score']:.4f}"
    )

    print(
        f"Mean absolute calibration error: "
        f"{metrics['mean_absolute_calibration_error']:.4f}"
    )

    print(
        f"Maximum calibration error: "
        f"{metrics['max_calibration_error']:.4f}"
    )

    print()
    print("CALIBRATION TABLE")

    print(
        metrics[
            "calibration_table"
        ].to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )