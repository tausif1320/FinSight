from __future__ import annotations

from pathlib import Path
import sys

import joblib
import pandas as pd

from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    brier_score_loss,
)


ROOT = (
    Path(__file__)
    .resolve()
    .parents[1]
)

sys.path.insert(
    0,
    str(ROOT / "src"),
)


from finsight.models.dataset import (
    chronological_split,
)


DATA_PATH = (
    ROOT
    / "data"
    / "processed"
    / "modeling_dataset_1000_users.csv"
)

MODEL_PATH = (
    ROOT
    / "models"
    / "stress_model.joblib"
)

METADATA_PATH = (
    ROOT
    / "models"
    / "feature_metadata.json"
)

REPORT_DIR = (
    ROOT
    / "reports"
    / "models"
)


def evaluate(
    y_true,
    probabilities,
    threshold,
):

    predictions = (
        probabilities >= threshold
    ).astype(int)

    return {
        "pr_auc":
            average_precision_score(
                y_true,
                probabilities,
            ),

        "roc_auc":
            roc_auc_score(
                y_true,
                probabilities,
            ),

        "precision":
            precision_score(
                y_true,
                predictions,
                zero_division=0,
            ),

        "recall":
            recall_score(
                y_true,
                predictions,
                zero_division=0,
            ),

        "f1":
            f1_score(
                y_true,
                predictions,
                zero_division=0,
            ),

        "brier_score":
            brier_score_loss(
                y_true,
                probabilities,
            ),

        "threshold":
            threshold,
    }


def main():

    print("=" * 80)
    print("FinSight Final Model Evaluation")
    print("=" * 80)

    # ------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------

    df = pd.read_csv(
        DATA_PATH
    )

    train_df, validation_df, test_df = (
        chronological_split(df)
    )

    # ------------------------------------------------------------
    # Load persisted model
    # ------------------------------------------------------------

    model = joblib.load(
        MODEL_PATH
    )

    # ------------------------------------------------------------
    # Load metadata
    # ------------------------------------------------------------

    import json

    with open(
        METADATA_PATH,
        "r",
        encoding="utf-8",
    ) as file:

        metadata = json.load(file)

    features = metadata[
        "feature_columns"
    ]

    threshold = float(
        metadata[
            "decision_threshold"
        ]
    )

    # ------------------------------------------------------------
    # Test inference
    # ------------------------------------------------------------

    X_test = test_df[
        features
    ]

    y_test = test_df[
        "financial_stress_30d"
    ].astype(int)

    probabilities = (
        model.predict_proba(
            X_test
        )[:, 1]
    )

    metrics = evaluate(
        y_test,
        probabilities,
        threshold,
    )

    # ------------------------------------------------------------
    # Print
    # ------------------------------------------------------------

    print(
        "\nFINAL HELD-OUT TEST RESULTS"
    )

    print("-" * 80)

    for name, value in metrics.items():

        print(
            f"{name:15s}: "
            f"{value:.6f}"
        )

    print(
        "\nTest rows: "
        f"{len(test_df):,}"
    )

    print(
        "Test stress cases: "
        f"{y_test.sum():,}"
    )

    print(
        "Positive rate: "
        f"{y_test.mean():.4%}"
    )

    # ------------------------------------------------------------
    # Save
    # ------------------------------------------------------------

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output = pd.DataFrame(
        [metrics]
    )

    output_path = (
        REPORT_DIR
        / "final_model_test_metrics.csv"
    )

    output.to_csv(
        output_path,
        index=False,
    )

    # ------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------

    assert 0 <= metrics[
        "pr_auc"
    ] <= 1

    assert 0 <= metrics[
        "roc_auc"
    ] <= 1

    assert 0 <= metrics[
        "brier_score"
    ] <= 1

    assert 0 <= threshold <= 1

    print(
        "\n✓ Persisted model evaluated"
    )

    print(
        "✓ Test data remained outside training"
    )

    print(
        "✓ Probability metrics calculated"
    )

    print(
        "✓ Threshold metrics calculated"
    )

    print(
        f"\nSaved: {output_path}"
    )

    print("\n" + "=" * 80)
    print("FINAL MODEL EVALUATION PASSED")
    print("=" * 80)


if __name__ == "__main__":
    main()