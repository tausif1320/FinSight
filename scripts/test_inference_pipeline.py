from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd


ROOT = (
    Path(__file__)
    .resolve()
    .parents[1]
)

sys.path.insert(
    0,
    str(ROOT / "src"),
)


from finsight.inference.predictor import (
    FinSightPredictor,
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
    / "inference"
)


def main():

    print("=" * 80)
    print("FinSight Production Inference Pipeline")
    print("=" * 80)

    # ================================================================
    # 1. LOAD MODEL
    # ================================================================

    print(
        "\nLoading persisted model..."
    )

    predictor = FinSightPredictor(
        model_path=MODEL_PATH,
        metadata_path=METADATA_PATH,
    )

    print(
        f"Model: "
        f"{predictor.model_name}"
    )

    print(
        f"Features: "
        f"{len(predictor.feature_columns)}"
    )

    # ================================================================
    # 2. LOAD DATA
    # ================================================================

    df = pd.read_csv(
        DATA_PATH
    )

    # Use the final test period ONLY for
    # inference demonstration.
    #
    # It is NOT used for training.

    test_months = sorted(
        df["month"].unique()
    )[-2:]

    inference_df = df[
        df["month"].isin(
            test_months
        )
    ].copy()

    print(
        "\nInference rows: "
        f"{len(inference_df):,}"
    )

    print(
        f"Test months: "
        f"{test_months}"
    )

    # ================================================================
    # 3. RUN INFERENCE
    # ================================================================

    print(
        "\nRunning model inference..."
    )

    predictions = predictor.predict(
        inference_df
    )

    print(
        f"Predictions: "
        f"{len(predictions):,}"
    )

    # ================================================================
    # 4. SUMMARY
    # ================================================================

    print("\n" + "=" * 80)
    print("INFERENCE SUMMARY")
    print("=" * 80)

    print(
        "\nRisk levels:"
    )

    print(
        predictions[
            "risk_level"
        ]
        .value_counts()
        .to_string()
    )

    print(
        "\nFinancial pressure:"
    )

    print(
        predictions[
            "financial_pressure"
        ]
        .value_counts()
        .to_string()
    )

    print(
        "\nInterventions:"
    )

    print(
        predictions[
            "intervention"
        ]
        .value_counts()
        .to_string()
    )

    print(
        "\nRisk probability statistics:"
    )

    print(
        predictions[
            "risk_probability"
        ]
        .describe()
        .to_string()
    )

    # ================================================================
    # 5. SAMPLE
    # ================================================================

    print("\n" + "=" * 80)
    print("SAMPLE INFERENCE RESULTS")
    print("=" * 80)

    columns = [
        "user_id",
        "month",
        "risk_probability",
        "risk_level",
        "financial_pressure",
        "intervention",
        "recommendation_status",
    ]

    print(
        predictions[
            columns
        ]
        .head(15)
        .to_string(
            index=False
        )
    )

    # ================================================================
    # 6. SAVE
    # ================================================================

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        REPORT_DIR
        / "production_inference_results.csv"
    )

    predictions.to_csv(
        output_path,
        index=False,
    )

    # ================================================================
    # 7. VALIDATION
    # ================================================================

    print("\n" + "=" * 80)
    print("INFERENCE VALIDATION")
    print("=" * 80)

    assert len(
        predictions
    ) == len(
        inference_df
    )

    assert predictions[
        "risk_probability"
    ].between(
        0,
        1,
    ).all()

    assert predictions[
        "risk_level"
    ].isin(
        [
            "LOW",
            "MODERATE",
            "HIGH",
        ]
    ).all()

    assert predictions[
        "financial_pressure"
    ].notna().all()

    assert predictions[
        "intervention"
    ].notna().all()

    assert predictions[
        "explanation"
    ].notna().all()

    print(
        "✓ Persisted model loaded"
    )

    print(
        "✓ Feature metadata loaded"
    )

    print(
        "✓ Test-period inference completed"
    )

    print(
        "✓ Probabilities are valid"
    )

    print(
        "✓ Risk levels generated"
    )

    print(
        "✓ Financial pressure generated"
    )

    print(
        "✓ Intervention recommendations generated"
    )

    print(
        "✓ Explanations generated"
    )

    print(
        "\nSaved:"
    )

    print(
        f"  {output_path}"
    )

    print("\n" + "=" * 80)
    print("PRODUCTION INFERENCE PIPELINE PASSED")
    print("=" * 80)


if __name__ == "__main__":
    main()