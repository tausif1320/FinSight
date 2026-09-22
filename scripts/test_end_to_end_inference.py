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


from finsight.inference.pipeline import (
    FinSightInferencePipeline,
)


DATA_PATH = (
    ROOT
    / "data"
    / "processed"
    / "modeling_dataset_1000_users.csv"
)

STRESS_MODEL_PATH = (
    ROOT
    / "models"
    / "stress_model.joblib"
)

STRESS_METADATA_PATH = (
    ROOT
    / "models"
    / "feature_metadata.json"
)

CATE_MODEL_PATH = (
    ROOT
    / "models"
    / "cate_models.joblib"
)

OUTPUT_DIR = (
    ROOT
    / "reports"
    / "inference"
)


def main():

    print("=" * 80)
    print("FinSight End-to-End Inference")
    print("=" * 80)

    # ========================================================================
    # 1. LOAD PIPELINE
    # ========================================================================

    print(
        "\nLoading production artifacts..."
    )

    pipeline = FinSightInferencePipeline(
        stress_model_path=STRESS_MODEL_PATH,
        stress_metadata_path=STRESS_METADATA_PATH,
        cate_model_path=CATE_MODEL_PATH,
    )

    print(
        "✓ Stress model loaded"
    )

    print(
        "✓ Stress metadata loaded"
    )

    print(
        "✓ CATE models loaded"
    )

    # ========================================================================
    # 2. LOAD DATA
    # ========================================================================

    df = pd.read_csv(
        DATA_PATH
    )

    # Use the chronological test period.
    #
    # This is inference only. No fitting happens here.

    test_months = sorted(
        df[
            "month"
        ].unique()
    )[-2:]

    inference_df = (
        df[
            df[
                "month"
            ].isin(
                test_months
            )
        ]
        .copy()
        .reset_index(drop=True)
    )

    print(
        f"\nInference observations: "
        f"{len(inference_df):,}"
    )

    print(
        f"Months: "
        f"{test_months}"
    )

    # ========================================================================
    # 3. RUN END-TO-END INFERENCE
    # ========================================================================

    print(
        "\nRunning end-to-end inference..."
    )

    results = pipeline.predict(
        inference_df
    )

    print(
        f"Predictions generated: "
        f"{len(results):,}"
    )

    # ========================================================================
    # 4. SUMMARY
    # ========================================================================

    print(
        "\n" + "=" * 80
    )

    print(
        "RISK LEVELS"
    )

    print(
        results[
            "risk_level"
        ]
        .value_counts()
        .to_string()
    )

    print(
        "\nFINANCIAL PRESSURE"
    )

    print(
        results[
            "financial_pressure"
        ]
        .value_counts()
        .to_string()
    )

    print(
        "\nINTERVENTIONS"
    )

    print(
        results[
            "intervention"
        ]
        .value_counts()
        .to_string()
    )

    print(
        "\nRECOMMENDATION STATUS"
    )

    print(
        results[
            "recommendation_status"
        ]
        .value_counts()
        .to_string()
    )

    # ========================================================================
    # 5. CATE COVERAGE
    # ========================================================================

    cate_available = (
        results[
            "cate"
        ].notna()
    )

    print(
        "\nCATE coverage:"
    )

    print(
        f"  With CATE: "
        f"{cate_available.sum():,}"
    )

    print(
        f"  Without CATE: "
        f"{(~cate_available).sum():,}"
    )

    # ========================================================================
    # 6. SAMPLE OUTPUT
    # ========================================================================

    print(
        "\n" + "=" * 80
    )

    print(
        "SAMPLE END-TO-END RESULTS"
    )

    print(
        "=" * 80
    )

    display_columns = [
        "user_id",
        "month",
        "risk_probability",
        "risk_level",
        "financial_pressure",
        "intervention",
        "cate",
        "recommendation_status",
    ]

    print(
        results[
            display_columns
        ]
        .head(20)
        .to_string(
            index=False
        )
    )

    # ========================================================================
    # 7. VALIDATION
    # ========================================================================

    print(
        "\n" + "=" * 80
    )

    print(
        "PIPELINE VALIDATION"
    )

    print(
        "=" * 80
    )

    assert (
        len(results)
        == len(inference_df)
    )

    assert (
        results[
            "risk_probability"
        ]
        .between(
            0,
            1,
        )
        .all()
    )

    assert (
        results[
            "risk_level"
        ]
        .isin(
            [
                "LOW",
                "MODERATE",
                "HIGH",
            ]
        )
        .all()
    )

    assert (
        results[
            "financial_pressure"
        ]
        .isin(
            [
                "STABLE",
                "LIQUIDITY_PRESSURE",
                "SPENDING_PRESSURE",
                "OBLIGATION_PRESSURE",
            ]
        )
        .all()
    )

    # CATE should exist exactly for users with
    # an active intervention pressure category.

    active_pressure = (
        results[
            "financial_pressure"
        ].isin(
            [
                "LIQUIDITY_PRESSURE",
                "SPENDING_PRESSURE",
                "OBLIGATION_PRESSURE",
            ]
        )
    )

    assert (
        results.loc[
            active_pressure,
            "cate",
        ]
        .notna()
        .all()
    )

    assert (
        results.loc[
            ~active_pressure,
            "cate",
        ]
        .isna()
        .all()
    )

    # Intervention must correspond to pressure.

    expected_interventions = {
        "LIQUIDITY_PRESSURE":
            "LIQUIDITY_PROTECTION",

        "SPENDING_PRESSURE":
            "SPENDING_CONTROL",

        "OBLIGATION_PRESSURE":
            "OBLIGATION_MANAGEMENT",
    }

    for pressure, intervention in (
        expected_interventions.items()
    ):

        mask = (
            results[
                "financial_pressure"
            ]
            == pressure
        )

        if mask.any():

            assert (
                results.loc[
                    mask,
                    "intervention",
                ]
                == intervention
            ).all()

    print(
        "✓ Correct number of predictions"
    )

    print(
        "✓ Risk probabilities valid"
    )

    print(
        "✓ Risk levels valid"
    )

    print(
        "✓ Financial pressure valid"
    )

    print(
        "✓ CATE generated for active intervention cohorts"
    )

    print(
        "✓ Stable users have no CATE intervention estimate"
    )

    print(
        "✓ Pressure/intervention mapping is consistent"
    )

    # ========================================================================
    # 8. SAVE
    # ========================================================================

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        OUTPUT_DIR
        / "end_to_end_inference_results.csv"
    )

    results.to_csv(
        output_path,
        index=False,
    )

    print(
        f"\nSaved: "
        f"{output_path}"
    )

    print(
        "\n" + "=" * 80
    )

    print(
        "END-TO-END INFERENCE PASSED"
    )

    print(
        "=" * 80
    )


if __name__ == "__main__":
    main()