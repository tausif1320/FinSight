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


from finsight.inference.cate_predictor import (
    CATEPredictor,
)


DATA_PATH = (
    ROOT
    / "data"
    / "processed"
    / "modeling_dataset_1000_users.csv"
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
    print("FinSight Production CATE Inference")
    print("=" * 80)

    # ------------------------------------------------------------
    # Load model
    # ------------------------------------------------------------

    predictor = CATEPredictor(
        CATE_MODEL_PATH
    )

    print(
        "\nLoaded intervention models:"
    )

    for intervention in predictor.models:

        print(
            f"  ✓ {intervention}"
        )

    # ------------------------------------------------------------
    # Load modeling data
    # ------------------------------------------------------------

    df = pd.read_csv(
        DATA_PATH
    )

    # Use a small sample only for
    # inference validation.

    sample = (
        df
        .sort_values(
            ["user_id", "month"]
        )
        .head(20)
        .copy()
    )

    # ------------------------------------------------------------
    # Generate all policy-specific CATEs
    # ------------------------------------------------------------

    result = (
        predictor.predict_for_interventions(
            sample
        )
    )

    cate_columns = [
        column
        for column in result.columns
        if column.startswith("cate_")
    ]

    print(
        "\nCATE columns:"
    )

    for column in cate_columns:

        print(
            f"  {column}"
        )

    print(
        "\nSample CATE estimates:"
    )

    display_columns = [
        "user_id",
        "month",
    ] + cate_columns

    print(
        result[
            display_columns
        ]
        .to_string(
            index=False
        )
    )

    # ------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------

    assert len(
        result
    ) == len(
        sample
    )

    assert len(
        cate_columns
    ) == len(
        predictor.models
    )

    for column in cate_columns:

        assert result[
            column
        ].notna().all()

    # ------------------------------------------------------------
    # Save
    # ------------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        OUTPUT_DIR
        / "production_cate_inference.csv"
    )

    result[
        display_columns
    ].to_csv(
        output_path,
        index=False,
    )

    print(
        "\n✓ CATE models loaded"
    )

    print(
        "✓ Treatment effects generated"
    )

    print(
        "✓ All intervention models returned estimates"
    )

    print(
        "✓ No missing CATE estimates"
    )

    print(
        f"\nSaved: {output_path}"
    )

    print("\n" + "=" * 80)
    print("PRODUCTION CATE INFERENCE PASSED")
    print("=" * 80)


if __name__ == "__main__":
    main()