from __future__ import annotations

import gc
from pathlib import Path

import numpy as np
import pandas as pd

from finsight.models.dataset import (
    get_feature_columns,
    chronological_split,
)

from finsight.models.baselines import (
    prepare_xy,
)

from finsight.models.tree_models import (
    build_hist_gradient_boosting,
)

from finsight.explainability.shap_explainer import (
    build_shap_explainer,
    calculate_shap_values,
    build_feature_importance_table,
    build_local_explanation,
)


DATASET_PATH = (
    "data/processed/modeling_dataset_1000_users.csv"
)

OUTPUT_DIR = Path(
    "reports/shap"
)

TARGET = "financial_stress_30d"

RANDOM_STATE = 42

BACKGROUND_SIZE = 200

EXPLANATION_SIZE = 100

TOP_FEATURES = 20

TOP_LOCAL_FEATURES = 10


def main():

    print("=" * 80)
    print("FinSight SHAP Explainability Experiment")
    print("=" * 80)

    # ========================================================
    # LOAD DATA
    # ========================================================

    print()
    print("Loading dataset...")

    df = pd.read_csv(
        DATASET_PATH,
        parse_dates=["month"],
    )

    print(
        f"Rows: {len(df):,}"
    )

    print(
        f"Users: "
        f"{df['user_id'].nunique():,}"
    )

    print(
        f"Stress cases: "
        f"{int(df[TARGET].sum()):,}"
    )

    # ========================================================
    # VALIDATION
    # ========================================================

    assert (
        df["user_id"].nunique() == 1000
    )

    assert (
        df.duplicated(
            subset=["user_id", "month"]
        ).sum()
        == 0
    )

    assert TARGET in df.columns

    # ========================================================
    # FEATURES
    # ========================================================

    feature_columns = get_feature_columns(
        df
    )

    print()
    print(
        f"Predictor columns: "
        f"{len(feature_columns)}"
    )

    # ========================================================
    # CHRONOLOGICAL SPLIT
    # ========================================================

    train_df, validation_df, test_df = (
        chronological_split(df)
    )

    print()
    print("=" * 80)
    print("CHRONOLOGICAL SPLIT")
    print("=" * 80)

    print(
        f"Train: "
        f"{len(train_df):,}"
    )

    print(
        f"Validation: "
        f"{len(validation_df):,}"
    )

    print(
        f"Test: "
        f"{len(test_df):,}"
    )

    # ========================================================
    # PREPARE FEATURES
    # ========================================================

    X_train, y_train = prepare_xy(
        train_df,
        feature_columns,
        TARGET,
    )

    X_test, y_test = prepare_xy(
        test_df,
        feature_columns,
        TARGET,
    )

    # ========================================================
    # FIT MODEL
    # ========================================================

    print()
    print("=" * 80)
    print("TRAINING HISTGRADIENTBOOSTING")
    print("=" * 80)

    pipeline = (
        build_hist_gradient_boosting()
    )

    pipeline.fit(
        X_train,
        y_train,
    )

    print(
        "Model training completed."
    )

    # ========================================================
    # EXTRACT PREPROCESSOR + MODEL
    # ========================================================

    imputer = pipeline.named_steps[
        "imputer"
    ]

    model = pipeline.named_steps[
        "model"
    ]

    X_train_imputed = pd.DataFrame(
        imputer.transform(X_train),
        columns=feature_columns,
        index=X_train.index,
    )

    X_test_imputed = pd.DataFrame(
        imputer.transform(X_test),
        columns=feature_columns,
        index=X_test.index,
    )

    # ========================================================
    # BACKGROUND DATA
    # ========================================================

    rng = np.random.default_rng(
        RANDOM_STATE
    )

    background_size = min(
        BACKGROUND_SIZE,
        len(X_train_imputed),
    )

    background_indices = rng.choice(
        len(X_train_imputed),
        size=background_size,
        replace=False,
    )

    background_data = (
        X_train_imputed.iloc[
            background_indices
        ].copy()
    )

    print()
    print(
        f"SHAP background rows: "
        f"{len(background_data)}"
    )

    # ========================================================
    # BUILD EXPLAINER
    # ========================================================

    print()
    print(
        "Building SHAP explainer..."
    )

    explainer = build_shap_explainer(
        model=model,
        background_data=background_data,
    )

    print(
        "SHAP explainer created."
    )

    # ========================================================
    # SELECT TEST OBSERVATIONS
    # ========================================================

    explanation_size = min(
        EXPLANATION_SIZE,
        len(X_test_imputed),
    )

    explanation_indices = rng.choice(
        len(X_test_imputed),
        size=explanation_size,
        replace=False,
    )

    X_explain = (
        X_test_imputed.iloc[
            explanation_indices
        ].copy()
    )

    explain_metadata = (
        test_df.iloc[
            explanation_indices
        ][
            [
                "user_id",
                "month",
                TARGET,
            ]
        ]
        .reset_index(drop=True)
    )

    print()
    print(
        f"Explaining "
        f"{len(X_explain)} test observations."
    )

    # ========================================================
    # CALCULATE SHAP VALUES
    # ========================================================

    print()
    print(
        "Calculating SHAP values..."
    )

    shap_values = calculate_shap_values(
        explainer,
        X_explain,
    )

    print(
        f"SHAP matrix shape: "
        f"{shap_values.shape}"
    )

    # ========================================================
    # GLOBAL IMPORTANCE
    # ========================================================

    importance = (
        build_feature_importance_table(
            feature_columns,
            shap_values,
        )
    )

    print()
    print("=" * 80)
    print("GLOBAL SHAP FEATURE IMPORTANCE")
    print("=" * 80)

    print(
        importance.head(
            TOP_FEATURES
        ).to_string(
            index=False,
            float_format=lambda x: f"{x:.6f}",
        )
    )

    # ========================================================
    # SAVE GLOBAL IMPORTANCE
    # ========================================================

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    importance_path = (
        OUTPUT_DIR
        / "global_feature_importance.csv"
    )

    importance.to_csv(
        importance_path,
        index=False,
    )

    # ========================================================
    # LOCAL EXPLANATIONS
    # ========================================================

    local_rows = []

    print()
    print("=" * 80)
    print("GENERATING LOCAL EXPLANATIONS")
    print("=" * 80)

    for row_number in range(
        len(X_explain)
    ):

        row_features = (
            X_explain.iloc[
                row_number
            ]
        )

        row_shap = (
            shap_values[
                row_number
            ]
        )

        local = build_local_explanation(
            feature_names=feature_columns,
            feature_values=row_features.values,
            shap_values=row_shap,
            top_n=TOP_LOCAL_FEATURES,
        )

        metadata = (
            explain_metadata.iloc[
                row_number
            ]
        )

        for _, item in local.iterrows():

            local_rows.append(
                {
                    "user_id":
                        metadata["user_id"],

                    "month":
                        metadata["month"],

                    TARGET:
                        metadata[TARGET],

                    "feature":
                        item["feature"],

                    "feature_value":
                        item["feature_value"],

                    "shap_value":
                        item["shap_value"],

                    "direction":
                        item["direction"],

                    "absolute_shap":
                        item["absolute_shap"],
                }
            )

    local_df = pd.DataFrame(
        local_rows
    )

    local_path = (
        OUTPUT_DIR
        / "local_explanations.csv"
    )

    local_df.to_csv(
        local_path,
        index=False,
    )

    # ========================================================
    # TOP POSITIVE DRIVERS
    # ========================================================

    positive_drivers = (
        importance.copy()
    )

    positive_drivers[
        "mean_signed_shap"
    ] = (
        shap_values.mean(axis=0)
    )

    positive_drivers = (
        positive_drivers
        .sort_values(
            "mean_signed_shap",
            ascending=False,
        )
        .reset_index(drop=True)
    )

    positive_path = (
        OUTPUT_DIR
        / "positive_risk_drivers.csv"
    )

    positive_drivers.to_csv(
        positive_path,
        index=False,
    )

    # ========================================================
    # NEGATIVE / PROTECTIVE DRIVERS
    # ========================================================

    protective_drivers = (
        positive_drivers
        .sort_values(
            "mean_signed_shap",
            ascending=True,
        )
        .reset_index(drop=True)
    )

    protective_path = (
        OUTPUT_DIR
        / "protective_drivers.csv"
    )

    protective_drivers.to_csv(
        protective_path,
        index=False,
    )

    # ========================================================
    # DISPLAY TOP DRIVERS
    # ========================================================

    print()
    print(
        "TOP FEATURES PUSHING "
        "TOWARD FINANCIAL STRESS"
    )

    print(
        positive_drivers[
            [
                "feature",
                "mean_signed_shap",
                "mean_abs_shap",
            ]
        ]
        .head(10)
        .to_string(
            index=False,
            float_format=lambda x: f"{x:.6f}",
        )
    )

    print()
    print(
        "TOP PROTECTIVE FEATURES"
    )

    print(
        protective_drivers[
            [
                "feature",
                "mean_signed_shap",
                "mean_abs_shap",
            ]
        ]
        .head(10)
        .to_string(
            index=False,
            float_format=lambda x: f"{x:.6f}",
        )
    )

    # ========================================================
    # VALIDATION
    # ========================================================

    assert (
        shap_values.shape
        == X_explain.shape
    )

    assert (
        len(importance)
        == len(feature_columns)
    )

    assert (
        len(local_df)
        == explanation_size
        * TOP_LOCAL_FEATURES
    )

    assert (
        importance["feature"]
        .nunique()
        == len(feature_columns)
    )

    assert np.isfinite(
        shap_values
    ).all()

    print()
    print("=" * 80)
    print("SHAP VALIDATION PASSED")
    print("=" * 80)

    print()
    print("Saved:")
    print(
        f"  {importance_path}"
    )
    print(
        f"  {local_path}"
    )
    print(
        f"  {positive_path}"
    )
    print(
        f"  {protective_path}"
    )

    # Release memory
    del X_train_imputed
    del X_test_imputed
    del background_data
    del X_explain
    del shap_values

    gc.collect()


if __name__ == "__main__":
    main()