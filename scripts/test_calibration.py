from __future__ import annotations

import numpy as np
import pandas as pd

from finsight.models.dataset import (
    get_feature_columns,
    chronological_split,
)

from finsight.models.baselines import (
    build_logistic_regression,
    prepare_xy,
)

from finsight.models.tree_models import (
    build_random_forest,
    build_hist_gradient_boosting,
)

from finsight.models.calibration import (
    calculate_calibration_metrics,
    print_calibration_report,
)


DATASET_PATH = (
    "data/processed/modeling_dataset_1000_users.csv"
)

TARGET = "financial_stress_30d"


def main():

    print("=" * 80)
    print("FinSight Probability Calibration Experiment")
    print("1,000-User Dataset")
    print("=" * 80)

    # ========================================================
    # LOAD DATASET
    # ========================================================

    print()
    print("Loading saved modeling dataset...")

    model_df = pd.read_csv(
        DATASET_PATH,
        parse_dates=["month"],
    )

    print(
        f"Rows: {len(model_df):,}"
    )

    print(
        f"Unique users: "
        f"{model_df['user_id'].nunique():,}"
    )

    print(
        f"Stress cases: "
        f"{int(model_df[TARGET].sum()):,}"
    )

    print(
        f"Stress rate: "
        f"{model_df[TARGET].mean():.4f}"
    )

    # ========================================================
    # VALIDATION
    # ========================================================

    assert TARGET in model_df.columns

    assert (
        model_df["user_id"].nunique() == 1000
    )

    duplicate_keys = model_df.duplicated(
        subset=["user_id", "month"]
    ).sum()

    assert duplicate_keys == 0

    print()
    print("Dataset validation passed.")

    # ========================================================
    # FEATURES
    # ========================================================

    feature_columns = get_feature_columns(
        model_df
    )

    print()
    print("=" * 80)
    print("FEATURES")
    print("=" * 80)

    print(
        f"Predictor columns: "
        f"{len(feature_columns)}"
    )

    # ========================================================
    # CHRONOLOGICAL SPLIT
    # ========================================================

    train_df, validation_df, test_df = (
        chronological_split(model_df)
    )

    print()
    print("=" * 80)
    print("CHRONOLOGICAL SPLIT")
    print("=" * 80)

    print(
        f"Train: "
        f"{len(train_df):,} rows | "
        f"{int(train_df[TARGET].sum()):,} stress"
    )

    print(
        f"Validation: "
        f"{len(validation_df):,} rows | "
        f"{int(validation_df[TARGET].sum()):,} stress"
    )

    print(
        f"Test: "
        f"{len(test_df):,} rows | "
        f"{int(test_df[TARGET].sum()):,} stress"
    )

    # ========================================================
    # PREPARE DATA
    # ========================================================

    X_train, y_train = prepare_xy(
        train_df,
        feature_columns,
        TARGET,
    )

    X_validation, y_validation = prepare_xy(
        validation_df,
        feature_columns,
        TARGET,
    )

    X_test, y_test = prepare_xy(
        test_df,
        feature_columns,
        TARGET,
    )

    # ========================================================
    # MODELS
    # ========================================================

    models = {
        "Logistic Regression":
            build_logistic_regression(),

        "Random Forest":
            build_random_forest(),

        "HistGradientBoosting":
            build_hist_gradient_boosting(),
    }

    validation_results = []
    test_results = []

    # ========================================================
    # TRAIN AND CALIBRATE
    # ========================================================

    for model_name, model in models.items():

        print()
        print("=" * 80)
        print(f"TRAINING: {model_name}")
        print("=" * 80)

        model.fit(
            X_train,
            y_train,
        )

        validation_probability = (
            model.predict_proba(
                X_validation
            )[:, 1]
        )

        test_probability = (
            model.predict_proba(
                X_test
            )[:, 1]
        )

        # ----------------------------------------------------
        # VALIDATION CALIBRATION
        # ----------------------------------------------------

        validation_metrics = (
            calculate_calibration_metrics(
                y_true=y_validation,
                probabilities=validation_probability,
                n_bins=10,
            )
        )

        print_calibration_report(
            model_name
            + " - VALIDATION",
            validation_metrics,
        )

        validation_results.append(
            {
                "model": model_name,
                "brier_score":
                    validation_metrics[
                        "brier_score"
                    ],
                "mean_absolute_calibration_error":
                    validation_metrics[
                        "mean_absolute_calibration_error"
                    ],
                "max_calibration_error":
                    validation_metrics[
                        "max_calibration_error"
                    ],
            }
        )

        # ----------------------------------------------------
        # TEST CALIBRATION
        # ----------------------------------------------------

        test_metrics = (
            calculate_calibration_metrics(
                y_true=y_test,
                probabilities=test_probability,
                n_bins=10,
            )
        )

        print_calibration_report(
            model_name
            + " - TEST",
            test_metrics,
        )

        test_results.append(
            {
                "model": model_name,
                "brier_score":
                    test_metrics[
                        "brier_score"
                    ],
                "mean_absolute_calibration_error":
                    test_metrics[
                        "mean_absolute_calibration_error"
                    ],
                "max_calibration_error":
                    test_metrics[
                        "max_calibration_error"
                    ],
            }
        )

    # ========================================================
    # VALIDATION SUMMARY
    # ========================================================

    validation_summary = pd.DataFrame(
        validation_results
    )

    print()
    print("=" * 80)
    print("VALIDATION CALIBRATION COMPARISON")
    print("=" * 80)

    print(
        validation_summary.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    # ========================================================
    # TEST SUMMARY
    # ========================================================

    test_summary = pd.DataFrame(
        test_results
    )

    print()
    print("=" * 80)
    print("TEST CALIBRATION COMPARISON")
    print("=" * 80)

    print(
        test_summary.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    # ========================================================
    # CHECKS
    # ========================================================

    assert len(validation_summary) == 3
    assert len(test_summary) == 3

    numeric_columns = [
        "brier_score",
        "mean_absolute_calibration_error",
        "max_calibration_error",
    ]

    for column in numeric_columns:

        assert (
            validation_summary[column]
            .notna()
            .all()
        )

        assert (
            test_summary[column]
            .notna()
            .all()
        )

        assert np.isfinite(
            validation_summary[column]
        ).all()

        assert np.isfinite(
            test_summary[column]
        ).all()

    print()
    print("=" * 80)
    print("CALIBRATION VALIDATION PASSED")
    print("=" * 80)


if __name__ == "__main__":
    main()