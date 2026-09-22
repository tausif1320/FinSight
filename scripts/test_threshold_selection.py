from __future__ import annotations

import numpy as np
import pandas as pd

from sklearn.metrics import (
    f1_score,
    fbeta_score,
    precision_score,
    recall_score,
)

from finsight.models.dataset import (
    get_feature_columns,
    chronological_split,
)

from finsight.models.baselines import (
    build_logistic_regression,
    prepare_xy,
)

from finsight.models.feature_sets import (
    get_static_features,
    get_dynamic_features,
)

from finsight.models.model_evaluation import (
    evaluate_binary_classifier,
)


# ============================================================
# CONFIGURATION
# ============================================================

DATASET_PATH = (
    "data/processed/modeling_dataset_100_users.csv"
)

TARGET = "financial_stress_30d"

DEFAULT_THRESHOLD = 0.50

THRESHOLDS = np.array([
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
    0.65,
    0.70,
    0.75,
    0.80,
])


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 80)
    print("FinSight Threshold Selection Experiment")
    print("100-User Dataset")
    print("=" * 80)

    # ========================================================
    # LOAD DATASET
    # ========================================================

    print()
    print("Loading modeling dataset...")

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
    # DATA VALIDATION
    # ========================================================

    duplicate_keys = model_df.duplicated(
        subset=["user_id", "month"]
    ).sum()

    assert duplicate_keys == 0, (
        f"Found {duplicate_keys} duplicate user-month rows."
    )

    assert TARGET in model_df.columns

    assert set(
        model_df[TARGET].unique()
    ).issubset({0, 1})

    print()
    print("Dataset validation passed.")

    # ========================================================
    # FEATURE SETS
    # ========================================================

    all_features = get_feature_columns(
        model_df
    )

    static_features = get_static_features(
        all_features
    )

    dynamic_features = get_dynamic_features(
        all_features
    )

    combined_features = list(
        dict.fromkeys(
            static_features + dynamic_features
        )
    )

    print()
    print("=" * 80)
    print("FEATURE SET")
    print("=" * 80)

    print(
        f"Total available predictors: "
        f"{len(all_features)}"
    )

    print(
        f"Static features: "
        f"{len(static_features)}"
    )

    print(
        f"Dynamic features: "
        f"{len(dynamic_features)}"
    )

    print(
        f"Combined features: "
        f"{len(combined_features)}"
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
        f"{len(train_df)} rows | "
        f"{int(train_df[TARGET].sum())} stress"
    )

    print(
        f"Validation: "
        f"{len(validation_df)} rows | "
        f"{int(validation_df[TARGET].sum())} stress"
    )

    print(
        f"Test: "
        f"{len(test_df)} rows | "
        f"{int(test_df[TARGET].sum())} stress"
    )

    print()
    print(
        f"Train period: "
        f"{train_df['month'].min().date()} "
        f"-> "
        f"{train_df['month'].max().date()}"
    )

    print(
        f"Validation period: "
        f"{validation_df['month'].min().date()} "
        f"-> "
        f"{validation_df['month'].max().date()}"
    )

    print(
        f"Test period: "
        f"{test_df['month'].min().date()} "
        f"-> "
        f"{test_df['month'].max().date()}"
    )

    # ========================================================
    # VALIDATION SAMPLE SIZE
    # ========================================================

    validation_positive_count = int(
        validation_df[TARGET].sum()
    )

    print()
    print("=" * 80)
    print("VALIDATION SAMPLE SIZE")
    print("=" * 80)

    print(
        f"Validation stress cases: "
        f"{validation_positive_count}"
    )

    if validation_positive_count < 10:

        print()

        print(
            "WARNING: Validation contains fewer than "
            "10 positive cases."
        )

        print(
            "Threshold selection is highly unstable "
            "on this dataset."
        )

        print(
            "The threshold analysis is therefore treated "
            "as an engineering diagnostic."
        )

    # ========================================================
    # PREPARE DATA
    # ========================================================

    X_train, y_train = prepare_xy(
        train_df,
        combined_features,
        TARGET,
    )

    X_validation, y_validation = prepare_xy(
        validation_df,
        combined_features,
        TARGET,
    )

    X_test, y_test = prepare_xy(
        test_df,
        combined_features,
        TARGET,
    )

    # ========================================================
    # TRAIN LOGISTIC REGRESSION
    # ========================================================

    print()
    print("=" * 80)
    print("TRAINING STATIC + DYNAMIC LOGISTIC REGRESSION")
    print("=" * 80)

    model = build_logistic_regression()

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

    # ========================================================
    # VALIDATION THRESHOLD ANALYSIS
    # ========================================================

    threshold_results = []

    for threshold in THRESHOLDS:

        validation_prediction = (
            validation_probability >= threshold
        ).astype(int)

        precision = precision_score(
            y_validation,
            validation_prediction,
            zero_division=0,
        )

        recall = recall_score(
            y_validation,
            validation_prediction,
            zero_division=0,
        )

        f1 = f1_score(
            y_validation,
            validation_prediction,
            zero_division=0,
        )

        f2 = fbeta_score(
            y_validation,
            validation_prediction,
            beta=2,
            zero_division=0,
        )

        predicted_positive = int(
            validation_prediction.sum()
        )

        threshold_results.append(
            {
                "threshold": threshold,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "f2": f2,
                "predicted_positive": predicted_positive,
            }
        )

    threshold_df = pd.DataFrame(
        threshold_results
    )

    # ========================================================
    # PRINT VALIDATION THRESHOLDS
    # ========================================================

    print()
    print("=" * 80)
    print("VALIDATION THRESHOLD ANALYSIS")
    print("=" * 80)

    print(
        threshold_df.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    # ========================================================
    # SELECT THRESHOLD
    # ========================================================

    best_row = threshold_df.loc[
        threshold_df["f2"].idxmax()
    ]

    selected_threshold = float(
        best_row["threshold"]
    )

    print()
    print("=" * 80)
    print("SELECTED THRESHOLD")
    print("=" * 80)

    print(
        f"Default threshold: "
        f"{DEFAULT_THRESHOLD:.2f}"
    )

    print(
        f"Validation-selected threshold: "
        f"{selected_threshold:.2f}"
    )

    print(
        f"Validation F2: "
        f"{best_row['f2']:.4f}"
    )

    # ========================================================
    # TEST AT DEFAULT THRESHOLD
    # ========================================================

    default_test_metrics = (
        evaluate_binary_classifier(
            y_true=y_test,
            y_probability=test_probability,
            threshold=DEFAULT_THRESHOLD,
        )
    )

    # ========================================================
    # TEST AT SELECTED THRESHOLD
    # ========================================================

    selected_test_metrics = (
        evaluate_binary_classifier(
            y_true=y_test,
            y_probability=test_probability,
            threshold=selected_threshold,
        )
    )

    # ========================================================
    # TEST COMPARISON
    # ========================================================

    comparison = pd.DataFrame(
        [
            {
                "threshold_type": "Default 0.50",
                **default_test_metrics,
            },
            {
                "threshold_type": "Validation Selected",
                **selected_test_metrics,
            },
        ]
    )

    print()
    print("=" * 80)
    print("TEST SET THRESHOLD COMPARISON")
    print("=" * 80)

    print(
        comparison[
            [
                "threshold_type",
                "threshold",
                "precision",
                "recall",
                "f1",
                "pr_auc",
                "roc_auc",
                "brier_score",
            ]
        ].to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    # ========================================================
    # SANITY CHECKS
    # ========================================================

    assert len(threshold_df) == len(
        THRESHOLDS
    )

    assert (
        threshold_df["threshold"]
        .between(0, 1)
        .all()
    )

    assert (
        comparison["threshold"]
        .between(0, 1)
        .all()
    )

    assert np.all(
        np.isfinite(validation_probability)
    )

    assert np.all(
        np.isfinite(test_probability)
    )

    print()
    print("=" * 80)
    print("THRESHOLD SELECTION VALIDATION PASSED")
    print("=" * 80)


if __name__ == "__main__":
    main()