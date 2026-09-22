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

from finsight.models.tree_models import (
    build_random_forest,
    build_hist_gradient_boosting,
)

from finsight.models.model_evaluation import (
    evaluate_binary_classifier,
)


# ============================================================
# CONFIGURATION
# ============================================================

DATASET_PATH = (
    "data/processed/modeling_dataset_1000_users.csv"
)

TARGET = "financial_stress_30d"

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
    0.85,
    0.90,
])


# ============================================================
# MODEL BUILDERS
# ============================================================

def build_models():

    return {
        "Logistic Regression":
            build_logistic_regression(),

        "Random Forest":
            build_random_forest(),

        "HistGradientBoosting":
            build_hist_gradient_boosting(),
    }


# ============================================================
# THRESHOLD SEARCH
# ============================================================

def find_best_threshold(
    y_true,
    probabilities,
):
    """
    Select threshold using validation F2.

    F2 gives more weight to recall than precision,
    which is appropriate when missing a financially
    stressed user is more costly than generating an
    additional intervention candidate.
    """

    rows = []

    for threshold in THRESHOLDS:

        predictions = (
            probabilities >= threshold
        ).astype(int)

        precision = precision_score(
            y_true,
            predictions,
            zero_division=0,
        )

        recall = recall_score(
            y_true,
            predictions,
            zero_division=0,
        )

        f1 = f1_score(
            y_true,
            predictions,
            zero_division=0,
        )

        f2 = fbeta_score(
            y_true,
            predictions,
            beta=2,
            zero_division=0,
        )

        predicted_positive = int(
            predictions.sum()
        )

        rows.append(
            {
                "threshold": threshold,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "f2": f2,
                "predicted_positive": predicted_positive,
            }
        )

    threshold_df = pd.DataFrame(rows)

    # Highest F2.
    # If tied, choose the higher threshold because
    # it produces fewer positive predictions.
    best_f2 = threshold_df["f2"].max()

    candidates = threshold_df[
        threshold_df["f2"] == best_f2
    ]

    best_row = candidates.sort_values(
        "threshold",
        ascending=False,
    ).iloc[0]

    return (
        float(best_row["threshold"]),
        threshold_df,
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 80)
    print("FinSight Model-Specific Threshold Experiment")
    print("1,000-User Dataset")
    print("=" * 80)

    # ========================================================
    # LOAD DATA
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

    assert set(
        model_df[TARGET].unique()
    ).issubset({0, 1})

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
    # PREPARE MATRICES
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
    # BUILD MODELS
    # ========================================================

    models = build_models()

    validation_results = []
    test_results = []

    threshold_tables = {}

    # ========================================================
    # TRAIN EACH MODEL
    # ========================================================

    for model_name, model in models.items():

        print()
        print("=" * 80)
        print(f"TRAINING: {model_name}")
        print("=" * 80)

        # ----------------------------------------------------
        # TRAIN
        # ----------------------------------------------------

        model.fit(
            X_train,
            y_train,
        )

        # ----------------------------------------------------
        # VALIDATION PROBABILITIES
        # ----------------------------------------------------

        validation_probability = (
            model.predict_proba(
                X_validation
            )[:, 1]
        )

        # ----------------------------------------------------
        # SELECT THRESHOLD
        # ----------------------------------------------------

        selected_threshold, threshold_df = (
            find_best_threshold(
                y_true=y_validation,
                probabilities=validation_probability,
            )
        )

        threshold_tables[
            model_name
        ] = threshold_df

        selected_row = threshold_df[
            threshold_df["threshold"]
            == selected_threshold
        ].iloc[0]

        print()
        print(
            f"Selected threshold: "
            f"{selected_threshold:.2f}"
        )

        print(
            f"Validation precision: "
            f"{selected_row['precision']:.4f}"
        )

        print(
            f"Validation recall: "
            f"{selected_row['recall']:.4f}"
        )

        print(
            f"Validation F1: "
            f"{selected_row['f1']:.4f}"
        )

        print(
            f"Validation F2: "
            f"{selected_row['f2']:.4f}"
        )

        # ----------------------------------------------------
        # TEST PROBABILITIES
        # ----------------------------------------------------

        test_probability = (
            model.predict_proba(
                X_test
            )[:, 1]
        )

        # ----------------------------------------------------
        # TEST AT SELECTED THRESHOLD
        # ----------------------------------------------------

        selected_metrics = (
            evaluate_binary_classifier(
                y_true=y_test,
                y_probability=test_probability,
                threshold=selected_threshold,
            )
        )

        selected_metrics["model"] = model_name
        selected_metrics["threshold_source"] = (
            "validation"
        )

        test_results.append(
            selected_metrics
        )

        # ----------------------------------------------------
        # TEST AT 0.50
        # ----------------------------------------------------

        default_metrics = (
            evaluate_binary_classifier(
                y_true=y_test,
                y_probability=test_probability,
                threshold=0.50,
            )
        )

        default_metrics["model"] = model_name
        default_metrics["threshold_source"] = (
            "default_0.50"
        )

        validation_results.append(
            default_metrics
        )

    # ========================================================
    # DEFAULT THRESHOLD RESULTS
    # ========================================================

    default_df = pd.DataFrame(
        validation_results
    )

    print()
    print("=" * 80)
    print("TEST RESULTS AT DEFAULT THRESHOLD = 0.50")
    print("=" * 80)

    print(
        default_df[
            [
                "model",
                "threshold",
                "pr_auc",
                "roc_auc",
                "precision",
                "recall",
                "f1",
                "brier_score",
            ]
        ].to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    # ========================================================
    # VALIDATION-SELECTED RESULTS
    # ========================================================

    selected_df = pd.DataFrame(
        test_results
    )

    print()
    print("=" * 80)
    print("TEST RESULTS AT VALIDATION-SELECTED THRESHOLDS")
    print("=" * 80)

    print(
        selected_df[
            [
                "model",
                "threshold",
                "pr_auc",
                "roc_auc",
                "precision",
                "recall",
                "f1",
                "brier_score",
            ]
        ].to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    # ========================================================
    # THRESHOLD SUMMARY
    # ========================================================

    threshold_summary = selected_df[
        [
            "model",
            "threshold",
        ]
    ].copy()

    print()
    print("=" * 80)
    print("MODEL-SPECIFIC THRESHOLDS")
    print("=" * 80)

    print(
        threshold_summary.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    # ========================================================
    # DELTA FROM DEFAULT
    # ========================================================

    comparison_rows = []

    for model_name in models:

        default_row = default_df[
            default_df["model"] == model_name
        ].iloc[0]

        selected_row = selected_df[
            selected_df["model"] == model_name
        ].iloc[0]

        comparison_rows.append(
            {
                "model": model_name,
                "threshold": selected_row[
                    "threshold"
                ],
                "precision_change": (
                    selected_row["precision"]
                    - default_row["precision"]
                ),
                "recall_change": (
                    selected_row["recall"]
                    - default_row["recall"]
                ),
                "f1_change": (
                    selected_row["f1"]
                    - default_row["f1"]
                ),
                "brier_change": (
                    selected_row["brier_score"]
                    - default_row["brier_score"]
                ),
            }
        )

    delta_df = pd.DataFrame(
        comparison_rows
    )

    print()
    print("=" * 80)
    print("EFFECT OF THRESHOLD OPTIMIZATION")
    print("=" * 80)

    print(
        delta_df.to_string(
            index=False,
            float_format=lambda x: f"{x:+.4f}",
        )
    )

    # ========================================================
    # SANITY CHECKS
    # ========================================================

    assert len(default_df) == 3
    assert len(selected_df) == 3

    assert (
        selected_df["threshold"]
        .between(0, 1)
        .all()
    )

    assert (
        default_df["pr_auc"]
        .notna()
        .all()
    )

    assert (
        selected_df["pr_auc"]
        .notna()
        .all()
    )

    assert (
        np.isfinite(
            selected_df[
                [
                    "pr_auc",
                    "roc_auc",
                    "precision",
                    "recall",
                    "f1",
                    "brier_score",
                ]
            ].to_numpy()
        ).all()
    )

    print()
    print("=" * 80)
    print("MODEL THRESHOLD VALIDATION PASSED")
    print("=" * 80)


if __name__ == "__main__":
    main()