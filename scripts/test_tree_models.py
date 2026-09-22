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

from finsight.models.model_evaluation import (
    evaluate_binary_classifier,
    compare_models,
)


DATASET_PATH = (
    "data/processed/modeling_dataset_1000_users.csv"
)

TARGET = "financial_stress_30d"


def train_and_evaluate(
    model,
    model_name,
    train_df,
    test_df,
    feature_columns,
):
    """
    Train a model on the chronological training set
    and evaluate it on the untouched test set.
    """

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

    print()
    print("=" * 80)
    print(f"TRAINING: {model_name}")
    print("=" * 80)

    model.fit(
        X_train,
        y_train,
    )

    test_probability = model.predict_proba(
        X_test
    )[:, 1]

    metrics = evaluate_binary_classifier(
        y_true=y_test,
        y_probability=test_probability,
        threshold=0.50,
    )

    metrics["model"] = model_name

    return metrics


def main():

    print("=" * 80)
    print("FinSight Tree Model Experiment")
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
    # DATA VALIDATION
    # ========================================================

    assert TARGET in model_df.columns

    assert (
        model_df["user_id"].nunique() == 1000
    )

    duplicate_keys = model_df.duplicated(
        subset=["user_id", "month"]
    ).sum()

    assert duplicate_keys == 0, (
        f"Found {duplicate_keys} duplicate user-month rows."
    )

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
    # MODELS
    # ========================================================

    models = [
        (
            "Logistic Regression",
            build_logistic_regression(),
        ),
        (
            "Random Forest",
            build_random_forest(),
        ),
        (
            "HistGradientBoosting",
            build_hist_gradient_boosting(),
        ),
    ]

    results = []

    # ========================================================
    # TRAINING
    # ========================================================

    for model_name, model in models:

        metrics = train_and_evaluate(
            model=model,
            model_name=model_name,
            train_df=train_df,
            test_df=test_df,
            feature_columns=feature_columns,
        )

        results.append(metrics)

        print()
        print(
            f"{model_name} TEST RESULTS"
        )
        print(
            f"PR-AUC:       {metrics['pr_auc']:.4f}"
        )
        print(
            f"ROC-AUC:      {metrics['roc_auc']:.4f}"
        )
        print(
            f"Precision:    {metrics['precision']:.4f}"
        )
        print(
            f"Recall:       {metrics['recall']:.4f}"
        )
        print(
            f"F1:           {metrics['f1']:.4f}"
        )
        print(
            f"Brier score:  {metrics['brier_score']:.4f}"
        )

    # ========================================================
    # COMPARISON
    # ========================================================

    results_dict = {
        result["model"]: {
            key: value
            for key, value in result.items()
            if key != "model"
        }
        for result in results
    }

    comparison = compare_models(
        results_dict
    )

    print()
    print("=" * 80)
    print("FINAL TEST SET MODEL COMPARISON")
    print("=" * 80)

    print(
        comparison.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    # ========================================================
    # SANITY CHECKS
    # ========================================================

    assert len(results) == 3

    assert set(
        comparison["model"]
    ) == {
        "Logistic Regression",
        "Random Forest",
        "HistGradientBoosting",
    }

    metric_columns = [
        "pr_auc",
        "roc_auc",
        "precision",
        "recall",
        "f1",
        "brier_score",
    ]

    for column in metric_columns:

        assert comparison[column].notna().all()

        assert np.isfinite(
            comparison[column]
        ).all()

    print()
    print("=" * 80)
    print("TREE MODEL VALIDATION PASSED")
    print("=" * 80)


if __name__ == "__main__":
    main()