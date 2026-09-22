from __future__ import annotations

import pandas as pd

from finsight.models.dataset import (
    get_feature_columns,
    chronological_split,
)

from finsight.models.feature_sets import (
    get_static_features,
    get_dynamic_features,
)

from finsight.models.feature_experiment import (
    compare_feature_sets,
)


DATASET_PATH = (
    "data/processed/modeling_dataset_1000_users.csv"
)

TARGET = "financial_stress_30d"


def print_result_table(title, results):

    print()
    print("=" * 80)
    print(title)
    print("=" * 80)

    columns = [
        "model",
        "feature_count",
        "pr_auc",
        "roc_auc",
        "precision",
        "recall",
        "f1",
        "brier_score",
    ]

    available_columns = [
        column
        for column in columns
        if column in results.columns
    ]

    print(
        results[available_columns].to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )


def main():

    print("=" * 80)
    print("FinSight Static vs Dynamic Experiment")
    print("Saved 1,000-User Dataset")
    print("=" * 80)

    # --------------------------------------------------
    # LOAD DATA
    # --------------------------------------------------

    print()
    print("Loading modeling dataset...")

    model_df = pd.read_csv(
        DATASET_PATH,
        parse_dates=["month"],
    )

    print(
        f"Loaded rows: {len(model_df):,}"
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

    # --------------------------------------------------
    # BASIC VALIDATION
    # --------------------------------------------------

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
    print(
        "Dataset validation passed."
    )

    # --------------------------------------------------
    # FEATURE COLUMNS
    # --------------------------------------------------

    all_features = get_feature_columns(
        model_df
    )

    static_features = get_static_features(
        all_features
    )

    dynamic_features = get_dynamic_features(
        all_features
    )

    print()
    print("=" * 80)
    print("FEATURE SETS")
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
        f"{len(static_features) + len(dynamic_features)}"
    )

    # --------------------------------------------------
    # CHRONOLOGICAL SPLIT
    # --------------------------------------------------

    train_df, validation_df, test_df = chronological_split(
        model_df
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

    # --------------------------------------------------
    # EXPERIMENT
    # --------------------------------------------------

    print()
    print("=" * 80)
    print("STATIC VS DYNAMIC EXPERIMENT")
    print("=" * 80)

    validation_results, test_results = (
        compare_feature_sets(
            train_df=train_df,
            validation_df=validation_df,
            test_df=test_df,
            static_features=static_features,
            dynamic_features=dynamic_features,
            target_column=TARGET,
        )
    )

    # --------------------------------------------------
    # RESULTS
    # --------------------------------------------------

    print_result_table(
        "VALIDATION RESULTS",
        validation_results,
    )

    print_result_table(
        "TEST RESULTS",
        test_results,
    )

    # --------------------------------------------------
    # TEST DELTAS
    # --------------------------------------------------

    static_test = test_results[
        test_results["model"] == "Static"
    ].iloc[0]

    dynamic_test = test_results[
        test_results["model"] == "Static + Dynamic"
    ].iloc[0]

    print()
    print("=" * 80)
    print("TEST SET DELTAS")
    print("=" * 80)

    print(
        f"PR-AUC: "
        f"{dynamic_test['pr_auc'] - static_test['pr_auc']:+.4f}"
    )

    print(
        f"ROC-AUC: "
        f"{dynamic_test['roc_auc'] - static_test['roc_auc']:+.4f}"
    )

    print(
        f"Precision: "
        f"{dynamic_test['precision'] - static_test['precision']:+.4f}"
    )

    print(
        f"Recall: "
        f"{dynamic_test['recall'] - static_test['recall']:+.4f}"
    )

    print(
        f"F1: "
        f"{dynamic_test['f1'] - static_test['f1']:+.4f}"
    )

    print(
        f"Brier score: "
        f"{dynamic_test['brier_score'] - static_test['brier_score']:+.4f}"
    )

    # --------------------------------------------------
    # SANITY CHECKS
    # --------------------------------------------------

    assert len(validation_results) == 2
    assert len(test_results) == 2

    assert (
        validation_results["pr_auc"].notna().all()
    )

    assert (
        test_results["pr_auc"].notna().all()
    )

    print()
    print("=" * 80)
    print("STATIC VS DYNAMIC EXPERIMENT PASSED")
    print("=" * 80)


if __name__ == "__main__":
    main()