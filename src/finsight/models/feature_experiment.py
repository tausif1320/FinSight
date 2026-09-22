from __future__ import annotations

import pandas as pd

from finsight.models.baselines import (
    build_logistic_regression,
    prepare_xy,
)
from finsight.models.model_evaluation import evaluate_binary_classifier


def train_and_evaluate_feature_set(
    train_df: pd.DataFrame,
    evaluation_df: pd.DataFrame,
    feature_columns: list[str],
    target_column: str = "financial_stress_30d",
    model_name: str = "Model",
) -> dict:
    """
    Train Logistic Regression on one feature set and
    evaluate it on the supplied evaluation dataset.
    """

    X_train, y_train = prepare_xy(
        train_df,
        feature_columns,
        target_column,
    )

    X_eval, y_eval = prepare_xy(
        evaluation_df,
        feature_columns,
        target_column,
    )

    model = build_logistic_regression()

    model.fit(X_train, y_train)

    probabilities = model.predict_proba(X_eval)[:, 1]

    metrics = evaluate_binary_classifier(
        y_true=y_eval,
        y_probability=probabilities,
        threshold=0.50,
    )

    metrics["model"] = model_name
    metrics["feature_count"] = len(feature_columns)

    return metrics


def compare_feature_sets(
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    test_df: pd.DataFrame,
    static_features: list[str],
    dynamic_features: list[str],
    target_column: str = "financial_stress_30d",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compare:

    1. Static features
    2. Static + Dynamic features

    Returns:
        validation_results
        test_results
    """

    all_features = static_features + dynamic_features

    results_validation = []
    results_test = []

    feature_sets = {
        "Static": static_features,
        "Static + Dynamic": all_features,
    }

    for model_name, features in feature_sets.items():

        validation_metrics = train_and_evaluate_feature_set(
            train_df=train_df,
            evaluation_df=validation_df,
            feature_columns=features,
            target_column=target_column,
            model_name=model_name,
        )

        test_metrics = train_and_evaluate_feature_set(
            train_df=train_df,
            evaluation_df=test_df,
            feature_columns=features,
            target_column=target_column,
            model_name=model_name,
        )

        results_validation.append(validation_metrics)
        results_test.append(test_metrics)

    validation_df_results = pd.DataFrame(results_validation)
    test_df_results = pd.DataFrame(results_test)

    return validation_df_results, test_df_results