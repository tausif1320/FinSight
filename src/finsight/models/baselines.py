from __future__ import annotations

import pandas as pd

from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def build_dummy_baseline() -> DummyClassifier:
    """
    Majority/prior baseline.
    """

    return DummyClassifier(
        strategy="prior",
    )


def build_logistic_regression() -> Pipeline:
    """
    Logistic regression baseline.

    Pipeline:
        median imputation
        -> standardization
        -> class-balanced logistic regression
    """

    return Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="median",
                ),
            ),
            (
                "scaler",
                StandardScaler(),
            ),
            (
                "model",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=2000,
                    random_state=42,
                ),
            ),
        ]
    )


def prepare_xy(
    df: pd.DataFrame,
    feature_columns: list[str],
    target_column: str = "financial_stress_30d",
):
    """
    Prepare X and y for model training.
    """

    missing_features = [
        column
        for column in feature_columns
        if column not in df.columns
    ]

    if missing_features:
        raise ValueError(
            f"Missing feature columns: {missing_features}"
        )

    if target_column not in df.columns:
        raise ValueError(
            f"Missing target column: {target_column}"
        )

    X = df[feature_columns].copy()
    y = df[target_column].astype(int).copy()

    return X, y