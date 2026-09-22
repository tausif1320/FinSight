from __future__ import annotations

import pandas as pd


KEY_COLUMNS = ["user_id", "month"]

TARGET_COLUMN = "financial_stress_30d"

# These columns describe the future outcome.
# They must NEVER be used as model predictors.
FUTURE_LEAKAGE_COLUMNS = [
    # Future dates
    "prediction_date",
    "horizon_end",

    # Future financial outcomes
    "future_income",
    "future_expenses",
    "future_net_cash_flow",
    "minimum_future_balance",
    "ending_future_balance",

    # Future transaction / spending information
    "future_transaction_count",
    "future_discretionary_spending",
    "future_essential_spending",

    # Future cash-flow behavior
    "negative_cashflow_days",

    # Future obligation information
    "required_obligations",
    "obligation_coverage",

    # Future stress components
    "critical_liquidity",
    "obligation_failure",
    "severe_liquidity_drawdown",
    "liquidity_ratio",

    # Target
    "financial_stress_30d",
]


def _validate_input_frames(
    feature_df: pd.DataFrame,
    outcome_df: pd.DataFrame,
) -> None:
    """Validate the basic structure of feature and outcome data."""

    if not isinstance(feature_df, pd.DataFrame):
        raise TypeError("feature_df must be a pandas DataFrame")

    if not isinstance(outcome_df, pd.DataFrame):
        raise TypeError("outcome_df must be a pandas DataFrame")

    for column in KEY_COLUMNS:
        if column not in feature_df.columns:
            raise ValueError(
                f"Feature dataframe missing required column: {column}"
            )

        if column not in outcome_df.columns:
            raise ValueError(
                f"Outcome dataframe missing required column: {column}"
            )

    if TARGET_COLUMN not in outcome_df.columns:
        raise ValueError(
            f"Outcome dataframe missing target column: {TARGET_COLUMN}"
        )

    if feature_df.duplicated(KEY_COLUMNS).any():
        raise ValueError(
            "Feature dataframe contains duplicate user-month observations"
        )

    if outcome_df.duplicated(KEY_COLUMNS).any():
        raise ValueError(
            "Outcome dataframe contains duplicate user-month observations"
        )


def build_modeling_dataset(
    feature_df: pd.DataFrame,
    outcome_df: pd.DataFrame,
    drop_incomplete_history: bool = True,
) -> pd.DataFrame:
    """
    Combine historical behavioral features with future outcomes.

    Each row represents:

        user + prediction month
        -> historical features
        -> future 30-day outcome

    Future outcome columns remain in the dataframe for validation,
    but are excluded later from model predictors.
    """

    _validate_input_frames(feature_df, outcome_df)

    features = feature_df.copy()
    outcomes = outcome_df.copy()

    features["month"] = pd.to_datetime(features["month"])
    outcomes["month"] = pd.to_datetime(outcomes["month"])

    model_df = features.merge(
        outcomes,
        on=KEY_COLUMNS,
        how="inner",
        validate="one_to_one",
        suffixes=("", "_outcome"),
    )

    if model_df.empty:
        raise ValueError(
            "Modeling dataset is empty after merging features and outcomes"
        )

    if drop_incomplete_history:
        if "has_full_history" not in model_df.columns:
            raise ValueError(
                "has_full_history column is required when "
                "drop_incomplete_history=True"
            )

        model_df = model_df[
            model_df["has_full_history"].astype(bool)
        ].copy()

    if model_df.empty:
        raise ValueError(
            "No observations remain after removing incomplete history"
        )

    # Target must be binary.
    target_values = set(model_df[TARGET_COLUMN].dropna().unique())

    if not target_values.issubset({0, 1, False, True}):
        raise ValueError(
            f"{TARGET_COLUMN} must be binary 0/1. "
            f"Found values: {target_values}"
        )

    model_df[TARGET_COLUMN] = (
        model_df[TARGET_COLUMN]
        .astype(int)
    )

    model_df = (
        model_df
        .sort_values(KEY_COLUMNS)
        .reset_index(drop=True)
    )

    return model_df


def get_feature_columns(model_df: pd.DataFrame) -> list[str]:
    """
    Return numeric/bool predictor columns while excluding identifiers,
    target, and future outcome columns.
    """

    excluded = (
    set(KEY_COLUMNS)
    | set(FUTURE_LEAKAGE_COLUMNS)
    | {"prediction_balance"}
    )

    feature_columns = []

    for column in model_df.columns:
        if column in excluded:
            continue

        if pd.api.types.is_numeric_dtype(model_df[column]):
            feature_columns.append(column)

    if not feature_columns:
        raise ValueError("No numeric predictor columns found")

    return feature_columns


def chronological_split(
    model_df: pd.DataFrame,
    train_fraction: float = 0.70,
    validation_fraction: float = 0.15,
):
    """
    Split data chronologically by month.

    No random row-level splitting is used because observations are
    temporal and random splitting could leak future information into
    training.
    """

    if not 0 < train_fraction < 1:
        raise ValueError("train_fraction must be between 0 and 1")

    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between 0 and 1")

    if train_fraction + validation_fraction >= 1:
        raise ValueError(
            "train_fraction + validation_fraction must be < 1"
        )

    if "month" not in model_df.columns:
        raise ValueError("model_df must contain month")

    df = model_df.copy()
    df["month"] = pd.to_datetime(df["month"])

    months = sorted(df["month"].dropna().unique())

    if len(months) < 3:
        raise ValueError(
            "At least 3 unique months are required for "
            "train/validation/test splitting"
        )

    n_months = len(months)

    train_end = max(1, int(n_months * train_fraction))

    validation_end = max(
        train_end + 1,
        int(n_months * (train_fraction + validation_fraction)),
    )

    # Ensure at least one test month.
    validation_end = min(validation_end, n_months - 1)

    train_months = months[:train_end]
    validation_months = months[train_end:validation_end]
    test_months = months[validation_end:]

    if not validation_months or not test_months:
        raise ValueError(
            "Unable to create non-empty validation and test periods"
        )

    train_df = (
        df[df["month"].isin(train_months)]
        .sort_values(KEY_COLUMNS)
        .reset_index(drop=True)
    )

    validation_df = (
        df[df["month"].isin(validation_months)]
        .sort_values(KEY_COLUMNS)
        .reset_index(drop=True)
    )

    test_df = (
        df[df["month"].isin(test_months)]
        .sort_values(KEY_COLUMNS)
        .reset_index(drop=True)
    )

    # Temporal ordering validation.
    if train_df["month"].max() >= validation_df["month"].min():
        raise ValueError("Training period overlaps validation period")

    if validation_df["month"].max() >= test_df["month"].min():
        raise ValueError("Validation period overlaps test period")

    return train_df, validation_df, test_df


def summarize_split(
    name: str,
    df: pd.DataFrame,
) -> dict:
    """Return and print a compact summary of a temporal split."""

    if df.empty:
        raise ValueError(f"{name} split is empty")

    positive = int(df[TARGET_COLUMN].sum())
    negative = int(len(df) - positive)

    summary = {
        "name": name,
        "rows": len(df),
        "positive": positive,
        "negative": negative,
        "positive_rate": float(df[TARGET_COLUMN].mean()),
        "min_month": df["month"].min(),
        "max_month": df["month"].max(),
    }

    print(f"\n{name.upper()} SPLIT")
    print(f"Rows: {summary['rows']}")
    print(f"Positive: {summary['positive']}")
    print(f"Negative: {summary['negative']}")
    print(f"Positive rate: {summary['positive_rate']:.4f}")
    print(f"Months: {summary['min_month']} -> {summary['max_month']}")

    return summary