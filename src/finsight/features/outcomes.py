"""
Future outcome construction for FinSight.

This module creates future-looking outcome variables using transactions
strictly after the prediction date.

Temporal convention
-------------------
For a prediction date D:

    Feature window:
        historical observations up to and including D

    Outcome window:
        D + 1 through D + horizon_days

No transaction occurring after the outcome window is used.

Primary outcome
---------------
financial_stress_30d

A user is considered financially stressed during the future horizon when
at least one of the following occurs:

1. Critical liquidity:
       minimum_future_balance <= 0

2. Obligation coverage failure:
       obligation_coverage < 1

3. Severe liquidity drawdown:
       minimum_future_balance / prediction_balance < 0.25
       AND
       future_net_cash_flow < 0

Daily negative cash flow is NOT used as a stress criterion because salary
is periodic while expenses occur on many individual days. Therefore,
negative daily cash-flow days are descriptive only.

Transaction schema
------------------
The FinSight transaction generator provides:

    transaction_id
    user_id
    timestamp
    transaction_type
    merchant_category
    amount
    payment_method
    balance_before
    balance_after

Recurring obligations are represented through merchant_category:

    Rent
    EMI
    Utility
    Subscription
    Other_Recurring
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------

DEFAULT_HORIZON_DAYS = 30
DEFAULT_MIN_BALANCE_THRESHOLD = 0.0
DEFAULT_MAX_OBLIGATION_COVERAGE = 1.0
DEFAULT_SEVERE_BALANCE_RETENTION = 0.25


# ---------------------------------------------------------------------
# Required schemas
# ---------------------------------------------------------------------

REQUIRED_TRANSACTION_COLUMNS = {
    "user_id",
    "timestamp",
    "transaction_type",
    "merchant_category",
    "amount",
}


REQUIRED_MONTHLY_COLUMNS = {
    "user_id",
    "month",
}


# ---------------------------------------------------------------------
# Transaction categories
# ---------------------------------------------------------------------

OBLIGATION_CATEGORIES = {
    "rent",
    "emi",
    "utility",
    "utilities",
    "subscription",
    "other_recurring",
}


ESSENTIAL_CATEGORIES = {
    "food",
    "groceries",
    "transport",
    "healthcare",
    "education",
    "rent",
    "emi",
    "utility",
    "utilities",
    "subscription",
    "other_recurring",
}


# ---------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------

def _validate_transactions(
    transactions: pd.DataFrame,
) -> None:
    """Validate the FinSight transaction dataframe."""

    if not isinstance(
        transactions,
        pd.DataFrame,
    ):
        raise TypeError(
            "transactions must be a pandas DataFrame."
        )

    missing = (
        REQUIRED_TRANSACTION_COLUMNS
        - set(transactions.columns)
    )

    if missing:
        raise ValueError(
            "transactions is missing required columns: "
            f"{sorted(missing)}"
        )

    if transactions.empty:
        raise ValueError(
            "transactions cannot be empty."
        )

    if transactions["user_id"].isna().any():
        raise ValueError(
            "transactions contains missing user_id values."
        )

    if transactions["timestamp"].isna().any():
        raise ValueError(
            "transactions contains missing timestamp values."
        )

    if transactions["amount"].isna().any():
        raise ValueError(
            "transactions contains missing amount values."
        )

    if transactions["transaction_type"].isna().any():
        raise ValueError(
            "transactions contains missing transaction_type values."
        )

    if transactions["merchant_category"].isna().any():
        raise ValueError(
            "transactions contains missing merchant_category values."
        )


def _validate_monthly_profile(
    monthly_profile: pd.DataFrame,
) -> None:
    """Validate the monthly profile dataframe."""

    if not isinstance(
        monthly_profile,
        pd.DataFrame,
    ):
        raise TypeError(
            "monthly_profile must be a pandas DataFrame."
        )

    missing = (
        REQUIRED_MONTHLY_COLUMNS
        - set(monthly_profile.columns)
    )

    if missing:
        raise ValueError(
            "monthly_profile is missing required columns: "
            f"{sorted(missing)}"
        )

    if monthly_profile.empty:
        raise ValueError(
            "monthly_profile cannot be empty."
        )

    if monthly_profile["user_id"].isna().any():
        raise ValueError(
            "monthly_profile contains missing user_id values."
        )

    if monthly_profile["month"].isna().any():
        raise ValueError(
            "monthly_profile contains missing month values."
        )


# ---------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------

def _prepare_transactions(
    transactions: pd.DataFrame,
) -> pd.DataFrame:
    """
    Normalize the FinSight transaction dataframe.

    The generator uses `timestamp`.
    The outcome engine internally uses `date`.
    """

    _validate_transactions(
        transactions
    )

    tx = transactions.copy()

    tx["date"] = pd.to_datetime(
        tx["timestamp"],
        errors="raise",
    )

    tx["amount"] = pd.to_numeric(
        tx["amount"],
        errors="raise",
    )

    sort_columns = [
        "user_id",
        "date",
    ]

    if "transaction_id" in tx.columns:
        sort_columns.append(
            "transaction_id"
        )

    tx = (
        tx.sort_values(
            sort_columns,
            kind="mergesort",
        )
        .reset_index(drop=True)
    )

    return tx


def _prepare_monthly_profile(
    monthly_profile: pd.DataFrame,
) -> pd.DataFrame:
    """Normalize the monthly profile dataframe."""

    _validate_monthly_profile(
        monthly_profile
    )

    profile = monthly_profile.copy()

    profile["month"] = pd.to_datetime(
        profile["month"],
        errors="raise",
    )

    profile = (
        profile.sort_values(
            [
                "user_id",
                "month",
            ],
            kind="mergesort",
        )
        .reset_index(drop=True)
    )

    return profile


# ---------------------------------------------------------------------
# Numeric helpers
# ---------------------------------------------------------------------

def _safe_divide(
    numerator: float,
    denominator: float,
) -> float:
    """Safely divide two scalar values."""

    if not np.isfinite(
        denominator
    ):
        return np.nan

    if denominator == 0:
        return np.nan

    return float(
        numerator / denominator
    )


# ---------------------------------------------------------------------
# Transaction classification
# ---------------------------------------------------------------------

def _normalized_transaction_type(
    transactions: pd.DataFrame,
) -> pd.Series:
    """Return normalized transaction types."""

    return (
        transactions["transaction_type"]
        .astype(str)
        .str.lower()
        .str.strip()
    )


def _infer_income_mask(
    transactions: pd.DataFrame,
) -> pd.Series:
    """
    Identify income transactions.

    FinSight currently uses CREDIT for salary and bonus transactions.
    """

    transaction_type = (
        _normalized_transaction_type(
            transactions
        )
    )

    return transaction_type.isin(
        {
            "income",
            "salary",
            "bonus",
            "credit",
        }
    )


def _infer_expense_mask(
    transactions: pd.DataFrame,
) -> pd.Series:
    """
    Identify expense transactions.

    FinSight currently uses DEBIT for expenses.
    """

    transaction_type = (
        _normalized_transaction_type(
            transactions
        )
    )

    return transaction_type.isin(
        {
            "expense",
            "debit",
            "spending",
            "purchase",
        }
    )


def _infer_obligation_mask(
    transactions: pd.DataFrame,
) -> pd.Series:
    """
    Identify recurring financial obligations.

    The FinSight transaction generator stores obligation semantics
    in merchant_category.
    """

    if "is_obligation" in transactions.columns:

        return (
            transactions["is_obligation"]
            .fillna(False)
            .astype(bool)
        )

    if "merchant_category" in transactions.columns:

        merchant_category = (
            transactions[
                "merchant_category"
            ]
            .astype(str)
            .str.lower()
            .str.strip()
        )

        return merchant_category.isin(
            OBLIGATION_CATEGORIES
        )

    if "category" in transactions.columns:

        category = (
            transactions["category"]
            .astype(str)
            .str.lower()
            .str.strip()
        )

        return category.isin(
            OBLIGATION_CATEGORIES
        )

    return pd.Series(
        False,
        index=transactions.index,
        dtype=bool,
    )


# ---------------------------------------------------------------------
# Future outcome calculations
# ---------------------------------------------------------------------

def _calculate_future_outcomes(
    future_transactions: pd.DataFrame,
    prediction_balance: float,
) -> dict:
    """
    Calculate all future-horizon financial outcomes.

    The returned dictionary contains only raw outcome metrics.
    Stress component columns are added later in the public API so
    that the final target and its components are guaranteed to use
    exactly the same conditions.
    """

    if future_transactions.empty:

        return {
            "minimum_future_balance": float(
                prediction_balance
            ),
            "ending_future_balance": float(
                prediction_balance
            ),
            "future_net_cash_flow": 0.0,
            "future_income": 0.0,
            "future_expenses": 0.0,
            "negative_cashflow_days": 0,
            "required_obligations": 0.0,
            "obligation_coverage": np.inf,
            "future_transaction_count": 0,
            "future_discretionary_spending": 0.0,
            "future_essential_spending": 0.0,
        }

    future = future_transactions.copy()

    sort_columns = ["date"]

    if "transaction_id" in future.columns:
        sort_columns.append(
            "transaction_id"
        )

    future = (
        future.sort_values(
            sort_columns,
            kind="mergesort",
        )
        .reset_index(drop=True)
    )

    income_mask = _infer_income_mask(
        future
    )

    expense_mask = _infer_expense_mask(
        future
    )

    # -------------------------------------------------------------
    # Validate transaction types before calculating anything.
    # -------------------------------------------------------------

    normalized_types = (
        _normalized_transaction_type(
            future
        )
    )

    supported_types = {
        "income",
        "salary",
        "bonus",
        "credit",
        "expense",
        "debit",
        "spending",
        "purchase",
    }

    unknown_types = sorted(
        set(
            normalized_types
        )
        - supported_types
    )

    if unknown_types:

        raise ValueError(
            "Unknown transaction type(s) encountered: "
            f"{unknown_types}"
        )

    # -------------------------------------------------------------
    # Income and expenses
    # -------------------------------------------------------------

    income = float(
        future.loc[
            income_mask,
            "amount",
        ]
        .abs()
        .sum()
    )

    expenses = float(
        future.loc[
            expense_mask,
            "amount",
        ]
        .abs()
        .sum()
    )

    future_net_cash_flow = (
        income
        - expenses
    )

    # -------------------------------------------------------------
    # Sequential balance reconstruction
    # -------------------------------------------------------------

    balance = float(
        prediction_balance
    )

    minimum_balance = balance

    for _, transaction in future.iterrows():

        transaction_type = str(
            transaction[
                "transaction_type"
            ]
        ).lower().strip()

        amount = float(
            abs(
                transaction["amount"]
            )
        )

        if transaction_type in {
            "income",
            "salary",
            "bonus",
            "credit",
        }:

            balance += amount

        elif transaction_type in {
            "expense",
            "debit",
            "spending",
            "purchase",
        }:

            balance -= amount

        minimum_balance = min(
            minimum_balance,
            balance,
        )

    ending_balance = balance

    # -------------------------------------------------------------
    # Daily negative cash-flow days
    # -------------------------------------------------------------

    daily = future.copy()

    daily["calendar_day"] = (
        daily["date"]
        .dt.normalize()
    )

    daily["income_amount"] = np.where(
        income_mask,
        daily["amount"].abs(),
        0.0,
    )

    daily["expense_amount"] = np.where(
        expense_mask,
        daily["amount"].abs(),
        0.0,
    )

    daily = (
        daily.groupby(
            "calendar_day",
            as_index=False,
        )
        .agg(
            daily_income=(
                "income_amount",
                "sum",
            ),
            daily_expense=(
                "expense_amount",
                "sum",
            ),
        )
    )

    daily["daily_net_cash_flow"] = (
        daily["daily_income"]
        - daily["daily_expense"]
    )

    negative_cashflow_days = int(
        (
            daily[
                "daily_net_cash_flow"
            ]
            < 0
        ).sum()
    )

    # -------------------------------------------------------------
    # Required obligations
    # -------------------------------------------------------------

    obligation_mask = (
        _infer_obligation_mask(
            future
        )
    )

    required_obligations = float(
        future.loc[
            obligation_mask
            & expense_mask,
            "amount",
        ]
        .abs()
        .sum()
    )

    # -------------------------------------------------------------
    # Obligation coverage
    # -------------------------------------------------------------

    if required_obligations > 0:

        available_resources = (
            prediction_balance
            + income
        )

        obligation_coverage = _safe_divide(
            available_resources,
            required_obligations,
        )

    else:

        obligation_coverage = np.inf

    # -------------------------------------------------------------
    # Spending categories
    # -------------------------------------------------------------

    future_discretionary_spending = 0.0
    future_essential_spending = 0.0

    if "spending_type" in future.columns:

        spending_type = (
            future[
                "spending_type"
            ]
            .astype(str)
            .str.lower()
            .str.strip()
        )

        discretionary_mask = (
            spending_type
            == "discretionary"
        )

        essential_mask = (
            spending_type
            == "essential"
        )

        future_discretionary_spending = float(
            future.loc[
                discretionary_mask
                & expense_mask,
                "amount",
            ]
            .abs()
            .sum()
        )

        future_essential_spending = float(
            future.loc[
                essential_mask
                & expense_mask,
                "amount",
            ]
            .abs()
            .sum()
        )

    elif "merchant_category" in future.columns:

        merchant_category = (
            future[
                "merchant_category"
            ]
            .astype(str)
            .str.lower()
            .str.strip()
        )

        essential_mask = (
            merchant_category.isin(
                ESSENTIAL_CATEGORIES
            )
        )

        future_essential_spending = float(
            future.loc[
                essential_mask
                & expense_mask,
                "amount",
            ]
            .abs()
            .sum()
        )

        future_discretionary_spending = float(
            max(
                expenses
                - future_essential_spending,
                0.0,
            )
        )

    elif "category" in future.columns:

        category = (
            future[
                "category"
            ]
            .astype(str)
            .str.lower()
            .str.strip()
        )

        essential_mask = (
            category.isin(
                ESSENTIAL_CATEGORIES
            )
        )

        future_essential_spending = float(
            future.loc[
                essential_mask
                & expense_mask,
                "amount",
            ]
            .abs()
            .sum()
        )

        future_discretionary_spending = float(
            max(
                expenses
                - future_essential_spending,
                0.0,
            )
        )

    return {
        "minimum_future_balance": float(
            minimum_balance
        ),
        "ending_future_balance": float(
            ending_balance
        ),
        "future_net_cash_flow": float(
            future_net_cash_flow
        ),
        "future_income": float(
            income
        ),
        "future_expenses": float(
            expenses
        ),
        "negative_cashflow_days": int(
            negative_cashflow_days
        ),
        "required_obligations": float(
            required_obligations
        ),
        "obligation_coverage": float(
            obligation_coverage
        ),
        "future_transaction_count": int(
            len(future)
        ),
        "future_discretionary_spending": float(
            future_discretionary_spending
        ),
        "future_essential_spending": float(
            future_essential_spending
        ),
    }


# ---------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------

def build_future_outcomes(
    transactions: pd.DataFrame,
    monthly_profile: pd.DataFrame,
    horizon_days: int = DEFAULT_HORIZON_DAYS,
    min_balance_threshold: float = (
        DEFAULT_MIN_BALANCE_THRESHOLD
    ),
    max_obligation_coverage: float = (
        DEFAULT_MAX_OBLIGATION_COVERAGE
    ),
    severe_balance_retention: float = (
        DEFAULT_SEVERE_BALANCE_RETENTION
    ),
) -> pd.DataFrame:
    """
    Build future financial outcomes for every valid prediction point.

    Parameters
    ----------
    transactions:
        Actual FinSight transaction-level dataframe.

    monthly_profile:
        Monthly dataframe containing at least:

            user_id
            month

        It must also contain one balance column. The function searches
        for:

            ending_balance
            balance
            current_balance
            average_balance

    horizon_days:
        Number of future calendar days.

    min_balance_threshold:
        Balance threshold defining critical liquidity.

    max_obligation_coverage:
        Coverage threshold below which obligations are considered
        insufficiently covered.

    severe_balance_retention:
        Fraction of prediction-date balance below which a negative
        future cash-flow trajectory is considered a severe drawdown.

    Returns
    -------
    pd.DataFrame
        One row per user-month with a complete future horizon.

    Temporal convention
    -------------------
    For prediction date D:

        D + 1 <= transaction date <= D + horizon_days
    """

    # -------------------------------------------------------------
    # Parameter validation
    # -------------------------------------------------------------

    if horizon_days <= 0:

        raise ValueError(
            "horizon_days must be greater than zero."
        )

    if not (
        0 < severe_balance_retention <= 1
    ):

        raise ValueError(
            "severe_balance_retention must "
            "be in (0, 1]."
        )

    if max_obligation_coverage <= 0:

        raise ValueError(
            "max_obligation_coverage must "
            "be greater than zero."
        )

    # -------------------------------------------------------------
    # Prepare inputs
    # -------------------------------------------------------------

    tx = _prepare_transactions(
        transactions
    )

    profile = _prepare_monthly_profile(
        monthly_profile
    )

    # -------------------------------------------------------------
    # Determine balance column
    # -------------------------------------------------------------

    balance_candidates = [
        "ending_balance",
        "balance",
        "current_balance",
        "average_balance",
    ]

    balance_column = next(
        (
            column
            for column in balance_candidates
            if column in profile.columns
        ),
        None,
    )

    if balance_column is None:

        raise ValueError(
            "monthly_profile must contain one of: "
            f"{balance_candidates}"
        )

    # -------------------------------------------------------------
    # Generate raw future outcomes
    # -------------------------------------------------------------

    records: list[dict] = []

    for (
        user_id,
        user_profile,
    ) in profile.groupby(
        "user_id",
        sort=False,
    ):

        user_profile = (
            user_profile
            .sort_values(
                "month",
                kind="mergesort",
            )
        )

        user_transactions = (
            tx[
                tx["user_id"] == user_id
            ]
            .sort_values(
                "date",
                kind="mergesort",
            )
        )

        if user_transactions.empty:
            continue

        available_max_date = (
            user_transactions[
                "date"
            ].max()
        )

        for _, row in user_profile.iterrows():

            prediction_month = pd.Timestamp(
                row["month"]
            )

            # Prediction date is the final calendar day
            # of the observation month.
            prediction_date = (
                prediction_month
                + pd.offsets.MonthEnd(0)
            )

            horizon_end = (
                prediction_date
                + pd.Timedelta(
                    days=horizon_days
                )
            )

            # -----------------------------------------------------
            # Require complete future horizon.
            # -----------------------------------------------------

            if (
                available_max_date
                < horizon_end
            ):
                continue

            # -----------------------------------------------------
            # Prediction balance
            # -----------------------------------------------------

            prediction_balance = float(
                row[
                    balance_column
                ]
            )

            if not np.isfinite(
                prediction_balance
            ):
                continue

            # -----------------------------------------------------
            # Strictly future transactions:
            #
            # prediction_date + 1
            # through
            # prediction_date + horizon_days
            # -----------------------------------------------------

            future_mask = (
                (
                    user_transactions[
                        "date"
                    ]
                    > prediction_date
                )
                & (
                    user_transactions[
                        "date"
                    ]
                    <= horizon_end
                )
            )

            future_transactions = (
                user_transactions.loc[
                    future_mask
                ].copy()
            )

            outcome_values = (
                _calculate_future_outcomes(
                    future_transactions=(
                        future_transactions
                    ),
                    prediction_balance=(
                        prediction_balance
                    ),
                )
            )

            record = {
                "user_id": user_id,
                "month": prediction_month,
                "prediction_date": prediction_date,
                "prediction_balance": (
                    prediction_balance
                ),
                "horizon_end": horizon_end,
                **outcome_values,
            }

            records.append(
                record
            )

    if not records:

        raise ValueError(
            "No complete future horizons were available."
        )

    outcomes = pd.DataFrame(
        records
    )

    # -------------------------------------------------------------
    # Stress component 1:
    # Critical liquidity
    # -------------------------------------------------------------

    outcomes[
        "critical_liquidity"
    ] = (
        outcomes[
            "minimum_future_balance"
        ]
        <= min_balance_threshold
    ).astype(bool)

    # -------------------------------------------------------------
    # Liquidity ratio
    #
    # Ratio:
    #
    #     minimum future balance
    #     -----------------------
    #     prediction-date balance
    #
    # For non-positive prediction balances we cannot meaningfully
    # compute a retention ratio. Use 0.0 so the column contains
    # finite values and severe drawdown cannot accidentally be
    # triggered from an invalid denominator.
    # -------------------------------------------------------------

    positive_prediction_balance = (
        outcomes[
            "prediction_balance"
        ]
        > 0
    )

    outcomes[
        "liquidity_ratio"
    ] = 0.0

    outcomes.loc[
        positive_prediction_balance,
        "liquidity_ratio",
    ] = (
        outcomes.loc[
            positive_prediction_balance,
            "minimum_future_balance",
        ]
        / outcomes.loc[
            positive_prediction_balance,
            "prediction_balance",
        ]
    )

    outcomes[
        "liquidity_ratio"
    ] = pd.to_numeric(
        outcomes[
            "liquidity_ratio"
        ],
        errors="coerce",
    )

    # -------------------------------------------------------------
    # Stress component 2:
    # Obligation coverage failure
    # -------------------------------------------------------------

    outcomes[
        "obligation_failure"
    ] = (
        outcomes[
            "obligation_coverage"
        ]
        < max_obligation_coverage
    ).astype(bool)

    # -------------------------------------------------------------
    # Stress component 3:
    # Severe liquidity drawdown
    # -------------------------------------------------------------

    outcomes[
        "severe_liquidity_drawdown"
    ] = (
        positive_prediction_balance
        & (
            outcomes[
                "liquidity_ratio"
            ]
            < severe_balance_retention
        )
        & (
            outcomes[
                "future_net_cash_flow"
            ]
            < 0
        )
    ).astype(bool)

    # -------------------------------------------------------------
    # Primary target
    #
    # IMPORTANT:
    # The final target is exactly the union of the three stored
    # component columns. This guarantees consistency with the
    # validation script.
    # -------------------------------------------------------------

    outcomes[
        "financial_stress_30d"
    ] = (
        outcomes[
            "critical_liquidity"
        ]
        | outcomes[
            "obligation_failure"
        ]
        | outcomes[
            "severe_liquidity_drawdown"
        ]
    ).astype(int)

    # -------------------------------------------------------------
    # Final column order
    # -------------------------------------------------------------

    ordered_columns = [
        "user_id",
        "month",
        "prediction_date",
        "prediction_balance",
        "horizon_end",
        "minimum_future_balance",
        "ending_future_balance",
        "future_net_cash_flow",
        "future_income",
        "future_expenses",
        "negative_cashflow_days",
        "required_obligations",
        "obligation_coverage",
        "future_transaction_count",
        "future_discretionary_spending",
        "future_essential_spending",
        "critical_liquidity",
        "obligation_failure",
        "severe_liquidity_drawdown",
        "liquidity_ratio",
        "financial_stress_30d",
    ]

    missing_output_columns = (
        set(ordered_columns)
        - set(outcomes.columns)
    )

    if missing_output_columns:

        raise AssertionError(
            "Internal error: outcomes is missing "
            "expected output columns: "
            f"{sorted(missing_output_columns)}"
        )

    outcomes = (
        outcomes[
            ordered_columns
        ]
        .sort_values(
            [
                "user_id",
                "month",
            ],
            kind="mergesort",
        )
        .reset_index(
            drop=True
        )
    )

    # -------------------------------------------------------------
    # Final validation
    # -------------------------------------------------------------

    if outcomes[
        [
            "user_id",
            "month",
        ]
    ].duplicated().any():

        raise AssertionError(
            "Duplicate user-month prediction "
            "points detected."
        )

    if outcomes[
        "financial_stress_30d"
    ].isna().any():

        raise AssertionError(
            "financial_stress_30d contains "
            "missing values."
        )

    if not outcomes[
        "financial_stress_30d"
    ].isin(
        [0, 1]
    ).all():

        raise AssertionError(
            "financial_stress_30d must be binary."
        )

    # Boolean component columns must remain boolean.
    component_columns = [
        "critical_liquidity",
        "obligation_failure",
        "severe_liquidity_drawdown",
    ]

    for column in component_columns:

        if outcomes[
            column
        ].dtype != bool:

            raise AssertionError(
                f"{column} must have boolean dtype."
            )

    # Liquidity ratio must be finite.
    if not np.isfinite(
        outcomes[
            "liquidity_ratio"
        ]
    ).all():

        raise AssertionError(
            "liquidity_ratio contains "
            "non-finite values."
        )

    # Required outcomes must be finite whenever
    # obligations actually exist.
    obligation_rows = (
        outcomes[
            "required_obligations"
        ]
        > 0
    )

    if obligation_rows.any():

        coverage = outcomes.loc[
            obligation_rows,
            "obligation_coverage",
        ]

        if coverage.isna().any():

            raise AssertionError(
                "obligation_coverage contains "
                "NaN for observations with "
                "required obligations."
            )

        if not np.isfinite(
            coverage
        ).all():

            raise AssertionError(
                "obligation_coverage contains "
                "non-finite values despite "
                "required obligations being present."
            )

    # -------------------------------------------------------------
    # Most important consistency check:
    #
    # financial_stress_30d must exactly equal the union of the
    # three documented stress components.
    # -------------------------------------------------------------

    component_union = (
        outcomes[
            "critical_liquidity"
        ]
        | outcomes[
            "obligation_failure"
        ]
        | outcomes[
            "severe_liquidity_drawdown"
        ]
    ).astype(int)

    if not (
        component_union
        == outcomes[
            "financial_stress_30d"
        ]
    ).all():

        raise AssertionError(
            "financial_stress_30d does not equal "
            "the union of the documented stress "
            "components."
        )

    return outcomes


# ---------------------------------------------------------------------
# Validation / inspection
# ---------------------------------------------------------------------

def inspect_stress_distribution(
    outcome_df: pd.DataFrame,
    verbose: bool = True,
) -> dict:
    """
    Inspect the distribution and components of the stress target.

    Returns
    -------
    dict
        Summary statistics for the stress target.
    """

    required_columns = {
        "prediction_balance",
        "minimum_future_balance",
        "future_net_cash_flow",
        "required_obligations",
        "obligation_coverage",
        "critical_liquidity",
        "obligation_failure",
        "severe_liquidity_drawdown",
        "liquidity_ratio",
        "financial_stress_30d",
    }

    missing = (
        required_columns
        - set(outcome_df.columns)
    )

    if missing:

        raise ValueError(
            "Outcome dataframe is missing columns: "
            f"{sorted(missing)}"
        )

    # -------------------------------------------------------------
    # Use the actual stored component columns.
    # -------------------------------------------------------------

    critical_liquidity = (
        outcome_df[
            "critical_liquidity"
        ]
        .astype(bool)
    )

    obligation_failure = (
        outcome_df[
            "obligation_failure"
        ]
        .astype(bool)
    )

    severe_drawdown = (
        outcome_df[
            "severe_liquidity_drawdown"
        ]
        .astype(bool)
    )

    stress = (
        outcome_df[
            "financial_stress_30d"
        ]
        == 1
    )

    # -------------------------------------------------------------
    # Verify component consistency.
    # -------------------------------------------------------------

    component_union = (
        critical_liquidity
        | obligation_failure
        | severe_drawdown
    )

    if not (
        component_union
        == stress
    ).all():

        raise AssertionError(
            "Stored stress components do not "
            "match financial_stress_30d."
        )

    stress_rate = float(
        stress.mean()
    )

    result = {
        "n_rows": int(
            len(outcome_df)
        ),
        "stress_count": int(
            stress.sum()
        ),
        "non_stress_count": int(
            (~stress).sum()
        ),
        "stress_rate": stress_rate,
        "critical_liquidity_count": int(
            critical_liquidity.sum()
        ),
        "critical_liquidity_rate": float(
            critical_liquidity.mean()
        ),
        "obligation_failure_count": int(
            obligation_failure.sum()
        ),
        "obligation_failure_rate": float(
            obligation_failure.mean()
        ),
        "severe_drawdown_count": int(
            severe_drawdown.sum()
        ),
        "severe_drawdown_rate": float(
            severe_drawdown.mean()
        ),
        "non_zero_obligation_count": int(
            (
                outcome_df[
                    "required_obligations"
                ]
                > 0
            ).sum()
        ),
        "non_zero_obligation_rate": float(
            (
                outcome_df[
                    "required_obligations"
                ]
                > 0
            ).mean()
        ),
    }

    if verbose:

        print(
            "\n"
            + "=" * 70
        )

        print(
            "FINANCIAL STRESS TARGET AUDIT"
        )

        print(
            "=" * 70
        )

        print(
            f"Rows: "
            f"{result['n_rows']:,}"
        )

        print(
            f"Stress cases: "
            f"{result['stress_count']:,}"
        )

        print(
            f"Non-stress cases: "
            f"{result['non_stress_count']:,}"
        )

        print(
            f"Stress rate: "
            f"{result['stress_rate']:.2%}"
        )

        print(
            "\nStress components:"
        )

        print(
            f"Critical liquidity: "
            f"{result['critical_liquidity_count']:,} "
            f"({result['critical_liquidity_rate']:.2%})"
        )

        print(
            f"Obligation failure: "
            f"{result['obligation_failure_count']:,} "
            f"({result['obligation_failure_rate']:.2%})"
        )

        print(
            f"Severe drawdown: "
            f"{result['severe_drawdown_count']:,} "
            f"({result['severe_drawdown_rate']:.2%})"
        )

        print(
            "\nObligation representation:"
        )

        print(
            f"Non-zero obligation horizons: "
            f"{result['non_zero_obligation_count']:,} "
            f"({result['non_zero_obligation_rate']:.2%})"
        )

        print(
            "\nOutcome distributions:"
        )

        print(
            outcome_df[
                [
                    "prediction_balance",
                    "minimum_future_balance",
                    "future_net_cash_flow",
                    "required_obligations",
                    "obligation_coverage",
                ]
            ].describe()
        )

        print(
            "\nTarget counts:"
        )

        print(
            outcome_df[
                "financial_stress_30d"
            ]
            .value_counts()
            .sort_index()
        )

        print(
            "=" * 70
        )

    # Both classes must exist for classification.
    if (
        outcome_df[
            "financial_stress_30d"
        ].nunique()
        < 2
    ):

        raise AssertionError(
            "Stress target contains only one class."
        )

    return result