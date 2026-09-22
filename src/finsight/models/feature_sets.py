from __future__ import annotations

from typing import List


STATIC_FEATURES: List[str] = [
    "salary_income",
    "bonus_income",
    "total_income",
    "total_debits",
    "variable_spending",
    "essential_spending",
    "discretionary_spending",
    "recurring_spending",
    "transaction_count",
    "average_transaction",
    "starting_balance",
    "ending_balance",
    "minimum_balance",
    "maximum_balance",
    "average_balance",
    "net_cash_flow",
    "savings_rate",
    "income_to_expense_ratio",
    "essential_spending_ratio",
    "discretionary_spending_ratio",
    "minimum_balance_ratio",
    "average_balance_ratio",
]


DYNAMIC_FEATURES: List[str] = [
    "state_changed",
    "state_duration",
    "recent_state_change",
    "historical_spending_mean",
    "historical_spending_std",
    "spending_baseline_deviation",
    "spending_baseline_ratio",
    "transaction_count_change",
    "savings_rate_change",
    "rolling_spending_mean",
    "rolling_spending_std",
    "rolling_spending_min",
    "rolling_spending_max",
    "rolling_income_mean",
    "income_volatility",
    "rolling_average_balance",
    "rolling_minimum_balance",
    "rolling_savings_rate",
    "rolling_transaction_count",
    "rolling_net_cash_flow",
    "spending_growth",
    "discretionary_spending_growth",
    "income_growth",
    "balance_growth",
    "spending_acceleration",
    "cash_buffer",
    "cash_buffer_ratio",
    "expense_pressure",
    "discretionary_pressure",
    "negative_cash_flow",
    "category_shift",
]


def get_static_features(available_features: List[str]) -> List[str]:
    """
    Return static features that are actually available
    in the modeling dataset.
    """
    available = set(available_features)

    missing = [
        feature
        for feature in STATIC_FEATURES
        if feature not in available
    ]

    if missing:
        raise ValueError(
            "Missing static features:\n"
            + "\n".join(f"  - {feature}" for feature in missing)
        )

    return STATIC_FEATURES.copy()


def get_dynamic_features(available_features: List[str]) -> List[str]:
    """
    Return dynamic behavioral features that are actually available
    in the modeling dataset.
    """
    available = set(available_features)

    missing = [
        feature
        for feature in DYNAMIC_FEATURES
        if feature not in available
    ]

    if missing:
        raise ValueError(
            "Missing dynamic features:\n"
            + "\n".join(f"  - {feature}" for feature in missing)
        )

    return DYNAMIC_FEATURES.copy()


def get_static_plus_dynamic_features(
    available_features: List[str],
) -> List[str]:
    """
    Return the complete feature set:
    static financial features + dynamic behavioral features.
    """
    static = get_static_features(available_features)
    dynamic = get_dynamic_features(available_features)

    return static + dynamic