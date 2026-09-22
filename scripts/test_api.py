from __future__ import annotations

from pathlib import Path

import pandas as pd
import requests


ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = (
    ROOT
    / "data"
    / "processed"
    / "modeling_dataset_1000_users.csv"
)

API_URL = (
    "http://127.0.0.1:8000/predict"
)


# =============================================================================
# PRODUCTION FEATURE SET
# =============================================================================

FEATURE_COLUMNS = [
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


# =============================================================================
# MAIN TEST
# =============================================================================

def main():

    print("=" * 80)
    print("FinSight Production API Test")
    print("=" * 80)

    # -------------------------------------------------------------------------
    # 1. LOAD DATA
    # -------------------------------------------------------------------------

    print(
        "\nLoading modeling dataset..."
    )

    df = pd.read_csv(
        DATA_PATH
    )

    print(
        f"Dataset rows: {len(df):,}"
    )

    # -------------------------------------------------------------------------
    # 2. SELECT A REAL OBSERVATION
    # -------------------------------------------------------------------------

    row = df.iloc[9]

    print(
        f"Testing user_id: {row['user_id']}"
    )

    print(
        f"Testing month: {row['month']}"
    )

    # -------------------------------------------------------------------------
    # 3. VALIDATE REQUIRED FEATURES
    # -------------------------------------------------------------------------

    missing_features = [
        column
        for column in FEATURE_COLUMNS
        if column not in df.columns
    ]

    if missing_features:

        raise ValueError(
            "Missing production features:\n"
            + "\n".join(
                missing_features
            )
        )

    # -------------------------------------------------------------------------
    # 4. BUILD API PAYLOAD
    # -------------------------------------------------------------------------

    payload = {
        "user_id": int(
            row["user_id"]
        ),

        "month": str(
            row["month"]
        ),
    }

    for column in FEATURE_COLUMNS:

        value = row[column]

        # Convert NumPy scalar values into normal
        # Python values so requests/json can serialize them.

        if pd.isna(value):
            value = 0.0

        else:
            value = float(value)

        payload[column] = value

    # -------------------------------------------------------------------------
    # 5. SEND REQUEST
    # -------------------------------------------------------------------------

    print(
        "\nSending prediction request..."
    )

    response = requests.post(
        API_URL,
        json=payload,
        timeout=30,
    )

    print(
        f"HTTP status: "
        f"{response.status_code}"
    )

    if response.status_code != 200:

        print(
            "\nAPI response:"
        )

        print(
            response.text
        )

        raise RuntimeError(
            "Prediction API request failed."
        )

    result = response.json()

    # -------------------------------------------------------------------------
    # 6. DISPLAY RESULT
    # -------------------------------------------------------------------------

    print(
        "\n" + "=" * 80
    )

    print(
        "PREDICTION RESULT"
    )

    print(
        "=" * 80
    )

    for key, value in result.items():

        print(
            f"{key}: {value}"
        )

    # -------------------------------------------------------------------------
    # 7. VALIDATE RESPONSE
    # -------------------------------------------------------------------------

    assert (
        result["user_id"]
        == int(row["user_id"])
    )

    assert (
        result["month"]
        == str(row["month"])
    )

    assert (
        0.0
        <= result["risk_probability"]
        <= 1.0
    )

    assert result[
        "risk_level"
    ] in {
        "LOW",
        "MODERATE",
        "HIGH",
    }

    assert result[
        "financial_pressure"
    ] in {
        "STABLE",
        "LIQUIDITY_PRESSURE",
        "SPENDING_PRESSURE",
        "OBLIGATION_PRESSURE",
    }

    assert result[
        "intervention"
    ] in {
        "SAVINGS_REINFORCEMENT",
        "LIQUIDITY_PROTECTION",
        "SPENDING_CONTROL",
        "OBLIGATION_MANAGEMENT",
    }

    assert isinstance(
        result[
            "intervention_priority"
        ],
        int,
    )

    assert result[
        "recommendation_status"
    ] in {
        "MONITOR",
        "STRONG_SIMULATED_SUPPORT",
        "SIMULATED_SUPPORT",
        "RULE_BASED_RECOMMENDATION",
    }

    # -------------------------------------------------------------------------
    # 8. VALIDATE PRESSURE -> INTERVENTION MAPPING
    # -------------------------------------------------------------------------

    expected_mapping = {
        "STABLE":
            "SAVINGS_REINFORCEMENT",

        "LIQUIDITY_PRESSURE":
            "LIQUIDITY_PROTECTION",

        "SPENDING_PRESSURE":
            "SPENDING_CONTROL",

        "OBLIGATION_PRESSURE":
            "OBLIGATION_MANAGEMENT",
    }

    expected_intervention = (
        expected_mapping[
            result[
                "financial_pressure"
            ]
        ]
    )

    assert (
        result["intervention"]
        == expected_intervention
    ), (
        "Pressure/intervention mismatch: "
        f"{result['financial_pressure']} "
        f"-> {result['intervention']}, "
        f"expected {expected_intervention}"
    )

    # -------------------------------------------------------------------------
    # 9. VALIDATE CATE LOGIC
    # -------------------------------------------------------------------------

    if (
        result["financial_pressure"]
        == "STABLE"
    ):

        assert (
            result["cate"]
            is None
        )

    else:

        assert (
            result["cate"]
            is not None
        )

        assert isinstance(
            result["cate"],
            float,
        )

    # -------------------------------------------------------------------------
    # 10. SUCCESS
    # -------------------------------------------------------------------------

    print(
        "\n" + "=" * 80
    )

    print(
        "✓ API response valid"
    )

    print(
        "✓ Risk probability valid"
    )

    print(
        "✓ Risk level valid"
    )

    print(
        "✓ Financial pressure valid"
    )

    print(
        "✓ Intervention valid"
    )

    print(
        "✓ Pressure/intervention mapping valid"
    )

    print(
        "✓ CATE logic valid"
    )

    print(
        "=" * 80
    )

    print(
        "FINDSIGHT API TEST PASSED"
    )

    print(
        "=" * 80
    )


if __name__ == "__main__":
    main()