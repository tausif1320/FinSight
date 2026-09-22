from __future__ import annotations

import pandas as pd

from finsight.inference.pipeline import (
    FinSightInferencePipeline,
)


DATA_PATH = (
    "data/processed/modeling_dataset_1000_users.csv"
)


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


def main() -> None:

    print("=" * 70)
    print("FinSight End-to-End Pipeline Test")
    print("=" * 70)

    df = pd.read_csv(DATA_PATH)

    print(f"Dataset rows: {len(df)}")

    pipeline = FinSightInferencePipeline(
        stress_model_path="models/stress_model.joblib",
        stress_metadata_path="models/feature_metadata.json",
        cate_model_path="models/cate_models.joblib",
        shap_background_path="models/shap_background.csv",
    )

    test_df = df[
        ["user_id", "month"] + FEATURE_COLUMNS
    ].copy()

    # Use a small but representative sample for integration testing.
    test_df = test_df.sample(
        n=min(20, len(test_df)),
        random_state=42,
    )

    print(
        f"Testing {len(test_df)} observations..."
    )

    results = pipeline.predict(
        test_df
    )

    required_columns = [
        "user_id",
        "month",
        "risk_probability",
        "risk_level",
        "financial_pressure",
        "intervention",
        "intervention_priority",
        "cate",
        "recommendation_status",
        "explanation",
        "model_drivers",
    ]

    missing = [
        column
        for column in required_columns
        if column not in results.columns
    ]

    assert not missing, (
        f"Missing output columns: {missing}"
    )

    # ------------------------------------------------------------------
    # Validate probability
    # ------------------------------------------------------------------

    assert results[
        "risk_probability"
    ].between(0, 1).all()

    # ------------------------------------------------------------------
    # Validate pressure -> intervention
    # ------------------------------------------------------------------

    expected_mapping = {
        "STABLE": "SAVINGS_REINFORCEMENT",
        "LIQUIDITY_PRESSURE": "LIQUIDITY_PROTECTION",
        "SPENDING_PRESSURE": "SPENDING_CONTROL",
        "OBLIGATION_PRESSURE": "OBLIGATION_MANAGEMENT",
    }

    for _, row in results.iterrows():

        pressure = row[
            "financial_pressure"
        ]

        intervention = row[
            "intervention"
        ]

        assert (
            intervention
            == expected_mapping[pressure]
        ), (
            f"Invalid intervention mapping: "
            f"{pressure} -> {intervention}"
        )

    # ------------------------------------------------------------------
    # Validate CATE behavior
    # ------------------------------------------------------------------

    for _, row in results.iterrows():

        pressure = row[
            "financial_pressure"
        ]

        cate = row["cate"]

        if pressure == "STABLE":

            assert pd.isna(cate), (
                "Stable user should not have "
                "an intervention-specific CATE."
            )

        else:

            assert pd.notna(cate), (
                f"Missing CATE for "
                f"{pressure}"
            )

    # ------------------------------------------------------------------
    # Validate SHAP model drivers
    # ------------------------------------------------------------------

    for _, row in results.iterrows():

        drivers = row[
            "model_drivers"
        ]

        assert isinstance(
            drivers,
            str,
        )

        assert len(
            drivers.strip()
        ) > 0, (
            "model_drivers is empty."
        )

    # ------------------------------------------------------------------
    # Print results
    # ------------------------------------------------------------------

    print()
    print("Risk levels:")
    print(
        results[
            "risk_level"
        ].value_counts()
    )

    print()
    print("Financial pressure:")
    print(
        results[
            "financial_pressure"
        ].value_counts()
    )

    print()
    print("Interventions:")
    print(
        results[
            "intervention"
        ].value_counts()
    )

    print()
    print("Sample result:")
    print(
        results.iloc[0].to_dict()
    )

    print()
    print("=" * 70)
    print("END-TO-END PIPELINE TEST PASSED")
    print("=" * 70)


if __name__ == "__main__":
    main()