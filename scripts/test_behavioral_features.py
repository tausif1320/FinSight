import numpy as np
import pandas as pd

from finsight.data_generation.population import generate_population
from finsight.data_generation.income import (
    generate_base_income,
    generate_salary_days,
    generate_monthly_income,
)
from finsight.data_generation.obligations import (
    generate_obligations,
)
from finsight.data_generation.spending import (
    generate_spending_profile,
)
from finsight.data_generation.balance import (
    generate_initial_balances,
)
from finsight.data_generation.behavior import (
    generate_behavioral_profile,
)
from finsight.data_generation.shocks import (
    generate_shocks,
)
from finsight.data_generation.transactions import (
    generate_transactions,
)

from finsight.features.monthly_profile import (
    build_monthly_profile,
)

from finsight.features.financial_state import (
    assign_financial_states,
)

from finsight.features.behavioral_features import (
    build_behavioral_features,
)


SEED = 42
N_USERS = 100
N_MONTHS = 12


# ============================================================
# DATASET
# ============================================================

def build_dataset():

    rng = np.random.default_rng(SEED)

    population = generate_population(
        n_users=N_USERS,
        rng=rng,
    )

    population = generate_base_income(
        population,
        rng,
    )

    population = generate_salary_days(
        population,
        rng,
    )

    income_df = generate_monthly_income(
        population,
        N_MONTHS,
        rng,
    )

    obligations_df = generate_obligations(
        population,
        rng,
    )

    spending_df = generate_spending_profile(
        population,
        rng,
    )

    initial_balance_df = generate_initial_balances(
        population,
        income_df,
        rng,
    )

    behavior_df = generate_behavioral_profile(
        population,
        N_MONTHS,
        rng,
    )

    shock_df = generate_shocks(
        population,
        N_MONTHS,
        rng,
    )

    transactions_df = generate_transactions(
        population=population,
        income_df=income_df,
        obligations_df=obligations_df,
        spending_df=spending_df,
        initial_balance_df=initial_balance_df,
        rng=rng,
        n_months=N_MONTHS,
        behavior_df=behavior_df,
        shock_df=shock_df,
    )

    monthly_df = build_monthly_profile(
        transactions_df
    )

    state_df = assign_financial_states(
        monthly_df
    )

    feature_df = build_behavioral_features(
        monthly_profile=monthly_df,
        state_df=state_df,
        transactions_df=transactions_df,
        window_months=3,
    )

    return (
        transactions_df,
        monthly_df,
        state_df,
        feature_df,
    )


# ============================================================
# TEST 1
# SCHEMA
# ============================================================

def test_schema(feature_df):

    print("\n" + "=" * 70)
    print("1. FEATURE SCHEMA")
    print("=" * 70)

    print(
        f"\nRows: {len(feature_df)}"
    )

    print(
        f"Columns: {len(feature_df.columns)}"
    )

    print("\nColumns:")

    for column in feature_df.columns:
        print(
            f"  - {column}"
        )

    required = {
        "user_id",
        "month",
        "financial_state",
        "state_duration",
        "rolling_spending_mean",
        "rolling_spending_std",
        "rolling_income_mean",
        "rolling_average_balance",
        "rolling_savings_rate",
        "spending_growth",
        "discretionary_spending_growth",
        "income_growth",
        "balance_growth",
        "spending_acceleration",
        "spending_baseline_deviation",
        "spending_baseline_ratio",
        "category_shift",
        "cash_buffer_ratio",
        "expense_pressure",
        "discretionary_pressure",
        "negative_cash_flow",
        "has_full_history",
    }

    missing = (
        required
        - set(feature_df.columns)
    )

    assert not missing, (
        "Missing expected behavioral features: "
        f"{sorted(missing)}"
    )

    print(
        "\nFeature schema validation passed."
    )


# ============================================================
# TEST 2
# USER-MONTH UNIQUENESS
# ============================================================

def test_uniqueness(feature_df):

    print("\n" + "=" * 70)
    print("2. USER-MONTH UNIQUENESS")
    print("=" * 70)

    duplicates = feature_df.duplicated(
        subset=["user_id", "month"],
        keep=False,
    )

    assert not duplicates.any(), (
        "Duplicate user-month feature observations found."
    )

    expected_rows = (
        N_USERS * N_MONTHS
    )

    assert len(feature_df) == expected_rows, (
        f"Expected {expected_rows} rows, "
        f"got {len(feature_df)}."
    )

    print(
        f"Expected rows: {expected_rows}"
    )

    print(
        f"Actual rows:   {len(feature_df)}"
    )

    print(
        "\nUser-month uniqueness validation passed."
    )


# ============================================================
# TEST 3
# HISTORY
# ============================================================

def test_history(feature_df):

    print("\n" + "=" * 70)
    print("3. TEMPORAL HISTORY")
    print("=" * 70)

    first_three = (
        feature_df
        .sort_values(["user_id", "month"])
        .groupby("user_id")
        .head(3)
    )

    # First two months cannot have complete 3-month history.
    first_two = (
        feature_df
        .sort_values(["user_id", "month"])
        .groupby("user_id")
        .head(2)
    )

    third_month = (
        feature_df
        .sort_values(["user_id", "month"])
        .groupby("user_id")
        .nth(2)
        .reset_index()
    )

    assert not first_two[
        "has_full_history"
    ].any(), (
        "First two months incorrectly marked "
        "as having full history."
    )

    assert third_month[
        "has_full_history"
    ].all(), (
        "Third month should have full 3-month history."
    )

    print(
        "First two months correctly marked "
        "as incomplete history."
    )

    print(
        "Third month correctly marked "
        "as complete history."
    )

    print(
        "\nTemporal history validation passed."
    )


# ============================================================
# TEST 4
# NO FUTURE LEAKAGE
# ============================================================

def test_no_future_leakage(
    monthly_df,
    feature_df,
):

    print("\n" + "=" * 70)
    print("4. FUTURE-LEAKAGE CHECK")
    print("=" * 70)

    monthly = monthly_df.copy()

    monthly["month"] = pd.to_datetime(
        monthly["month"],
        format="%Y-%m",
    )

    features = feature_df.copy()

    features["month"] = pd.to_datetime(
        features["month"]
    )

    # For every feature row, verify that rolling features
    # can be reproduced using data up to and including
    # the current month only.

    merged = features[
        [
            "user_id",
            "month",
            "rolling_spending_mean",
        ]
    ].merge(
        monthly[
            [
                "user_id",
                "month",
                "variable_spending",
            ]
        ],
        on=["user_id", "month"],
        how="left",
        validate="one_to_one",
    )

    assert merged["variable_spending"].notna().all()

    # Explicitly test the first three months.
    for user_id, group in monthly.groupby(
        "user_id"
    ):

        group = group.sort_values(
            "month"
        )

        feature_group = features[
            features["user_id"]
            == user_id
        ].sort_values("month")

        for i in range(len(group)):

            current_month = group.iloc[i][
                "month"
            ]

            feature_row = feature_group.iloc[i]

            if i < 2:
                assert pd.isna(
                    feature_row[
                        "rolling_spending_mean"
                    ]
                ), (
                    "Rolling feature exists before "
                    "enough historical observations."
                )

            else:
                historical_window = group.iloc[
                    i - 2 : i + 1
                ][
                    "variable_spending"
                ]

                expected = (
                    historical_window.mean()
                )

                actual = feature_row[
                    "rolling_spending_mean"
                ]

                assert np.isclose(
                    actual,
                    expected,
                    rtol=1e-8,
                    atol=1e-8,
                    equal_nan=True,
                ), (
                    f"Rolling spending mismatch for "
                    f"{user_id}, {current_month}."
                )

    print(
        "Rolling features verified using only "
        "current/past observations."
    )

    print(
        "\nFuture-leakage validation passed."
    )


# ============================================================
# TEST 5
# BASELINE LEAKAGE
# ============================================================

def test_personal_baseline_leakage(
    monthly_df,
    feature_df,
):

    print("\n" + "=" * 70)
    print("5. PERSONAL BASELINE LEAKAGE")
    print("=" * 70)

    monthly = monthly_df.copy()

    monthly["month"] = pd.to_datetime(
        monthly["month"],
        format="%Y-%m",
    )

    features = feature_df.copy()

    features["month"] = pd.to_datetime(
        features["month"]
    )

    for user_id, group in monthly.groupby(
        "user_id"
    ):

        group = group.sort_values(
            "month"
        ).reset_index(drop=True)

        feature_group = features[
            features["user_id"]
            == user_id
        ].sort_values("month").reset_index(
            drop=True
        )

        for i in range(len(group)):

            actual = feature_group.iloc[i][
                "historical_spending_mean"
            ]

            if i == 0:

                assert pd.isna(actual), (
                    "First observation has a "
                    "historical baseline."
                )

            else:

                expected = group.iloc[
                    :i
                ][
                    "variable_spending"
                ].mean()

                assert np.isclose(
                    actual,
                    expected,
                    rtol=1e-8,
                    atol=1e-8,
                    equal_nan=True,
                ), (
                    f"Historical baseline leakage/mismatch "
                    f"for {user_id}, index {i}."
                )

    print(
        "Personal baselines use only observations "
        "strictly before the current month."
    )

    print(
        "\nPersonal baseline leakage validation passed."
    )


# ============================================================
# TEST 6
# FEATURE SANITY
# ============================================================

def test_feature_sanity(feature_df):

    print("\n" + "=" * 70)
    print("6. FEATURE SANITY")
    print("=" * 70)

    numeric_columns = feature_df.select_dtypes(
        include=np.number
    ).columns

    infinite_values = np.isinf(
        feature_df[numeric_columns]
    ).sum().sum()

    assert infinite_values == 0, (
        "Infinite numeric feature values detected."
    )

    print(
        "No infinite numeric values."
    )

    print(
        "\nMissing values by selected feature:"
    )

    selected = [
        "rolling_spending_mean",
        "rolling_spending_std",
        "rolling_income_mean",
        "spending_growth",
        "income_growth",
        "category_shift",
        "cash_buffer_ratio",
        "spending_baseline_ratio",
    ]

    print(
        feature_df[selected]
        .isna()
        .sum()
    )

    print(
        "\nFeature sanity validation passed."
    )


# ============================================================
# TEST 7
# FEATURE SAMPLE
# ============================================================

def print_feature_sample(feature_df):

    print("\n" + "=" * 70)
    print("7. FEATURE SAMPLE")
    print("=" * 70)

    columns = [
        "user_id",
        "month",
        "financial_state",
        "state_duration",
        "rolling_spending_mean",
        "rolling_spending_std",
        "spending_growth",
        "spending_acceleration",
        "spending_baseline_deviation",
        "category_shift",
        "cash_buffer_ratio",
        "expense_pressure",
        "negative_cash_flow",
        "has_full_history",
    ]

    print(
        feature_df[columns]
        .head(15)
        .to_string(index=False)
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print(
        "Generating FinSight dataset..."
    )

    (
        transactions_df,
        monthly_df,
        state_df,
        feature_df,
    ) = build_dataset()

    print(
        f"Transactions: "
        f"{len(transactions_df)}"
    )

    print(
        f"Monthly observations: "
        f"{len(monthly_df)}"
    )

    print(
        f"Feature observations: "
        f"{len(feature_df)}"
    )

    test_schema(
        feature_df
    )

    test_uniqueness(
        feature_df
    )

    test_history(
        feature_df
    )

    test_no_future_leakage(
        monthly_df,
        feature_df,
    )

    test_personal_baseline_leakage(
        monthly_df,
        feature_df,
    )

    test_feature_sanity(
        feature_df
    )

    print_feature_sample(
        feature_df
    )

    print("\n" + "=" * 70)
    print(
        "ALL BEHAVIORAL FEATURE VALIDATIONS PASSED"
    )
    print("=" * 70)


if __name__ == "__main__":
    main()