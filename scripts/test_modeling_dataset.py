"""
Validation script for the FinSight modeling dataset.
"""

from __future__ import annotations

import numpy as np

from finsight.data_generation.population import (
    generate_population,
)
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

from finsight.features.outcomes import (
    build_future_outcomes,
)

from finsight.models.dataset import (
    build_modeling_dataset,
    get_feature_columns,
    chronological_split,
    summarize_split,
)


SEED = 42
N_USERS = 100
N_MONTHS = 12
HORIZON_DAYS = 30


def build_dataset():

    rng = np.random.default_rng(
        SEED
    )

    population = generate_population(
        N_USERS,
        rng,
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

    outcome_df = build_future_outcomes(
        transactions=transactions_df,
        monthly_profile=monthly_df,
        horizon_days=HORIZON_DAYS,
    )

    return (
        feature_df,
        outcome_df,
    )


def main():

    print(
        "Building FinSight modeling dataset..."
    )

    feature_df, outcome_df = (
        build_dataset()
    )

    print(
        f"\nFeature observations: "
        f"{len(feature_df)}"
    )

    print(
        f"Outcome observations: "
        f"{len(outcome_df)}"
    )

    # -------------------------------------------------------------
    # Build modeling dataset
    # -------------------------------------------------------------

    model_df = build_modeling_dataset(
        feature_df,
        outcome_df,
    )

    print(
        "\n"
        + "=" * 70
    )

    print(
        "MODELING DATASET"
    )

    print(
        "=" * 70
    )

    print(
        f"Rows: "
        f"{len(model_df)}"
    )

    print(
        f"Columns: "
        f"{len(model_df.columns)}"
    )

    print(
        f"Stress cases: "
        f"{model_df['financial_stress_30d'].sum()}"
    )

    print(
        f"Non-stress cases: "
        f"{(
            model_df['financial_stress_30d'] == 0
        ).sum()}"
    )

    # -------------------------------------------------------------
    # Feature columns
    # -------------------------------------------------------------

    feature_columns = get_feature_columns(
        model_df
    )

    print(
        f"\nPredictor columns: "
        f"{len(feature_columns)}"
    )

    print(
        "\nPredictors:"
    )

    for column in feature_columns:
        print(
            f"  - {column}"
        )

    # -------------------------------------------------------------
    # Leakage checks
    # -------------------------------------------------------------

    forbidden = {
        "prediction_date",
        "horizon_end",
        "prediction_balance",
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
    }

    leakage_columns = (
        set(feature_columns)
        & forbidden
    )

    assert not leakage_columns, (
        "Future leakage detected in "
        f"predictors: {sorted(leakage_columns)}"
    )

    print(
        "\nNo future outcome columns "
        "are present in predictors."
    )

    # -------------------------------------------------------------
    # Chronological split
    # -------------------------------------------------------------

    (
        train_df,
        validation_df,
        test_df,
    ) = chronological_split(
        model_df
    )

    print(
        "\n"
        + "=" * 70
    )

    print(
        "CHRONOLOGICAL SPLIT"
    )

    print(
        "=" * 70
    )

    train_summary = summarize_split(
        "Train",
        train_df,
    )

    validation_summary = summarize_split(
        "Validation",
        validation_df,
    )

    test_summary = summarize_split(
        "Test",
        test_df,
    )

    # -------------------------------------------------------------
    # Temporal assertions
    # -------------------------------------------------------------

    assert (
        train_summary["max_month"]
        < validation_summary["min_month"]
    )

    assert (
        validation_summary["max_month"]
        < test_summary["min_month"]
    )

    print(
        "\nTemporal split validation passed."
    )

    # -------------------------------------------------------------
    # Target checks
    # -------------------------------------------------------------

    for name, split in [
        ("Train", train_df),
        ("Validation", validation_df),
        ("Test", test_df),
    ]:

        assert split[
            "financial_stress_30d"
        ].isin([0, 1]).all()

        assert not split[
            "financial_stress_30d"
        ].isna().any()

    # -------------------------------------------------------------
    # Feature matrix sanity
    # -------------------------------------------------------------

    X_train = train_df[
        feature_columns
    ]

    X_validation = validation_df[
        feature_columns
    ]

    X_test = test_df[
        feature_columns
    ]

    assert X_train.shape[1] == len(
        feature_columns
    )

    assert X_validation.shape[1] == len(
        feature_columns
    )

    assert X_test.shape[1] == len(
        feature_columns
    )

    # Check for infinite values.
    for name, X in [
        ("Train", X_train),
        ("Validation", X_validation),
        ("Test", X_test),
    ]:

        numeric = X.select_dtypes(
            include=[np.number]
        )

        infinite_count = np.isinf(
            numeric
        ).sum().sum()

        assert infinite_count == 0, (
            f"{name} contains "
            f"{infinite_count} infinite values."
        )

    print(
        "Feature matrix validation passed."
    )

    print(
        "\n"
        + "=" * 70
    )

    print(
        "MODELING DATASET VALIDATION PASSED"
    )

    print(
        "=" * 70
    )


if __name__ == "__main__":
    main()