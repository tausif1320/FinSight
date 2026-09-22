from __future__ import annotations

import numpy as np

from finsight.data_generation.population import generate_population
from finsight.data_generation.income import (
    generate_base_income,
    generate_salary_days,
    generate_monthly_income,
)
from finsight.data_generation.obligations import generate_obligations
from finsight.data_generation.spending import generate_spending_profile
from finsight.data_generation.balance import generate_initial_balances
from finsight.data_generation.behavior import generate_behavioral_profile
from finsight.data_generation.shocks import generate_shocks
from finsight.data_generation.transactions import generate_transactions

from finsight.features.monthly_profile import build_monthly_profile
from finsight.features.financial_state import assign_financial_states
from finsight.features.behavioral_features import build_behavioral_features
from finsight.features.outcomes import build_future_outcomes

from finsight.models.dataset import (
    build_modeling_dataset,
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


SEED = 42
N_USERS = 10000
N_MONTHS = 12
HORIZON_DAYS = 30

TARGET = "financial_stress_30d"


def build_dataset():
    """
    Build the exact same validated FinSight dataset
    used by test_modeling_dataset.py and test_baselines.py.
    """

    rng = np.random.default_rng(SEED)

    # --------------------------------------------------
    # POPULATION
    # --------------------------------------------------

    population = generate_population(
        N_USERS,
        rng,
    )

    # --------------------------------------------------
    # INCOME
    # --------------------------------------------------

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

    # --------------------------------------------------
    # OBLIGATIONS
    # --------------------------------------------------

    obligations_df = generate_obligations(
        population,
        rng,
    )

    # --------------------------------------------------
    # SPENDING
    # --------------------------------------------------

    spending_df = generate_spending_profile(
        population,
        rng,
    )

    # --------------------------------------------------
    # INITIAL BALANCES
    # --------------------------------------------------

    initial_balance_df = generate_initial_balances(
        population,
        income_df,
        rng,
    )

    # --------------------------------------------------
    # BEHAVIOR
    # --------------------------------------------------

    behavior_df = generate_behavioral_profile(
        population,
        N_MONTHS,
        rng,
    )

    # --------------------------------------------------
    # SHOCKS
    # --------------------------------------------------

    shock_df = generate_shocks(
        population,
        N_MONTHS,
        rng,
    )

    # --------------------------------------------------
    # TRANSACTIONS
    # --------------------------------------------------

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

    print(f"Transactions: {len(transactions_df):,}")

    # --------------------------------------------------
    # MONTHLY PROFILE
    # --------------------------------------------------

    monthly_df = build_monthly_profile(
        transactions_df
    )

    print(f"Monthly observations: {len(monthly_df):,}")

    # --------------------------------------------------
    # FINANCIAL STATES
    # --------------------------------------------------

    state_df = assign_financial_states(
        monthly_df
    )

    # --------------------------------------------------
    # BEHAVIORAL FEATURES
    # --------------------------------------------------

    feature_df = build_behavioral_features(
        monthly_profile=monthly_df,
        state_df=state_df,
        transactions_df=transactions_df,
        window_months=3,
    )

    print(f"Feature observations: {len(feature_df):,}")

    # --------------------------------------------------
    # FUTURE OUTCOMES
    # --------------------------------------------------

    outcome_df = build_future_outcomes(
        transactions=transactions_df,
        monthly_profile=monthly_df,
        horizon_days=HORIZON_DAYS,
    )

    print(f"Outcome observations: {len(outcome_df):,}")

    return feature_df, outcome_df


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

    print("Building FinSight static vs dynamic dataset...")

    feature_df, outcome_df = build_dataset()

    # --------------------------------------------------
    # MODELING DATASET
    # --------------------------------------------------

    model_df = build_modeling_dataset(
        feature_df,
        outcome_df,
        drop_incomplete_history=True,
    )

    all_features = get_feature_columns(model_df)

    print()
    print("=" * 80)
    print("MODELING DATASET")
    print("=" * 80)

    print(f"Rows: {len(model_df):,}")
    print(
        f"Stress cases: "
        f"{int(model_df[TARGET].sum()):,}"
    )
    print(
        f"Stress rate: "
        f"{model_df[TARGET].mean():.4f}"
    )
    print(
        f"Total predictors: "
        f"{len(all_features)}"
    )

    # --------------------------------------------------
    # FEATURE SETS
    # --------------------------------------------------

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
        f"{len(train_df)} rows | "
        f"{int(train_df[TARGET].sum())} stress"
    )

    print(
        f"Validation: "
        f"{len(validation_df)} rows | "
        f"{int(validation_df[TARGET].sum())} stress"
    )

    print(
        f"Test: "
        f"{len(test_df)} rows | "
        f"{int(test_df[TARGET].sum())} stress"
    )

    # --------------------------------------------------
    # STATIC VS DYNAMIC
    # --------------------------------------------------

    print()
    print("=" * 80)
    print("STATIC VS DYNAMIC EXPERIMENT")
    print("=" * 80)

    validation_results, test_results = compare_feature_sets(
        train_df=train_df,
        validation_df=validation_df,
        test_df=test_df,
        static_features=static_features,
        dynamic_features=dynamic_features,
        target_column=TARGET,
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
    # DELTA
    # --------------------------------------------------

    static_test = test_results[
        test_results["model"] == "Static"
    ].iloc[0]

    dynamic_test = test_results[
        test_results["model"] == "Static + Dynamic"
    ].iloc[0]

    print()
    print("=" * 80)
    print("DYNAMIC FEATURE IMPROVEMENT")
    print("=" * 80)

    pr_auc_delta = (
        dynamic_test["pr_auc"]
        - static_test["pr_auc"]
    )

    roc_auc_delta = (
        dynamic_test["roc_auc"]
        - static_test["roc_auc"]
    )

    brier_delta = (
        dynamic_test["brier_score"]
        - static_test["brier_score"]
    )

    recall_delta = (
        dynamic_test["recall"]
        - static_test["recall"]
    )

    f1_delta = (
        dynamic_test["f1"]
        - static_test["f1"]
    )

    print(
        f"PR-AUC improvement: "
        f"{pr_auc_delta:+.4f}"
    )

    print(
        f"ROC-AUC improvement: "
        f"{roc_auc_delta:+.4f}"
    )

    print(
        f"Recall improvement: "
        f"{recall_delta:+.4f}"
    )

    print(
        f"F1 improvement: "
        f"{f1_delta:+.4f}"
    )

    print(
        f"Brier score change: "
        f"{brier_delta:+.4f}"
    )

    # --------------------------------------------------
    # BASIC VALIDATION
    # --------------------------------------------------

    assert len(static_features) > 0
    assert len(dynamic_features) > 0

    assert set(static_features).issubset(
        set(all_features)
    )

    assert set(dynamic_features).issubset(
        set(all_features)
    )

    assert not set(dynamic_features).intersection(
        {
            "financial_stress_30d",
            "minimum_future_balance",
            "ending_future_balance",
            "future_net_cash_flow",
            "future_income",
            "future_expenses",
            "required_obligations",
            "obligation_coverage",
        }
    )

    assert len(validation_results) == 2
    assert len(test_results) == 2

    assert test_results["pr_auc"].notna().all()
    assert test_results["roc_auc"].notna().all()
    assert test_results["brier_score"].notna().all()

    print()
    print("=" * 80)
    print("STATIC VS DYNAMIC VALIDATION PASSED")
    print("=" * 80)


if __name__ == "__main__":
    main()