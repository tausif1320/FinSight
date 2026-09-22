"""
Baseline model validation for FinSight.

Uses the exact same validated data-generation pipeline
as test_modeling_dataset.py.

Models:
    1. Dummy baseline
    2. Logistic Regression baseline

Evaluation:
    - PR-AUC
    - ROC-AUC
    - Precision
    - Recall
    - F1
    - Brier score
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
)

from finsight.models.baselines import (
    build_dummy_baseline,
    build_logistic_regression,
    prepare_xy,
)

from finsight.models.model_evaluation import (
    evaluate_binary_classifier,
    print_metrics,
    compare_models,
)


# ================================================================
# CONFIGURATION
# ================================================================

SEED = 42
N_USERS = 100
N_MONTHS = 12
HORIZON_DAYS = 30

TARGET = "financial_stress_30d"


# ================================================================
# DATASET
# ================================================================

def build_dataset():

    rng = np.random.default_rng(
        SEED
    )

    # ------------------------------------------------------------
    # Population
    # ------------------------------------------------------------

    population = generate_population(
        N_USERS,
        rng,
    )

    # ------------------------------------------------------------
    # Income
    # ------------------------------------------------------------

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

    # ------------------------------------------------------------
    # Obligations
    # ------------------------------------------------------------

    obligations_df = generate_obligations(
        population,
        rng,
    )

    # ------------------------------------------------------------
    # Spending
    # ------------------------------------------------------------

    spending_df = generate_spending_profile(
        population,
        rng,
    )

    # ------------------------------------------------------------
    # Initial balance
    # ------------------------------------------------------------

    initial_balance_df = generate_initial_balances(
        population,
        income_df,
        rng,
    )

    # ------------------------------------------------------------
    # Behavioral regime
    # ------------------------------------------------------------

    behavior_df = generate_behavioral_profile(
        population,
        N_MONTHS,
        rng,
    )

    # ------------------------------------------------------------
    # Financial shocks
    # ------------------------------------------------------------

    shock_df = generate_shocks(
        population,
        N_MONTHS,
        rng,
    )

    # ------------------------------------------------------------
    # Transactions
    # ------------------------------------------------------------

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

    print(
        f"Transactions: {len(transactions_df)}"
    )

    # ------------------------------------------------------------
    # Monthly profile
    # EXACT SAME CALL AS VALIDATED SCRIPT
    # ------------------------------------------------------------

    monthly_df = build_monthly_profile(
        transactions_df
    )

    print(
        f"Monthly observations: {len(monthly_df)}"
    )

    # ------------------------------------------------------------
    # Financial states
    # EXACT SAME CALL AS VALIDATED SCRIPT
    # ------------------------------------------------------------

    state_df = assign_financial_states(
        monthly_df
    )

    # ------------------------------------------------------------
    # Behavioral features
    # EXACT SAME CALL AS VALIDATED SCRIPT
    # ------------------------------------------------------------

    feature_df = build_behavioral_features(
        monthly_profile=monthly_df,
        state_df=state_df,
        transactions_df=transactions_df,
        window_months=3,
    )

    print(
        f"Feature observations: {len(feature_df)}"
    )

    # ------------------------------------------------------------
    # Future outcomes
    # EXACT SAME CALL AS VALIDATED SCRIPT
    # ------------------------------------------------------------

    outcome_df = build_future_outcomes(
        transactions=transactions_df,
        monthly_profile=monthly_df,
        horizon_days=HORIZON_DAYS,
    )

    print(
        f"Outcome observations: {len(outcome_df)}"
    )

    return (
        feature_df,
        outcome_df,
    )


# ================================================================
# MAIN
# ================================================================

def main():

    print(
        "Building FinSight baseline dataset..."
    )

    feature_df, outcome_df = build_dataset()

    # ------------------------------------------------------------
    # Modeling dataset
    # ------------------------------------------------------------

    model_df = build_modeling_dataset(
        feature_df,
        outcome_df,
    )

    print()
    print("=" * 70)
    print("MODELING DATASET")
    print("=" * 70)

    print(
        f"Rows: {len(model_df)}"
    )

    stress_cases = int(
        model_df[TARGET].sum()
    )

    print(
        f"Stress cases: {stress_cases}"
    )

    print(
        f"Non-stress cases: "
        f"{len(model_df) - stress_cases}"
    )

    print(
        f"Stress rate: "
        f"{model_df[TARGET].mean():.4f}"
    )

    # ------------------------------------------------------------
    # Feature selection
    # ------------------------------------------------------------

    feature_columns = get_feature_columns(
        model_df
    )

    print(
        f"Predictor columns: "
        f"{len(feature_columns)}"
    )

    # ------------------------------------------------------------
    # Chronological split
    # ------------------------------------------------------------

    (
        train_df,
        validation_df,
        test_df,
    ) = chronological_split(
        model_df
    )

    print()
    print("=" * 70)
    print("CHRONOLOGICAL SPLIT")
    print("=" * 70)

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

    # ------------------------------------------------------------
    # Prepare X / y
    # ------------------------------------------------------------

    X_train, y_train = prepare_xy(
        train_df,
        feature_columns,
        TARGET,
    )

    X_validation, y_validation = prepare_xy(
        validation_df,
        feature_columns,
        TARGET,
    )

    X_test, y_test = prepare_xy(
        test_df,
        feature_columns,
        TARGET,
    )

    # ------------------------------------------------------------
    # Baseline models
    # ------------------------------------------------------------

    models = {
        "Dummy Baseline": build_dummy_baseline(),
        "Logistic Regression": build_logistic_regression(),
    }

    results = {}

    # ------------------------------------------------------------
    # Train and evaluate
    # ------------------------------------------------------------

    for name, model in models.items():

        print()
        print(
            "=" * 70
        )

        print(
            f"TRAINING: {name}"
        )

        print(
            "=" * 70
        )

        model.fit(
            X_train,
            y_train,
        )

        # --------------------------------------------------------
        # Validation
        # --------------------------------------------------------

        validation_probability = (
            model.predict_proba(
                X_validation
            )[:, 1]
        )

        validation_metrics = (
            evaluate_binary_classifier(
                y_validation,
                validation_probability,
            )
        )

        print_metrics(
            f"{name} - Validation",
            validation_metrics,
        )

        # --------------------------------------------------------
        # Test
        # --------------------------------------------------------

        test_probability = (
            model.predict_proba(
                X_test
            )[:, 1]
        )

        test_metrics = (
            evaluate_binary_classifier(
                y_test,
                test_probability,
            )
        )

        print_metrics(
            f"{name} - Test",
            test_metrics,
        )

        results[name] = test_metrics

    # ------------------------------------------------------------
    # Model comparison
    # ------------------------------------------------------------

    comparison = compare_models(
        results
    )

    print()
    print("=" * 70)
    print("TEST SET MODEL COMPARISON")
    print("=" * 70)

    print(
        comparison.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    # ------------------------------------------------------------
    # Sanity checks
    # ------------------------------------------------------------

    assert len(X_train) == len(y_train)

    assert len(X_validation) == len(
        y_validation
    )

    assert len(X_test) == len(
        y_test
    )

    # No infinite values.
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

    print()
    print("=" * 70)
    print("BASELINE MODEL VALIDATION PASSED")
    print("=" * 70)


if __name__ == "__main__":
    main()