import numpy as np
import pandas as pd

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

from finsight.features.financial_state import (
    assign_financial_states,
    FINANCIAL_STATES,
)


SEED = 42
N_USERS = 100
N_MONTHS = 12


# ============================================================
# DATASET GENERATION
# ============================================================

def build_dataset():

    rng = np.random.default_rng(SEED)

    # --------------------------------------------------------
    # 1. Population
    # --------------------------------------------------------

    population = generate_population(
        n_users=N_USERS,
        rng=rng,
    )

    # --------------------------------------------------------
    # 2. Income
    # --------------------------------------------------------

    population = generate_base_income(
        population=population,
        rng=rng,
    )

    population = generate_salary_days(
        population=population,
        rng=rng,
    )

    income_df = generate_monthly_income(
        population=population,
        n_months=N_MONTHS,
        rng=rng,
    )

    # --------------------------------------------------------
    # 3. Obligations
    # --------------------------------------------------------

    obligations_df = generate_obligations(
        population=population,
        rng=rng,
    )

    # --------------------------------------------------------
    # 4. Spending profile
    # --------------------------------------------------------

    spending_df = generate_spending_profile(
        population=population,
        rng=rng,
    )

    # --------------------------------------------------------
    # 5. Initial balances
    # --------------------------------------------------------

    initial_balance_df = generate_initial_balances(
        population=population,
        income_df=income_df,
        rng=rng,
    )

    # --------------------------------------------------------
    # 6. Behavioral regimes
    # --------------------------------------------------------

    behavior_df = generate_behavioral_profile(
        population=population,
        n_months=N_MONTHS,
        rng=rng,
    )

    # --------------------------------------------------------
    # 7. Financial shocks
    # --------------------------------------------------------

    shock_df = generate_shocks(
        population=population,
        n_months=N_MONTHS,
        rng=rng,
    )

    # --------------------------------------------------------
    # 8. Transactions
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # 9. Monthly financial profile
    # --------------------------------------------------------

    monthly_df = build_monthly_profile(
        transactions_df
    )

    # --------------------------------------------------------
    # 10. Financial states
    # --------------------------------------------------------

    state_df = assign_financial_states(
        monthly_df
    )

    return (
        population,
        income_df,
        obligations_df,
        spending_df,
        initial_balance_df,
        behavior_df,
        shock_df,
        transactions_df,
        monthly_df,
        state_df,
    )


# ============================================================
# TEST 1
# STATE CHARACTERISTICS
# ============================================================

def test_state_characteristics(state_df):

    print("\n" + "=" * 70)
    print("1. STATE CHARACTERISTICS")
    print("=" * 70)

    metrics = [
        "cash_buffer_ratio",
        "expense_to_income_ratio",
        "spending_growth",
        "savings_rate",
        "net_cash_flow",
        "discretionary_spending_ratio_income",
    ]

    available_metrics = [
        column
        for column in metrics
        if column in state_df.columns
    ]

    summary = (
        state_df
        .groupby("financial_state")[available_metrics]
        .agg(["mean", "median"])
        .round(4)
    )

    print(summary)

    assert not summary.empty, (
        "State characteristic summary is empty."
    )

    stress = state_df[
        state_df["financial_state"]
        == "LIQUIDITY_STRESS"
    ]

    non_stress = state_df[
        state_df["financial_state"]
        != "LIQUIDITY_STRESS"
    ]

    assert len(stress) > 0, (
        "No LIQUIDITY_STRESS observations found."
    )

    assert len(non_stress) > 0, (
        "No non-stress observations found."
    )

    stress_buffer = (
        stress["cash_buffer_ratio"]
        .mean()
    )

    non_stress_buffer = (
        non_stress["cash_buffer_ratio"]
        .mean()
    )

    print(
        "\nMean cash-buffer ratio:"
    )

    print(
        f"LIQUIDITY_STRESS : {stress_buffer:.4f}"
    )

    print(
        f"NON-STRESS       : {non_stress_buffer:.4f}"
    )

    assert stress_buffer <= non_stress_buffer, (
        "LIQUIDITY_STRESS does not have a lower "
        "cash-buffer ratio than non-stress states."
    )

    print(
        "\nState characteristic validation passed."
    )


# ============================================================
# TEST 2
# STATE PERSISTENCE
# ============================================================

def test_state_persistence(state_df):

    print("\n" + "=" * 70)
    print("2. STATE PERSISTENCE")
    print("=" * 70)

    df = state_df.sort_values(
        ["user_id", "month"]
    ).copy()

    df["new_run"] = (
        df["financial_state"]
        != df.groupby("user_id")[
            "financial_state"
        ].shift()
    )

    df["run_id"] = (
        df.groupby("user_id")["new_run"]
        .cumsum()
    )

    durations = (
        df.groupby(
            ["user_id", "run_id"],
            as_index=False,
        )
        .size()
        .rename(
            columns={"size": "duration"}
        )
    )

    print("\nState duration statistics:")

    print(
        durations["duration"]
        .describe()
        .round(2)
    )

    one_month_pct = (
        durations["duration"]
        .eq(1)
        .mean()
    )

    print(
        f"\nOne-month state runs: "
        f"{one_month_pct:.2%}"
    )

    assert one_month_pct < 0.90, (
        "More than 90% of state runs last exactly "
        "one month. The state engine may be too reactive."
    )

    print(
        "\nState persistence validation passed."
    )


# ============================================================
# TEST 3
# TRANSITION SEMANTICS
# ============================================================

def test_transition_semantics(state_df):

    print("\n" + "=" * 70)
    print("3. TRANSITION SEMANTICS")
    print("=" * 70)

    transitions = state_df[
        state_df["state_changed"]
    ].copy()

    print("\nTransition distribution:")

    print(
        transitions[
            "state_transition"
        ]
        .value_counts()
        .sort_index()
    )

    assert len(transitions) > 0, (
        "No state transitions were detected."
    )

    transition_metrics = [
        "cash_buffer_change",
        "savings_rate_change",
        "spending_growth",
        "discretionary_spending_growth",
    ]

    available = [
        column
        for column in transition_metrics
        if column in transitions.columns
    ]

    print("\nAverage transition signals:")

    print(
        transitions
        .groupby("state_transition")[available]
        .mean()
        .round(4)
    )

    # --------------------------------------------------------
    # STABLE -> SPENDING_PRESSURE
    # --------------------------------------------------------

    stable_to_pressure = transitions[
        transitions["state_transition"]
        == "STABLE->SPENDING_PRESSURE"
    ]

    if len(stable_to_pressure) > 0:

        mean_growth = (
            stable_to_pressure[
                "spending_growth"
            ]
            .mean()
        )

        print(
            "\nSTABLE -> SPENDING_PRESSURE "
            f"mean spending growth: {mean_growth:.4f}"
        )

        assert mean_growth > 0, (
            "STABLE -> SPENDING_PRESSURE does not show "
            "positive spending growth."
        )

    # --------------------------------------------------------
    # SPENDING_PRESSURE -> RECOVERY
    # --------------------------------------------------------

    pressure_to_recovery = transitions[
        transitions["state_transition"]
        == "SPENDING_PRESSURE->RECOVERY"
    ]

    if len(pressure_to_recovery) > 0:

        mean_growth = (
            pressure_to_recovery[
                "spending_growth"
            ]
            .mean()
        )

        print(
            "\nSPENDING_PRESSURE -> RECOVERY "
            f"mean spending growth: {mean_growth:.4f}"
        )

        assert mean_growth < 0, (
            "SPENDING_PRESSURE -> RECOVERY does not show "
            "declining spending."
        )

    print(
        "\nTransition semantic validation passed."
    )


# ============================================================
# TEST 4
# SHOCK -> STATE DETERIORATION
# ============================================================

def test_shocks_and_state_deterioration(
    state_df,
    shock_df,
):

    print("\n" + "=" * 70)
    print("4. SHOCK -> STATE DETERIORATION")
    print("=" * 70)

    if shock_df.empty:

        print(
            "No shocks generated. Test skipped."
        )

        return

    shock_months = shock_df[
        [
            "user_id",
            "month",
            "shock_type",
        ]
    ].copy()

    shock_months["month"] = (
        "2026-"
        + shock_months["month"]
        .astype(int)
        .astype(str)
        .str.zfill(2)
    )

    merged = state_df.merge(
        shock_months,
        on=["user_id", "month"],
        how="left",
    )

    merged["has_shock"] = (
        merged["shock_type"].notna()
    )

    shock_states = merged[
        merged["has_shock"]
    ]

    non_shock_states = merged[
        ~merged["has_shock"]
    ]

    print(
        "\nObservations with shock:",
        len(shock_states),
    )

    print(
        "Observations without shocks:",
        len(non_shock_states),
    )

    if len(shock_states) > 0:

        print(
            "\nState distribution after shocks:"
        )

        print(
            shock_states[
                "financial_state"
            ]
            .value_counts(
                normalize=True
            )
            .round(4)
        )

        print(
            "\nState distribution without shocks:"
        )

        print(
            non_shock_states[
                "financial_state"
            ]
            .value_counts(
                normalize=True
            )
            .round(4)
        )

        print(
            "\nShock types:"
        )

        print(
            shock_states[
                "shock_type"
            ]
            .value_counts()
        )

    print(
        "\nShock/state validation completed."
    )


# ============================================================
# TEST 5
# STATE DISTRIBUTION
# ============================================================

def test_state_distribution(state_df):

    print("\n" + "=" * 70)
    print("5. STATE DISTRIBUTION")
    print("=" * 70)

    distribution = (
        state_df["financial_state"]
        .value_counts(
            normalize=True
        )
        .sort_index()
    )

    print(
        distribution.round(4)
    )

    assert len(distribution) >= 3, (
        "Too few financial states are represented."
    )

    assert distribution.max() < 0.70, (
        "One financial state dominates more than "
        "70% of observations."
    )

    print(
        "\nState distribution validation passed."
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print(
        "Generating FinSight dataset..."
    )

    (
        population,
        income_df,
        obligations_df,
        spending_df,
        initial_balance_df,
        behavior_df,
        shock_df,
        transactions_df,
        monthly_df,
        state_df,
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
        f"Financial-state observations: "
        f"{len(state_df)}"
    )

    test_state_characteristics(
        state_df
    )

    test_state_persistence(
        state_df
    )

    test_transition_semantics(
        state_df
    )

    test_shocks_and_state_deterioration(
        state_df,
        shock_df,
    )

    test_state_distribution(
        state_df
    )

    print("\n" + "=" * 70)
    print(
        "ALL STATE SEMANTIC VALIDATIONS PASSED"
    )
    print("=" * 70)


if __name__ == "__main__":
    main()