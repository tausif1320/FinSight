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
from finsight.data_generation.behavior import (
    generate_behavioral_profile,
)
from finsight.data_generation.shocks import generate_shocks
from finsight.data_generation.transactions import (
    generate_transactions,
)
from finsight.features.monthly_profile import (
    build_monthly_profile,
)
from finsight.features.financial_state import (
    assign_financial_states,
    FINANCIAL_STATES,
)


SEED = 42
N_USERS = 100
N_MONTHS = 12


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
        n_months=N_MONTHS,
        rng=rng,
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
        n_months=N_MONTHS,
        rng=rng,
    )

    shock_df = generate_shocks(
        population,
        n_months=N_MONTHS,
        rng=rng,
    )

    transactions = generate_transactions(
        population=population,
        income_df=income_df,
        obligations_df=obligations_df,
        spending_df=spending_df,
        initial_balance_df=initial_balance_df,
        n_months=N_MONTHS,
        behavior_df=behavior_df,
        shock_df=shock_df,
        rng=np.random.default_rng(SEED + 1),
    )

    monthly_profile = build_monthly_profile(
        transactions
    )

    return (
        population,
        behavior_df,
        shock_df,
        transactions,
        monthly_profile,
    )


def test_schema(state_df):

    required_columns = {
        "user_id",
        "month",
        "financial_state",
        "previous_financial_state",
        "state_changed",
        "state_transition",
        "cash_buffer",
        "cash_buffer_ratio",
        "expense_to_income_ratio",
        "spending_growth",
        "spending_baseline_deviation",
        "liquidity_pressure",
        "spending_pressure",
    }

    missing = (
        required_columns
        - set(state_df.columns)
    )

    assert not missing, (
        f"Missing state columns: "
        f"{sorted(missing)}"
    )


def test_row_count(state_df):

    expected_rows = N_USERS * N_MONTHS

    assert len(state_df) == expected_rows, (
        f"Expected {expected_rows} user-month rows, "
        f"found {len(state_df)}."
    )


def test_state_values(state_df):

    invalid_states = set(
        state_df["financial_state"].dropna().unique()
    ) - set(FINANCIAL_STATES)

    assert not invalid_states, (
        f"Invalid financial states: "
        f"{sorted(invalid_states)}"
    )


def test_one_row_per_user_month(state_df):

    duplicates = state_df.duplicated(
        subset=["user_id", "month"]
    )

    assert not duplicates.any(), (
        "Duplicate user-month observations found."
    )


def test_no_future_leakage(state_df):
    """
    Verify that the first chronological observation for every user
    does not contain historical features.

    Important:
    groupby().first() cannot be used here because it skips NaN values.
    We explicitly select the first chronological row instead.
    """

    ordered = state_df.sort_values(
        ["user_id", "month"],
        kind="mergesort",
    )

    first_month = (
        ordered
        .groupby("user_id", sort=False)
        .head(1)
        .copy()
    )

    assert len(first_month) == N_USERS, (
        f"Expected {N_USERS} first-month observations, "
        f"found {len(first_month)}."
    )

    assert (
        first_month[
            "previous_variable_spending"
        ].isna().all()
    ), (
        "First-month observations contain "
        "previous spending values."
    )

    assert (
        first_month[
            "previous_discretionary_spending"
        ].isna().all()
    ), (
        "First-month observations contain "
        "previous discretionary spending values."
    )

    assert (
        first_month[
            "previous_savings_rate"
        ].isna().all()
    ), (
        "First-month observations contain "
        "previous savings-rate values."
    )

    assert (
        first_month[
            "previous_cash_buffer"
        ].isna().all()
    ), (
        "First-month observations contain "
        "previous cash-buffer values."
    )

    assert (
        first_month[
            "historical_spending_mean"
        ].isna().all()
    ), (
        "First-month observations contain "
        "historical spending baseline values."
    )

    print(
        f"First-month leakage check passed "
        f"for {len(first_month)} users."
    )


def test_state_change_logic(state_df):

    changed_rows = state_df[
        state_df["state_changed"]
    ]

    for _, row in changed_rows.iterrows():

        assert pd.notna(
            row["previous_financial_state"]
        ), (
            "A state change cannot occur without "
            "a previous state."
        )

        expected_transition = (
            row["previous_financial_state"]
            + "->"
            + row["financial_state"]
        )

        assert (
            row["state_transition"]
            == expected_transition
        ), (
            "Incorrect state transition label."
        )


def test_transition_consistency(state_df):

    ordered = state_df.sort_values(
        ["user_id", "month"]
    )

    for user_id, group in ordered.groupby(
        "user_id",
        sort=False,
    ):

        rows = group.reset_index(drop=True)

        assert pd.isna(
            rows.loc[
                0,
                "previous_financial_state",
            ]
        )

        for i in range(1, len(rows)):

            previous_state = rows.loc[
                i - 1,
                "financial_state",
            ]

            current_previous_state = rows.loc[
                i,
                "previous_financial_state",
            ]

            assert (
                previous_state
                == current_previous_state
            ), (
                f"Previous-state mismatch for "
                f"{user_id}, row {i}."
            )


def test_liquidity_stress_logic(state_df):

    stress_rows = state_df[
        state_df["financial_state"]
        == "LIQUIDITY_STRESS"
    ]

    assert len(stress_rows) > 0, (
        "No LIQUIDITY_STRESS observations "
        "were generated. The state engine may "
        "be too conservative."
    )

    stress_condition = (
        stress_rows["liquidity_pressure"]
        |
        stress_rows["negative_cash_flow"]
    )

    assert stress_condition.all(), (
        "A LIQUIDITY_STRESS observation was created "
        "without liquidity pressure or negative "
        "cash flow."
    )


def test_state_distribution(state_df):

    distribution = (
        state_df["financial_state"]
        .value_counts()
        .sort_index()
    )

    print(
        "\nFinancial-state distribution:"
    )

    print(distribution)

    # We want the synthetic world to contain
    # more than one state.
    assert (
        distribution.shape[0] >= 2
    ), (
        "Only one financial state was generated. "
        "The state engine is not differentiating "
        "financial conditions."
    )


def test_behavior_and_financial_state_are_separate(
    state_df,
    behavior_df,
):
    """
    Verify that behavioral regimes and financial states
    are represented as separate concepts.

    The behavioral generator stores month as an integer
    from 1..N_MONTHS, while the monthly financial profile
    stores month as YYYY-MM. Convert the behavioral month
    explicitly rather than passing integers to pd.to_datetime().
    """

    print(
        "\nBehavior dataframe columns:"
    )

    print(
        list(behavior_df.columns)
    )

    # ---------------------------------------------------------
    # Identify behavioral regime column.
    # ---------------------------------------------------------

    regime_candidates = [
        "regime",
        "behavior_regime",
        "behavioral_regime",
        "behavior_state",
    ]

    regime_column = next(
        (
            column
            for column in regime_candidates
            if column in behavior_df.columns
        ),
        None,
    )

    assert regime_column is not None, (
        "Could not identify the behavioral regime column. "
        f"Available columns: {list(behavior_df.columns)}"
    )

    # ---------------------------------------------------------
    # Validate required columns.
    # ---------------------------------------------------------

    required_columns = {
        "user_id",
        "month",
        regime_column,
    }

    missing = (
        required_columns
        - set(behavior_df.columns)
    )

    assert not missing, (
        "Behavior dataframe is missing required "
        f"columns: {sorted(missing)}"
    )

    behavior_subset = behavior_df[
        [
            "user_id",
            "month",
            regime_column,
        ]
    ].copy()

    # ---------------------------------------------------------
    # Normalize behavioral month.
    #
    # behavior.py uses:
    #
    #     month = 1, 2, ..., 12
    #
    # monthly_profile uses:
    #
    #     2026-01, 2026-02, ..., 2026-12
    #
    # Do NOT use pd.to_datetime(integer) here because pandas
    # interprets integer values as timestamps from 1970.
    # ---------------------------------------------------------

    behavior_subset["month"] = (
        pd.to_numeric(
            behavior_subset["month"],
            errors="raise",
        )
        .astype(int)
    )

    invalid_months = behavior_subset[
        ~behavior_subset["month"].between(
            1,
            N_MONTHS,
        )
    ]

    assert invalid_months.empty, (
        "Behavior dataframe contains invalid month values: "
        f"{invalid_months['month'].unique().tolist()}"
    )

    behavior_subset["month"] = (
        "2026-"
        + behavior_subset["month"]
        .astype(str)
        .str.zfill(2)
    )

    # ---------------------------------------------------------
    # Validate one behavioral observation per user-month.
    # ---------------------------------------------------------

    duplicate_behavior_rows = (
        behavior_subset.duplicated(
            subset=["user_id", "month"],
            keep=False,
        )
    )

    assert not duplicate_behavior_rows.any(), (
        "Duplicate behavioral user-month observations found."
    )

    # ---------------------------------------------------------
    # Merge behavior with financial state.
    # ---------------------------------------------------------

    merged = state_df.merge(
        behavior_subset,
        on=[
            "user_id",
            "month",
        ],
        how="left",
        validate="one_to_one",
    )

    assert (
        merged[regime_column].notna().all()
    ), (
        "Behavioral regime could not be aligned "
        "with every financial-state observation."
    )

    # ---------------------------------------------------------
    # Print distributions.
    # ---------------------------------------------------------

    print(
        "\nBehavioral regime distribution:"
    )

    print(
        merged[regime_column]
        .value_counts()
        .sort_index()
    )

    print(
        "\nFinancial state distribution:"
    )

    print(
        merged["financial_state"]
        .value_counts()
        .sort_index()
    )

    # ---------------------------------------------------------
    # Verify that behavior and financial state are not
    # identical concepts.
    # ---------------------------------------------------------

    identical_rate = (
        merged[regime_column]
        == merged["financial_state"]
    ).mean()

    print(
        "\nBehavior regime == financial state rate: "
        f"{identical_rate:.2%}"
    )

    assert (
        identical_rate < 1.0
    ), (
        "Behavioral regime and financial state are "
        "identical for every observation. They must "
        "remain separate concepts."
    )

    # ---------------------------------------------------------
    # Cross-tabulation.
    #
    # A behavioral regime should be capable of producing
    # multiple financial states depending on liquidity,
    # obligations, history, etc.
    # ---------------------------------------------------------

    cross_tab = pd.crosstab(
        merged[regime_column],
        merged["financial_state"],
    )

    print(
        "\nBehavior regime × financial state:"
    )

    print(
        cross_tab
    )

    multi_state_behavior_count = (
        (cross_tab > 0)
        .sum(axis=1)
        > 1
    ).sum()

    assert (
        multi_state_behavior_count > 0
    ), (
        "No behavioral regime appears across multiple "
        "financial states. The behavioral and financial "
        "state systems may be too tightly coupled."
    )

    print(
        "\nBehavior/state separation validation passed."
    )

def main():

    print(
        "\nGenerating FinSight dataset..."
    )

    (
        population,
        behavior_df,
        shock_df,
        transactions,
        monthly_profile,
    ) = build_dataset()

    print(
        f"Transactions: {len(transactions)}"
    )

    print(
        f"Monthly observations: "
        f"{len(monthly_profile)}"
    )

    print(
        "\nBuilding financial states..."
    )

    state_df = assign_financial_states(
        monthly_profile
    )

    print(
        f"State observations: {len(state_df)}"
    )

    print(
        "\nState columns:"
    )

    print(
        [
            "financial_state",
            "previous_financial_state",
            "state_changed",
            "state_transition",
        ]
    )

    test_schema(
        state_df
    )

    print(
        "Schema validation passed."
    )

    test_row_count(
        state_df
    )

    print(
        "Row-count validation passed."
    )

    test_state_values(
        state_df
    )

    print(
        "State-value validation passed."
    )

    test_one_row_per_user_month(
        state_df
    )

    print(
        "User-month uniqueness validation passed."
    )

    test_no_future_leakage(
        state_df
    )

    print(
        "Historical-feature validation passed."
    )

    test_state_change_logic(
        state_df
    )

    print(
        "State-change validation passed."
    )

    test_transition_consistency(
        state_df
    )

    print(
        "Transition consistency validation passed."
    )

    test_liquidity_stress_logic(
        state_df
    )

    print(
        "Liquidity-stress validation passed."
    )

    test_state_distribution(
        state_df
    )

    test_behavior_and_financial_state_are_separate(
        state_df,
        behavior_df,
    )

    print(
        "Behavior/state separation validation passed."
    )

    print(
        "\nExample financial states:"
    )

    print(
        state_df[
            [
                "user_id",
                "month",
                "financial_state",
                "previous_financial_state",
                "state_changed",
                "state_transition",
                "cash_buffer_ratio",
                "expense_to_income_ratio",
                "spending_growth",
                "savings_rate",
            ]
        ]
        .head(20)
        .to_string(index=False)
    )

    print(
        "\nFinancial-state validation passed."
    )


if __name__ == "__main__":
    main()