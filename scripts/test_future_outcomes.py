import numpy as np
import pandas as pd

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


SEED = 42
N_USERS = 100
N_MONTHS = 12
HORIZON_DAYS = 30


# ============================================================
# DATASET
# ============================================================

def build_dataset():

    rng = np.random.default_rng(SEED)

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
        transactions_df,
        monthly_df,
        state_df,
        feature_df,
        outcome_df,
    )


# ============================================================
# TEST 1
# SCHEMA
# ============================================================

def test_schema(outcome_df):

    print("\n" + "=" * 70)
    print("1. OUTCOME SCHEMA")
    print("=" * 70)

    required = {
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
    }

    missing = (
        required
        - set(outcome_df.columns)
    )

    assert not missing, (
        "Missing outcome columns: "
        f"{sorted(missing)}"
    )

    print(
        f"Rows: {len(outcome_df)}"
    )

    print(
        f"Columns: {len(outcome_df.columns)}"
    )

    print(
        "\nOutcome schema validation passed."
    )


# ============================================================
# TEST 2
# ROW COUNT
# ============================================================

def test_row_count(outcome_df):

    print("\n" + "=" * 70)
    print("2. OUTCOME ROW COUNT")
    print("=" * 70)

    # December cannot have a complete 30-day future
    # horizon because our synthetic data ends on Dec 31.
    expected = (
        N_USERS
        * (N_MONTHS - 1)
    )

    print(
        f"Expected complete horizons: {expected}"
    )

    print(
        f"Actual: {len(outcome_df)}"
    )

    assert len(outcome_df) == expected, (
        "Unexpected number of complete "
        "future-horizon observations."
    )

    print(
        "\nOutcome row-count validation passed."
    )


# ============================================================
# TEST 3
# USER-MONTH UNIQUENESS
# ============================================================

def test_uniqueness(outcome_df):

    print("\n" + "=" * 70)
    print("3. USER-MONTH UNIQUENESS")
    print("=" * 70)

    duplicates = outcome_df.duplicated(
        subset=["user_id", "month"],
        keep=False,
    )

    assert not duplicates.any(), (
        "Duplicate outcome user-month rows found."
    )

    print(
        "\nUser-month uniqueness validation passed."
    )


# ============================================================
# TEST 4
# FUTURE WINDOW
# ============================================================

def test_future_window(
    outcome_df,
    transactions_df,
):

    print("\n" + "=" * 70)
    print("4. FUTURE WINDOW VALIDATION")
    print("=" * 70)

    transactions = transactions_df.copy()

    transactions["timestamp"] = (
        pd.to_datetime(
            transactions["timestamp"]
        )
    )

    for _, row in outcome_df.iterrows():

        prediction_date = pd.Timestamp(
            row["prediction_date"]
        )

        horizon_end = pd.Timestamp(
            row["horizon_end"]
        )

        expected_end = (
            prediction_date
            + pd.Timedelta(
                days=HORIZON_DAYS
            )
        )

        assert horizon_end == expected_end, (
            f"Incorrect horizon for "
            f"{row['user_id']} "
            f"{row['month']}."
        )

        future_start = (
            prediction_date
            + pd.Timedelta(days=1)
        )

        future = transactions[
            (
                transactions["user_id"]
                == row["user_id"]
            )
            & (
                transactions["timestamp"]
                >= future_start
            )
            & (
                transactions["timestamp"]
                <= horizon_end
            )
        ]

        assert len(future) > 0, (
            f"No future transactions found for "
            f"{row['user_id']} {row['month']}."
        )

    print(
        "Every outcome uses the next "
        f"{HORIZON_DAYS} days."
    )

    print(
        "\nFuture-window validation passed."
    )


# ============================================================
# TEST 5
# ACCOUNTING
# ============================================================

def test_accounting(outcome_df):

    print("\n" + "=" * 70)
    print("5. FUTURE ACCOUNTING")
    print("=" * 70)

    calculated_net = (
        outcome_df["future_income"]
        - outcome_df["future_expenses"]
    )

    difference = (
        calculated_net
        - outcome_df["future_net_cash_flow"]
    ).abs()

    assert (
        difference.max() < 1e-6
    ), (
        "Future cash-flow accounting mismatch."
    )

    print(
        "Future income - expenses = "
        "future net cash flow."
    )

    print(
        "\nFuture accounting validation passed."
    )


# ============================================================
# TEST 7
# OUTCOME SANITY
# ============================================================

def test_balance_rollforward(outcome_df):

    print("\n" + "=" * 70)
    print("6. FUTURE BALANCE ROLL-FORWARD")
    print("=" * 70)

    expected_ending_balance = (
        outcome_df["prediction_balance"]
        + outcome_df["future_net_cash_flow"]
    )

    difference = (
        expected_ending_balance
        - outcome_df["ending_future_balance"]
    ).abs()

    assert difference.max() < 1e-6, (
        "Ending future balance does not reconcile with "
        "prediction balance + future net cash flow."
    )

    assert (
        outcome_df["minimum_future_balance"]
        <= outcome_df["prediction_balance"] + 1e-6
    ).all(), (
        "Minimum future balance cannot be greater than "
        "prediction-date balance."
    )

    print(
        "Prediction balance + future net cash flow = "
        "ending future balance."
    )

    print("\nFuture balance roll-forward validation passed.")


# ============================================================
# TEST 7
# OUTCOME SANITY
# ============================================================

def test_outcome_sanity(outcome_df):

    print("\n" + "=" * 70)
    print("6. OUTCOME SANITY")
    print("=" * 70)

    numeric_columns = [
        "minimum_future_balance",
        "ending_future_balance",
        "future_net_cash_flow",
        "future_income",
        "future_expenses",
        "negative_cashflow_days",
        "required_obligations",
        "future_transaction_count",
        "future_discretionary_spending",
        "future_essential_spending",
    ]

    infinite_values = np.isinf(
        outcome_df[numeric_columns]
    ).sum().sum()

    assert infinite_values == 0, (
        "Infinite outcome values detected."
    )

    assert not outcome_df[
        "liquidity_ratio"
    ].isna().any(), (
        "Missing liquidity_ratio values detected."
    )

    assert outcome_df[
        "critical_liquidity"
    ].dtype == bool

    assert outcome_df[
        "obligation_failure"
    ].dtype == bool

    assert outcome_df[
        "severe_liquidity_drawdown"
    ].dtype == bool

    assert (
        outcome_df["negative_cashflow_days"]
        >= 0
    ).all()

    assert (
        outcome_df["negative_cashflow_days"]
        <= HORIZON_DAYS
    ).all()

    assert (
        outcome_df["future_income"]
        >= 0
    ).all()

    assert (
        outcome_df["future_expenses"]
        >= 0
    ).all()

    print(
        "No invalid numeric outcomes."
    )

    print(
        "\nOutcome sanity validation passed."
    )


# ============================================================
# TEST 8
# STRESS DISTRIBUTION
# ============================================================

def inspect_stress_distribution(
    outcome_df,
):

    print("\n" + "=" * 70)
    print("7. STRESS DISTRIBUTION")
    print("=" * 70)

    stress_rate = (
        outcome_df["financial_stress_30d"]
        .mean()
    )

    print(
        f"\nFinancial stress rate: "
        f"{stress_rate:.2%}"
    )

    print(
        "\nStress counts:"
    )

    print(
        outcome_df[
            "financial_stress_30d"
        ]
        .value_counts()
        .sort_index()
    )

    print(
        "\nOutcome distributions:"
    )

    print(
        outcome_df[
            [
                "minimum_future_balance",
                "future_net_cash_flow",
                "negative_cashflow_days",
                "obligation_coverage",
            ]
        ]
        .describe()
        .round(2)
    )

    # At this stage we intentionally don't impose
    # a "correct" stress prevalence. We inspect it first.
    assert (
        outcome_df[
            "financial_stress_30d"
        ].nunique()
        == 2
    ), (
        "Stress target contains only one class. "
        "Thresholds need adjustment."
    )

    print(
        "\nBoth stress classes are represented."
    )


# ============================================================
# TEST 8B
# STRESS COMPONENT CONSISTENCY
# ============================================================

def test_stress_component_consistency(outcome_df):

    print("\n" + "=" * 70)
    print("8B. STRESS COMPONENT CONSISTENCY")
    print("=" * 70)

    component_union = (
        outcome_df["critical_liquidity"]
        | outcome_df["obligation_failure"]
        | outcome_df["severe_liquidity_drawdown"]
    ).astype(int)

    assert (
        component_union
        == outcome_df["financial_stress_30d"]
    ).all(), (
        "Final stress label does not equal the union of "
        "the documented stress components."
    )

    print(
        "financial_stress_30d exactly matches the documented "
        "stress components."
    )


# ============================================================
# TEST 9
# FEATURE/OUTCOME SEPARATION
# ============================================================

def test_temporal_separation(
    feature_df,
    outcome_df,
):

    print("\n" + "=" * 70)
    print("8. FEATURE / OUTCOME SEPARATION")
    print("=" * 70)

    features = feature_df[
        [
            "user_id",
            "month",
        ]
    ].copy()

    outcomes = outcome_df[
        [
            "user_id",
            "month",
            "prediction_date",
            "horizon_end",
        ]
    ].copy()

    features["month"] = pd.to_datetime(
        features["month"]
    )

    outcomes["month"] = pd.to_datetime(
        outcomes["month"]
    )

    merged = features.merge(
        outcomes,
        on=["user_id", "month"],
        how="inner",
        validate="one_to_one",
    )

    assert len(merged) == len(
        outcomes
    )

    assert (
        merged["prediction_date"]
        >= merged["month"]
    ).all()

    # The prediction date must be the end of the observation month.
    expected_prediction_date = (
        merged["month"]
        + pd.offsets.MonthEnd(0)
    )

    assert (
        merged["prediction_date"]
        == expected_prediction_date
    ).all()

    print(
        "Prediction date occurs after the "
        "feature month."
    )

    print(
        "Outcome horizon occurs after "
        "the prediction date."
    )

    print(
        "\nFeature/outcome temporal separation "
        "validation passed."
    )


# ============================================================
# SAMPLE
# ============================================================

def print_sample(outcome_df):

    print("\n" + "=" * 70)
    print("9. OUTCOME SAMPLE")
    print("=" * 70)

    columns = [
        "user_id",
        "month",
        "prediction_date",
        "horizon_end",
        "prediction_balance",
        "minimum_future_balance",
        "future_net_cash_flow",
        "negative_cashflow_days",
        "required_obligations",
        "obligation_coverage",
        "financial_stress_30d",
    ]

    print(
        outcome_df[
            columns
        ]
        .head(15)
        .to_string(index=False)
    )

def inspect_stress_components(
    outcome_df,
):
    print("\n" + "=" * 70)
    print("9. STRESS COMPONENT ANALYSIS")
    print("=" * 70)

    critical_liquidity = (
        outcome_df["minimum_future_balance"] <= 0
    )

    obligation_failure = (
        outcome_df["obligation_coverage"] < 1
    )

    positive_balance = outcome_df["prediction_balance"] > 0

    liquidity_ratio = pd.Series(
        0.0,
        index=outcome_df.index,
        dtype=float,
    )

    liquidity_ratio.loc[positive_balance] = (
        outcome_df.loc[
            positive_balance,
            "minimum_future_balance",
        ]
        / outcome_df.loc[
            positive_balance,
            "prediction_balance",
        ]
    )

    severe_drawdown = (
        positive_balance
        & (liquidity_ratio < 0.25)
        & (outcome_df["future_net_cash_flow"] < 0)
    )

    negative_future_cashflow = (
        outcome_df["future_net_cash_flow"] < 0
    )

    print("\nCritical liquidity:")
    print(f"Count: {critical_liquidity.sum()}")
    print(f"Rate:  {critical_liquidity.mean():.2%}")

    print("\nObligation coverage failure:")
    print(f"Count: {obligation_failure.sum()}")
    print(f"Rate:  {obligation_failure.mean():.2%}")

    print("\nSevere liquidity drawdown:")
    print(f"Count: {severe_drawdown.sum()}")
    print(f"Rate:  {severe_drawdown.mean():.2%}")

    print("\nNegative 30-day net cash flow:")
    print(f"Count: {negative_future_cashflow.sum()}")
    print(f"Rate:  {negative_future_cashflow.mean():.2%}")

    component_table = pd.DataFrame({
        "critical_liquidity": critical_liquidity,
        "obligation_failure": obligation_failure,
        "severe_liquidity_drawdown": severe_drawdown,
        "negative_future_cashflow": negative_future_cashflow,
        "financial_stress_30d": (
            outcome_df["financial_stress_30d"] == 1
        ),
    })

    print("\nComponent combinations:")
    print(
        component_table[
            [
                "critical_liquidity",
                "obligation_failure",
                "severe_liquidity_drawdown",
                "financial_stress_30d",
            ]
        ]
        .value_counts()
        .sort_index()
    )

    # The final target must be exactly the union of the three documented
    # stress components.
    component_union = (
        critical_liquidity
        | obligation_failure
        | severe_drawdown
    )

    assert (
        component_union
        == (outcome_df["financial_stress_30d"] == 1)
    ).all(), (
        "Stress target is not exactly explained by its "
        "documented component conditions."
    )

    print(
        "\nNegative 30-day net cash flow is retained only as a "
        "descriptive diagnostic, not as a stress criterion."
    )


def inspect_obligation_data(
    transactions_df,
    monthly_df,
    outcome_df,
):
    print("\n" + "=" * 70)
    print("OBLIGATION DATA DIAGNOSTICS")
    print("=" * 70)

    # ------------------------------------------------------------
    # Monthly profile
    # ------------------------------------------------------------
    print("\nMONTHLY PROFILE COLUMNS:")
    print(monthly_df.columns.tolist())

    monthly_obligation_columns = [
        c
        for c in monthly_df.columns
        if any(
            keyword in c.lower()
            for keyword in [
                "oblig",
                "emi",
                "rent",
                "utility",
                "subscription",
            ]
        )
    ]

    print("\nOBLIGATION-RELATED MONTHLY COLUMNS:")
    print(monthly_obligation_columns)

    # ------------------------------------------------------------
    # Transactions
    # ------------------------------------------------------------
    print("\nTRANSACTION COLUMNS:")
    print(transactions_df.columns.tolist())

    transaction_obligation_columns = [
        c
        for c in transactions_df.columns
        if any(
            keyword in c.lower()
            for keyword in [
                "oblig",
                "emi",
                "rent",
                "utility",
                "subscription",
            ]
        )
    ]

    print("\nOBLIGATION-RELATED TRANSACTION COLUMNS:")
    print(transaction_obligation_columns)

    # ------------------------------------------------------------
    # Transaction types
    # ------------------------------------------------------------
    print("\nTRANSACTION TYPES:")
    print(
        transactions_df["transaction_type"]
        .value_counts(dropna=False)
    )

    # ------------------------------------------------------------
    # Categories
    # ------------------------------------------------------------
    if "category" in transactions_df.columns:
        print("\nTRANSACTION CATEGORIES:")
        print(
            transactions_df["category"]
            .value_counts(dropna=False)
        )

    # ------------------------------------------------------------
    # Explicit obligation flag
    # ------------------------------------------------------------
    if "is_obligation" in transactions_df.columns:
        print("\nIS_OBLIGATION:")
        print(
            transactions_df["is_obligation"]
            .value_counts(dropna=False)
        )

        print("\nOBLIGATION TRANSACTION SAMPLE:")
        print(
            transactions_df[
                transactions_df["is_obligation"] == True
            ]
            .head(20)
            .to_string(index=False)
        )

    # ------------------------------------------------------------
    # Monthly profile sample
    # ------------------------------------------------------------
    print("\nMONTHLY PROFILE SAMPLE:")
    print(
        monthly_df.head(5).to_string(index=False)
    )

    # ------------------------------------------------------------
    # Outcome obligation distribution
    # ------------------------------------------------------------
    print("\nOUTCOME OBLIGATION DISTRIBUTION:")
    print(
        outcome_df[
            [
                "user_id",
                "month",
                "required_obligations",
                "obligation_coverage",
                "obligation_failure",
            ]
        ]
        .head(20)
        .to_string(index=False)
    )

    print("\nNON-ZERO REQUIRED OBLIGATIONS:")
    non_zero = outcome_df[
        outcome_df["required_obligations"] > 0
    ]

    print(
        f"{len(non_zero)} / {len(outcome_df)}"
    )

    print("=" * 70)

def test_obligations_are_present(
    outcome_df,
):
    print(
        "\n"
        + "=" * 70
    )
    print(
        "OBLIGATION REPRESENTATION"
    )
    print(
        "=" * 70
    )

    non_zero = (
        outcome_df[
            "required_obligations"
        ] > 0
    )

    print(
        "Non-zero obligation horizons: "
        f"{non_zero.sum()} / {len(outcome_df)}"
    )

    print(
        "Maximum required obligations: "
        f"{outcome_df['required_obligations'].max():,.2f}"
    )

    assert non_zero.any(), (
        "No future horizons contain required "
        "obligations. Check merchant_category "
        "mapping in outcomes.py."
    )

    covered_rows = (
        outcome_df.loc[
            non_zero,
            "obligation_coverage",
        ]
    )

    assert covered_rows.notna().all(), (
        "Obligation coverage contains NaN "
        "for horizons with obligations."
    )

    assert np.isfinite(
        covered_rows
    ).all(), (
        "Obligation coverage contains "
        "non-finite values despite required "
        "obligations being present."
    )

    print(
        "\nObligation representation validation passed."
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
        outcome_df,
    ) = build_dataset()

    print(
        f"Transactions: {len(transactions_df)}"
    )

    print(
        f"Feature observations: {len(feature_df)}"
    )

    print(
        f"Outcome observations: {len(outcome_df)}"
    )

    test_schema(
        outcome_df
    )

    test_row_count(
        outcome_df
    )

    test_uniqueness(
        outcome_df
    )

    test_future_window(
        outcome_df,
        transactions_df,
    )

    test_accounting(
        outcome_df
    )

    test_balance_rollforward(
        outcome_df
    )

    test_outcome_sanity(
        outcome_df
    )

    test_obligations_are_present(
    outcome_df
    )

    inspect_obligation_data(
    transactions_df,
    monthly_df,
    outcome_df,
)

    inspect_stress_components(
        outcome_df
    )

    test_stress_component_consistency(
        outcome_df
    )

    test_temporal_separation(
        feature_df,
        outcome_df,
    )

    print_sample(
        outcome_df
    )

    print("\n" + "=" * 70)
    print(
        "ALL FUTURE OUTCOME VALIDATIONS PASSED"
    )
    print("=" * 70)


if __name__ == "__main__":
    main()