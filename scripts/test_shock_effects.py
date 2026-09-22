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
from finsight.data_generation.transactions import generate_transactions
from finsight.features.monthly_profile import build_monthly_profile


SEED = 42
N_USERS = 100
N_MONTHS = 12
CONTROLLED_SHOCK_MONTH = 6

# Because this is a stochastic synthetic population, we do not require
# every individual user to move in the same direction.
MIN_AFFECTED_RATE = 0.80


def build_common_inputs():
    """
    Generate all inputs that remain identical across
    baseline and controlled-shock scenarios.
    """
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

    return (
        population,
        income_df,
        obligations_df,
        spending_df,
        initial_balance_df,
    )


def create_empty_shock_df():
    """
    Create an empty shock dataframe with the schema expected
    by generate_transactions().
    """
    return pd.DataFrame(
        columns=[
            "user_id",
            "month",
            "shock_type",
            "shock_amount",
            "income_impact",
            "duration_months",
            "severity",
        ]
    )


def generate_no_shock_dataset(common_inputs):
    """
    Generate the control dataset with no shocks.
    """
    (
        population,
        income_df,
        obligations_df,
        spending_df,
        initial_balance_df,
    ) = common_inputs

    empty_shock_df = create_empty_shock_df()

    transactions = generate_transactions(
        population=population,
        income_df=income_df,
        obligations_df=obligations_df,
        spending_df=spending_df,
        initial_balance_df=initial_balance_df,
        n_months=N_MONTHS,
        shock_df=empty_shock_df,
        rng=np.random.default_rng(SEED + 1),
    )

    profile = build_monthly_profile(
        transactions
    )

    return transactions, profile


def create_controlled_shock_df(
    population,
    shock_type,
):
    """
    Create a deterministic shock for every user in month 6.

    This intentionally avoids the stochastic shock generator because
    this test is trying to isolate the causal direction of each shock.
    """
    records = []

    amount_ratios = {
        "MEDICAL": 0.20,
        "TRAVEL": 0.15,
        "LARGE_PURCHASE": 0.25,
        "EMERGENCY": 0.30,
        "UNEXPECTED_BILL": 0.15,
    }

    for _, user in population.iterrows():

        if shock_type == "INCOME_DISRUPTION":

            records.append(
                {
                    "user_id": user["user_id"],
                    "month": CONTROLLED_SHOCK_MONTH,
                    "shock_type": shock_type,
                    "shock_amount": 0.0,
                    "income_impact": 0.30,
                    "duration_months": 1,
                    "severity": "MEDIUM",
                }
            )

        else:

            income = user["base_income"]

            amount = (
                income
                * amount_ratios[shock_type]
            )

            records.append(
                {
                    "user_id": user["user_id"],
                    "month": CONTROLLED_SHOCK_MONTH,
                    "shock_type": shock_type,
                    "shock_amount": round(
                        amount,
                        2,
                    ),
                    "income_impact": 0.0,
                    "duration_months": 1,
                    "severity": "MEDIUM",
                }
            )

    return pd.DataFrame(records)


def generate_controlled_shock_dataset(
    common_inputs,
    shock_type,
):
    """
    Generate a dataset containing one controlled shock per user
    in month 6.
    """
    (
        population,
        income_df,
        obligations_df,
        spending_df,
        initial_balance_df,
    ) = common_inputs

    shock_df = create_controlled_shock_df(
        population,
        shock_type,
    )

    transactions = generate_transactions(
        population=population,
        income_df=income_df,
        obligations_df=obligations_df,
        spending_df=spending_df,
        initial_balance_df=initial_balance_df,
        n_months=N_MONTHS,
        shock_df=shock_df,
        rng=np.random.default_rng(SEED + 1),
    )

    profile = build_monthly_profile(
        transactions
    )

    return transactions, profile, shock_df


def compare_profiles(
    baseline_profile,
    shock_profile,
    month=CONTROLLED_SHOCK_MONTH,
):
    """
    Compare baseline and shocked monthly profiles
    for the controlled shock month.
    """
    month_string = f"2026-{month:02d}"

    baseline = baseline_profile[
        baseline_profile["month"] == month_string
    ].copy()

    shocked = shock_profile[
        shock_profile["month"] == month_string
    ].copy()

    merged = baseline.merge(
        shocked,
        on="user_id",
        suffixes=(
            "_baseline",
            "_shock",
        ),
        validate="one_to_one",
    )

    if len(merged) != N_USERS:
        raise AssertionError(
            f"Expected {N_USERS} users in comparison, "
            f"found {len(merged)}."
        )

    return merged


def validate_sequential_accounting(transactions):
    """
    Validate the ledger row by row.

    For every user:

        CREDIT:
            balance_after = balance_before + amount

        DEBIT:
            balance_after = balance_before - amount

    Also verify that each transaction's balance_before
    equals the previous transaction's balance_after.
    """
    ordered = transactions.sort_values(
        [
            "user_id",
            "timestamp",
            "transaction_id",
        ],
        kind="mergesort",
    )

    for user_id, group in ordered.groupby(
        "user_id",
        sort=False,
    ):
        previous_balance = None

        for _, row in group.iterrows():

            balance_before = row["balance_before"]
            amount = row["amount"]
            balance_after = row["balance_after"]

            if previous_balance is not None:
                assert np.isclose(
                    balance_before,
                    previous_balance,
                    atol=0.01,
                ), (
                    f"Balance continuity failure for "
                    f"{user_id}"
                )

            if row["transaction_type"] == "CREDIT":

                expected_balance = (
                    balance_before + amount
                )

            elif row["transaction_type"] == "DEBIT":

                expected_balance = (
                    balance_before - amount
                )

            else:

                raise AssertionError(
                    f"Unknown transaction type: "
                    f"{row['transaction_type']}"
                )

            assert np.isclose(
                expected_balance,
                balance_after,
                atol=0.01,
            ), (
                f"Accounting mismatch for "
                f"{user_id} at transaction "
                f"{row['transaction_id']}"
            )

            previous_balance = balance_after


def validate_shock_transaction(
    transactions,
    shock_df,
    shock_type,
):
    """
    Verify that every controlled expense shock produces
    a corresponding transaction in the ledger.

    Income disruption is not represented as an expense
    transaction, so it is excluded from this check.
    """
    if shock_type == "INCOME_DISRUPTION":
        return

    expected_amounts = shock_df[
        [
            "user_id",
            "month",
            "shock_amount",
        ]
    ].copy()

    shock_transaction_map = {
        "MEDICAL": "Healthcare",
        "TRAVEL": "Travel",
        "LARGE_PURCHASE": "Shopping",
        "EMERGENCY": "Other",
        "UNEXPECTED_BILL": "Other",
    }

    expected_category = shock_transaction_map[
        shock_type
    ]

    shock_transactions = transactions[
        (
            transactions["merchant_category"]
            == expected_category
        )
        &
        (
            transactions["transaction_type"]
            == "DEBIT"
        )
    ].copy()

    matched = 0

    for _, expected in expected_amounts.iterrows():

        user_rows = shock_transactions[
            shock_transactions["user_id"]
            == expected["user_id"]
        ]

        amount_match = np.isclose(
            user_rows["amount"].to_numpy(),
            expected["shock_amount"],
            atol=0.01,
        )

        if amount_match.any():
            matched += 1

    assert matched == len(expected_amounts), (
        f"{shock_type}: expected "
        f"{len(expected_amounts)} shock transactions, "
        f"matched {matched}."
    )


def main():

    common_inputs = build_common_inputs()

    print("\nGenerating baseline dataset...")

    baseline_transactions, baseline_profile = (
        generate_no_shock_dataset(
            common_inputs
        )
    )

    print(
        f"Baseline transactions: "
        f"{len(baseline_transactions)}"
    )

    shock_types = [
        "MEDICAL",
        "TRAVEL",
        "LARGE_PURCHASE",
        "EMERGENCY",
        "UNEXPECTED_BILL",
        "INCOME_DISRUPTION",
    ]

    results = []

    for shock_type in shock_types:

        print(
            f"\nTesting {shock_type}..."
        )

        (
            shock_transactions,
            shock_profile,
            shock_df,
        ) = generate_controlled_shock_dataset(
            common_inputs,
            shock_type,
        )

        comparison = compare_profiles(
            baseline_profile,
            shock_profile,
            month=CONTROLLED_SHOCK_MONTH,
        )

        if shock_type == "INCOME_DISRUPTION":

            # -------------------------------------------------
            # Income disruption
            # -------------------------------------------------

            income_difference = (
                comparison["total_income_baseline"]
                - comparison["total_income_shock"]
            )

            cashflow_difference = (
                comparison["net_cash_flow_baseline"]
                - comparison["net_cash_flow_shock"]
            )

            balance_difference = (
                comparison["ending_balance_baseline"]
                - comparison["ending_balance_shock"]
            )

            income_positive_rate = (
                income_difference > 0
            ).mean()

            cashflow_positive_rate = (
                cashflow_difference > 0
            ).mean()

            balance_positive_rate = (
                balance_difference > 0
            ).mean()

            assert (
                income_difference.mean() > 0
            ), (
                "Income disruption did not reduce "
                "average income."
            )

            assert (
                cashflow_difference.mean() > 0
            ), (
                "Income disruption did not reduce "
                "average net cash flow."
            )

            assert (
                balance_difference.mean() > 0
            ), (
                "Income disruption did not reduce "
                "average ending balance."
            )

            assert (
                income_positive_rate
                >= MIN_AFFECTED_RATE
            ), (
                "Income disruption affected too few "
                "users' income."
            )

            assert (
                cashflow_positive_rate
                >= MIN_AFFECTED_RATE
            ), (
                "Income disruption affected too few "
                "users' cash flow."
            )

            assert (
                balance_positive_rate
                >= MIN_AFFECTED_RATE
            ), (
                "Income disruption affected too few "
                "users' ending balances."
            )

            results.append(
                {
                    "shock_type": shock_type,
                    "mean_income_impact":
                        income_difference.mean(),
                    "mean_cashflow_impact":
                        cashflow_difference.mean(),
                    "mean_balance_impact":
                        balance_difference.mean(),
                    "income_positive_rate":
                        income_positive_rate,
                    "cashflow_positive_rate":
                        cashflow_positive_rate,
                    "balance_positive_rate":
                        balance_positive_rate,
                }
            )

        else:

            # -------------------------------------------------
            # Expense shock
            # -------------------------------------------------

            debit_difference = (
                comparison["total_debits_shock"]
                - comparison["total_debits_baseline"]
            )

            balance_difference = (
                comparison["ending_balance_baseline"]
                - comparison["ending_balance_shock"]
            )

            debit_positive_rate = (
                debit_difference > 0
            ).mean()

            balance_positive_rate = (
                balance_difference > 0
            ).mean()

            # Population-level effect.
            assert (
                debit_difference.mean() > 0
            ), (
                f"{shock_type} did not increase "
                "average monthly debits."
            )

            assert (
                balance_difference.mean() > 0
            ), (
                f"{shock_type} did not reduce "
                "average ending balance."
            )

            # Individual-level directional consistency.
            # We intentionally do not require 100%.
            assert (
                debit_positive_rate
                >= MIN_AFFECTED_RATE
            ), (
                f"{shock_type} affected too few "
                "users' monthly debits."
            )

            assert (
                balance_positive_rate
                >= MIN_AFFECTED_RATE
            ), (
                f"{shock_type} affected too few "
                "users' ending balances."
            )

            results.append(
                {
                    "shock_type": shock_type,
                    "mean_debit_impact":
                        debit_difference.mean(),
                    "mean_balance_impact":
                        balance_difference.mean(),
                    "debit_positive_rate":
                        debit_positive_rate,
                    "balance_positive_rate":
                        balance_positive_rate,
                }
            )

        # Validate that the controlled shock actually
        # appears in the transaction ledger.
        validate_shock_transaction(
            transactions=shock_transactions,
            shock_df=shock_df,
            shock_type=shock_type,
        )

    results_df = pd.DataFrame(results)

    print("\nShock effect summary:")

    print(
        results_df.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    print(
        "\nShock direction checks passed."
    )

    # ---------------------------------------------------------
    # Validate sequential accounting for every shock scenario.
    # ---------------------------------------------------------

    print(
        "\nChecking transaction accounting..."
    )

    for shock_type in shock_types:

        (
            transactions,
            _,
            _,
        ) = generate_controlled_shock_dataset(
            common_inputs,
            shock_type,
        )

        validate_sequential_accounting(
            transactions
        )

    print(
        "Transaction accounting passed."
    )

    # ---------------------------------------------------------
    # Final explicit shock transaction check.
    # ---------------------------------------------------------

    print(
        "\nChecking shock transaction creation..."
    )

    emergency_transactions, _, emergency_shock_df = (
        generate_controlled_shock_dataset(
            common_inputs,
            "EMERGENCY",
        )
    )

    validate_shock_transaction(
        transactions=emergency_transactions,
        shock_df=emergency_shock_df,
        shock_type="EMERGENCY",
    )

    shock_category_rows = emergency_transactions[
        (
            emergency_transactions[
                "merchant_category"
            ]
            == "Other"
        )
        &
        (
            emergency_transactions[
                "transaction_type"
            ]
            == "DEBIT"
        )
    ]

    assert len(shock_category_rows) > 0, (
        "No emergency shock transactions "
        "were created."
    )

    print(
        f"Emergency shock transactions found: "
        f"{len(shock_category_rows)}"
    )

    print(
        "\nShock effect validation passed."
    )


if __name__ == "__main__":
    main()
