
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

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
from finsight.data_generation.transactions import (
    generate_transactions,
)
from finsight.utils.random import create_rng


SEED = 42

N_USERS = 10
N_MONTHS = 2


def generate_test_transactions():
    """
    Generate a small deterministic transaction dataset.
    """

    rng = create_rng(SEED)

    # ---------------------------------------------------------
    # 1. Population
    # ---------------------------------------------------------

    population = generate_population(
        N_USERS,
        rng,
    )

    # ---------------------------------------------------------
    # 2. Income
    # ---------------------------------------------------------

    population = generate_base_income(
        population,
        rng,
    )

    population = generate_salary_days(
        population,
        rng,
    )

    income = generate_monthly_income(
        population,
        N_MONTHS,
        rng,
    )

    # ---------------------------------------------------------
    # 3. Obligations
    # ---------------------------------------------------------

    obligations = generate_obligations(
        population,
        rng,
    )

    # ---------------------------------------------------------
    # 4. Spending profile
    # ---------------------------------------------------------

    spending = generate_spending_profile(
        population,
        rng,
    )

    # ---------------------------------------------------------
    # 5. Initial balance
    # ---------------------------------------------------------

    initial_balances = generate_initial_balances(
        population,
        income,
        rng,
    )

    # ---------------------------------------------------------
    # 6. Transactions
    # ---------------------------------------------------------

    transactions = generate_transactions(
        population=population,
        income_df=income,
        obligations_df=obligations,
        spending_df=spending,
        initial_balance_df=initial_balances,
        n_months=N_MONTHS,
        rng=rng,
    )

    return transactions


def main():

    # =========================================================
    # GENERATE DATA
    # =========================================================

    transactions = generate_test_transactions()

    # ---------------------------------------------------------
    # BASIC VALIDATION
    # ---------------------------------------------------------

    print("\nShape:")
    print(transactions.shape)

    print("\nColumns:")
    print(transactions.columns.tolist())

    print("\nMissing values:")
    print(transactions.isna().sum())

    # Required columns
    required_columns = {
        "transaction_id",
        "user_id",
        "timestamp",
        "transaction_type",
        "merchant_category",
        "amount",
        "payment_method",
        "balance_before",
        "balance_after",
    }

    missing_columns = (
        required_columns
        - set(transactions.columns)
    )

    assert not missing_columns, (
        "Missing transaction columns: "
        f"{sorted(missing_columns)}"
    )

    # Transaction ID must be unique
    assert transactions["transaction_id"].is_unique

    # Correct number of users
    assert (
        transactions["user_id"].nunique()
        == N_USERS
    )

    # No missing critical values
    assert transactions["timestamp"].notna().all()

    assert transactions["amount"].notna().all()

    assert transactions["balance_before"].notna().all()

    assert transactions["balance_after"].notna().all()

    # Amounts must be positive
    assert (
        transactions["amount"] > 0
    ).all()

    # Only valid transaction types
    assert transactions[
        "transaction_type"
    ].isin(
        ["CREDIT", "DEBIT"]
    ).all()

    # =========================================================
    # TRANSACTION TYPES
    # =========================================================

    print("\nTransaction type distribution:")

    print(
        transactions[
            "transaction_type"
        ].value_counts()
    )

    # Both transaction types must exist
    assert (
        transactions["transaction_type"]
        == "CREDIT"
    ).any()

    assert (
        transactions["transaction_type"]
        == "DEBIT"
    ).any()

    # =========================================================
    # CATEGORIES
    # =========================================================

    print("\nCategory distribution:")

    print(
        transactions[
            "merchant_category"
        ].value_counts()
    )

    # Salary must exist
    assert "SALARY" in set(
        transactions["merchant_category"]
    )

    # =========================================================
    # SALARY COUNT
    # =========================================================

    salary_count = (
        transactions[
            transactions["merchant_category"]
            == "SALARY"
        ]
        .groupby("user_id")
        .size()
    )

    print("\nSalary transactions per user:")

    print(salary_count)

    # Every user must receive salary once per month
    assert (
        salary_count == N_MONTHS
    ).all()

    # =========================================================
    # BALANCE VALIDATION
    # =========================================================

    print(
        "\nChecking accounting invariant..."
    )

    # Sort exactly as the transaction generator
    # defines the ledger order.
    transactions_sorted = (
        transactions
        .sort_values(
            [
                "user_id",
                "timestamp",
                "transaction_id",
            ],
            kind="mergesort",
        )
        .reset_index(drop=True)
    )

    for user_id, user_transactions in (
        transactions_sorted.groupby(
            "user_id",
            sort=False,
        )
    ):

        # -----------------------------------------------------
        # Starting point
        # -----------------------------------------------------
        #
        # The first transaction's balance_before is the
        # starting balance represented by the transaction
        # ledger.
        #
        # We deliberately do NOT reconstruct this from
        # initial_balances here. The purpose of this test is
        # to validate the transaction ledger itself.

        expected_balance = float(
            user_transactions.iloc[0][
                "balance_before"
            ]
        )

        for _, row in user_transactions.iterrows():

            # -------------------------------------------------
            # Check balance_before
            # -------------------------------------------------

            assert np.isclose(
                expected_balance,
                row["balance_before"],
                atol=0.01,
            ), (
                "Balance-before mismatch for "
                f"{user_id} at "
                f"{row['timestamp']}"
            )

            # -------------------------------------------------
            # Apply transaction
            # -------------------------------------------------

            if row["transaction_type"] == "CREDIT":

                expected_balance += float(
                    row["amount"]
                )

            elif row["transaction_type"] == "DEBIT":

                expected_balance -= float(
                    row["amount"]
                )

            else:

                raise AssertionError(
                    "Invalid transaction type: "
                    f"{row['transaction_type']}"
                )

            # -------------------------------------------------
            # Check balance_after
            # -------------------------------------------------

            assert np.isclose(
                expected_balance,
                row["balance_after"],
                atol=0.01,
            ), (
                "Balance-after mismatch for "
                f"{user_id} at "
                f"{row['timestamp']}"
            )

    print(
        "Accounting invariant passed."
    )

    # =========================================================
    # BALANCE CONTINUITY
    # =========================================================

    print(
        "\nChecking balance continuity..."
    )

    for user_id, user_transactions in (
        transactions_sorted.groupby(
            "user_id",
            sort=False,
        )
    ):

        balance_after = (
            user_transactions[
                "balance_after"
            ]
            .iloc[:-1]
            .to_numpy()
        )

        next_balance_before = (
            user_transactions[
                "balance_before"
            ]
            .iloc[1:]
            .to_numpy()
        )

        assert np.allclose(
            balance_after,
            next_balance_before,
            atol=0.01,
        ), (
            "Balance continuity failed for "
            f"{user_id}"
        )

    print(
        "Balance continuity passed."
    )

    # =========================================================
    # FINAL BALANCE CHECK
    # =========================================================

    print("\nFinal balances:")

    final_balances = (
        transactions_sorted
        .groupby("user_id")
        .tail(1)
        [
            [
                "user_id",
                "balance_after",
            ]
        ]
    )

    print(final_balances)

    assert (
        final_balances["balance_after"]
        .notna()
        .all()
    )

    # =========================================================
    # CATEGORY SANITY
    # =========================================================

    variable_categories = {
        "Food",
        "Groceries",
        "Transport",
        "Shopping",
        "Entertainment",
        "Travel",
        "Healthcare",
        "Education",
        "Other",
    }

    recurring_categories = {
        "Rent",
        "EMI",
        "Utility",
        "Subscription",
        "Other_Recurring",
    }

    generated_categories = set(
        transactions[
            "merchant_category"
        ]
    )

    # At least one variable category
    assert generated_categories.intersection(
        variable_categories
    )

    # At least one recurring category
    assert generated_categories.intersection(
        recurring_categories
    )

    # =========================================================
    # CREDIT / DEBIT ACCOUNTING SANITY
    # =========================================================

    print(
        "\nChecking transaction accounting..."
    )

    credit_total = transactions.loc[
        transactions["transaction_type"]
        == "CREDIT",
        "amount",
    ].sum()

    debit_total = transactions.loc[
        transactions["transaction_type"]
        == "DEBIT",
        "amount",
    ].sum()

    assert credit_total > 0
    assert debit_total > 0

    print(
        f"Total credits: "
        f"{credit_total:.2f}"
    )

    print(
        f"Total debits: "
        f"{debit_total:.2f}"
    )

    # =========================================================
    # REPRODUCIBILITY TEST
    # =========================================================

    print(
        "\nChecking reproducibility..."
    )

    rng2 = create_rng(SEED)

    population2 = generate_population(
        N_USERS,
        rng2,
    )

    population2 = generate_base_income(
        population2,
        rng2,
    )

    population2 = generate_salary_days(
        population2,
        rng2,
    )

    income2 = generate_monthly_income(
        population2,
        N_MONTHS,
        rng2,
    )

    obligations2 = generate_obligations(
        population2,
        rng2,
    )

    spending2 = generate_spending_profile(
        population2,
        rng2,
    )

    initial_balances2 = (
        generate_initial_balances(
            population2,
            income2,
            rng2,
        )
    )

    transactions2 = generate_transactions(
        population=population2,
        income_df=income2,
        obligations_df=obligations2,
        spending_df=spending2,
        initial_balance_df=initial_balances2,
        n_months=N_MONTHS,
        rng=rng2,
    )

    # Include both balance columns because they are now part
    # of the transaction ledger.
    comparison_columns = [
        "transaction_id",
        "user_id",
        "timestamp",
        "transaction_type",
        "merchant_category",
        "amount",
        "payment_method",
        "balance_before",
        "balance_after",
    ]

    pd.testing.assert_frame_equal(
        transactions[
            comparison_columns
        ].reset_index(drop=True),
        transactions2[
            comparison_columns
        ].reset_index(drop=True),
    )

    print(
        "Reproducibility test passed."
    )

    # =========================================================
    # FINAL RESULT
    # =========================================================

    print(
        "\nTransaction generation validation passed."
    )


if __name__ == "__main__":
    main()
