
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

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
from finsight.data_generation.transactions import (
    generate_transactions,
)
from finsight.features.monthly_profile import (
    build_monthly_profile,
)
from finsight.utils.random import create_rng


SEED = 42
N_USERS = 10
N_MONTHS = 2


def generate_test_transactions():
    """Generate a small deterministic transaction dataset."""

    rng = create_rng(SEED)

    # ---------------------------------------------------------
    # POPULATION
    # ---------------------------------------------------------

    population = generate_population(
        N_USERS,
        rng,
    )

    # ---------------------------------------------------------
    # INCOME
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
    # OBLIGATIONS
    # ---------------------------------------------------------

    obligations = generate_obligations(
        population,
        rng,
    )

    # ---------------------------------------------------------
    # SPENDING PROFILE
    # ---------------------------------------------------------

    spending = generate_spending_profile(
        population,
        rng,
    )

    # ---------------------------------------------------------
    # INITIAL BALANCE
    # ---------------------------------------------------------

    initial_balances = generate_initial_balances(
        population,
        income,
        rng,
    )

    # ---------------------------------------------------------
    # TRANSACTIONS
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

    transactions = generate_test_transactions()

    # ---------------------------------------------------------
    # BUILD MONTHLY PROFILE
    # ---------------------------------------------------------

    profile = build_monthly_profile(
        transactions
    )

    # ---------------------------------------------------------
    # BASIC INFORMATION
    # ---------------------------------------------------------

    print("\nMonthly profile shape:")
    print(profile.shape)

    print("\nColumns:")
    print(profile.columns.tolist())

    print("\nMissing values:")
    print(profile.isna().sum())

    # ---------------------------------------------------------
    # REQUIRED COLUMNS
    # ---------------------------------------------------------

    required_columns = {
        "user_id",
        "month",

        # Income
        "salary_income",
        "bonus_income",
        "total_income",

        # Spending
        "total_debits",
        "variable_spending",
        "essential_spending",
        "discretionary_spending",
        "recurring_spending",

        # Balance
        "starting_balance",
        "ending_balance",
        "balance_change",
        "minimum_balance",
        "maximum_balance",
        "average_balance",

        # Transactions
        "transaction_count",
        "debit_transaction_count",
        "credit_transaction_count",

        # Financial metrics
        "net_cash_flow",
        "savings_rate",
        "income_to_expense_ratio",
    }

    missing_columns = (
        required_columns
        - set(profile.columns)
    )

    assert not missing_columns, (
        "Missing profile columns: "
        f"{sorted(missing_columns)}"
    )

    # ---------------------------------------------------------
    # BASIC STRUCTURE
    # ---------------------------------------------------------

    assert not profile.empty

    assert (
        profile[
            ["user_id", "month"]
        ]
        .duplicated()
        .sum()
        == 0
    )

    assert (
        profile["user_id"].nunique()
        == N_USERS
    )

    assert (
        profile["month"].nunique()
        == N_MONTHS
    )

    expected_rows = (
        N_USERS * N_MONTHS
    )

    assert len(profile) == expected_rows

    # ---------------------------------------------------------
    # MISSING VALUES
    # ---------------------------------------------------------

    assert not profile.isna().any().any()

    # ---------------------------------------------------------
    # INCOME VALIDATION
    # ---------------------------------------------------------

    assert (
        profile["salary_income"] >= 0
    ).all()

    assert (
        profile["bonus_income"] >= 0
    ).all()

    assert (
        profile["total_income"] >= 0
    ).all()

    calculated_total_income = (
        profile["salary_income"]
        + profile["bonus_income"]
    )

    assert np.allclose(
        calculated_total_income,
        profile["total_income"],
        atol=0.01,
    )

    # ---------------------------------------------------------
    # SPENDING VALIDATION
    # ---------------------------------------------------------

    assert (
        profile["total_debits"] >= 0
    ).all()

    assert (
        profile["variable_spending"] >= 0
    ).all()

    assert (
        profile["essential_spending"] >= 0
    ).all()

    assert (
        profile["discretionary_spending"] >= 0
    ).all()

    assert (
        profile["recurring_spending"] >= 0
    ).all()

    # ---------------------------------------------------------
    # SPENDING COMPOSITION
    # ---------------------------------------------------------

    calculated_variable_spending = (
        profile["essential_spending"]
        + profile["discretionary_spending"]
    )

    # Essential + discretionary are subsets
    # of variable spending.
    assert (
        calculated_variable_spending
        <= profile["variable_spending"] + 0.01
    ).all()

    # ---------------------------------------------------------
    # CASH FLOW
    # ---------------------------------------------------------

    calculated_cash_flow = (
        profile["total_income"]
        - profile["total_debits"]
    )

    assert np.allclose(
        calculated_cash_flow,
        profile["net_cash_flow"],
        atol=0.01,
    )

    # ---------------------------------------------------------
    # BALANCE VALIDATION
    # ---------------------------------------------------------

    # Starting balance must be finite.
    assert np.isfinite(
        profile["starting_balance"]
    ).all()

    # Ending balance must be finite.
    assert np.isfinite(
        profile["ending_balance"]
    ).all()

    # Balance change must be finite.
    assert np.isfinite(
        profile["balance_change"]
    ).all()

    # Starting balance + net cash flow
    # should equal ending balance.
    calculated_ending_balance = (
        profile["starting_balance"]
        + profile["net_cash_flow"]
    )

    assert np.allclose(
        calculated_ending_balance,
        profile["ending_balance"],
        atol=0.01,
    )

    # Balance change should equal
    # ending balance - starting balance.
    calculated_balance_change = (
        profile["ending_balance"]
        - profile["starting_balance"]
    )

    assert np.allclose(
        calculated_balance_change,
        profile["balance_change"],
        atol=0.01,
    )

    # ---------------------------------------------------------
    # SAVINGS RATE
    # ---------------------------------------------------------

    valid_income = (
        profile["total_income"] > 0
    )

    calculated_savings_rate = (
        profile.loc[
            valid_income,
            "net_cash_flow",
        ]
        / profile.loc[
            valid_income,
            "total_income",
        ]
    )

    assert np.allclose(
        calculated_savings_rate,
        profile.loc[
            valid_income,
            "savings_rate",
        ],
        atol=1e-10,
    )

    # ---------------------------------------------------------
    # INCOME / EXPENSE RATIO
    # ---------------------------------------------------------

    valid_expenses = (
        profile["total_debits"] > 0
    )

    calculated_income_expense_ratio = (
        profile.loc[
            valid_expenses,
            "total_income",
        ]
        / profile.loc[
            valid_expenses,
            "total_debits",
        ]
    )

    assert np.allclose(
        calculated_income_expense_ratio,
        profile.loc[
            valid_expenses,
            "income_to_expense_ratio",
        ],
        atol=1e-10,
    )

    # ---------------------------------------------------------
    # TRANSACTION COUNTS
    # ---------------------------------------------------------

    assert (
        profile["transaction_count"] > 0
    ).all()

    assert (
        profile["debit_transaction_count"] > 0
    ).all()

    assert (
        profile["credit_transaction_count"] > 0
    ).all()

    calculated_transaction_count = (
        profile["debit_transaction_count"]
        + profile["credit_transaction_count"]
    )

    assert (
        calculated_transaction_count
        == profile["transaction_count"]
    ).all()

    # ---------------------------------------------------------
    # BALANCE SANITY
    # ---------------------------------------------------------

    assert (
        profile["minimum_balance"]
        <= profile["maximum_balance"]
    ).all()

    assert (
        profile["minimum_balance"]
        <= profile["average_balance"] + 0.01
    ).all()

    assert (
        profile["average_balance"]
        <= profile["maximum_balance"] + 0.01
    ).all()

    # Starting and ending balances should
    # also be within the observed balance range.
    assert (
        profile["starting_balance"]
        <= profile["maximum_balance"] + 0.01
    ).all()

    assert (
        profile["ending_balance"]
        <= profile["maximum_balance"] + 0.01
    ).all()

    # ---------------------------------------------------------
    # DISPLAY
    # ---------------------------------------------------------

    display_columns = [
        "user_id",
        "month",
        "total_income",
        "total_debits",
        "variable_spending",
        "essential_spending",
        "discretionary_spending",
        "recurring_spending",
        "starting_balance",
        "ending_balance",
        "balance_change",
        "net_cash_flow",
        "savings_rate",
        "minimum_balance",
        "transaction_count",
    ]

    print("\nMonthly financial profile:")

    print(
        profile[display_columns]
        .round(2)
        .to_string(index=False)
    )

    print(
        "\nMonthly profile validation passed."
    )


if __name__ == "__main__":
    main()