
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
from finsight.data_generation.behavior import (
    generate_behavioral_profile,
)
from finsight.data_generation.shocks import (
    generate_shocks,
)
from finsight.data_generation.transactions import (
    generate_transactions,
)
from finsight.utils.random import create_rng


SEED = 42
N_USERS = 100
N_MONTHS = 12


def build_inputs(seed):

    rng = create_rng(seed)

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

    income = generate_monthly_income(
        population,
        N_MONTHS,
        rng,
    )

    obligations = generate_obligations(
        population,
        rng,
    )

    spending = generate_spending_profile(
        population,
        rng,
    )

    initial_balances = generate_initial_balances(
        population,
        income,
        rng,
    )

    behavior = generate_behavioral_profile(
        population,
        N_MONTHS,
        rng,
    )

    shocks = generate_shocks(
        population,
        N_MONTHS,
        rng,
    )

    return (
        population,
        income,
        obligations,
        spending,
        initial_balances,
        behavior,
        shocks,
    )


def main():

    (
        population,
        income,
        obligations,
        spending,
        initial_balances,
        behavior,
        shocks,
    ) = build_inputs(SEED)

    rng = create_rng(SEED + 1)

    # =========================================================
    # INTEGRATED TRANSACTIONS
    # =========================================================

    transactions = generate_transactions(
        population=population,
        income_df=income,
        obligations_df=obligations,
        spending_df=spending,
        initial_balance_df=initial_balances,
        n_months=N_MONTHS,
        rng=rng,
        behavior_df=behavior,
        shock_df=shocks,
    )

    print("\nIntegrated transaction shape:")
    print(transactions.shape)

    print("\nCategories:")
    print(
        transactions[
            "merchant_category"
        ].value_counts()
    )

    # =========================================================
    # BASIC VALIDATION
    # =========================================================

    assert not transactions.empty

    assert transactions[
        "transaction_id"
    ].is_unique

    assert (
        transactions["user_id"].nunique()
        == N_USERS
    )

    assert not transactions.isna().any().any()

    assert (
        transactions["amount"] > 0
    ).all()

    assert transactions[
        "transaction_type"
    ].isin(
        ["CREDIT", "DEBIT"]
    ).all()

    # =========================================================
    # SHOCK TRANSACTION VALIDATION
    # =========================================================

    expense_shocks = shocks[
        shocks["shock_type"]
        != "INCOME_DISRUPTION"
    ]

    expected_shock_types = set(
        expense_shocks["shock_type"]
    )

    shock_category_map = {
        "MEDICAL": "Healthcare",
        "TRAVEL": "Travel",
        "LARGE_PURCHASE": "Shopping",
        "EMERGENCY": "Other",
        "UNEXPECTED_BILL": "Other",
    }

    for shock_type in expected_shock_types:

        shock = expense_shocks[
            expense_shocks["shock_type"]
            == shock_type
        ]

        for _, shock_row in shock.iterrows():

            user_transactions = transactions[
                transactions["user_id"]
                == shock_row["user_id"]
            ]

            matching = user_transactions[
                (
                    user_transactions[
                        "merchant_category"
                    ]
                    == shock_category_map[
                        shock_type
                    ]
                )
                & (
                    user_transactions[
                        "amount"
                    ].sub(
                        shock_row["shock_amount"]
                    ).abs()
                    < 0.01
                )
                & (
                    user_transactions[
                        "transaction_type"
                    ]
                    == "DEBIT"
                )
            ]

            assert not matching.empty, (
                f"Missing integrated shock "
                f"transaction for "
                f"{shock_row['user_id']} "
                f"{shock_type}"
            )

    # =========================================================
    # ACCOUNTING INVARIANT
    # =========================================================

    print(
        "\nChecking accounting invariant..."
    )

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

        expected_balance = float(
            group.iloc[0][
                "balance_before"
            ]
        )

        for _, row in group.iterrows():

            assert np.isclose(
                expected_balance,
                row["balance_before"],
                atol=0.01,
            )

            if row["transaction_type"] == "CREDIT":
                expected_balance += float(
                    row["amount"]
                )
            else:
                expected_balance -= float(
                    row["amount"]
                )

            assert np.isclose(
                expected_balance,
                row["balance_after"],
                atol=0.01,
            )

    print(
        "Accounting invariant passed."
    )

    # =========================================================
    # BALANCE CONTINUITY
    # =========================================================

    for user_id, group in ordered.groupby(
        "user_id",
        sort=False,
    ):

        assert np.allclose(
            group["balance_after"]
            .iloc[:-1]
            .to_numpy(),

            group["balance_before"]
            .iloc[1:]
            .to_numpy(),

            atol=0.01,
        )

    print(
        "Balance continuity passed."
    )

    # =========================================================
    # INCOME DISRUPTION VALIDATION
    # =========================================================

    income_disruptions = shocks[
        shocks["shock_type"]
        == "INCOME_DISRUPTION"
    ]

    for _, shock in income_disruptions.iterrows():

        user_id = shock["user_id"]
        month = int(shock["month"])

        income_row = income[
            (
                income["user_id"]
                == user_id
            )
            & (
                income["month"]
                == month
            )
        ].iloc[0]

        expected_salary = (
            income_row["monthly_salary"]
            * (
                1.0
                - shock["income_impact"]
            )
        )

        user_transactions = transactions[
            transactions["user_id"]
            == user_id
        ].copy()

        user_transactions[
            "timestamp"
        ] = pd.to_datetime(
            user_transactions["timestamp"]
        )

        month_transactions = (
            user_transactions[
                user_transactions[
                    "timestamp"
                ].dt.month
                == month
            ]
        )

        salary_transactions = (
            month_transactions[
                month_transactions[
                    "merchant_category"
                ]
                == "SALARY"
            ]
        )

        assert not salary_transactions.empty

        actual_salary = float(
            salary_transactions.iloc[0][
                "amount"
            ]
        )

        assert np.isclose(
            actual_salary,
            expected_salary,
            atol=0.01,
        ), (
            f"Income disruption mismatch "
            f"for {user_id}, month {month}"
        )

    print(
        "Income disruption validation passed."
    )

    # =========================================================
    # REPRODUCIBILITY
    # =========================================================

    (
        population2,
        income2,
        obligations2,
        spending2,
        initial_balances2,
        behavior2,
        shocks2,
    ) = build_inputs(SEED)

    rng2 = create_rng(SEED + 1)

    transactions2 = generate_transactions(
        population=population2,
        income_df=income2,
        obligations_df=obligations2,
        spending_df=spending2,
        initial_balance_df=initial_balances2,
        n_months=N_MONTHS,
        rng=rng2,
        behavior_df=behavior2,
        shock_df=shocks2,
    )

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
        "Integrated reproducibility test passed."
    )

    # =========================================================
    # FINAL
    # =========================================================

    print(
        "\nBehavior + shock integration validation passed."
    )


if __name__ == "__main__":
    main()
