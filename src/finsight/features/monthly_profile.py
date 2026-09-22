import numpy as np
import pandas as pd


VARIABLE_CATEGORIES = {
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

ESSENTIAL_CATEGORIES = {
    "Food",
    "Groceries",
    "Transport",
    "Healthcare",
    "Education",
}

DISCRETIONARY_CATEGORIES = {
    "Shopping",
    "Entertainment",
    "Travel",
    "Other",
}

RECURRING_CATEGORIES = {
    "Rent",
    "EMI",
    "Utility",
    "Subscription",
    "Other_Recurring",
}


def _validate_transactions(transactions: pd.DataFrame) -> None:
    """Validate the transaction-level input."""

    required_columns = {
        "transaction_id",
        "user_id",
        "timestamp",
        "transaction_type",
        "merchant_category",
        "amount",
        "balance_after",
    }

    missing = required_columns - set(transactions.columns)

    if missing:
        raise ValueError(
            "transactions is missing columns: "
            f"{sorted(missing)}"
        )

    if transactions.empty:
        raise ValueError(
            "transactions cannot be empty."
        )

    if transactions["transaction_id"].duplicated().any():
        raise ValueError(
            "transaction_id must be unique."
        )

    if transactions["user_id"].isna().any():
        raise ValueError(
            "user_id cannot contain missing values."
        )

    if transactions["timestamp"].isna().any():
        raise ValueError(
            "timestamp cannot contain missing values."
        )

    if transactions["amount"].isna().any():
        raise ValueError(
            "amount cannot contain missing values."
        )

    if (transactions["amount"] <= 0).any():
        raise ValueError(
            "Transaction amounts must be positive."
        )

    if not transactions["transaction_type"].isin(
        ["CREDIT", "DEBIT"]
    ).all():
        raise ValueError(
            "Invalid transaction type found."
        )


def build_monthly_profile(
    transactions: pd.DataFrame,
) -> pd.DataFrame:
    """
    Convert transaction-level data into one row per user-month.

    This function is an aggregation layer only.

    Input:
        Transaction-level financial ledger.

    Output:
        One observation per user-month containing:
        - income
        - spending
        - recurring obligations
        - transaction behavior
        - cash-flow measures
        - balance statistics

    No future-looking features are created here.
    """

    _validate_transactions(transactions)

    df = transactions.copy()

    # ---------------------------------------------------------
    # DATETIME
    # ---------------------------------------------------------

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        errors="raise",
    )

    # Make sure transactions are chronologically ordered.
    df = df.sort_values(
        [
            "user_id",
            "timestamp",
            "transaction_id",
        ],
        kind="mergesort",
    ).reset_index(drop=True)

    # ---------------------------------------------------------
    # MONTH
    # ---------------------------------------------------------

    df["month"] = (
        df["timestamp"]
        .dt.to_period("M")
        .astype(str)
    )

    # ---------------------------------------------------------
    # CREDIT / DEBIT FLAGS
    # ---------------------------------------------------------

    df["credit_amount"] = np.where(
        df["transaction_type"] == "CREDIT",
        df["amount"],
        0.0,
    )

    df["debit_amount"] = np.where(
        df["transaction_type"] == "DEBIT",
        df["amount"],
        0.0,
    )

    # ---------------------------------------------------------
    # TRANSACTION CLASSIFICATION
    # ---------------------------------------------------------

    df["is_salary"] = (
        df["merchant_category"] == "SALARY"
    )

    df["is_bonus"] = (
        df["merchant_category"] == "BONUS"
    )

    df["is_variable_spending"] = (
        df["merchant_category"].isin(
            VARIABLE_CATEGORIES
        )
    )

    df["is_essential"] = (
        df["merchant_category"].isin(
            ESSENTIAL_CATEGORIES
        )
    )

    df["is_discretionary"] = (
        df["merchant_category"].isin(
            DISCRETIONARY_CATEGORIES
        )
    )

    df["is_recurring"] = (
        df["merchant_category"].isin(
            RECURRING_CATEGORIES
        )
    )

    # ---------------------------------------------------------
    # USER-MONTH AGGREGATION
    # ---------------------------------------------------------

    grouped = df.groupby(
        ["user_id", "month"],
        sort=True,
    )

    records = []

    for (user_id, month), group in grouped:

        # -----------------------------------------------------
        # INCOME
        # -----------------------------------------------------

        salary_income = group.loc[
            group["is_salary"],
            "amount",
        ].sum()

        bonus_income = group.loc[
            group["is_bonus"],
            "amount",
        ].sum()

        total_income = (
            salary_income
            + bonus_income
        )

        # -----------------------------------------------------
        # TOTAL SPENDING
        # -----------------------------------------------------

        total_debits = group[
            "debit_amount"
        ].sum()

        # -----------------------------------------------------
        # VARIABLE SPENDING
        # -----------------------------------------------------

        variable_spending = group.loc[
            group["is_variable_spending"],
            "amount",
        ].sum()

        essential_spending = group.loc[
            group["is_essential"],
            "amount",
        ].sum()

        discretionary_spending = group.loc[
            group["is_discretionary"],
            "amount",
        ].sum()

        recurring_spending = group.loc[
            group["is_recurring"],
            "amount",
        ].sum()

        # -----------------------------------------------------
        # TRANSACTION STATISTICS
        # -----------------------------------------------------

        transaction_count = len(group)

        debit_transaction_count = (
            group["transaction_type"]
            .eq("DEBIT")
            .sum()
        )

        credit_transaction_count = (
            group["transaction_type"]
            .eq("CREDIT")
            .sum()
        )

        debit_amounts = group.loc[
            group["transaction_type"] == "DEBIT",
            "amount",
        ]

        average_transaction = (
            group["amount"].mean()
        )

        median_transaction = (
            group["amount"].median()
        )

        transaction_amount_std = (
            group["amount"].std()
        )

        # -----------------------------------------------------
        # BALANCE STATISTICS
        # -----------------------------------------------------

        minimum_balance = (
            group["balance_after"].min()
        )

        maximum_balance = (
            group["balance_after"].max()
        )

        average_balance = (
            group["balance_after"].mean()
        )

        ending_balance = (
            group
            .sort_values(
                [
                    "timestamp",
                    "transaction_id",
                ],
                kind="mergesort",
            )
            .iloc[-1]["balance_after"]
        )

        starting_balance = (
        group
        .sort_values(
            [
                "timestamp",
                "transaction_id",
            ],
            kind="mergesort",
        )
        .iloc[0]["balance_before"]
    )

        # -----------------------------------------------------
        # CASH FLOW
        # -----------------------------------------------------

        net_cash_flow = (
            total_income
            - total_debits
        )

        if total_income > 0:
            savings_rate = (
                net_cash_flow
                / total_income
            )
        else:
            savings_rate = np.nan

        if total_debits > 0:
            income_to_expense_ratio = (
                total_income
                / total_debits
            )
        else:
            income_to_expense_ratio = np.nan

        # -----------------------------------------------------
        # SPENDING COMPOSITION
        # -----------------------------------------------------

        if variable_spending > 0:

            essential_spending_ratio = (
                essential_spending
                / variable_spending
            )

            discretionary_spending_ratio = (
                discretionary_spending
                / variable_spending
            )

        else:

            essential_spending_ratio = np.nan
            discretionary_spending_ratio = np.nan

        # -----------------------------------------------------
        # LIQUIDITY
        # -----------------------------------------------------

        if total_income > 0:

            minimum_balance_ratio = (
                minimum_balance
                / total_income
            )

            average_balance_ratio = (
                average_balance
                / total_income
            )

        else:

            minimum_balance_ratio = np.nan
            average_balance_ratio = np.nan

        # -----------------------------------------------------
        # TRANSACTION FREQUENCY
        # -----------------------------------------------------

        transactions_per_day = (
            transaction_count / 30.0
        )

        debit_transactions_per_day = (
            debit_transaction_count / 30.0
        )

        # -----------------------------------------------------
        # RECORD
        # -----------------------------------------------------

        records.append(
            {
                "user_id": user_id,
                "month": month,

                # Income
                "salary_income": salary_income,
                "bonus_income": bonus_income,
                "total_income": total_income,

                # Spending
                "total_debits": total_debits,
                "variable_spending": variable_spending,
                "essential_spending": essential_spending,
                "discretionary_spending": (
                    discretionary_spending
                ),
                "recurring_spending": (
                    recurring_spending
                ),

                # Transactions
                "transaction_count": (
                    transaction_count
                ),
                "debit_transaction_count": (
                    debit_transaction_count
                ),
                "credit_transaction_count": (
                    credit_transaction_count
                ),
                "transactions_per_day": (
                    transactions_per_day
                ),
                "debit_transactions_per_day": (
                    debit_transactions_per_day
                ),
                "average_transaction": (
                    average_transaction
                ),
                "median_transaction": (
                    median_transaction
                ),
                "transaction_amount_std": (
                    transaction_amount_std
                ),

                # Balance / liquidity
                #
                # first_balance_after is deliberately named
                # this way because it is NOT the true beginning
                # balance of the month.
                "starting_balance": (
                    starting_balance
                ),
                "ending_balance": (
                    ending_balance
                ),
                "minimum_balance": (
                    minimum_balance
                ),
                "maximum_balance": (
                    maximum_balance
                ),
                "average_balance": (
                    average_balance
                ),

                # Cash flow
                "net_cash_flow": (
                    net_cash_flow
                ),
                "savings_rate": (
                    savings_rate
                ),
                "income_to_expense_ratio": (
                    income_to_expense_ratio
                ),

                # Spending composition
                "essential_spending_ratio": (
                    essential_spending_ratio
                ),
                "discretionary_spending_ratio": (
                    discretionary_spending_ratio
                ),

                # Liquidity ratios
                "minimum_balance_ratio": (
                    minimum_balance_ratio
                ),
                "average_balance_ratio": (
                    average_balance_ratio
                ),
            }
        )

    profile = pd.DataFrame(records)
    profile["balance_change"] = (
        profile["ending_balance"]
        - profile["starting_balance"]
    )

    # ---------------------------------------------------------
    # NUMERIC CLEANUP
    # ---------------------------------------------------------

    numeric_columns = profile.select_dtypes(
        include=[np.number]
    ).columns

    profile[numeric_columns] = (
        profile[numeric_columns]
        .replace(
            [np.inf, -np.inf],
            np.nan,
        )
    )

    # ---------------------------------------------------------
    # SORT FINAL OUTPUT
    # ---------------------------------------------------------

    profile = profile.sort_values(
        ["user_id", "month"],
        kind="mergesort",
    ).reset_index(drop=True)

    return profile