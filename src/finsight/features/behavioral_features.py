import numpy as np
import pandas as pd


REQUIRED_MONTHLY_COLUMNS = {
    "user_id",
    "month",
    "total_income",
    "total_debits",
    "variable_spending",
    "essential_spending",
    "discretionary_spending",
    "recurring_spending",
    "transaction_count",
    "average_transaction",
    "median_transaction",
    "transaction_amount_std",
    "starting_balance",
    "ending_balance",
    "minimum_balance",
    "maximum_balance",
    "average_balance",
    "net_cash_flow",
    "savings_rate",
}


REQUIRED_STATE_COLUMNS = {
    "user_id",
    "month",
    "financial_state",
    "previous_financial_state",
    "state_changed",
    "state_transition",
}


REQUIRED_TRANSACTION_COLUMNS = {
    "user_id",
    "timestamp",
    "amount",
    "transaction_type",
    "merchant_category",
}


# ============================================================
# VALIDATION
# ============================================================

def _validate_inputs(
    monthly_profile: pd.DataFrame,
    state_df: pd.DataFrame,
    transactions_df: pd.DataFrame,
):
    missing_monthly = (
        REQUIRED_MONTHLY_COLUMNS
        - set(monthly_profile.columns)
    )

    if missing_monthly:
        raise ValueError(
            "Monthly profile is missing columns: "
            f"{sorted(missing_monthly)}"
        )

    missing_state = (
        REQUIRED_STATE_COLUMNS
        - set(state_df.columns)
    )

    if missing_state:
        raise ValueError(
            "State dataframe is missing columns: "
            f"{sorted(missing_state)}"
        )

    missing_transactions = (
        REQUIRED_TRANSACTION_COLUMNS
        - set(transactions_df.columns)
    )

    if missing_transactions:
        raise ValueError(
            "Transactions dataframe is missing columns: "
            f"{sorted(missing_transactions)}"
        )


# ============================================================
# MONTH HANDLING
# ============================================================

def _prepare_monthly_profile(
    monthly_profile: pd.DataFrame,
) -> pd.DataFrame:

    df = monthly_profile.copy()

    df["month"] = pd.to_datetime(
        df["month"],
        format="%Y-%m",
    )

    df = df.sort_values(
        ["user_id", "month"]
    ).reset_index(drop=True)

    duplicate_rows = df.duplicated(
        subset=["user_id", "month"],
        keep=False,
    )

    if duplicate_rows.any():
        raise ValueError(
            "Duplicate user-month observations found "
            "in monthly profile."
        )

    return df


# ============================================================
# MONTHLY CATEGORY FEATURES
# ============================================================

def _build_category_features(
    transactions_df: pd.DataFrame,
) -> pd.DataFrame:

    df = transactions_df.copy()

    df["timestamp"] = pd.to_datetime(
        df["timestamp"]
    )

    df["month"] = (
        df["timestamp"]
        .dt.to_period("M")
        .dt.to_timestamp()
    )

    # Only debit transactions represent spending.
    debit = df[
        df["transaction_type"].astype(str).str.upper()
        == "DEBIT"
    ].copy()

    if debit.empty:
        return pd.DataFrame(
            columns=[
                "user_id",
                "month",
                "category_spending",
            ]
        )

    category_spending = (
        debit
        .groupby(
            [
                "user_id",
                "month",
                "merchant_category",
            ],
            as_index=False,
        )["amount"]
        .sum()
        .rename(
            columns={
                "amount": "category_spending"
            }
        )
    )

    return category_spending


# ============================================================
# PERSONAL BASELINES
# ============================================================

def _add_personal_baselines(
    df: pd.DataFrame,
) -> pd.DataFrame:

    df = df.copy()

    grouped = df.groupby("user_id")

    # Shift first so the current month is NEVER included
    # in its own historical baseline.
    previous_spending = (
        grouped["variable_spending"]
        .shift(1)
    )

    previous_transaction_count = (
        grouped["transaction_count"]
        .shift(1)
    )

    previous_savings_rate = (
        grouped["savings_rate"]
        .shift(1)
    )

    df["historical_spending_mean"] = (
        grouped["variable_spending"]
        .transform(
            lambda x: x.shift(1)
            .expanding(min_periods=1)
            .mean()
        )
    )

    df["historical_spending_std"] = (
        grouped["variable_spending"]
        .transform(
            lambda x: x.shift(1)
            .expanding(min_periods=2)
            .std()
        )
    )

    df["spending_baseline_deviation"] = (
        df["variable_spending"]
        - df["historical_spending_mean"]
    )

    df["spending_baseline_ratio"] = (
        df["variable_spending"]
        / df["historical_spending_mean"]
        .replace(0, np.nan)
    )

    df["transaction_count_change"] = (
        df["transaction_count"]
        - previous_transaction_count
    )

    df["savings_rate_change"] = (
        df["savings_rate"]
        - previous_savings_rate
    )

    return df


# ============================================================
# ROLLING FEATURES
# ============================================================

def _add_rolling_features(
    df: pd.DataFrame,
    window_months: int,
) -> pd.DataFrame:

    df = df.copy()

    grouped = df.groupby(
        "user_id",
        group_keys=False,
    )

    # --------------------------------------------------------
    # Rolling spending
    # --------------------------------------------------------

    df["rolling_spending_mean"] = (
        grouped["variable_spending"]
        .transform(
            lambda x: x.rolling(
                window_months,
                min_periods=window_months,
            ).mean()
        )
    )

    df["rolling_spending_std"] = (
        grouped["variable_spending"]
        .transform(
            lambda x: x.rolling(
                window_months,
                min_periods=window_months,
            ).std()
        )
    )

    df["rolling_spending_min"] = (
        grouped["variable_spending"]
        .transform(
            lambda x: x.rolling(
                window_months,
                min_periods=window_months,
            ).min()
        )
    )

    df["rolling_spending_max"] = (
        grouped["variable_spending"]
        .transform(
            lambda x: x.rolling(
                window_months,
                min_periods=window_months,
            ).max()
        )
    )

    # --------------------------------------------------------
    # Income
    # --------------------------------------------------------

    df["rolling_income_mean"] = (
        grouped["total_income"]
        .transform(
            lambda x: x.rolling(
                window_months,
                min_periods=window_months,
            ).mean()
        )
    )

    df["income_volatility"] = (
        grouped["total_income"]
        .transform(
            lambda x: x.rolling(
                window_months,
                min_periods=window_months,
            ).std()
        )
    )

    # --------------------------------------------------------
    # Balance
    # --------------------------------------------------------

    df["rolling_average_balance"] = (
        grouped["average_balance"]
        .transform(
            lambda x: x.rolling(
                window_months,
                min_periods=window_months,
            ).mean()
        )
    )

    df["rolling_minimum_balance"] = (
        grouped["minimum_balance"]
        .transform(
            lambda x: x.rolling(
                window_months,
                min_periods=window_months,
            ).min()
        )
    )

    # --------------------------------------------------------
    # Savings
    # --------------------------------------------------------

    df["rolling_savings_rate"] = (
        grouped["savings_rate"]
        .transform(
            lambda x: x.rolling(
                window_months,
                min_periods=window_months,
            ).mean()
        )
    )

    # --------------------------------------------------------
    # Transactions
    # --------------------------------------------------------

    df["rolling_transaction_count"] = (
        grouped["transaction_count"]
        .transform(
            lambda x: x.rolling(
                window_months,
                min_periods=window_months,
            ).mean()
        )
    )

    # --------------------------------------------------------
    # Cash flow
    # --------------------------------------------------------

    df["rolling_net_cash_flow"] = (
        grouped["net_cash_flow"]
        .transform(
            lambda x: x.rolling(
                window_months,
                min_periods=window_months,
            ).mean()
        )
    )

    return df


# ============================================================
# GROWTH / ACCELERATION
# ============================================================

def _add_growth_features(
    df: pd.DataFrame,
) -> pd.DataFrame:

    df = df.copy()

    grouped = df.groupby("user_id")

    previous_spending = (
        grouped["variable_spending"]
        .shift(1)
    )

    previous_discretionary = (
        grouped["discretionary_spending"]
        .shift(1)
    )

    previous_income = (
        grouped["total_income"]
        .shift(1)
    )

    previous_balance = (
        grouped["ending_balance"]
        .shift(1)
    )

    df["spending_growth"] = (
        df["variable_spending"]
        / previous_spending.replace(0, np.nan)
        - 1
    )

    df["discretionary_spending_growth"] = (
        df["discretionary_spending"]
        / previous_discretionary.replace(
            0,
            np.nan,
        )
        - 1
    )

    df["income_growth"] = (
        df["total_income"]
        / previous_income.replace(0, np.nan)
        - 1
    )

    df["balance_growth"] = (
        df["ending_balance"]
        / previous_balance.replace(0, np.nan)
        - 1
    )

    # Acceleration = current growth - previous growth.
    previous_growth = (
        grouped["spending_growth"]
        .shift(1)
    )

    df["spending_acceleration"] = (
        df["spending_growth"]
        - previous_growth
    )

    return df


# ============================================================
# FINANCIAL STATE FEATURES
# ============================================================

def _add_state_features(
    df: pd.DataFrame,
    state_df: pd.DataFrame,
) -> pd.DataFrame:

    states = state_df[
        [
            "user_id",
            "month",
            "financial_state",
            "previous_financial_state",
            "state_changed",
            "state_transition",
        ]
    ].copy()

    states["month"] = pd.to_datetime(
        states["month"],
        format="%Y-%m",
    )

    df = df.merge(
        states,
        on=["user_id", "month"],
        how="left",
        validate="one_to_one",
    )

    if df["financial_state"].isna().any():
        raise ValueError(
            "Some monthly observations could not be "
            "matched with financial states."
        )

    # Duration of current state.
    df["state_duration"] = (
        df.groupby(
            [
                "user_id",
                "financial_state",
            ]
        )
        .cumcount()
        + 1
    )

    # Recent transition indicator.
    df["recent_state_change"] = (
        df["state_changed"]
        .astype(int)
    )

    return df


# ============================================================
# LIQUIDITY / PRESSURE FEATURES
# ============================================================

def _add_pressure_features(
    df: pd.DataFrame,
) -> pd.DataFrame:

    df = df.copy()

    df["cash_buffer"] = (
        df["minimum_balance"]
        .clip(lower=0)
    )

    df["cash_buffer_ratio"] = (
        df["cash_buffer"]
        / df["total_income"].replace(0, np.nan)
    )

    df["expense_pressure"] = (
        df["total_debits"]
        / df["total_income"].replace(0, np.nan)
    )

    df["discretionary_pressure"] = (
        df["discretionary_spending"]
        / df["total_income"].replace(0, np.nan)
    )

    df["negative_cash_flow"] = (
        df["net_cash_flow"] < 0
    ).astype(int)

    return df


# ============================================================
# CATEGORY SHIFT
# ============================================================

def _add_category_shift(
    df: pd.DataFrame,
    transactions_df: pd.DataFrame,
) -> pd.DataFrame:

    category_df = _build_category_features(
        transactions_df
    )

    if category_df.empty:
        df["category_shift"] = np.nan
        return df

    category_df["month"] = pd.to_datetime(
        category_df["month"]
    )

    # Convert category totals into monthly shares.
    totals = (
        category_df
        .groupby(
            ["user_id", "month"]
        )["category_spending"]
        .transform("sum")
    )

    category_df["category_share"] = (
        category_df["category_spending"]
        / totals.replace(0, np.nan)
    )

    category_pivot = (
        category_df
        .pivot_table(
            index=["user_id", "month"],
            columns="merchant_category",
            values="category_share",
            fill_value=0,
        )
        .reset_index()
    )

    category_columns = [
        column
        for column in category_pivot.columns
        if column not in ["user_id", "month"]
    ]

    category_pivot = category_pivot.sort_values(
        ["user_id", "month"]
    )

    # Month-to-month absolute distribution change.
    shifts = []

    for user_id, group in category_pivot.groupby(
        "user_id"
    ):
        group = group.sort_values("month").copy()

        previous = group[
            category_columns
        ].shift(1)

        current = group[
            category_columns
        ]

        shift = (
            current - previous
        ).abs().sum(axis=1)

        temp = pd.DataFrame({
            "user_id": group["user_id"].values,
            "month": group["month"].values,
            "category_shift": shift.values,
        })

        shifts.append(temp)

    shift_df = pd.concat(
        shifts,
        ignore_index=True,
    )

    df = df.merge(
        shift_df,
        on=["user_id", "month"],
        how="left",
        validate="one_to_one",
    )

    return df


# ============================================================
# MAIN FEATURE BUILDER
# ============================================================

def build_behavioral_features(
    monthly_profile: pd.DataFrame,
    state_df: pd.DataFrame,
    transactions_df: pd.DataFrame,
    window_months: int = 3,
) -> pd.DataFrame:

    if window_months < 2:
        raise ValueError(
            "window_months must be at least 2."
        )

    _validate_inputs(
        monthly_profile,
        state_df,
        transactions_df,
    )

    df = _prepare_monthly_profile(
        monthly_profile
    )

    # State information.
    df = _add_state_features(
        df,
        state_df,
    )

    # Historical baselines.
    df = _add_personal_baselines(
        df
    )

    # Rolling temporal features.
    df = _add_rolling_features(
        df,
        window_months=window_months,
    )

    # Growth and acceleration.
    df = _add_growth_features(
        df
    )

    # Liquidity and pressure.
    df = _add_pressure_features(
        df
    )

    # Category distribution changes.
    df = _add_category_shift(
        df,
        transactions_df,
    )

    # --------------------------------------------------------
    # Minimum history requirement
    # --------------------------------------------------------

    # A 3-month behavioral window requires 3 observations.
    # We do NOT create fake values for the first two months.
    df["has_full_history"] = (
        df.groupby("user_id")
        .cumcount()
        >= window_months - 1
    )

    # --------------------------------------------------------
    # Final ordering
    # --------------------------------------------------------

    df = df.sort_values(
        ["user_id", "month"]
    ).reset_index(drop=True)

    return df