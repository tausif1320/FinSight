import numpy as np
import pandas as pd


FINANCIAL_STATES = [
    "STABLE",
    "SPENDING_PRESSURE",
    "LIQUIDITY_STRESS",
    "RECOVERY",
]


REQUIRED_COLUMNS = {
    "user_id",
    "month",
    "total_income",
    "total_debits",
    "variable_spending",
    "discretionary_spending",
    "recurring_spending",
    "starting_balance",
    "ending_balance",
    "minimum_balance",
    "average_balance",
    "net_cash_flow",
    "savings_rate",
}


def _validate_monthly_profile(monthly_profile: pd.DataFrame):
    """
    Validate that the monthly financial profile contains
    the fields required by the financial-state engine.
    """
    missing = REQUIRED_COLUMNS - set(
        monthly_profile.columns
    )

    if missing:
        raise ValueError(
            "monthly_profile is missing required "
            f"columns: {sorted(missing)}"
        )

    if monthly_profile.empty:
        raise ValueError(
            "monthly_profile cannot be empty."
        )

    if monthly_profile[
        ["user_id", "month"]
    ].isna().any().any():
        raise ValueError(
            "user_id and month cannot contain missing values."
        )


def _prepare_profile(monthly_profile):
    """
    Prepare monthly profile for chronological
    user-level calculations.
    """
    df = monthly_profile.copy()

    df["month"] = pd.to_datetime(
        df["month"],
        format="%Y-%m",
        errors="raise",
    ).dt.to_period("M").astype(str)

    df = df.sort_values(
        [
            "user_id",
            "month",
        ],
        kind="mergesort",
    ).reset_index(drop=True)

    return df


def _safe_divide(numerator, denominator):
    """
    Element-wise safe division.
    """
    denominator = denominator.replace(
        0,
        np.nan,
    )

    return numerator / denominator


def _add_state_features(df):
    """
    Construct state-related financial indicators.

    These features use only current and historical monthly
    information. No future values are used here.
    """

    grouped = df.groupby(
        "user_id",
        sort=False,
    )

    # ---------------------------------------------------------
    # Liquidity
    # ---------------------------------------------------------

    df["cash_buffer"] = df[
        "minimum_balance"
    ].clip(lower=0)

    df["cash_buffer_ratio"] = _safe_divide(
        df["cash_buffer"],
        df["total_income"],
    )

    df["ending_balance_ratio"] = _safe_divide(
        df["ending_balance"],
        df["total_income"],
    )

    # ---------------------------------------------------------
    # Spending pressure
    # ---------------------------------------------------------

    df["expense_to_income_ratio"] = _safe_divide(
        df["total_debits"],
        df["total_income"],
    )

    df["discretionary_spending_ratio_income"] = (
        _safe_divide(
            df["discretionary_spending"],
            df["total_income"],
        )
    )

    # ---------------------------------------------------------
    # Historical behavior
    # ---------------------------------------------------------

    df["previous_variable_spending"] = grouped[
        "variable_spending"
    ].shift(1)

    df["previous_discretionary_spending"] = grouped[
        "discretionary_spending"
    ].shift(1)

    df["previous_savings_rate"] = grouped[
        "savings_rate"
    ].shift(1)

    df["previous_cash_buffer"] = grouped[
        "cash_buffer"
    ].shift(1)

    # Spending growth.
    df["spending_growth"] = _safe_divide(
        df["variable_spending"]
        - df["previous_variable_spending"],
        df["previous_variable_spending"].abs(),
    )

    # Discretionary spending growth.
    df["discretionary_spending_growth"] = (
        _safe_divide(
            df["discretionary_spending"]
            - df["previous_discretionary_spending"],
            df["previous_discretionary_spending"].abs(),
        )
    )

    # Savings deterioration.
    df["savings_rate_change"] = (
        df["savings_rate"]
        - df["previous_savings_rate"]
    )

    # Cash-buffer deterioration.
    df["cash_buffer_change"] = (
        df["cash_buffer"]
        - df["previous_cash_buffer"]
    )

    # ---------------------------------------------------------
    # Rolling personal baseline
    # ---------------------------------------------------------

    # Shift first so the current month is never part of
    # its own historical baseline.
    historical_spending = grouped[
        "variable_spending"
    ].shift(1)

    df["historical_spending_mean"] = (
        historical_spending
        .groupby(df["user_id"])
        .transform(
            lambda x: x.expanding(
                min_periods=1
            ).mean()
        )
    )

    df["historical_spending_std"] = (
        historical_spending
        .groupby(df["user_id"])
        .transform(
            lambda x: x.expanding(
                min_periods=2
            ).std()
        )
    )

    df["spending_baseline_deviation"] = (
        _safe_divide(
            df["variable_spending"]
            - df["historical_spending_mean"],
            df["historical_spending_mean"].abs(),
        )
    )

    df["spending_zscore"] = _safe_divide(
        df["variable_spending"]
        - df["historical_spending_mean"],
        df["historical_spending_std"],
    )

    # ---------------------------------------------------------
    # Negative / weak cash-flow indicators
    # ---------------------------------------------------------

    df["negative_cash_flow"] = (
        df["net_cash_flow"] < 0
    )

    df["low_savings"] = (
        df["savings_rate"] < 0.10
    )

    df["high_expense_ratio"] = (
        df["expense_to_income_ratio"] > 0.90
    )

    # ---------------------------------------------------------
    # State pressure indicators
    # ---------------------------------------------------------

    df["liquidity_pressure"] = (
        (df["cash_buffer_ratio"] < 0.25)
        |
        (df["ending_balance"] < 0)
        |
        (df["negative_cash_flow"])
    )

    df["spending_pressure"] = (
        (
            df["spending_growth"] > 0.15
        )
        |
        (
            df["spending_baseline_deviation"] > 0.15
        )
        |
        (
            df["discretionary_spending_growth"] > 0.20
        )
    )

    return df


def _assign_initial_state(row):
    """
    Assign a state for the first available observation.

    The first observation does not have a historical
    behavioral baseline, so state assignment relies primarily
    on current financial condition.
    """

    if (
        row["liquidity_pressure"]
        or row["negative_cash_flow"]
    ):
        return "LIQUIDITY_STRESS"

    if (
        row["high_expense_ratio"]
        or row["low_savings"]
    ):
        return "SPENDING_PRESSURE"

    return "STABLE"


def _assign_subsequent_state(
    row,
    previous_state,
):
    """
    Assign the current state using current financial
    conditions and the previous financial state.

    This creates temporal state transitions instead of
    treating every month as an independent classification.
    """

    liquidity_stress = (
        row["liquidity_pressure"]
        or row["negative_cash_flow"]
    )

    spending_pressure = (
        row["spending_pressure"]
        or row["high_expense_ratio"]
        or row["low_savings"]
    )

    improving_liquidity = (
        row["cash_buffer_change"] > 0
        and row["savings_rate_change"] > 0
    )

    reducing_spending = (
        row["spending_growth"] < 0
        or row["spending_baseline_deviation"] < 0
    )

    # ---------------------------------------------------------
    # Highest-priority state: liquidity stress
    # ---------------------------------------------------------

    if liquidity_stress:
        return "LIQUIDITY_STRESS"

    # ---------------------------------------------------------
    # Recovery
    # ---------------------------------------------------------

    if previous_state == "LIQUIDITY_STRESS":

        if (
            improving_liquidity
            and reducing_spending
        ):
            return "RECOVERY"

        if improving_liquidity:
            return "RECOVERY"

    # ---------------------------------------------------------
    # Continued recovery
    # ---------------------------------------------------------

    if previous_state == "RECOVERY":

        if (
            liquidity_stress
        ):
            return "LIQUIDITY_STRESS"

        if (
            not spending_pressure
            and improving_liquidity
        ):
            return "STABLE"

        return "RECOVERY"

    # ---------------------------------------------------------
    # Spending pressure
    # ---------------------------------------------------------

    if spending_pressure:
        return "SPENDING_PRESSURE"

    # ---------------------------------------------------------
    # Stable
    # ---------------------------------------------------------

    if previous_state == "SPENDING_PRESSURE":

        if (
            improving_liquidity
            and not spending_pressure
        ):
            return "RECOVERY"

    return "STABLE"


def assign_financial_states(
    monthly_profile: pd.DataFrame,
):
    """
    Assign a financial state to every user-month.

    Returns a dataframe containing:
        - original monthly profile
        - state-related features
        - financial_state
        - previous_financial_state
        - state_changed
        - state_transition
    """

    _validate_monthly_profile(
        monthly_profile
    )

    df = _prepare_profile(
        monthly_profile
    )

    df = _add_state_features(df)

    states = []

    for user_id, group in df.groupby(
        "user_id",
        sort=False,
    ):

        previous_state = None

        for index, row in group.iterrows():

            if previous_state is None:
                current_state = _assign_initial_state(
                    row
                )
            else:
                current_state = _assign_subsequent_state(
                    row,
                    previous_state,
                )

            states.append(
                (
                    index,
                    previous_state,
                    current_state,
                )
            )

            previous_state = current_state

    state_map = pd.DataFrame(
        states,
        columns=[
            "index",
            "previous_financial_state",
            "financial_state",
        ],
    ).set_index("index")

    df = df.join(state_map)

    df["state_changed"] = (
        df["previous_financial_state"].notna()
        &
        (
            df["previous_financial_state"]
            != df["financial_state"]
        )
    )

    df["state_transition"] = np.where(
        df["state_changed"],
        (
            df["previous_financial_state"]
            + "->"
            + df["financial_state"]
        ),
        "NO_CHANGE",
    )

    return df.reset_index(drop=True)