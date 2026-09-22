import numpy as np
import pandas as pd


def generate_initial_balances(
    population: pd.DataFrame,
    income_df: pd.DataFrame,
    rng: np.random.Generator,
    min_buffer_months: float = 0.10,
    max_buffer_months: float = 2.50,
) -> pd.DataFrame:
    """
    Generate the starting account balance for each user.

    Initial balance is derived from the user's income rather than
    being sampled independently.

    initial_balance =
        reference_monthly_income * initial_buffer_months
    """

    if min_buffer_months < 0:
        raise ValueError("min_buffer_months must be >= 0")

    if max_buffer_months < min_buffer_months:
        raise ValueError(
            "max_buffer_months must be >= min_buffer_months"
        )

    if "user_id" not in population.columns:
        raise ValueError("population must contain user_id")

    if "user_id" not in income_df.columns:
        raise ValueError("income_df must contain user_id")

    reference_income = (
        income_df
        .groupby("user_id")["total_income"]
        .first()
        .rename("reference_income")
        .reset_index()
    )

    result = population[["user_id"]].merge(
        reference_income,
        on="user_id",
        how="left",
        validate="one_to_one",
    )

    if result["reference_income"].isna().any():
        raise ValueError(
            "Some users do not have a reference income."
        )

    buffer_months = rng.uniform(
        min_buffer_months,
        max_buffer_months,
        size=len(result),
    )

    initial_balance = (
        result["reference_income"] * buffer_months
    )

    result["initial_buffer_months"] = buffer_months
    result["initial_balance"] = initial_balance

    return result