import pandas as pd
import numpy as np
from finsight.data_generation.population import (
    generate_population,
)

from finsight.data_generation.income import (
    generate_base_income,
    generate_salary_days,
)

from finsight.data_generation.spending import (
    generate_spending_profile,
)

from finsight.utils.random import create_rng


def main():

    rng = create_rng(42)

    # --------------------------------------------------------
    # Population
    # --------------------------------------------------------

    population = generate_population(
        n_users=100,
        rng=rng
    )

    # --------------------------------------------------------
    # Income
    # --------------------------------------------------------

    population = generate_base_income(
        population,
        rng
    )

    population = generate_salary_days(
        population,
        rng
    )

    # --------------------------------------------------------
    # Spending profile
    # --------------------------------------------------------

    spending = generate_spending_profile(
        population,
        rng
    )

    users = population.merge(
        spending,
        on="user_id",
        how="left"
    )

    # --------------------------------------------------------
    # Basic validation
    # --------------------------------------------------------

    print("\nShape:")
    print(users.shape)

    print("\nMissing values:")
    print(users.isna().sum())

    # --------------------------------------------------------
    # Spending summary
    # --------------------------------------------------------

    print("\nEssential spending ratio:")
    print(
        users["essential_ratio"].describe()
    )

    print("\nDiscretionary spending ratio:")
    print(
        users["discretionary_ratio"].describe()
    )

    print("\nExpected daily transactions:")
    print(
        users[
            "expected_daily_transactions"
        ].describe()
    )

    print("\nWeekend multiplier:")
    print(
        users[
            "weekend_multiplier"
        ].describe()
    )

    # --------------------------------------------------------
    # Category probability validation
    # --------------------------------------------------------

    category_columns = [
        "food_probability",
        "groceries_probability",
        "transport_probability",
        "shopping_probability",
        "entertainment_probability",
        "travel_probability",
        "healthcare_probability",
        "education_probability",
        "other_probability",
    ]

    category_sum = users[
        category_columns
    ].sum(axis=1)

    print("\nCategory probability sums:")
    print(category_sum.describe())

    # --------------------------------------------------------
    # Assertions
    # --------------------------------------------------------

    assert (
        users["essential_ratio"]
        .between(0, 1)
        .all()
    )

    assert (
        users["discretionary_ratio"]
        .between(0, 1)
        .all()
    )

    assert (
        users["expected_daily_transactions"]
        > 0
    ).all()

    assert np.allclose(
        category_sum,
        1.0,
        atol=1e-4
    )

    print(
        "\nSpending profile validation passed."
    )


if __name__ == "__main__":
    main()