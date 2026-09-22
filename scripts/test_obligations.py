import pandas as pd

from finsight.data_generation.population import generate_population
from finsight.data_generation.income import (
    generate_base_income,
    generate_salary_days,
)
from finsight.data_generation.obligations import (
    generate_obligations,
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
    # Obligations
    # --------------------------------------------------------

    obligations = generate_obligations(
        population,
        rng
    )

    # --------------------------------------------------------
    # Merge for validation
    # --------------------------------------------------------

    users = population.merge(
        obligations,
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

    print("\nObligation summary:")
    print(
        users[
            [
                "rent",
                "emi",
                "utilities",
                "subscriptions",
                "other_recurring",
                "total_recurring_obligations",
                "obligation_to_income_ratio"
            ]
        ].describe()
    )

    print("\nUsers with EMI:")
    print(
        (users["emi"] > 0).mean()
    )

    print("\nUsers with rent:")
    print(
        (users["rent"] > 0).mean()
    )

    print("\nUsers with other recurring costs:")
    print(
        (users["other_recurring"] > 0).mean()
    )

    print("\nHighest obligation ratios:")
    print(
        users[
            [
                "user_id",
                "base_income",
                "rent",
                "emi",
                "total_recurring_obligations",
                "obligation_to_income_ratio"
            ]
        ]
        .sort_values(
            "obligation_to_income_ratio",
            ascending=False
        )
        .head(10)
    )

    # --------------------------------------------------------
    # Logical checks
    # --------------------------------------------------------

    assert (
        users["total_recurring_obligations"]
        >= users["rent"]
    ).all()

    assert (
        users["total_recurring_obligations"]
        >= users["emi"]
    ).all()

    assert (
        users["obligation_to_income_ratio"]
        >= 0
    ).all()

    print(
        "\nObligation generation validation passed."
    )


if __name__ == "__main__":
    main()