from typing import Dict

import numpy as np
import pandas as pd


def generate_population(
    n_users: int,
    rng: np.random.Generator
) -> pd.DataFrame:
    """
    Generate the static latent profile of FinSight users.
    """

    if n_users <= 0:
        raise ValueError("n_users must be greater than 0")

    user_ids = [
        f"U{i:05d}"
        for i in range(1, n_users + 1)
    ]

    # --------------------------------------------------------
    # Age
    # --------------------------------------------------------

    age = np.rint(
        rng.normal(
            loc=29,
            scale=6,
            size=n_users
        )
    ).astype(int)

    age = np.clip(age, 21, 55)

    # --------------------------------------------------------
    # City tier
    # --------------------------------------------------------

    city_tier = rng.choice(
        ["Tier_1", "Tier_2", "Tier_3"],
        size=n_users,
        p=[0.45, 0.35, 0.20]
    )

    # --------------------------------------------------------
    # Employment type
    # --------------------------------------------------------

    employment_type = rng.choice(
        [
            "Private",
            "Government",
            "Other_Salaried"
        ],
        size=n_users,
        p=[0.75, 0.15, 0.10]
    )

    # --------------------------------------------------------
    # Income stability
    # --------------------------------------------------------

    income_stability = rng.beta(
        8,
        2,
        size=n_users
    )

    # --------------------------------------------------------
    # Spending propensity
    # --------------------------------------------------------

    spending_propensity = rng.beta(
        4,
        4,
        size=n_users
    )

    # --------------------------------------------------------
    # Savings propensity
    # --------------------------------------------------------

    savings_noise = rng.normal(
        0,
        0.10,
        size=n_users
    )

    savings_propensity = (
        1
        - spending_propensity
        + savings_noise
    )

    savings_propensity = np.clip(
        savings_propensity,
        0,
        1
    )

    # --------------------------------------------------------
    # Transaction propensity
    # --------------------------------------------------------

    transaction_propensity = rng.beta(
        4,
        4,
        size=n_users
    )

    # --------------------------------------------------------
    # Financial obligation burden
    # --------------------------------------------------------

    obligation_burden = rng.beta(
        3,
        7,
        size=n_users
    )

    # --------------------------------------------------------
    # DataFrame
    # --------------------------------------------------------

    population = pd.DataFrame({
        "user_id": user_ids,
        "age": age,
        "city_tier": city_tier,
        "employment_type": employment_type,
        "income_stability": income_stability,
        "spending_propensity": spending_propensity,
        "savings_propensity": savings_propensity,
        "transaction_propensity": transaction_propensity,
        "obligation_burden": obligation_burden
    })

    return population