import numpy as np
import pandas as pd


CITY_RENT_MULTIPLIER = {
    "Tier_1": 1.35,
    "Tier_2": 1.00,
    "Tier_3": 0.75,
}


def generate_obligations(
    population: pd.DataFrame,
    rng: np.random.Generator
) -> pd.DataFrame:
    """
    Generate recurring financial obligations for each user.

    Obligations are generated from income, city tier, and
    latent financial burden rather than independently.
    """

    records = []

    for _, user in population.iterrows():

        income = user["base_income"]
        city_tier = user["city_tier"]
        burden = user["obligation_burden"]

        # ----------------------------------------------------
        # Rent
        # ----------------------------------------------------

        has_rent = (
            rng.random() < 0.80
        )

        if has_rent:

            base_rent_ratio = rng.beta(
                5,
                15
            )

            rent = (
                income
                * base_rent_ratio
                * CITY_RENT_MULTIPLIER[city_tier]
            )

            # Prevent unrealistic rent relative to income.
            rent = np.clip(
                rent,
                4_000,
                income * 0.45
            )

        else:
            rent = 0.0

        # ----------------------------------------------------
        # EMI
        # ----------------------------------------------------

        has_emi = (
            rng.random() < 0.45
        )

        if has_emi:

            emi_ratio = rng.beta(
                3,
                12
            )

            # Latent obligation burden increases EMI exposure,
            # but does not dominate the entire financial profile.
            emi_ratio *= (
            0.8
            + 0.5 * burden
            )

            emi = income * emi_ratio

            emi = np.clip(
            emi,
            1_000,
            income * 0.30
            )

        else:
            emi = 0.0

        # ----------------------------------------------------
        # Utilities
        # ----------------------------------------------------

        utilities_ratio = rng.beta(
            2,
            30
        )

        utilities = (
            income
            * utilities_ratio
        )

        utilities = np.clip(
            utilities,
            800,
            income * 0.12
        )

        # ----------------------------------------------------
        # Subscriptions
        # ----------------------------------------------------

        n_subscriptions = rng.poisson(
            2
        )

        n_subscriptions = min(
            n_subscriptions,
            6
        )

        subscription_total = 0.0

        if n_subscriptions > 0:

            subscription_amounts = rng.lognormal(
                mean=np.log(500),
                sigma=0.45,
                size=n_subscriptions
            )

            subscription_total = (
                subscription_amounts.sum()
            )

        subscription_total = np.clip(
            subscription_total,
            0,
            income * 0.08
        )

        # ----------------------------------------------------
        # Insurance / other recurring obligations
        # ----------------------------------------------------

        other_recurring = 0.0

        if rng.random() < 0.35:

            other_recurring = rng.lognormal(
                mean=np.log(1_500),
                sigma=0.50
            )

            other_recurring = np.clip(
                other_recurring,
                500,
                income * 0.10
            )

        # ----------------------------------------------------
        # Total recurring obligations
        # ----------------------------------------------------

        total_obligations = (
            rent
            + emi
            + utilities
            + subscription_total
            + other_recurring
        )

        obligation_ratio = (
            total_obligations
            / income
        )

        records.append({
            "user_id": user["user_id"],
            "rent": round(rent, 2),
            "emi": round(emi, 2),
            "utilities": round(utilities, 2),
            "subscriptions": round(
                subscription_total,
                2
            ),
            "other_recurring": round(
                other_recurring,
                2
            ),
            "total_recurring_obligations": round(
                total_obligations,
                2
            ),
            "obligation_to_income_ratio": round(
                obligation_ratio,
                4
            ),
        })

    return pd.DataFrame(records)