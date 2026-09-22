import numpy as np
import pandas as pd


CATEGORIES = [
    "Food",
    "Groceries",
    "Transport",
    "Shopping",
    "Entertainment",
    "Travel",
    "Healthcare",
    "Education",
    "Other",
]


BASE_CATEGORY_ALPHA = np.array([
    4.0,   # Food
    3.0,   # Groceries
    2.5,   # Transport
    2.0,   # Shopping
    1.5,   # Entertainment
    1.0,   # Travel
    1.0,   # Healthcare
    0.8,   # Education
    3.0,   # Other
])


def generate_spending_profile(
    population: pd.DataFrame,
    rng: np.random.Generator
) -> pd.DataFrame:
    """
    Generate persistent spending characteristics for each user.

    The profile determines:
    - essential spending
    - discretionary spending
    - category preferences
    - transaction frequency
    - weekend behavior
    """

    records = []

    for _, user in population.iterrows():

        income = user["base_income"]

        spending_propensity = (
            user["spending_propensity"]
        )

        savings_propensity = (
            user["savings_propensity"]
        )

        transaction_propensity = (
            user["transaction_propensity"]
        )

        # ----------------------------------------------------
        # Essential spending ratio
        # ----------------------------------------------------

        essential_ratio = rng.beta(
            6,
            4
        )

        # Slightly reduce essential ratio for high-income
        # users while retaining individual variation.
        income_adjustment = np.clip(
            np.log1p(income / 50_000) * 0.03,
            0,
            0.08
        )

        essential_ratio = np.clip(
            essential_ratio - income_adjustment,
            0.30,
            0.80
        )

        # ----------------------------------------------------
        # Discretionary spending ratio
        # ----------------------------------------------------

        discretionary_base = rng.beta(
            3,
            7
        )

        discretionary_ratio = (
            discretionary_base
            * (0.70 + 0.60 * spending_propensity)
        )

        discretionary_ratio = np.clip(
            discretionary_ratio,
            0.05,
            0.45
        )

        # ----------------------------------------------------
        # Category preferences
        # ----------------------------------------------------

        alpha = BASE_CATEGORY_ALPHA.copy()

        # High spending propensity users:
        # more likely to allocate money toward
        # shopping / entertainment / travel.
        alpha[3] *= (
            0.7 + 1.5 * spending_propensity
        )

        alpha[4] *= (
            0.7 + 1.5 * spending_propensity
        )

        alpha[5] *= (
            0.7 + 1.3 * spending_propensity
        )

        # Higher savings propensity:
        # slightly increase groceries/education and
        # reduce discretionary categories.
        alpha[1] *= (
            0.8 + 0.5 * savings_propensity
        )

        alpha[7] *= (
            0.8 + 0.5 * savings_propensity
        )

        alpha[3] *= (
            1.1 - 0.4 * savings_propensity
        )

        alpha[4] *= (
            1.1 - 0.4 * savings_propensity
        )

        category_probabilities = rng.dirichlet(
            alpha
        )

        # ----------------------------------------------------
        # Transaction frequency
        # ----------------------------------------------------

        min_transactions = 1.0
        max_transactions = 7.0

        expected_daily_transactions = (
            min_transactions
            + transaction_propensity
            * (
                max_transactions
                - min_transactions
            )
        )

        # ----------------------------------------------------
        # Weekend behavior
        # ----------------------------------------------------

        weekend_multiplier = rng.uniform(
            1.20,
            1.50
        )

        # Higher discretionary propensity produces
        # a stronger weekend effect.
        weekend_multiplier *= (
            0.90
            + 0.20 * spending_propensity
        )

        records.append({
        "user_id": user["user_id"],

        # Latent behavioral variables.
        # These are carried into the spending profile because
        # downstream transaction generation uses them to produce
        # observable financial behavior.
        "spending_propensity": spending_propensity,
        "savings_propensity": savings_propensity,
        "transaction_propensity": transaction_propensity,

        "essential_ratio": round(
            essential_ratio,
            4,
        ),

        "discretionary_ratio": round(
            discretionary_ratio,
            4,
        ),

        "expected_daily_transactions": round(
            expected_daily_transactions,
            3,
        ),

        "weekend_multiplier": round(
            weekend_multiplier,
            3,
        ),

        "food_probability": category_probabilities[0],
        "groceries_probability": category_probabilities[1],
        "transport_probability": category_probabilities[2],
        "shopping_probability": category_probabilities[3],
        "entertainment_probability": category_probabilities[4],
        "travel_probability": category_probabilities[5],
        "healthcare_probability": category_probabilities[6],
        "education_probability": category_probabilities[7],
        "other_probability": category_probabilities[8],
    })

    return pd.DataFrame(records)