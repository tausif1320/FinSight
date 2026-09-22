import numpy as np
import pandas as pd


def generate_base_income(
    population: pd.DataFrame,
    rng: np.random.Generator
) -> pd.DataFrame:
    """
    Generate a base monthly income for every user.

    Income is right-skewed using a log-normal distribution
    and constrained to a realistic range.
    """

    n_users = len(population)

    # Log-normal parameters chosen to create a right-skewed
    # salaried-income distribution.
    mean_income = 55000

    sigma = 0.65

    mu = np.log(mean_income) - (sigma ** 2) / 2

    base_income = rng.lognormal(
        mean=mu,
        sigma=sigma,
        size=n_users
    )

    base_income = np.clip(
        base_income,
        20_000,
        250_000
    )

    result = population.copy()

    result["base_income"] = np.round(
        base_income,
        2
    )

    return result


def generate_salary_days(
    population: pd.DataFrame,
    rng: np.random.Generator
) -> pd.DataFrame:
    """
    Generate a salary payment day for each user.

    Early-month and end-of-month salary dates are more common.
    """

    n_users = len(population)

    salary_days = np.array([
        1, 2, 3, 4, 5,
        10, 15, 20,
        25, 26, 27, 28, 29, 30, 31
    ])

    probabilities = np.array([
        0.10, 0.08, 0.07, 0.06, 0.05,
        0.04, 0.03, 0.03,
        0.07, 0.07, 0.07, 0.08, 0.08, 0.08, 0.09
    ])

    probabilities = probabilities / probabilities.sum()

    salary_day = rng.choice(
        salary_days,
        size=n_users,
        p=probabilities
    )

    result = population.copy()

    result["salary_day"] = salary_day

    return result


def generate_monthly_income(
    population: pd.DataFrame,
    n_months: int,
    rng: np.random.Generator
) -> pd.DataFrame:
    """
    Generate monthly income observations for every user.

    Income variation depends on the user's income stability.
    Occasional raises and bonuses are also introduced.
    """

    records = []

    for _, user in population.iterrows():

        base_income = user["base_income"]
        stability = user["income_stability"]

        # Less stable users have larger monthly variation.
        noise_std = (
            0.02
            + 0.12 * (1 - stability)
        )

        current_income = base_income

        # At most one raise during the 12-month simulation.
        raise_month = None

        if rng.random() < 0.15:
            raise_month = int(
                rng.integers(
                    1,
                    n_months + 1
                )
            )

        raise_rate = 0.0

        if raise_month is not None:
            raise_rate = rng.uniform(
                0.05,
                0.15
            )

        for month in range(1, n_months + 1):

            # Apply raise.
            if (
                raise_month is not None
                and month == raise_month
            ):
                current_income *= (
                    1 + raise_rate
                )

            # Monthly income noise.
            noise = rng.normal(
                0,
                noise_std
            )

            monthly_income = (
                current_income
                * (1 + noise)
            )

            monthly_income = np.clip(
            monthly_income,
            20_000,
            300_000
            )

            # Occasional bonus.
            bonus = 0.0

            if rng.random() < 0.10:

                bonus = rng.lognormal(
                    mean=np.log(
                        max(
                            current_income * 0.25,
                            1
                        )
                    ),
                    sigma=0.35
                )

            total_income = (
                monthly_income
                + bonus
            )

            records.append({
                "user_id": user["user_id"],
                "month": month,
                "base_income": round(
                    base_income,
                    2
                ),
                "salary_day": int(
                    user["salary_day"]
                ),
                "income_stability": round(
                    stability,
                    4
                ),
                "monthly_salary": round(
                    monthly_income,
                    2
                ),
                "bonus": round(
                    bonus,
                    2
                ),
                "total_income": round(
                    total_income,
                    2
                )
            })

    return pd.DataFrame(records)