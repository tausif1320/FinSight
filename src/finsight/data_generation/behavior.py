import numpy as np
import pandas as pd


# ============================================================
# BEHAVIORAL REGIMES
# ============================================================

REGIMES = [
    "STABLE",
    "SPENDING_EXPANSION",
    "FINANCIAL_PRESSURE",
    "RECOVERY",
]


# ============================================================
# REGIME PARAMETERS
# ============================================================
#
# These parameters describe behavioral changes.
#
# They are NOT financial stress labels.
#
# Financial state will be derived later from the resulting
# financial trajectory.
# ============================================================

REGIME_PARAMETERS = {
    "STABLE": {
        "spending_multiplier": 1.00,
        "discretionary_multiplier": 1.00,
        "transaction_multiplier": 1.00,
    },

    "SPENDING_EXPANSION": {
        "spending_multiplier": 1.15,
        "discretionary_multiplier": 1.35,
        "transaction_multiplier": 1.15,
    },

    "FINANCIAL_PRESSURE": {
        "spending_multiplier": 1.05,
        "discretionary_multiplier": 0.85,
        "transaction_multiplier": 0.95,
    },

    "RECOVERY": {
        "spending_multiplier": 0.90,
        "discretionary_multiplier": 0.75,
        "transaction_multiplier": 0.90,
    },
}


# ============================================================
# VALIDATION
# ============================================================


def _validate_population(population: pd.DataFrame) -> None:
    """Validate the population dataframe."""

    required_columns = {
        "user_id",
        "income_stability",
        "spending_propensity",
        "savings_propensity",
        "transaction_propensity",
        "obligation_burden",
    }

    missing_columns = (
        required_columns
        - set(population.columns)
    )

    if missing_columns:
        raise ValueError(
            "Population is missing required columns: "
            f"{sorted(missing_columns)}"
        )

    if population["user_id"].duplicated().any():
        raise ValueError(
            "population contains duplicate user_id values."
        )


# ============================================================
# INITIAL REGIME
# ============================================================


def _initial_regime(user: pd.Series) -> str:
    """
    Determine the initial behavioral regime.

    Most users begin in STABLE.
    Users with stronger spending propensity or obligation
    burden have a somewhat higher chance of beginning under
    financial pressure.

    This is a synthetic data-generating assumption.
    """

    spending_propensity = float(
        user["spending_propensity"]
    )

    savings_propensity = float(
        user["savings_propensity"]
    )

    obligation_burden = float(
        user["obligation_burden"]
    )

    # Base probability of a non-stable starting condition.
    pressure_score = (
        0.04
        + 0.08 * spending_propensity
        + 0.06 * obligation_burden
        + 0.04 * (1.0 - savings_propensity)
    )

    pressure_score = np.clip(
        pressure_score,
        0.02,
        0.20,
    )

    return pressure_score


# ============================================================
# REGIME TRANSITION
# ============================================================


def _transition_probability(
    current_regime: str,
    user: pd.Series,
) -> dict:
    """
    Return probabilities for the next behavioral regime.

    Transitions are influenced by persistent user
    characteristics.

    The probabilities are intentionally conservative so that
    users do not randomly change behavior every month.
    """

    spending_propensity = float(
        user["spending_propensity"]
    )

    savings_propensity = float(
        user["savings_propensity"]
    )

    obligation_burden = float(
        user["obligation_burden"]
    )

    income_stability = float(
        user["income_stability"]
    )

    if current_regime == "STABLE":

        expansion_probability = (
            0.04
            + 0.06 * spending_propensity
        )

        pressure_probability = (
            0.015
            + 0.04 * obligation_burden
            + 0.025 * (1.0 - income_stability)
        )

        recovery_probability = 0.005

    elif current_regime == "SPENDING_EXPANSION":

        expansion_probability = 0.55

        pressure_probability = (
            0.12
            + 0.08 * obligation_burden
        )

        recovery_probability = (
            0.08
            + 0.08 * savings_propensity
        )

    elif current_regime == "FINANCIAL_PRESSURE":

        expansion_probability = 0.02

        pressure_probability = 0.50

        recovery_probability = (
            0.15
            + 0.20 * savings_propensity
            + 0.10 * income_stability
        )

    elif current_regime == "RECOVERY":

        expansion_probability = (
            0.025
            + 0.025 * spending_propensity
        )

        pressure_probability = (
            0.015
            + 0.025 * obligation_burden
        )

        recovery_probability = 0.45

    else:
        raise ValueError(
            f"Unknown behavioral regime: {current_regime}"
        )

    # Remaining probability stays in the current regime.
    transition_probability = (
        expansion_probability
        + pressure_probability
        + recovery_probability
    )

    stay_probability = (
        1.0 - transition_probability
    )

    # Numerical safety.
    stay_probability = max(
        stay_probability,
        0.0,
    )

    probabilities = {
        "STABLE": 0.0,
        "SPENDING_EXPANSION": 0.0,
        "FINANCIAL_PRESSURE": 0.0,
        "RECOVERY": 0.0,
    }

    probabilities[current_regime] = stay_probability

    # --------------------------------------------------------
    # Transition rules
    # --------------------------------------------------------

    if current_regime == "STABLE":

        probabilities["SPENDING_EXPANSION"] = (
            expansion_probability
        )

        probabilities["FINANCIAL_PRESSURE"] = (
            pressure_probability
        )

        probabilities["RECOVERY"] = (
            recovery_probability
        )

    elif current_regime == "SPENDING_EXPANSION":

        probabilities["FINANCIAL_PRESSURE"] = (
            pressure_probability
        )

        probabilities["RECOVERY"] = (
            recovery_probability
        )

        # Expansion can continue.
        probabilities["SPENDING_EXPANSION"] += (
            expansion_probability
        )

    elif current_regime == "FINANCIAL_PRESSURE":

        probabilities["RECOVERY"] = (
            recovery_probability
        )

        probabilities["FINANCIAL_PRESSURE"] += (
            pressure_probability
        )

        probabilities["SPENDING_EXPANSION"] = (
            expansion_probability
        )

    elif current_regime == "RECOVERY":

        probabilities["SPENDING_EXPANSION"] = (
            expansion_probability
        )

        probabilities["FINANCIAL_PRESSURE"] = (
            pressure_probability
        )

        probabilities["RECOVERY"] += (
            recovery_probability
        )

    # Normalize to protect against floating-point drift.
    total = sum(probabilities.values())

    if total <= 0:
        raise ValueError(
            "Invalid behavioral transition probabilities."
        )

    probabilities = {
        regime: probability / total
        for regime, probability
        in probabilities.items()
    }

    return probabilities


# ============================================================
# BEHAVIORAL PROFILE GENERATION
# ============================================================


def generate_behavioral_profile(
    population: pd.DataFrame,
    n_months: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """
    Generate a longitudinal behavioral regime for each user.

    Output grain:
        one row per user-month.

    The output describes behavioral state and behavioral
    multipliers. It does not directly alter transactions.
    """

    if n_months <= 0:
        raise ValueError(
            "n_months must be greater than 0."
        )

    _validate_population(population)

    records = []

    for _, user in population.iterrows():

        # ----------------------------------------------------
        # Initial state
        # ----------------------------------------------------

        initial_pressure_probability = (
            _initial_regime(user)
        )

        random_value = rng.random()

        if random_value < initial_pressure_probability:

            current_regime = "FINANCIAL_PRESSURE"

        else:

            current_regime = "STABLE"

        # ----------------------------------------------------
        # Generate monthly behavioral states
        # ----------------------------------------------------

        for month in range(1, n_months + 1):

            parameters = REGIME_PARAMETERS[
                current_regime
            ]

            records.append(
                {
                    "user_id": user["user_id"],
                    "month": month,
                    "behavioral_regime": current_regime,
                    "spending_multiplier": (
                        parameters[
                            "spending_multiplier"
                        ]
                    ),
                    "discretionary_multiplier": (
                        parameters[
                            "discretionary_multiplier"
                        ]
                    ),
                    "transaction_multiplier": (
                        parameters[
                            "transaction_multiplier"
                        ]
                    ),
                }
            )

            # ------------------------------------------------
            # Determine next month's regime
            # ------------------------------------------------

            probabilities = _transition_probability(
                current_regime,
                user,
            )

            regimes = list(
                probabilities.keys()
            )

            probability_values = [
                probabilities[regime]
                for regime in regimes
            ]

            current_regime = rng.choice(
                regimes,
                p=probability_values,
            )

    result = pd.DataFrame(records)

    # --------------------------------------------------------
    # Final validation
    # --------------------------------------------------------

    result = result.sort_values(
        ["user_id", "month"],
        kind="mergesort",
    ).reset_index(drop=True)

    expected_rows = (
        len(population) * n_months
    )

    if len(result) != expected_rows:
        raise ValueError(
            "Behavioral profile has an unexpected "
            f"number of rows: {len(result)} "
            f"(expected {expected_rows})."
        )

    if result[
        ["user_id", "month"]
    ].duplicated().any():

        raise ValueError(
            "Behavioral profile contains duplicate "
            "user-month observations."
        )

    if not result[
        "behavioral_regime"
    ].isin(REGIMES).all():

        raise ValueError(
            "Unknown behavioral regime detected."
        )

    return result
