import numpy as np
import pandas as pd


# ============================================================
# SHOCK TYPES
# ============================================================

SHOCK_TYPES = [
    "MEDICAL",
    "TRAVEL",
    "LARGE_PURCHASE",
    "EMERGENCY",
    "UNEXPECTED_BILL",
    "INCOME_DISRUPTION",
]


# ============================================================
# SHOCK CONFIGURATION
# ============================================================

SHOCK_PARAMETERS = {
    "MEDICAL": {
        "base_probability": 0.018,
        "amount_ratio_min": 0.10,
        "amount_ratio_max": 0.35,
        "income_impact": 0.00,
        "duration_min": 1,
        "duration_max": 1,
    },

    "TRAVEL": {
        "base_probability": 0.020,
        "amount_ratio_min": 0.08,
        "amount_ratio_max": 0.25,
        "income_impact": 0.00,
        "duration_min": 1,
        "duration_max": 1,
    },

    "LARGE_PURCHASE": {
        "base_probability": 0.025,
        "amount_ratio_min": 0.10,
        "amount_ratio_max": 0.40,
        "income_impact": 0.00,
        "duration_min": 1,
        "duration_max": 1,
    },

    "EMERGENCY": {
        "base_probability": 0.012,
        "amount_ratio_min": 0.15,
        "amount_ratio_max": 0.50,
        "income_impact": 0.00,
        "duration_min": 1,
        "duration_max": 1,
    },

    "UNEXPECTED_BILL": {
        "base_probability": 0.030,
        "amount_ratio_min": 0.05,
        "amount_ratio_max": 0.20,
        "income_impact": 0.00,
        "duration_min": 1,
        "duration_max": 1,
    },

    "INCOME_DISRUPTION": {
        "base_probability": 0.008,
        "amount_ratio_min": 0.00,
        "amount_ratio_max": 0.00,
        "income_impact": 0.30,
        "duration_min": 1,
        "duration_max": 2,
    },
}


# ============================================================
# VALIDATION
# ============================================================


def _validate_population(population: pd.DataFrame) -> None:
    """Validate population required for shock generation."""

    required_columns = {
        "user_id",
        "base_income",
        "income_stability",
        "obligation_burden",
        "spending_propensity",
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
# SHOCK PROBABILITY
# ============================================================


def _shock_probability(
    shock_type: str,
    user: pd.Series,
) -> float:
    """
    Calculate user-specific probability of experiencing
    a particular shock during a month.

    These are synthetic data-generating assumptions.
    """

    parameters = SHOCK_PARAMETERS[shock_type]

    probability = parameters["base_probability"]

    income_stability = float(
        user["income_stability"]
    )

    obligation_burden = float(
        user["obligation_burden"]
    )

    spending_propensity = float(
        user["spending_propensity"]
    )

    if shock_type == "MEDICAL":
        # Slightly more exposure with higher financial
        # obligation burden.
        probability += (
            0.006 * obligation_burden
        )

    elif shock_type == "TRAVEL":
        probability += (
            0.008 * spending_propensity
        )

    elif shock_type == "LARGE_PURCHASE":
        probability += (
            0.012 * spending_propensity
        )

    elif shock_type == "EMERGENCY":
        probability += (
            0.005 * obligation_burden
        )

    elif shock_type == "UNEXPECTED_BILL":
        probability += (
            0.006 * obligation_burden
        )

    elif shock_type == "INCOME_DISRUPTION":
        # Lower income stability means greater exposure.
        probability += (
            0.012 * (1.0 - income_stability)
        )

    return float(
        np.clip(
            probability,
            0.0,
            0.10,
        )
    )


# ============================================================
# SHOCK SEVERITY
# ============================================================


def _severity_from_amount_ratio(
    amount_ratio: float,
) -> str:
    """Convert shock magnitude into a qualitative severity."""

    if amount_ratio < 0.10:
        return "LOW"

    if amount_ratio < 0.25:
        return "MEDIUM"

    if amount_ratio < 0.40:
        return "HIGH"

    return "SEVERE"


# ============================================================
# GENERATE SHOCKS
# ============================================================


def generate_shocks(
    population: pd.DataFrame,
    n_months: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """
    Generate longitudinal financial shocks.

    Grain:
        zero or more shock events per user-month.

    The function does NOT modify transactions or financial
    state. It only generates external events.
    """

    if n_months <= 0:
        raise ValueError(
            "n_months must be greater than 0."
        )

    _validate_population(population)

    records = []

    for _, user in population.iterrows():

        base_income = float(
            user["base_income"]
        )

        for month in range(
            1,
            n_months + 1,
        ):

            # ------------------------------------------------
            # Determine whether a shock occurs
            # ------------------------------------------------

            candidate_shocks = []

            for shock_type in SHOCK_TYPES:

                probability = _shock_probability(
                    shock_type,
                    user,
                )

                if rng.random() < probability:
                    candidate_shocks.append(
                        shock_type
                    )

            # ------------------------------------------------
            # Avoid multiple simultaneous shocks.
            #
            # This keeps the synthetic world interpretable.
            # If multiple candidates occur, select one.
            # ------------------------------------------------

            if len(candidate_shocks) > 1:

                selected_index = int(
                    rng.integers(
                        0,
                        len(candidate_shocks),
                    )
                )

                candidate_shocks = [
                    candidate_shocks[
                        selected_index
                    ]
                ]

            # ------------------------------------------------
            # No shock this month
            # ------------------------------------------------

            if not candidate_shocks:
                continue

            shock_type = candidate_shocks[0]

            parameters = SHOCK_PARAMETERS[
                shock_type
            ]

            # ------------------------------------------------
            # Shock amount
            # ------------------------------------------------

            amount_ratio = rng.uniform(
                parameters["amount_ratio_min"],
                parameters["amount_ratio_max"],
            )

            shock_amount = (
                base_income
                * amount_ratio
            )

            # ------------------------------------------------
            # Income impact
            # ------------------------------------------------

            income_impact = float(
                parameters["income_impact"]
            )

            duration_months = int(
                rng.integers(
                    parameters["duration_min"],
                    parameters["duration_max"] + 1,
                )
            )

            # ------------------------------------------------
            # Severity
            # ------------------------------------------------

            if shock_type == "INCOME_DISRUPTION":

                if income_impact < 0.20:
                    severity = "LOW"

                elif income_impact < 0.35:
                    severity = "MEDIUM"

                elif income_impact < 0.50:
                    severity = "HIGH"

                else:
                    severity = "SEVERE"

            else:

                severity = (
                    _severity_from_amount_ratio(
                        amount_ratio
                    )
                )

            records.append(
                {
                    "user_id": user["user_id"],
                    "month": month,
                    "shock_type": shock_type,
                    "shock_amount": round(
                        shock_amount,
                        2,
                    ),
                    "income_impact": round(
                        income_impact,
                        4,
                    ),
                    "duration_months": (
                        duration_months
                    ),
                    "severity": severity,
                }
            )

    # --------------------------------------------------------
    # Empty result handling
    # --------------------------------------------------------

    columns = [
        "user_id",
        "month",
        "shock_type",
        "shock_amount",
        "income_impact",
        "duration_months",
        "severity",
    ]

    if not records:
        return pd.DataFrame(
            columns=columns
        )

    result = pd.DataFrame(records)

    result = result.sort_values(
        [
            "user_id",
            "month",
            "shock_type",
        ],
        kind="mergesort",
    ).reset_index(drop=True)

    return result