from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pandas as pd


@dataclass(frozen=True)
class Intervention:
    intervention_id: str
    category: str
    priority: int
    title: str
    description: str
    rationale: str
    eligibility_reason: str


INTERVENTION_LIBRARY = {
    "LIQUIDITY_PROTECTION": Intervention(
        intervention_id="INT_LIQUIDITY_01",
        category="LIQUIDITY_PROTECTION",
        priority=1,
        title="Protect your short-term cash buffer",
        description=(
            "Highlight upcoming cash pressure and prioritize "
            "maintaining sufficient liquid funds."
        ),
        rationale=(
            "The user's recent balance trajectory and cash "
            "buffer indicate elevated short-term liquidity pressure."
        ),
        eligibility_reason=(
            "Low cash-buffer ratio combined with elevated "
            "expense pressure or negative cash flow."
        ),
    ),
    "SPENDING_CONTROL": Intervention(
        intervention_id="INT_SPENDING_01",
        category="SPENDING_CONTROL",
        priority=2,
        title="Reduce discretionary spending pressure",
        description=(
            "Highlight recent discretionary spending growth "
            "and identify categories where spending has moved "
            "above the user's historical pattern."
        ),
        rationale=(
            "Recent spending behavior is deviating from the "
            "user's historical baseline."
        ),
        eligibility_reason=(
            "Elevated spending growth or personal-baseline deviation "
            "combined with discretionary pressure."
        ),
    ),
    "OBLIGATION_MANAGEMENT": Intervention(
        intervention_id="INT_OBLIGATION_01",
        category="OBLIGATION_MANAGEMENT",
        priority=3,
        title="Review recurring financial commitments",
        description=(
            "Highlight recurring obligations and their relationship "
            "to available income and liquidity."
        ),
        rationale=(
            "Recurring commitments are consuming a substantial "
            "portion of available financial capacity."
        ),
        eligibility_reason=(
            "Elevated expense pressure with recurring spending "
            "and declining balance conditions."
        ),
    ),
    "SAVINGS_REINFORCEMENT": Intervention(
        intervention_id="INT_SAVINGS_01",
        category="SAVINGS_REINFORCEMENT",
        priority=4,
        title="Strengthen your financial buffer",
        description=(
            "Reinforce positive saving behavior and encourage "
            "maintenance of the existing liquidity buffer."
        ),
        rationale=(
            "The user currently shows relatively stable financial "
            "behavior and positive liquidity conditions."
        ),
        eligibility_reason=(
            "Healthy balance trajectory, positive savings behavior, "
            "and low financial-pressure indicators."
        ),
    ),
}


def _safe_float(
    row: pd.Series,
    column: str,
    default: float = 0.0,
) -> float:
    """Safely retrieve a numeric feature."""

    if column not in row.index:
        return default

    value = row[column]

    if pd.isna(value):
        return default

    return float(value)


def _safe_bool(
    row: pd.Series,
    column: str,
    default: bool = False,
) -> bool:
    """Safely retrieve a boolean feature."""

    if column not in row.index:
        return default

    value = row[column]

    if pd.isna(value):
        return default

    return bool(value)


def classify_financial_pressure(
    row: pd.Series,
) -> str:
    """
    Classify the user's current behavioral pressure.

    This is NOT the ML prediction.

    It is a policy-layer interpretation of observed
    behavioral features.
    """

    cash_buffer_ratio = _safe_float(
        row,
        "cash_buffer_ratio",
    )

    expense_pressure = _safe_float(
        row,
        "expense_pressure",
    )

    spending_growth = _safe_float(
        row,
        "spending_growth",
    )

    discretionary_pressure = _safe_float(
        row,
        "discretionary_pressure",
    )

    spending_baseline_deviation = _safe_float(
        row,
        "spending_baseline_deviation",
    )

    recurring_spending = _safe_float(
        row,
        "recurring_spending",
    )

    balance_growth = _safe_float(
        row,
        "balance_growth",
    )

    negative_cash_flow = _safe_bool(
        row,
        "negative_cash_flow",
    )

    # --------------------------------------------------------
    # HIGH LIQUIDITY PRESSURE
    # --------------------------------------------------------

    if (
        cash_buffer_ratio < 0.50
        and (
            expense_pressure > 0.85
            or negative_cash_flow
        )
    ):
        return "LIQUIDITY_PRESSURE"

    # --------------------------------------------------------
    # SPENDING PRESSURE
    # --------------------------------------------------------

    if (
        (
            spending_growth > 0.10
            or spending_baseline_deviation > 0.10
        )
        and discretionary_pressure > 0.50
    ):
        return "SPENDING_PRESSURE"

    # --------------------------------------------------------
    # OBLIGATION PRESSURE
    # --------------------------------------------------------

    if (
        recurring_spending > 0
        and expense_pressure > 0.75
        and balance_growth < 0
    ):
        return "OBLIGATION_PRESSURE"

    # --------------------------------------------------------
    # OTHERWISE
    # --------------------------------------------------------

    return "STABLE"


def select_intervention(
    row: pd.Series,
    risk_probability: Optional[float] = None,
) -> Intervention:
    """
    Select the highest-priority eligible intervention.

    Parameters
    ----------
    row:
        One user-month feature vector.

    risk_probability:
        Optional predicted probability of financial stress
        within the next 30 days.

    Returns
    -------
    Intervention
    """

    pressure = classify_financial_pressure(
        row
    )

    if risk_probability is None:
        risk_probability = 0.0

    risk_probability = float(
        risk_probability
    )

    # --------------------------------------------------------
    # HIGH RISK + LIQUIDITY PRESSURE
    # --------------------------------------------------------

    if (
        risk_probability >= 0.60
        and pressure == "LIQUIDITY_PRESSURE"
    ):
        return INTERVENTION_LIBRARY[
            "LIQUIDITY_PROTECTION"
        ]

    # --------------------------------------------------------
    # HIGH RISK + SPENDING PRESSURE
    # --------------------------------------------------------

    if (
        risk_probability >= 0.60
        and pressure == "SPENDING_PRESSURE"
    ):
        return INTERVENTION_LIBRARY[
            "SPENDING_CONTROL"
        ]

    # --------------------------------------------------------
    # HIGH RISK + OBLIGATION PRESSURE
    # --------------------------------------------------------

    if (
        risk_probability >= 0.60
        and pressure == "OBLIGATION_PRESSURE"
    ):
        return INTERVENTION_LIBRARY[
            "OBLIGATION_MANAGEMENT"
        ]

    # --------------------------------------------------------
    # MODERATE RISK + BEHAVIORAL PRESSURE
    # --------------------------------------------------------

    if (
        risk_probability >= 0.30
        and pressure == "SPENDING_PRESSURE"
    ):
        return INTERVENTION_LIBRARY[
            "SPENDING_CONTROL"
        ]

    if (
        risk_probability >= 0.30
        and pressure == "OBLIGATION_PRESSURE"
    ):
        return INTERVENTION_LIBRARY[
            "OBLIGATION_MANAGEMENT"
        ]

    # --------------------------------------------------------
    # LOW RISK / STABLE
    # --------------------------------------------------------

    return INTERVENTION_LIBRARY[
        "SAVINGS_REINFORCEMENT"
    ]


def generate_intervention_record(
    row: pd.Series,
    risk_probability: float,
) -> dict:
    """
    Generate a complete intervention decision record.
    """

    intervention = select_intervention(
        row=row,
        risk_probability=risk_probability,
    )

    pressure = classify_financial_pressure(
        row
    )

    return {
        "user_id": row.get(
            "user_id",
            None,
        ),
        "month": row.get(
            "month",
            None,
        ),
        "risk_probability": float(
            risk_probability
        ),
        "pressure_state": pressure,
        "intervention_id":
            intervention.intervention_id,
        "intervention_category":
            intervention.category,
        "intervention_priority":
            intervention.priority,
        "intervention_title":
            intervention.title,
        "intervention_description":
            intervention.description,
        "intervention_rationale":
            intervention.rationale,
        "eligibility_reason":
            intervention.eligibility_reason,
    }


def generate_intervention_batch(
    feature_df: pd.DataFrame,
    probabilities,
) -> pd.DataFrame:
    """
    Generate intervention decisions for multiple
    user-month observations.
    """

    probabilities = list(
        probabilities
    )

    if len(feature_df) != len(probabilities):
        raise ValueError(
            "feature_df and probabilities must "
            "have the same number of rows."
        )

    records = []

    for index, (_, row) in enumerate(
        feature_df.iterrows()
    ):
        records.append(
            generate_intervention_record(
                row=row,
                risk_probability=probabilities[index],
            )
        )

    return pd.DataFrame(records)