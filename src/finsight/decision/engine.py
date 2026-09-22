from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class DecisionConfig:
    """
    Configuration for the FinSight decision engine.

    The engine combines:
        1. predicted financial-stress risk
        2. financial pressure classification
        3. intervention policy
        4. estimated CATE
        5. SHAP-style driver information

    CATE values originate from the synthetic experiment.
    """

    high_risk_threshold: float = 0.60
    moderate_risk_threshold: float = 0.30

    strong_cate_threshold: float = -0.10

    minimum_cash_buffer_ratio: float = 0.50
    high_expense_pressure: float = 0.85
    spending_growth_threshold: float = 0.10


@dataclass
class DecisionResult:
    user_id: Any
    month: Any

    risk_probability: float
    risk_level: str

    financial_pressure: str

    intervention: str
    intervention_priority: int

    cate: float | None

    recommendation_status: str

    explanation: list[str]

    model_drivers: list[str]


INTERVENTION_PRIORITY = {
    "LIQUIDITY_PROTECTION": 1,
    "SPENDING_CONTROL": 2,
    "OBLIGATION_MANAGEMENT": 3,
    "SAVINGS_REINFORCEMENT": 4,
    "NO_INTERVENTION": 5,
}


INTERVENTION_TEXT = {
    "LIQUIDITY_PROTECTION":
        "Protect available cash and preserve a minimum liquidity buffer.",

    "SPENDING_CONTROL":
        "Reduce discretionary spending and slow spending growth.",

    "OBLIGATION_MANAGEMENT":
        "Review recurring obligations and reduce near-term payment pressure.",

    "SAVINGS_REINFORCEMENT":
        "Maintain current financial behavior and reinforce savings.",

    "NO_INTERVENTION":
        "No intervention is currently supported by the decision rules.",
}


def _safe_float(
    value: Any,
    default: float = 0.0,
) -> float:

    try:

        result = float(value)

        if not np.isfinite(result):
            return default

        return result

    except (
        TypeError,
        ValueError,
    ):

        return default


def classify_risk(
    risk_probability: float,
    config: DecisionConfig | None = None,
) -> str:

    config = (
        config
        or DecisionConfig()
    )

    risk_probability = _safe_float(
        risk_probability
    )

    if (
        risk_probability
        >= config.high_risk_threshold
    ):

        return "HIGH"

    if (
        risk_probability
        >= config.moderate_risk_threshold
    ):

        return "MODERATE"

    return "LOW"


def classify_financial_pressure(
    row: pd.Series,
    config: DecisionConfig | None = None,
) -> str:

    config = (
        config
        or DecisionConfig()
    )

    cash_buffer = _safe_float(
        row.get(
            "cash_buffer_ratio"
        )
    )

    expense_pressure = _safe_float(
        row.get(
            "expense_pressure"
        )
    )

    negative_cash_flow = bool(
        row.get(
            "negative_cash_flow",
            False,
        )
    )

    spending_growth = _safe_float(
        row.get(
            "spending_growth"
        )
    )

    baseline_deviation = _safe_float(
        row.get(
            "spending_baseline_deviation"
        )
    )

    discretionary_pressure = _safe_float(
        row.get(
            "discretionary_pressure"
        )
    )

    recurring_spending = _safe_float(
        row.get(
            "recurring_spending"
        )
    )

    balance_growth = _safe_float(
        row.get(
            "balance_growth"
        )
    )

    if (
        cash_buffer
        < config.minimum_cash_buffer_ratio
        and (
            expense_pressure
            > config.high_expense_pressure
            or negative_cash_flow
        )
    ):

        return "LIQUIDITY_PRESSURE"

    if (
        (
            spending_growth
            > config.spending_growth_threshold
            or baseline_deviation
            > config.spending_growth_threshold
        )
        and discretionary_pressure
        > 0.50
    ):

        return "SPENDING_PRESSURE"

    if (
        recurring_spending > 0
        and expense_pressure > 0.75
        and balance_growth < 0
    ):

        return "OBLIGATION_PRESSURE"

    return "STABLE"


def select_intervention(
    pressure: str,
    risk_probability: float,
    cate: float | None = None,
    config: DecisionConfig | None = None,
) -> str:

    """
    Select an intervention using the existing FinSight
    intervention policy.

    CATE is used as an additional validation signal,
    not as an invented cross-intervention comparison.
    """

    config = (
        config
        or DecisionConfig()
    )

    risk_probability = _safe_float(
        risk_probability
    )

    if (
        risk_probability
        < config.moderate_risk_threshold
    ):

        return "SAVINGS_REINFORCEMENT"

    if pressure == "LIQUIDITY_PRESSURE":

        return "LIQUIDITY_PROTECTION"

    if pressure == "SPENDING_PRESSURE":

        return "SPENDING_CONTROL"

    if pressure == "OBLIGATION_PRESSURE":

        return "OBLIGATION_MANAGEMENT"

    return "SAVINGS_REINFORCEMENT"


def build_explanation(
    row: pd.Series,
    risk_probability: float,
    pressure: str,
    intervention: str,
    cate: float | None,
) -> list[str]:

    reasons: list[str] = []

    cash_buffer = _safe_float(
        row.get(
            "cash_buffer_ratio"
        )
    )

    expense_pressure = _safe_float(
        row.get(
            "expense_pressure"
        )
    )

    spending_growth = _safe_float(
        row.get(
            "spending_growth"
        )
    )

    baseline_deviation = _safe_float(
        row.get(
            "spending_baseline_deviation"
        )
    )

    discretionary_pressure = _safe_float(
        row.get(
            "discretionary_pressure"
        )
    )

    recurring_spending = _safe_float(
        row.get(
            "recurring_spending"
        )
    )

    balance_growth = _safe_float(
        row.get(
            "balance_growth"
        )
    )

    # ========================================================================
    # STABLE
    # ========================================================================

    if pressure == "STABLE":

        reasons.append(
            "Current financial behavior is stable."
        )

        reasons.append(
            "No active financial-pressure condition "
            "exceeded the configured intervention thresholds."
        )

    # ========================================================================
    # LIQUIDITY PRESSURE
    # ========================================================================

    elif pressure == "LIQUIDITY_PRESSURE":

        if cash_buffer < 0.50:

            reasons.append(
                "Cash buffer is below the configured "
                "liquidity threshold."
            )

        if expense_pressure > 0.85:

            reasons.append(
                "Expense pressure is elevated."
            )

        if (
            bool(
                row.get(
                    "negative_cash_flow",
                    False,
                )
            )
        ):

            reasons.append(
                "Recent cash-flow behavior indicates "
                "negative cash-flow pressure."
            )

        reasons.append(
            "Liquidity protection is the applicable "
            "intervention."
        )

    # ========================================================================
    # SPENDING PRESSURE
    # ========================================================================

    elif pressure == "SPENDING_PRESSURE":

        if spending_growth > 0.10:

            reasons.append(
                "Spending is growing faster than the "
                "configured behavioral threshold."
            )

        if baseline_deviation > 0.10:

            reasons.append(
                "Spending is above the user's historical "
                "behavioral baseline."
            )

        if discretionary_pressure > 0.50:

            reasons.append(
                "Discretionary spending is contributing "
                "to the detected pressure."
            )

        reasons.append(
            "Spending control is the applicable intervention."
        )

    # ========================================================================
    # OBLIGATION PRESSURE
    # ========================================================================

    elif pressure == "OBLIGATION_PRESSURE":

        if recurring_spending > 0:

            reasons.append(
                "Recurring financial obligations contribute "
                "to monthly pressure."
            )

        if expense_pressure > 0.75:

            reasons.append(
                "Expense pressure is elevated."
            )

        if balance_growth < 0:

            reasons.append(
                "The recent balance trajectory is declining."
            )

        reasons.append(
            "Obligation management is the applicable "
            "intervention."
        )

    # ========================================================================
    # CATE
    # ========================================================================

    if cate is not None:

        if cate <= -0.10:

            reasons.append(
                "The assigned intervention has strong "
                "simulated support in the synthetic experiment."
            )

        elif cate < 0:

            reasons.append(
                "The assigned intervention has simulated "
                "support in the synthetic experiment."
            )

    # ========================================================================
    # FALLBACK
    # ========================================================================

    if not reasons:

        reasons.append(
            "No major behavioral pressure signal exceeded "
            "the configured thresholds."
        )

    return reasons[:5]

def generate_decision(
    row: pd.Series,
    risk_probability: float,
    cate: float | None = None,
    model_drivers: list[str] | None = None,
    config: DecisionConfig | None = None,
) -> DecisionResult:

    config = (
        config
        or DecisionConfig()
    )

    risk_probability = _safe_float(
        risk_probability
    )

    risk_level = classify_risk(
        risk_probability,
        config,
    )

    pressure = classify_financial_pressure(
        row,
        config,
    )

    intervention = select_intervention(
        pressure,
        risk_probability,
        cate,
        config,
    )

    priority = INTERVENTION_PRIORITY[
        intervention
    ]

    explanation = build_explanation(
        row=row,
        risk_probability=risk_probability,
        pressure=pressure,
        intervention=intervention,
        cate=cate,
    )

    if intervention == "SAVINGS_REINFORCEMENT":

        status = "MONITOR"

    elif (
        cate is not None
        and cate <= config.strong_cate_threshold
    ):

        status = "STRONG_SIMULATED_SUPPORT"

    elif (
        cate is not None
        and cate < 0
    ):

        status = "SIMULATED_SUPPORT"

    else:

        status = "RULE_BASED_RECOMMENDATION"

    if model_drivers is None:

        model_drivers = []

    return DecisionResult(
        user_id=row.get(
            "user_id"
        ),
        month=row.get(
            "month"
        ),
        risk_probability=risk_probability,
        risk_level=risk_level,
        financial_pressure=pressure,
        intervention=intervention,
        intervention_priority=priority,
        cate=cate,
        recommendation_status=status,
        explanation=explanation,
        model_drivers=model_drivers,
    )


def decision_to_dict(
    result: DecisionResult,
) -> dict:

    return {
        "user_id": result.user_id,
        "month": result.month,
        "risk_probability": result.risk_probability,
        "risk_level": result.risk_level,
        "financial_pressure": result.financial_pressure,
        "intervention": result.intervention,
        "intervention_priority": result.intervention_priority,
        "cate": result.cate,
        "recommendation_status": result.recommendation_status,
        "explanation": " | ".join(
            result.explanation
        ),
        "model_drivers": " | ".join(
            result.model_drivers
        ),
        "intervention_description":
            INTERVENTION_TEXT[
                result.intervention
            ],
    }


def generate_decision_batch(
    df: pd.DataFrame,
    risk_column: str = "risk_probability",
    cate_column: str = "cate",
    config: DecisionConfig | None = None,
) -> pd.DataFrame:

    config = (
        config
        or DecisionConfig()
    )

    results = []

    for _, row in df.iterrows():

        risk_probability = _safe_float(
            row.get(
                risk_column
            )
        )

        cate_value = row.get(
            cate_column
        )

        if pd.isna(
            cate_value
        ):

            cate_value = None

        else:

            cate_value = _safe_float(
                cate_value
            )

        result = generate_decision(
            row=row,
            risk_probability=risk_probability,
            cate=cate_value,
            config=config,
        )

        results.append(
            decision_to_dict(
                result
            )
        )

    return pd.DataFrame(
        results
    )


__all__ = [
    "DecisionConfig",
    "DecisionResult",
    "classify_risk",
    "classify_financial_pressure",
    "select_intervention",
    "build_explanation",
    "generate_decision",
    "decision_to_dict",
    "generate_decision_batch",
]