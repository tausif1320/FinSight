from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

from .intervention_policy import classify_financial_pressure


RANDOM_STATE = 42


@dataclass(frozen=True)
class ExperimentConfig:
    treatment_probability: float = 0.50
    confidence: float = 0.95
    max_smd: float = 0.20
    stress_effect: dict[str, float] | None = None
    cash_buffer_effect: dict[str, float] | None = None
    savings_rate_effect: dict[str, float] | None = None
    discretionary_spending_effect: dict[str, float] | None = None

    def __post_init__(self) -> None:
        if not 0 < self.treatment_probability < 1:
            raise ValueError("treatment_probability must be between 0 and 1.")
        if not 0 < self.confidence < 1:
            raise ValueError("confidence must be between 0 and 1.")
        if self.max_smd <= 0:
            raise ValueError("max_smd must be positive.")


DEFAULT_STRESS_EFFECT = {
    "LIQUIDITY_PROTECTION": 0.20,
    "SPENDING_CONTROL": 0.16,
    "OBLIGATION_MANAGEMENT": 0.14,
}

DEFAULT_CASH_BUFFER_EFFECT = {
    "LIQUIDITY_PROTECTION": 0.10,
    "SPENDING_CONTROL": 0.06,
    "OBLIGATION_MANAGEMENT": 0.07,
}

DEFAULT_SAVINGS_RATE_EFFECT = {
    "LIQUIDITY_PROTECTION": 0.02,
    "SPENDING_CONTROL": 0.03,
    "OBLIGATION_MANAGEMENT": 0.02,
}

DEFAULT_DISCRETIONARY_EFFECT = {
    "LIQUIDITY_PROTECTION": -0.02,
    "SPENDING_CONTROL": -0.08,
    "OBLIGATION_MANAGEMENT": -0.02,
}


INTERVENTION_MAP = {
    "LIQUIDITY_PRESSURE": "LIQUIDITY_PROTECTION",
    "SPENDING_PRESSURE": "SPENDING_CONTROL",
    "OBLIGATION_PRESSURE": "OBLIGATION_MANAGEMENT",
}


BALANCE_FEATURES = [
    "risk_probability",
    "cash_buffer_ratio",
    "expense_pressure",
    "spending_growth",
    "discretionary_pressure",
    "recurring_spending",
    "balance_growth",
]


CONTINUOUS_BASELINE_FEATURES = [
    "cash_buffer_ratio",
    "savings_rate",
    "discretionary_spending",
]


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return default
    return value if np.isfinite(value) else default


def _rank_percentile(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    median = values.median()
    values = values.fillna(median if pd.notna(median) else 0.0)
    if values.nunique() <= 1:
        return pd.Series(0.5, index=series.index, dtype=float)
    return values.rank(method="average", pct=True)


def build_pressure_scores(df: pd.DataFrame) -> pd.DataFrame:
    """Build normalized behavioral pressure scores from observed features.

    Scores are percentile-based within the experiment population, avoiding
    brittle absolute thresholds. Higher values indicate greater pressure.
    """
    result = df.copy().reset_index(drop=True)

    cash = _rank_percentile(result["cash_buffer_ratio"])
    expense = _rank_percentile(result["expense_pressure"])
    spending_growth = _rank_percentile(result["spending_growth"])
    baseline_dev = _rank_percentile(result["spending_baseline_deviation"])
    discretionary = _rank_percentile(result["discretionary_pressure"])
    recurring = _rank_percentile(result["recurring_spending"])
    balance_decline = _rank_percentile(-pd.to_numeric(result["balance_growth"], errors="coerce"))

    result["liquidity_pressure_score"] = (
        0.45 * (1.0 - cash)
        + 0.30 * expense
        + 0.25 * balance_decline
    )
    result["spending_pressure_score"] = (
        0.40 * spending_growth
        + 0.35 * baseline_dev
        + 0.25 * discretionary
    )
    result["obligation_pressure_score"] = (
        0.45 * recurring
        + 0.30 * expense
        + 0.25 * balance_decline
    )

    score_columns = [
        "liquidity_pressure_score",
        "spending_pressure_score",
        "obligation_pressure_score",
    ]
    result["max_pressure_score"] = result[score_columns].max(axis=1)
    result["pressure_state"] = result[score_columns].idxmax(axis=1).map({
        "liquidity_pressure_score": "LIQUIDITY_PRESSURE",
        "spending_pressure_score": "SPENDING_PRESSURE",
        "obligation_pressure_score": "OBLIGATION_PRESSURE",
    })
    result["pressure_severity"] = result["max_pressure_score"]
    return result


def _balance_score(
    df: pd.DataFrame,
    intervention_category: str,
    treatment_indices: np.ndarray,
    features: Iterable[str] = BALANCE_FEATURES,
) -> float:
    """Return the maximum finite SMD for a candidate assignment."""
    cohort = df[df["intervention_category"] == intervention_category]
    treatment_indices = np.asarray(treatment_indices, dtype=int)
    treatment_set = set(treatment_indices.tolist())
    treatment = cohort.loc[cohort.index.isin(treatment_set)]
    control = cohort.loc[~cohort.index.isin(treatment_set)]

    if treatment.empty or control.empty:
        return np.inf

    scores = []
    for feature in features:
        if feature not in cohort.columns:
            continue
        smd = standardized_mean_difference(treatment[feature], control[feature])
        if np.isfinite(smd):
            scores.append(float(smd))
        else:
            return np.inf
    return max(scores) if scores else np.inf


def _rerandomize_cohort(
    df: pd.DataFrame,
    intervention_category: str,
    random_state: int,
    treatment_probability: float,
    n_candidates: int = 3000,
) -> tuple[np.ndarray, float]:
    """Find a balanced assignment while preserving severity-stratum mix."""
    cohort = df[
        (df["eligible"]) & (df["intervention_category"] == intervention_category)
    ].copy()
    cohort_indices = cohort.index.to_numpy(dtype=int)
    n = len(cohort_indices)
    if n < 2:
        return np.array([], dtype=int), np.inf

    n_treatment = int(round(n * treatment_probability))
    n_treatment = min(max(n_treatment, 1), n - 1)

    severity_rank = cohort["pressure_severity"].rank(method="first", pct=True)
    cohort = cohort.assign(
        _severity_stratum=pd.cut(
            severity_rank,
            bins=[0.0, 1 / 3, 2 / 3, 1.0],
            labels=["LOW", "MEDIUM", "HIGH"],
            include_lowest=True,
        )
    )

    strata = {
        str(name): group.index.to_numpy(dtype=int)
        for name, group in cohort.groupby("_severity_stratum", observed=False)
    }
    rng = np.random.default_rng(random_state)

    # Target treatment counts are allocated proportionally by stratum, with
    # correction so their sum equals the requested total exactly.
    raw_targets = {
        name: len(indices) * treatment_probability
        for name, indices in strata.items()
    }
    targets = {name: int(np.floor(value)) for name, value in raw_targets.items()}
    remainder = n_treatment - sum(targets.values())
    order = sorted(
        strata,
        key=lambda name: raw_targets[name] - targets[name],
        reverse=True,
    )
    for name in order:
        if remainder <= 0:
            break
        if targets[name] < len(strata[name]):
            targets[name] += 1
            remainder -= 1

    best_indices = None
    best_score = np.inf
    for _ in range(max(1, n_candidates)):
        selected = []
        for name, indices in strata.items():
            target = targets[name]
            if target <= 0:
                continue
            if target >= len(indices):
                # Keep at least one control observation in every populated
                # stratum whenever the cohort size permits it.
                target = len(indices) - 1 if len(indices) > 1 else len(indices)
            if target > 0:
                selected.extend(
                    rng.choice(indices, size=target, replace=False).tolist()
                )

        # Adjust for edge cases caused by very small strata.
        if len(selected) < n_treatment:
            remaining = np.setdiff1d(cohort_indices, np.asarray(selected, dtype=int))
            extra = rng.choice(remaining, size=n_treatment - len(selected), replace=False)
            selected.extend(extra.tolist())
        elif len(selected) > n_treatment:
            selected = rng.choice(np.asarray(selected, dtype=int), size=n_treatment, replace=False).tolist()

        candidate = np.asarray(selected, dtype=int)
        score = _balance_score(df, intervention_category, candidate)
        if score < best_score:
            best_score = score
            best_indices = candidate.copy()
            if best_score <= 0.10:
                break

    return (
        best_indices if best_indices is not None else np.array([], dtype=int),
        best_score,
    )

def assign_policy_specific_cohorts(
    df: pd.DataFrame,
    random_state: int = RANDOM_STATE,
    treatment_probability: float = 0.50,
    eligibility_percentile: float = 0.70,
    min_cohort_size: int = 20,
    rerandomization_candidates: int = 3000,
) -> pd.DataFrame:
    """Assign final behavioral cohorts and balanced treatment/control groups.

    Eligibility is based on the upper tail of a normalized behavioral-pressure
    score. Each eligible observation is assigned to its strongest pressure
    domain. Treatment/control allocation is then rerandomized within each
    cohort to minimize maximum baseline SMD.
    """
    if not 0 < treatment_probability < 1:
        raise ValueError("treatment_probability must be between 0 and 1.")
    if not 0.50 <= eligibility_percentile < 1:
        raise ValueError("eligibility_percentile must be in [0.50, 1).")
    if min_cohort_size < 2:
        raise ValueError("min_cohort_size must be at least 2.")

    result = build_pressure_scores(df)
    threshold = float(result["max_pressure_score"].quantile(eligibility_percentile))
    result["eligibility_threshold"] = threshold
    result["eligible"] = result["max_pressure_score"] >= threshold
    result["intervention_category"] = result["pressure_state"].map(INTERVENTION_MAP)
    result.loc[~result["eligible"], "intervention_category"] = pd.NA
    result["treatment"] = 0
    result["control"] = 0
    result["randomization_stratum"] = "INELIGIBLE"
    result["randomization_balance_score"] = np.nan

    # A single observation cannot be meaningfully randomized. Drop it from
    # eligibility rather than manufacturing a treatment/control comparison.
    for intervention in INTERVENTION_MAP.values():
        mask = result["intervention_category"] == intervention
        n = int(mask.sum())
        if 0 < n < min_cohort_size:
            result.loc[mask, "eligible"] = False
            result.loc[mask, "intervention_category"] = pd.NA

    for offset, intervention in enumerate(INTERVENTION_MAP.values(), start=1):
        mask = result["eligible"] & (result["intervention_category"] == intervention)
        indices = result.index[mask].to_numpy(dtype=int)
        if len(indices) < min_cohort_size:
            continue

        # Tertile strata preserve severity representation while rerandomization
        # handles the multivariate baseline balance.
        severity = result.loc[indices, "pressure_severity"].rank(method="first", pct=True)
        result.loc[indices, "randomization_stratum"] = pd.cut(
            severity,
            bins=[0.0, 1 / 3, 2 / 3, 1.0],
            labels=["LOW", "MEDIUM", "HIGH"],
            include_lowest=True,
        ).astype(str).map(lambda x: f"{intervention}_{x}")

        treatment_indices, score = _rerandomize_cohort(
            result,
            intervention,
            random_state=random_state + offset * 1009,
            treatment_probability=treatment_probability,
            n_candidates=rerandomization_candidates,
        )
        if len(treatment_indices) == 0:
            raise AssertionError(f"Could not construct treatment assignment for {intervention}.")
        result.loc[treatment_indices, "treatment"] = 1
        result.loc[indices, "control"] = (result.loc[indices, "treatment"] == 0).astype(int)
        result.loc[indices, "randomization_balance_score"] = score

    result["control"] = (
        result["eligible"] & (result["treatment"] == 0)
    ).astype(int)
    return result

def calculate_cohort_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for intervention in INTERVENTION_MAP.values():
        cohort = df[
            df["intervention_category"] == intervention
        ]
        treatment_n = int((cohort["treatment"] == 1).sum())
        control_n = int((cohort["treatment"] == 0).sum())
        total_n = treatment_n + control_n

        rows.append(
            {
                "intervention_category": intervention,
                "total_n": total_n,
                "treatment_n": treatment_n,
                "control_n": control_n,
                "treatment_rate": (
                    treatment_n / total_n if total_n else np.nan
                ),
            }
        )

    return pd.DataFrame(rows)


def standardized_mean_difference(
    treatment: pd.Series,
    control: pd.Series,
) -> float:
    """Return absolute standardized mean difference."""
    treatment = pd.to_numeric(treatment, errors="coerce").dropna()
    control = pd.to_numeric(control, errors="coerce").dropna()

    if treatment.empty or control.empty:
        return np.nan

    treatment_mean = treatment.mean()
    control_mean = control.mean()

    treatment_var = treatment.var(ddof=1) if len(treatment) > 1 else 0.0
    control_var = control.var(ddof=1) if len(control) > 1 else 0.0

    pooled_sd = np.sqrt(
        max((treatment_var + control_var) / 2.0, 0.0)
    )

    if pooled_sd == 0:
        return 0.0 if np.isclose(treatment_mean, control_mean) else np.inf

    return float(abs(treatment_mean - control_mean) / pooled_sd)


def check_randomization_balance(
    df: pd.DataFrame,
    intervention_category: str | None = None,
    features: Iterable[str] = BALANCE_FEATURES,
    max_smd: float = 0.20,
) -> pd.DataFrame:
    """Calculate baseline balance and standardized mean differences."""
    if max_smd <= 0:
        raise ValueError("max_smd must be positive.")

    cohort = df.copy()
    if intervention_category is not None:
        cohort = cohort[
            cohort["intervention_category"] == intervention_category
        ].copy()

    treatment = cohort[cohort["treatment"] == 1]
    control = cohort[cohort["treatment"] == 0]

    rows = []
    for feature in features:
        if feature not in cohort.columns:
            continue

        treatment_values = treatment[feature]
        control_values = control[feature]

        t_mean = pd.to_numeric(treatment_values, errors="coerce").mean()
        c_mean = pd.to_numeric(control_values, errors="coerce").mean()
        smd = standardized_mean_difference(
            treatment_values,
            control_values,
        )

        rows.append(
            {
                "intervention_category": intervention_category,
                "feature": feature,
                "treatment_mean": float(t_mean),
                "control_mean": float(c_mean),
                "absolute_difference": float(abs(t_mean - c_mean)),
                "smd": float(smd),
                "balanced": bool(np.isfinite(smd) and smd <= max_smd),
            }
        )

    return pd.DataFrame(rows)


def _synthetic_event_probability(
    row: pd.Series,
    intervention: str,
) -> float:
    """
    Construct a synthetic adverse-event probability.

    This deliberately represents a simulation mechanism, not an observed
    causal model. It combines the observed target, model risk probability,
    and behavioral pressure characteristics so every eligible cohort can
    produce non-zero outcomes.
    """
    observed_stress = _safe_float(row.get("financial_stress_30d"))
    risk = np.clip(_safe_float(row.get("risk_probability")), 0.0, 1.0)
    pressure = str(row.get("pressure_state", "STABLE"))

    pressure_floor = {
        "LIQUIDITY_PRESSURE": 0.16,
        "SPENDING_PRESSURE": 0.12,
        "OBLIGATION_PRESSURE": 0.14,
    }.get(pressure, 0.03)

    severity = np.clip(
        0.50 * _safe_float(row.get("expense_pressure"))
        + 0.30 * max(0.0, _safe_float(row.get("spending_growth")))
        + 0.20 * max(0.0, -_safe_float(row.get("balance_growth"))),
        0.0,
        2.0,
    )

    probability = (
        0.30 * observed_stress
        + 0.30 * risk
        + pressure_floor
        + 0.08 * min(severity, 1.0)
    )

    return float(np.clip(probability, 0.02, 0.85))


def simulate_treatment_response(
    df: pd.DataFrame,
    random_state: int = RANDOM_STATE,
    config: ExperimentConfig | None = None,
) -> pd.DataFrame:
    """
    Simulate synthetic treatment outcomes for eligible observations.

    The output explicitly labels simulated outcomes so they cannot be confused
    with observed future outcomes.
    """
    config = config or ExperimentConfig()
    result = df.copy().reset_index(drop=True)
    rng = np.random.default_rng(random_state)

    stress_effect = config.stress_effect or DEFAULT_STRESS_EFFECT
    cash_effect = config.cash_buffer_effect or DEFAULT_CASH_BUFFER_EFFECT
    savings_effect = config.savings_rate_effect or DEFAULT_SAVINGS_RATE_EFFECT
    discretionary_effect = (
        config.discretionary_spending_effect
        or DEFAULT_DISCRETIONARY_EFFECT
    )

    synthetic_stress = np.zeros(len(result), dtype=int)
    synthetic_cash_change = np.zeros(len(result), dtype=float)
    synthetic_savings_change = np.zeros(len(result), dtype=float)
    synthetic_discretionary_change = np.zeros(len(result), dtype=float)

    for position, row in result.iterrows():
        if not bool(row.get("eligible", False)):
            continue

        intervention = str(row["intervention_category"])
        base_probability = _synthetic_event_probability(
            row,
            intervention,
        )

        if int(row.get("treatment", 0)) == 1:
            rescue = stress_effect.get(intervention, 0.0)
            event_probability = base_probability * (1.0 - rescue)
        else:
            event_probability = base_probability

        synthetic_stress[position] = int(
            rng.random() < np.clip(event_probability, 0.0, 1.0)
        )

        # Continuous outcomes receive user-level noise so the synthetic
        # experiment has realistic variance instead of a mechanically fixed
        # treatment-control difference.
        cash_baseline = 0.01 * _safe_float(row.get("balance_growth"))
        savings_baseline = 0.02 * _safe_float(row.get("savings_rate_change"))
        discretionary_baseline = -0.01 * _safe_float(
            row.get("spending_growth")
        )

        synthetic_cash_change[position] = (
            cash_baseline + rng.normal(0.0, 0.04)
        )
        synthetic_savings_change[position] = (
            savings_baseline + rng.normal(0.0, 0.015)
        )
        synthetic_discretionary_change[position] = (
            discretionary_baseline + rng.normal(0.0, 0.04)
        )

        if int(row.get("treatment", 0)) == 1:
            synthetic_cash_change[position] += cash_effect.get(
                intervention, 0.0
            )
            synthetic_savings_change[position] += savings_effect.get(
                intervention, 0.0
            )
            synthetic_discretionary_change[position] += discretionary_effect.get(
                intervention, 0.0
            )

    result["synthetic_financial_stress_30d"] = synthetic_stress
    result["synthetic_cash_buffer_change"] = synthetic_cash_change
    result["synthetic_savings_rate_change"] = synthetic_savings_change
    result["synthetic_discretionary_spending_change"] = (
        synthetic_discretionary_change
    )

    return result


def calculate_group_statistics(
    df: pd.DataFrame,
    outcome_column: str = "synthetic_financial_stress_30d",
) -> pd.DataFrame:
    rows = []

    eligible = df[df["eligible"]].copy()

    for treatment_value, label in [(1, "treatment"), (0, "control")]:
        group = eligible[eligible["treatment"] == treatment_value]
        outcomes = pd.to_numeric(group[outcome_column], errors="coerce").dropna()

        rows.append(
            {
                "group": label,
                "n": int(len(outcomes)),
                "outcome_rate": (
                    float(outcomes.mean()) if len(outcomes) else np.nan
                ),
            }
        )

    return pd.DataFrame(rows)


def calculate_ate(
    df: pd.DataFrame,
    outcome_column: str = "synthetic_financial_stress_30d",
    confidence: float = 0.95,
) -> dict:
    if not 0 < confidence < 1:
        raise ValueError("confidence must be between 0 and 1.")

    eligible = df[df["eligible"]].copy()
    treatment = pd.to_numeric(
        eligible.loc[eligible["treatment"] == 1, outcome_column],
        errors="coerce",
    ).dropna()
    control = pd.to_numeric(
        eligible.loc[eligible["treatment"] == 0, outcome_column],
        errors="coerce",
    ).dropna()

    if len(treatment) == 0 or len(control) == 0:
        raise ValueError("Both treatment and control groups are required.")

    treatment_mean = float(treatment.mean())
    control_mean = float(control.mean())
    ate = treatment_mean - control_mean

    treatment_var = float(treatment.var(ddof=1)) if len(treatment) > 1 else 0.0
    control_var = float(control.var(ddof=1)) if len(control) > 1 else 0.0

    standard_error = np.sqrt(
        treatment_var / len(treatment)
        + control_var / len(control)
    )

    z = 1.959963984540054
    if confidence != 0.95:
        from scipy.stats import norm
        z = float(norm.ppf(0.5 + confidence / 2.0))

    ci_lower = ate - z * standard_error
    ci_upper = ate + z * standard_error

    relative_effect = (
        ate / control_mean
        if not np.isclose(control_mean, 0.0)
        else np.nan
    )

    return {
        "treatment_n": int(len(treatment)),
        "control_n": int(len(control)),
        "treatment_outcome_rate": treatment_mean,
        "control_outcome_rate": control_mean,
        "ate": float(ate),
        "relative_effect": float(relative_effect),
        "standard_error": float(standard_error),
        "ci_lower": float(ci_lower),
        "ci_upper": float(ci_upper),
        "outcome_column": outcome_column,
    }


def calculate_intervention_effects(
    df: pd.DataFrame,
    outcome_column: str = "synthetic_financial_stress_30d",
    confidence: float = 0.95,
) -> pd.DataFrame:
    rows = []

    for intervention in INTERVENTION_MAP.values():
        cohort = df[
            (df["eligible"])
            & (df["intervention_category"] == intervention)
        ].copy()

        if cohort.empty:
            continue

        if cohort["treatment"].nunique() < 2:
            continue

        effect = calculate_ate(
            cohort,
            outcome_column=outcome_column,
            confidence=confidence,
        )
        effect["intervention_category"] = intervention
        rows.append(effect)

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows)[
        [
            "intervention_category",
            "treatment_n",
            "control_n",
            "treatment_outcome_rate",
            "control_outcome_rate",
            "ate",
            "relative_effect",
            "standard_error",
            "ci_lower",
            "ci_upper",
            "outcome_column",
        ]
    ]


def calculate_continuous_effects(
    df: pd.DataFrame,
    outcome_columns: Iterable[str],
    confidence: float = 0.95,
) -> pd.DataFrame:
    rows = []

    for outcome_column in outcome_columns:
        for intervention in INTERVENTION_MAP.values():
            cohort = df[
                (df["eligible"])
                & (df["intervention_category"] == intervention)
            ].copy()

            if cohort.empty or cohort["treatment"].nunique() < 2:
                continue

            effect = calculate_ate(
                cohort,
                outcome_column=outcome_column,
                confidence=confidence,
            )
            effect["intervention_category"] = intervention
            rows.append(effect)

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows)[
        [
            "intervention_category",
            "outcome_column",
            "treatment_n",
            "control_n",
            "treatment_outcome_rate",
            "control_outcome_rate",
            "ate",
            "standard_error",
            "ci_lower",
            "ci_upper",
        ]
    ]
