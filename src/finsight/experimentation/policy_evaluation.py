from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class PolicyEvaluationConfig:
    outcome_column: str = "synthetic_financial_stress_30d"

    treatment_column: str = "treatment"

    propensity: float = 0.50

    benefit_threshold: float = 0.0

    strong_benefit_threshold: float = -0.10


def validate_policy_data(
    df: pd.DataFrame,
    config: PolicyEvaluationConfig,
) -> None:

    required = [
        config.outcome_column,
        config.treatment_column,
        "cate",
        "intervention_category",
    ]

    missing = [
        column
        for column in required
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing policy evaluation columns: {missing}"
        )

    if df.empty:
        raise ValueError(
            "Policy evaluation dataset is empty."
        )

    if not 0 < config.propensity < 1:
        raise ValueError(
            "Propensity must be between 0 and 1."
        )

    treatment_values = set(
        df[
            config.treatment_column
        ]
        .dropna()
        .unique()
    )

    if not treatment_values.issubset({0, 1}):
        raise ValueError(
            "Treatment must contain only 0 and 1."
        )


def calculate_ipw_policy_value(
    df: pd.DataFrame,
    policy_treatment_rule: pd.Series,
    config: PolicyEvaluationConfig | None = None,
) -> dict:

    """
    Estimate policy value using inverse-propensity weighting.

    policy_treatment_rule:
        1 = policy would treat this observation
        0 = policy would leave untreated

    Because v3 used randomized treatment assignment,
    the assignment probability is known.

    Lower outcome means better outcome because the
    outcome is synthetic financial stress.
    """

    config = (
        config
        or PolicyEvaluationConfig()
    )

    validate_policy_data(
        df,
        config,
    )

    rule = (
        pd.Series(
            policy_treatment_rule,
            index=df.index,
        )
        .astype(int)
    )

    if len(rule) != len(df):
        raise ValueError(
            "Policy rule length does not match dataset."
        )

    if not set(
        rule.unique()
    ).issubset({0, 1}):

        raise ValueError(
            "Policy rule must contain only 0/1."
        )

    observed_treatment = (
        df[
            config.treatment_column
        ]
        .astype(int)
    )

    outcome = (
        df[
            config.outcome_column
        ]
        .astype(float)
    )

    p = config.propensity

    # Probability of the actually observed treatment.
    observed_probability = np.where(
        observed_treatment == 1,
        p,
        1.0 - p,
    )

    # Only observations whose randomized assignment
    # agrees with the policy contribute to the estimate.
    policy_match = (
        observed_treatment
        == rule
    )

    weights = (
        policy_match.astype(float)
        / observed_probability
    )

    estimated_value = float(
        np.mean(
            weights * outcome
        )
    )

    effective_sample_size = (
        weights.sum() ** 2
        / np.sum(weights ** 2)
        if np.sum(weights ** 2) > 0
        else 0.0
    )

    return {
        "policy_value": estimated_value,
        "n": len(df),
        "policy_treatment_rate": float(
            rule.mean()
        ),
        "policy_match_rate": float(
            policy_match.mean()
        ),
        "effective_sample_size": float(
            effective_sample_size
        ),
    }


def calculate_policy_value_standard_error(
    df: pd.DataFrame,
    policy_treatment_rule: pd.Series,
    config: PolicyEvaluationConfig | None = None,
) -> float:

    """
    Approximate standard error of the IPW policy value.

    This is a simple variance estimator for the
    randomized synthetic experiment.
    """

    config = (
        config
        or PolicyEvaluationConfig()
    )

    observed_treatment = (
        df[
            config.treatment_column
        ]
        .astype(int)
    )

    outcome = (
        df[
            config.outcome_column
        ]
        .astype(float)
    )

    rule = (
        pd.Series(
            policy_treatment_rule,
            index=df.index,
        )
        .astype(int)
    )

    p = config.propensity

    observed_probability = np.where(
        observed_treatment == 1,
        p,
        1.0 - p,
    )

    policy_match = (
        observed_treatment
        == rule
    )

    contributions = (
        policy_match.astype(float)
        * outcome
        / observed_probability
    )

    return float(
        contributions.std(
            ddof=1
        )
        / np.sqrt(
            len(df)
        )
    )


def build_policy_rules(
    df: pd.DataFrame,
) -> dict[str, pd.Series]:

    """
    Build policy rules from cross-fitted CATE.

    The current v3 design only supports deciding
    treatment vs control within the assigned
    intervention cohort.
    """

    return {

        "NO_INTERVENTION": pd.Series(
            0,
            index=df.index,
        ),

        "TREAT_ALL": pd.Series(
            1,
            index=df.index,
        ),

        "CATE_BENEFIT": (
            df["cate"] < 0
        ).astype(int),

        "CATE_STRONG_BENEFIT": (
            df["cate"] <= -0.10
        ).astype(int),

    }


def evaluate_policy_suite(
    df: pd.DataFrame,
    config: PolicyEvaluationConfig | None = None,
) -> pd.DataFrame:

    config = (
        config
        or PolicyEvaluationConfig()
    )

    validate_policy_data(
        df,
        config,
    )

    policies = build_policy_rules(
        df
    )

    rows = []

    for policy_name, rule in policies.items():

        result = calculate_ipw_policy_value(
            df,
            rule,
            config,
        )

        standard_error = (
            calculate_policy_value_standard_error(
                df,
                rule,
                config,
            )
        )

        result.update(
            {
                "policy": policy_name,
                "standard_error": standard_error,
                "ci_lower": (
                    result["policy_value"]
                    - 1.96 * standard_error
                ),
                "ci_upper": (
                    result["policy_value"]
                    + 1.96 * standard_error
                ),
            }
        )

        rows.append(
            result
        )

    result_df = pd.DataFrame(
        rows
    )

    return result_df[
        [
            "policy",
            "n",
            "policy_value",
            "standard_error",
            "ci_lower",
            "ci_upper",
            "policy_treatment_rate",
            "policy_match_rate",
            "effective_sample_size",
        ]
    ]


def build_policy_assignment(
    df: pd.DataFrame,
) -> pd.DataFrame:

    result = df.copy()

    result[
        "policy_treat"
    ] = (
        result["cate"] < 0
    ).astype(int)

    result[
        "strong_policy_treat"
    ] = (
        result["cate"] <= -0.10
    ).astype(int)

    result[
        "policy_action"
    ] = np.where(
        result["policy_treat"] == 1,
        result[
            "intervention_category"
        ],
        "NO_INTERVENTION",
    )

    result[
        "strong_policy_action"
    ] = np.where(
        result[
            "strong_policy_treat"
        ] == 1,
        result[
            "intervention_category"
        ],
        "NO_INTERVENTION",
    )

    return result


__all__ = [
    "PolicyEvaluationConfig",
    "validate_policy_data",
    "calculate_ipw_policy_value",
    "calculate_policy_value_standard_error",
    "build_policy_rules",
    "evaluate_policy_suite",
    "build_policy_assignment",
]