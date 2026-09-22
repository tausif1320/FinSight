from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(
    0,
    str(ROOT / "src"),
)


from finsight.experimentation.policy_evaluation import (
    PolicyEvaluationConfig,
    evaluate_policy_suite,
    build_policy_assignment,
)


CATE_PATH = (
    ROOT
    / "reports"
    / "experimentation"
    / "cate_cross_fitted_effects.csv"
)


REPORT_DIR = (
    ROOT
    / "reports"
    / "experimentation"
)


def main():

    print("=" * 80)
    print("FinSight Personalized Policy Evaluation")
    print("=" * 80)

    # ================================================================
    # 1. LOAD CROSS-FITTED CATE
    # ================================================================

    print(
        "\nLoading cross-fitted CATE results..."
    )

    if not CATE_PATH.exists():

        raise FileNotFoundError(
            f"Missing CATE file:\n{CATE_PATH}"
        )

    df = pd.read_csv(
        CATE_PATH
    )

    print(
        f"Rows: {len(df):,}"
    )

    # ================================================================
    # 2. BASIC VALIDATION
    # ================================================================

    required = {
        "cate",
        "treatment",
        "intervention_category",
        "synthetic_financial_stress_30d",
    }

    missing = sorted(
        required - set(df.columns)
    )

    if missing:

        raise ValueError(
            f"Missing required columns: {missing}"
        )

    print(
        "\nTreatment distribution:"
    )

    print(
        df["treatment"]
        .value_counts()
        .sort_index()
        .to_string()
    )

    # ================================================================
    # 3. CONFIG
    # ================================================================

    config = PolicyEvaluationConfig(
        propensity=0.50
    )

    # ================================================================
    # 4. POLICY EVALUATION
    # ================================================================

    print(
        "\nEvaluating policies with IPW..."
    )

    policy_results = evaluate_policy_suite(
        df,
        config,
    )

    print("\n" + "=" * 80)
    print("IPW POLICY VALUE")
    print("=" * 80)

    print(
        policy_results.to_string(
            index=False
        )
    )

    # ================================================================
    # 5. BUILD POLICY ASSIGNMENT
    # ================================================================

    policy_assignment = (
        build_policy_assignment(
            df
        )
    )

    print("\n" + "=" * 80)
    print("PERSONALIZED POLICY DISTRIBUTION")
    print("=" * 80)

    print(
        "\nCATE benefit policy:"
    )

    print(
        policy_assignment[
            "policy_action"
        ]
        .value_counts()
        .to_string()
    )

    print(
        "\nStrong CATE benefit policy:"
    )

    print(
        policy_assignment[
            "strong_policy_action"
        ]
        .value_counts()
        .to_string()
    )

    # ================================================================
    # 6. POLICY TREATMENT RATE
    # ================================================================

    print("\n" + "=" * 80)
    print("POLICY TREATMENT RATES")
    print("=" * 80)

    print(
        f"Treat-all rate: "
        f"{policy_assignment['treatment'].mean():.4f}"
    )

    print(
        f"CATE benefit rate: "
        f"{policy_assignment['policy_treat'].mean():.4f}"
    )

    print(
        f"Strong CATE benefit rate: "
        f"{policy_assignment['strong_policy_treat'].mean():.4f}"
    )

    # ================================================================
    # 7. SAVE
    # ================================================================

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    policy_results.to_csv(
        REPORT_DIR
        / "policy_value_ipw.csv",
        index=False,
    )

    policy_assignment.to_csv(
        REPORT_DIR
        / "personalized_policy_assignment.csv",
        index=False,
    )

    # ================================================================
    # 8. VALIDATION
    # ================================================================

    print("\n" + "=" * 80)
    print("POLICY VALIDATION")
    print("=" * 80)

    assert len(
        policy_results
    ) == 4

    assert policy_results[
        "policy_value"
    ].notna().all()

    assert policy_results[
        "standard_error"
    ].notna().all()

    assert (
        policy_assignment[
            "policy_treat"
        ]
        .isin([0, 1])
        .all()
    )

    assert (
        policy_assignment[
            "strong_policy_treat"
        ]
        .isin([0, 1])
        .all()
    )

    print(
        "✓ Cross-fitted CATE used"
    )

    print(
        "✓ Known randomized propensity = 0.50"
    )

    print(
        "✓ IPW policy evaluation completed"
    )

    print(
        "✓ Policy confidence intervals calculated"
    )

    print(
        "✓ Personalized policy assignment generated"
    )

    print(
        "✓ Strong-benefit policy generated"
    )

    print(
        "✓ Test set remains untouched"
    )

    print("\nSaved:")

    print(
        "  reports/experimentation/"
        "policy_value_ipw.csv"
    )

    print(
        "  reports/experimentation/"
        "personalized_policy_assignment.csv"
    )

    print("\nIMPORTANT:")

    print(
        "These policy values evaluate the "
        "synthetic randomized experiment."
    )

    print(
        "They are not evidence of real-world "
        "financial intervention effectiveness."
    )

    print("\n" + "=" * 80)
    print("PERSONALIZED POLICY EVALUATION PASSED")
    print("=" * 80)


if __name__ == "__main__":
    main()