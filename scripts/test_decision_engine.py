from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(
    0,
    str(ROOT / "src"),
)


from finsight.experimentation.cate import (
    DEFAULT_FEATURES,
)

from finsight.decision.engine import (
    DecisionConfig,
    generate_decision,
    generate_decision_batch,
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
    / "decision"
)


def main():

    print("=" * 80)
    print("FinSight Decision Engine")
    print("=" * 80)

    # ================================================================
    # 1. LOAD CATE DATA
    # ================================================================

    print(
        "\nLoading cross-fitted CATE data..."
    )

    if not CATE_PATH.exists():

        raise FileNotFoundError(
            f"Missing:\n{CATE_PATH}"
        )

    df = pd.read_csv(
        CATE_PATH
    )

    print(
        f"Rows: {len(df):,}"
    )

    # ================================================================
    # 2. VALIDATE INPUT
    # ================================================================

    required = [
        "user_id",
        "month",
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
            f"Missing columns: {missing}"
        )

    # ================================================================
    # 3. CREATE A DEMONSTRATION RISK SCORE
    # ================================================================

    print(
        "\nCreating decision-engine risk input..."
    )

    # The actual production system should receive
    # the calibrated HGB probability directly.
    #
    # This test intentionally creates a deterministic
    # demonstration risk score from available features.
    #
    # It is NOT a replacement for the trained stress model.

    df["risk_probability"] = (
        0.20
        + 0.30
        * df[
            "expense_pressure"
        ].clip(
            lower=0,
            upper=1,
        )
        + 0.20
        * (
            1
            - df[
                "cash_buffer_ratio"
            ].clip(
                lower=0,
                upper=1,
            )
        )
        + 0.15
        * df[
            "spending_growth"
        ].clip(
            lower=0,
            upper=1,
        )
        + 0.15
        * df[
            "discretionary_pressure"
        ].clip(
            lower=0,
            upper=1,
        )
    ).clip(
        lower=0,
        upper=1,
    )

    # ================================================================
    # 4. GENERATE DECISIONS
    # ================================================================

    print(
        "Generating decisions..."
    )

    config = DecisionConfig()

    decisions = generate_decision_batch(
        df,
        risk_column="risk_probability",
        cate_column="cate",
        config=config,
    )

    # ================================================================
    # 5. SUMMARY
    # ================================================================

    print("\n" + "=" * 80)
    print("DECISION SUMMARY")
    print("=" * 80)

    print(
        "\nRisk levels:"
    )

    print(
        decisions[
            "risk_level"
        ]
        .value_counts()
        .to_string()
    )

    print(
        "\nFinancial pressure:"
    )

    print(
        decisions[
            "financial_pressure"
        ]
        .value_counts()
        .to_string()
    )

    print(
        "\nRecommended intervention:"
    )

    print(
        decisions[
            "intervention"
        ]
        .value_counts()
        .to_string()
    )

    print(
        "\nRecommendation status:"
    )

    print(
        decisions[
            "recommendation_status"
        ]
        .value_counts()
        .to_string()
    )

    # ================================================================
    # 6. SHOW EXAMPLES
    # ================================================================

    print("\n" + "=" * 80)
    print("SAMPLE DECISIONS")
    print("=" * 80)

    display_columns = [
        "user_id",
        "month",
        "risk_probability",
        "risk_level",
        "financial_pressure",
        "intervention",
        "cate",
        "recommendation_status",
    ]

    print(
        decisions[
            display_columns
        ]
        .head(10)
        .to_string(
            index=False
        )
    )

    # ================================================================
    # 7. SAVE
    # ================================================================

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    decisions.to_csv(
        REPORT_DIR
        / "decision_engine_output.csv",
        index=False,
    )

    # ================================================================
    # 8. VALIDATION
    # ================================================================

    print("\n" + "=" * 80)
    print("DECISION ENGINE VALIDATION")
    print("=" * 80)

    assert len(
        decisions
    ) == len(df)

    assert decisions[
        "risk_probability"
    ].between(
        0,
        1,
    ).all()

    assert decisions[
        "risk_level"
    ].isin(
        [
            "LOW",
            "MODERATE",
            "HIGH",
        ]
    ).all()

    assert decisions[
        "financial_pressure"
    ].isin(
        [
            "LIQUIDITY_PRESSURE",
            "SPENDING_PRESSURE",
            "OBLIGATION_PRESSURE",
            "STABLE",
        ]
    ).all()

    assert decisions[
        "intervention"
    ].isin(
        [
            "LIQUIDITY_PROTECTION",
            "SPENDING_CONTROL",
            "OBLIGATION_MANAGEMENT",
            "SAVINGS_REINFORCEMENT",
        ]
    ).all()

    assert decisions[
        "explanation"
    ].notna().all()

    assert decisions[
        "intervention_description"
    ].notna().all()

    print(
        "✓ CATE input loaded"
    )

    print(
        "✓ Risk classification completed"
    )

    print(
        "✓ Financial pressure classification completed"
    )

    print(
        "✓ Intervention policy executed"
    )

    print(
        "✓ CATE integrated"
    )

    print(
        "✓ Explanation generated"
    )

    print(
        "✓ Batch decision generation completed"
    )

    print(
        "✓ Decision output validated"
    )

    print(
        "\nSaved:"
    )

    print(
        "  reports/decision/"
        "decision_engine_output.csv"
    )

    print("\n" + "=" * 80)
    print("DECISION ENGINE PASSED")
    print("=" * 80)


if __name__ == "__main__":
    main()