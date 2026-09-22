from __future__ import annotations

import numpy as np
import pandas as pd

from finsight.experimentation.intervention_policy import (
    classify_financial_pressure,
    select_intervention,
    generate_intervention_record,
    generate_intervention_batch,
)


def make_row(**overrides) -> pd.Series:
    """
    Create a minimal valid behavioral feature row.
    """

    defaults = {
        "user_id": 1,
        "month": pd.Timestamp("2026-09-01"),

        "cash_buffer_ratio": 2.0,
        "expense_pressure": 0.50,

        "spending_growth": 0.00,
        "spending_baseline_deviation": 0.00,
        "discretionary_pressure": 0.20,

        "recurring_spending": 10000.0,
        "balance_growth": 0.10,

        "negative_cash_flow": False,
    }

    defaults.update(overrides)

    return pd.Series(defaults)


def test_liquidity_pressure():

    row = make_row(
        cash_buffer_ratio=0.20,
        expense_pressure=0.95,
        negative_cash_flow=True,
    )

    state = classify_financial_pressure(
        row
    )

    assert state == "LIQUIDITY_PRESSURE"

    intervention = select_intervention(
        row,
        risk_probability=0.80,
    )

    assert (
        intervention.category
        == "LIQUIDITY_PROTECTION"
    )


def test_spending_pressure():

    row = make_row(
        spending_growth=0.25,
        spending_baseline_deviation=0.20,
        discretionary_pressure=0.80,
    )

    state = classify_financial_pressure(
        row
    )

    assert state == "SPENDING_PRESSURE"

    intervention = select_intervention(
        row,
        risk_probability=0.80,
    )

    assert (
        intervention.category
        == "SPENDING_CONTROL"
    )


def test_obligation_pressure():

    row = make_row(
        recurring_spending=25000,
        expense_pressure=0.90,
        balance_growth=-0.20,
    )

    state = classify_financial_pressure(
        row
    )

    assert state == "OBLIGATION_PRESSURE"

    intervention = select_intervention(
        row,
        risk_probability=0.80,
    )

    assert (
        intervention.category
        == "OBLIGATION_MANAGEMENT"
    )


def test_stable_user():

    row = make_row(
        cash_buffer_ratio=2.5,
        expense_pressure=0.45,
        spending_growth=0.02,
        spending_baseline_deviation=0.01,
        discretionary_pressure=0.20,
        balance_growth=0.15,
    )

    state = classify_financial_pressure(
        row
    )

    assert state == "STABLE"

    intervention = select_intervention(
        row,
        risk_probability=0.10,
    )

    assert (
        intervention.category
        == "SAVINGS_REINFORCEMENT"
    )


def test_high_risk_priority():

    row = make_row(
        cash_buffer_ratio=0.20,
        expense_pressure=0.95,
        spending_growth=0.30,
        discretionary_pressure=0.90,
        negative_cash_flow=True,
    )

    intervention = select_intervention(
        row,
        risk_probability=0.90,
    )

    # Liquidity protection has the highest priority.
    assert (
        intervention.category
        == "LIQUIDITY_PROTECTION"
    )


def test_batch_generation():

    rows = pd.DataFrame(
        [
            make_row(
                user_id=1,
                cash_buffer_ratio=0.20,
                expense_pressure=0.95,
            ),
            make_row(
                user_id=2,
                spending_growth=0.30,
                discretionary_pressure=0.80,
            ),
            make_row(
                user_id=3,
            ),
        ]
    )

    probabilities = np.array(
        [0.80, 0.70, 0.10]
    )

    result = generate_intervention_batch(
        rows,
        probabilities,
    )

    assert len(result) == 3

    assert (
        result["user_id"].tolist()
        == [1, 2, 3]
    )

    assert (
        result[
            "intervention_category"
        ].tolist()
        == [
            "LIQUIDITY_PROTECTION",
            "SPENDING_CONTROL",
            "SAVINGS_REINFORCEMENT",
        ]
    )


def test_record_generation():

    row = make_row(
        user_id=99,
        cash_buffer_ratio=0.20,
        expense_pressure=0.90,
    )

    record = generate_intervention_record(
        row,
        risk_probability=0.85,
    )

    assert record["user_id"] == 99

    assert (
        record[
            "intervention_category"
        ]
        == "LIQUIDITY_PROTECTION"
    )

    assert (
        record["risk_probability"]
        == 0.85
    )


def main():

    print("=" * 80)
    print("FinSight Intervention Policy Tests")
    print("=" * 80)

    test_liquidity_pressure()
    print("✓ Liquidity intervention")

    test_spending_pressure()
    print("✓ Spending intervention")

    test_obligation_pressure()
    print("✓ Obligation intervention")

    test_stable_user()
    print("✓ Stable-user intervention")

    test_high_risk_priority()
    print("✓ Intervention priority")

    test_batch_generation()
    print("✓ Batch generation")

    test_record_generation()
    print("✓ Intervention record")

    print()
    print("=" * 80)
    print("ALL INTERVENTION POLICY TESTS PASSED")
    print("=" * 80)


if __name__ == "__main__":
    main()