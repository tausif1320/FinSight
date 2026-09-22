import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from finsight.data_generation.population import (
    generate_population,
)
from finsight.data_generation.income import (
    generate_base_income,
)
from finsight.data_generation.obligations import (
    generate_obligations,
)
from finsight.data_generation.shocks import (
    generate_shocks,
    SHOCK_TYPES,
)
from finsight.utils.random import create_rng


SEED = 42

N_USERS = 1000
N_MONTHS = 12


def main():

    rng = create_rng(SEED)

    # =========================================================
    # POPULATION
    # =========================================================

    population = generate_population(
        N_USERS,
        rng,
    )

    population = generate_base_income(
        population,
        rng,
    )

    # Obligations are generated here because obligation burden
    # is part of the population, and this confirms the broader
    # data-generation pipeline remains compatible.
    _ = generate_obligations(
        population,
        rng,
    )

    # =========================================================
    # SHOCK GENERATION
    # =========================================================

    shocks = generate_shocks(
        population=population,
        n_months=N_MONTHS,
        rng=rng,
    )

    # =========================================================
    # BASIC INFORMATION
    # =========================================================

    print("\nShock profile shape:")
    print(shocks.shape)

    print("\nColumns:")
    print(shocks.columns.tolist())

    # =========================================================
    # EMPTY DATAFRAME HANDLING
    # =========================================================

    # With 1,000 users × 12 months, the expected probability
    # of observing no shocks at all is extremely small.
    assert not shocks.empty, (
        "No shocks were generated. "
        "Increase population or inspect probabilities."
    )

    # =========================================================
    # REQUIRED COLUMNS
    # =========================================================

    required_columns = {
        "user_id",
        "month",
        "shock_type",
        "shock_amount",
        "income_impact",
        "duration_months",
        "severity",
    }

    missing_columns = (
        required_columns
        - set(shocks.columns)
    )

    assert not missing_columns, (
        "Missing shock columns: "
        f"{sorted(missing_columns)}"
    )

    # =========================================================
    # MISSING VALUES
    # =========================================================

    print("\nMissing values:")
    print(shocks.isna().sum())

    assert not shocks.isna().any().any()

    # =========================================================
    # SHOCK TYPES
    # =========================================================

    print("\nShock type distribution:")
    print(
        shocks["shock_type"].value_counts()
    )

    assert shocks[
        "shock_type"
    ].isin(SHOCK_TYPES).all()

    # =========================================================
    # MONTH VALIDATION
    # =========================================================

    assert (
        shocks["month"] >= 1
    ).all()

    assert (
        shocks["month"] <= N_MONTHS
    ).all()

    # =========================================================
    # USER VALIDATION
    # =========================================================

    assert (
        shocks["user_id"].isin(
            population["user_id"]
        )
    ).all()

    # =========================================================
    # SHOCK AMOUNT VALIDATION
    # =========================================================

    assert np.isfinite(
        shocks["shock_amount"]
    ).all()

    assert (
        shocks["shock_amount"] >= 0
    ).all()

    # =========================================================
    # INCOME IMPACT VALIDATION
    # =========================================================

    assert np.isfinite(
        shocks["income_impact"]
    ).all()

    assert (
        shocks["income_impact"] >= 0
    ).all()

    assert (
        shocks["income_impact"] <= 1
    ).all()

    # =========================================================
    # DURATION VALIDATION
    # =========================================================

    assert (
        shocks["duration_months"] >= 1
    ).all()

    assert (
        shocks["duration_months"] <= 2
    ).all()

    # =========================================================
    # SEVERITY VALIDATION
    # =========================================================

    valid_severities = {
        "LOW",
        "MEDIUM",
        "HIGH",
        "SEVERE",
    }

    assert shocks[
        "severity"
    ].isin(valid_severities).all()

    # =========================================================
    # USER-MONTH UNIQUENESS
    # =========================================================

    # The current generator deliberately allows at most one
    # shock per user-month.
    assert not (
        shocks[
            ["user_id", "month"]
        ]
        .duplicated()
        .any()
    )

    # =========================================================
    # SHOCK TYPE SPECIFIC VALIDATION
    # =========================================================

    income_disruptions = shocks[
        shocks["shock_type"]
        == "INCOME_DISRUPTION"
    ]

    if not income_disruptions.empty:

        assert (
            income_disruptions[
                "shock_amount"
            ] == 0
        ).all()

        assert (
            income_disruptions[
                "income_impact"
            ] > 0
        ).all()

    non_income_shocks = shocks[
        shocks["shock_type"]
        != "INCOME_DISRUPTION"
    ]

    if not non_income_shocks.empty:

        assert (
            non_income_shocks[
                "shock_amount"
            ] > 0
        ).all()

        assert (
            non_income_shocks[
                "income_impact"
            ] == 0
        ).all()

    # =========================================================
    # SHOCK RATE
    # =========================================================

    total_possible_user_months = (
        N_USERS * N_MONTHS
    )

    shock_rate = (
        len(shocks)
        / total_possible_user_months
    )

    print(
        f"\nShock rate: "
        f"{shock_rate:.2%}"
    )

    assert shock_rate > 0

    # Keep the synthetic population from becoming
    # absurdly shock-heavy.
    assert shock_rate < 0.20

    # =========================================================
    # REPRODUCIBILITY
    # =========================================================

    rng2 = create_rng(SEED)

    population2 = generate_population(
        N_USERS,
        rng2,
    )

    population2 = generate_base_income(
        population2,
        rng2,
    )

    _ = generate_obligations(
        population2,
        rng2,
    )

    shocks2 = generate_shocks(
        population=population2,
        n_months=N_MONTHS,
        rng=rng2,
    )

    pd.testing.assert_frame_equal(
        shocks.reset_index(drop=True),
        shocks2.reset_index(drop=True),
    )

    print(
        "\nShock reproducibility test passed."
    )

    # =========================================================
    # SAMPLE
    # =========================================================

    print("\nShock sample:")

    print(
        shocks.head(20).to_string(
            index=False
        )
    )

    # =========================================================
    # FINAL RESULT
    # =========================================================

    print(
        "\nShock generation validation passed."
    )


if __name__ == "__main__":
    main()