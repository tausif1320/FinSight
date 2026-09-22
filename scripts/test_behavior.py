import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from finsight.data_generation.population import (
    generate_population,
)
from finsight.data_generation.behavior import (
    generate_behavioral_profile,
    REGIMES,
)
from finsight.utils.random import create_rng


SEED = 42

N_USERS = 100
N_MONTHS = 12


def main():

    rng = create_rng(SEED)

    # ---------------------------------------------------------
    # POPULATION
    # ---------------------------------------------------------

    population = generate_population(
        N_USERS,
        rng,
    )

    # ---------------------------------------------------------
    # BEHAVIORAL PROFILE
    # ---------------------------------------------------------

    behavior = generate_behavioral_profile(
        population=population,
        n_months=N_MONTHS,
        rng=rng,
    )

    # ---------------------------------------------------------
    # BASIC INFORMATION
    # ---------------------------------------------------------

    print("\nBehavioral profile shape:")
    print(behavior.shape)

    print("\nColumns:")
    print(behavior.columns.tolist())

    print("\nMissing values:")
    print(behavior.isna().sum())

    # ---------------------------------------------------------
    # REQUIRED COLUMNS
    # ---------------------------------------------------------

    required_columns = {
        "user_id",
        "month",
        "behavioral_regime",
        "spending_multiplier",
        "discretionary_multiplier",
        "transaction_multiplier",
    }

    missing_columns = (
        required_columns
        - set(behavior.columns)
    )

    assert not missing_columns, (
        "Missing behavioral columns: "
        f"{sorted(missing_columns)}"
    )

    # ---------------------------------------------------------
    # ROW COUNT
    # ---------------------------------------------------------

    expected_rows = (
        N_USERS * N_MONTHS
    )

    assert len(behavior) == expected_rows

    # ---------------------------------------------------------
    # USER COUNT
    # ---------------------------------------------------------

    assert (
        behavior["user_id"].nunique()
        == N_USERS
    )

    # ---------------------------------------------------------
    # MONTH COUNT
    # ---------------------------------------------------------

    assert (
        behavior["month"].nunique()
        == N_MONTHS
    )

    # ---------------------------------------------------------
    # USER-MONTH UNIQUENESS
    # ---------------------------------------------------------

    assert not (
        behavior[
            ["user_id", "month"]
        ]
        .duplicated()
        .any()
    )

    # ---------------------------------------------------------
    # MISSING VALUES
    # ---------------------------------------------------------

    assert not behavior.isna().any().any()

    # ---------------------------------------------------------
    # REGIME VALIDATION
    # ---------------------------------------------------------

    print("\nBehavioral regime distribution:")

    print(
        behavior[
            "behavioral_regime"
        ].value_counts()
    )

    assert behavior[
        "behavioral_regime"
    ].isin(REGIMES).all()

    # ---------------------------------------------------------
    # MULTIPLIER VALIDATION
    # ---------------------------------------------------------

    multiplier_columns = [
        "spending_multiplier",
        "discretionary_multiplier",
        "transaction_multiplier",
    ]

    for column in multiplier_columns:

        assert np.isfinite(
            behavior[column]
        ).all()

        assert (
            behavior[column] > 0
        ).all()

    # ---------------------------------------------------------
    # REGIME PARAMETER VALIDATION
    # ---------------------------------------------------------

    stable = behavior[
        behavior["behavioral_regime"]
        == "STABLE"
    ]

    if not stable.empty:

        assert np.allclose(
            stable["spending_multiplier"],
            1.00,
        )

        assert np.allclose(
            stable["discretionary_multiplier"],
            1.00,
        )

        assert np.allclose(
            stable["transaction_multiplier"],
            1.00,
        )

    # ---------------------------------------------------------
    # TEMPORAL ORDER
    # ---------------------------------------------------------

    sorted_behavior = behavior.sort_values(
        ["user_id", "month"],
        kind="mergesort",
    )

    for user_id, group in sorted_behavior.groupby(
        "user_id",
        sort=False,
    ):

        months = group["month"].tolist()

        expected_months = list(
            range(1, N_MONTHS + 1)
        )

        assert months == expected_months, (
            f"Invalid month sequence for "
            f"{user_id}: {months}"
        )

    # ---------------------------------------------------------
    # REPRODUCIBILITY
    # ---------------------------------------------------------

    rng2 = create_rng(SEED)

    population2 = generate_population(
        N_USERS,
        rng2,
    )

    behavior2 = generate_behavioral_profile(
        population=population2,
        n_months=N_MONTHS,
        rng=rng2,
    )

    comparison_columns = [
        "user_id",
        "month",
        "behavioral_regime",
        "spending_multiplier",
        "discretionary_multiplier",
        "transaction_multiplier",
    ]

    pd.testing.assert_frame_equal(
        behavior[
            comparison_columns
        ].reset_index(drop=True),
        behavior2[
            comparison_columns
        ].reset_index(drop=True),
    )

    print(
        "\nBehavior reproducibility test passed."
    )

    # ---------------------------------------------------------
    # DISPLAY SAMPLE
    # ---------------------------------------------------------

    print("\nBehavioral profile sample:")

    print(
        behavior.head(20).to_string(
            index=False
        )
    )

    # ---------------------------------------------------------
    # FINAL RESULT
    # ---------------------------------------------------------

    print(
        "\nBehavioral profile validation passed."
    )


if __name__ == "__main__":
    main()
