from __future__ import annotations

import gc
from pathlib import Path

import numpy as np
import pandas as pd

from finsight.data_generation.population import generate_population
from finsight.data_generation.income import (
    generate_base_income,
    generate_salary_days,
    generate_monthly_income,
)
from finsight.data_generation.obligations import generate_obligations
from finsight.data_generation.spending import generate_spending_profile
from finsight.data_generation.balance import generate_initial_balances
from finsight.data_generation.behavior import generate_behavioral_profile
from finsight.data_generation.shocks import generate_shocks
from finsight.data_generation.transactions import generate_transactions

from finsight.features.monthly_profile import build_monthly_profile
from finsight.features.financial_state import assign_financial_states
from finsight.features.behavioral_features import build_behavioral_features
from finsight.features.outcomes import build_future_outcomes

from finsight.models.dataset import build_modeling_dataset


# ============================================================
# CONFIGURATION
# ============================================================

SEED = 42

TOTAL_USERS = 100
N_MONTHS = 12
HORIZON_DAYS = 30

BATCH_SIZE = 100

TARGET = "financial_stress_30d"

OUTPUT_DIR = Path("data/processed")
OUTPUT_FILE = OUTPUT_DIR / "modeling_dataset_1000_users.csv"


# ============================================================
# BUILD ONE BATCH
# ============================================================

def build_batch(
    user_ids: list[int],
    seed: int,
) -> pd.DataFrame:

    print()
    print("-" * 70)
    print(
        f"Building batch: "
        f"users {user_ids[0]} -> {user_ids[-1]}"
    )
    print("-" * 70)

    rng = np.random.default_rng(seed)

    n_users = len(user_ids)

    # --------------------------------------------------------
    # POPULATION
    # --------------------------------------------------------

    population = generate_population(
        n_users,
        rng,
    )

    # Preserve globally unique user IDs.
    population["user_id"] = user_ids

    # --------------------------------------------------------
    # INCOME
    # --------------------------------------------------------

    population = generate_base_income(
        population,
        rng,
    )

    population = generate_salary_days(
        population,
        rng,
    )

    income_df = generate_monthly_income(
        population,
        N_MONTHS,
        rng,
    )

    # --------------------------------------------------------
    # OBLIGATIONS
    # --------------------------------------------------------

    obligations_df = generate_obligations(
        population,
        rng,
    )

    # --------------------------------------------------------
    # SPENDING
    # --------------------------------------------------------

    spending_df = generate_spending_profile(
        population,
        rng,
    )

    # --------------------------------------------------------
    # INITIAL BALANCES
    # --------------------------------------------------------

    initial_balance_df = generate_initial_balances(
        population,
        income_df,
        rng,
    )

    # --------------------------------------------------------
    # BEHAVIOR
    # --------------------------------------------------------

    behavior_df = generate_behavioral_profile(
        population,
        N_MONTHS,
        rng,
    )

    # --------------------------------------------------------
    # SHOCKS
    # --------------------------------------------------------

    shock_df = generate_shocks(
        population,
        N_MONTHS,
        rng,
    )

    # --------------------------------------------------------
    # TRANSACTIONS
    # --------------------------------------------------------

    transactions_df = generate_transactions(
        population=population,
        income_df=income_df,
        obligations_df=obligations_df,
        spending_df=spending_df,
        initial_balance_df=initial_balance_df,
        rng=rng,
        n_months=N_MONTHS,
        behavior_df=behavior_df,
        shock_df=shock_df,
    )

    print(
        f"Transactions generated: "
        f"{len(transactions_df):,}"
    )

    # --------------------------------------------------------
    # MONTHLY PROFILE
    # --------------------------------------------------------

    monthly_df = build_monthly_profile(
        transactions_df
    )

    # --------------------------------------------------------
    # FINANCIAL STATES
    # --------------------------------------------------------

    state_df = assign_financial_states(
        monthly_df
    )

    # --------------------------------------------------------
    # BEHAVIORAL FEATURES
    # --------------------------------------------------------

    feature_df = build_behavioral_features(
        monthly_profile=monthly_df,
        state_df=state_df,
        transactions_df=transactions_df,
        window_months=3,
    )

    # --------------------------------------------------------
    # FUTURE OUTCOMES
    # --------------------------------------------------------

    outcome_df = build_future_outcomes(
        transactions=transactions_df,
        monthly_profile=monthly_df,
        horizon_days=HORIZON_DAYS,
    )

    # --------------------------------------------------------
    # MODELING DATASET
    # --------------------------------------------------------

    model_df = build_modeling_dataset(
        feature_df,
        outcome_df,
        drop_incomplete_history=True,
    )

    print(
        f"Modeling rows: "
        f"{len(model_df):,}"
    )

    print(
        f"Stress cases: "
        f"{int(model_df[TARGET].sum()):,}"
    )

    # --------------------------------------------------------
    # MEMORY CLEANUP
    # --------------------------------------------------------

    del population
    del income_df
    del obligations_df
    del spending_df
    del initial_balance_df
    del behavior_df
    del shock_df
    del transactions_df
    del monthly_df
    del state_df
    del feature_df
    del outcome_df

    gc.collect()

    return model_df


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("FinSight Batched Dataset Builder")
    print("=" * 70)

    print(f"Total users: {TOTAL_USERS:,}")
    print(f"Users per batch: {BATCH_SIZE:,}")
    print(f"Months: {N_MONTHS}")
    print(f"Prediction horizon: {HORIZON_DAYS} days")
    print()

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    batches = []

    n_batches = (
        TOTAL_USERS + BATCH_SIZE - 1
    ) // BATCH_SIZE

    # --------------------------------------------------------
    # GENERATE BATCHES
    # --------------------------------------------------------

    for batch_number in range(n_batches):

        start_user = (
            batch_number * BATCH_SIZE
        )

        end_user = min(
            start_user + BATCH_SIZE,
            TOTAL_USERS,
        )

        user_ids = list(
            range(
                start_user,
                end_user,
            )
        )

        batch_seed = SEED + batch_number

        batch_df = build_batch(
            user_ids=user_ids,
            seed=batch_seed,
        )

        batches.append(batch_df)

        print(
            f"Completed batch "
            f"{batch_number + 1}/{n_batches}"
        )

        print(
            f"Accumulated modeling rows: "
            f"{sum(len(x) for x in batches):,}"
        )

        print(
            f"Accumulated stress cases: "
            f"{sum(int(x[TARGET].sum()) for x in batches):,}"
        )

    # --------------------------------------------------------
    # COMBINE MODELING DATA
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("COMBINING BATCHES")
    print("=" * 70)

    modeling_df = pd.concat(
        batches,
        ignore_index=True,
    )

    del batches
    gc.collect()

    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("FINAL DATASET VALIDATION")
    print("=" * 70)

    expected_users = TOTAL_USERS

    actual_users = modeling_df[
        "user_id"
    ].nunique()

    print(
        f"Unique users: "
        f"{actual_users:,}"
    )

    assert actual_users == expected_users, (
        f"Expected {expected_users} users, "
        f"got {actual_users}"
    )

    duplicate_keys = modeling_df.duplicated(
        subset=["user_id", "month"]
    ).sum()

    print(
        f"Duplicate user-month rows: "
        f"{duplicate_keys}"
    )

    assert duplicate_keys == 0

    stress_cases = int(
        modeling_df[TARGET].sum()
    )

    total_rows = len(modeling_df)

    stress_rate = (
        stress_cases / total_rows
        if total_rows > 0
        else 0
    )

    print(
        f"Modeling rows: "
        f"{total_rows:,}"
    )

    print(
        f"Stress cases: "
        f"{stress_cases:,}"
    )

    print(
        f"Stress rate: "
        f"{stress_rate:.4f}"
    )

    # --------------------------------------------------------
    # SORT
    # --------------------------------------------------------

    modeling_df = modeling_df.sort_values(
        ["month", "user_id"]
    ).reset_index(drop=True)

    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    modeling_df.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print()
    print(
        f"Dataset saved to:"
    )
    print(
        f"  {OUTPUT_FILE}"
    )

    print()
    print("=" * 70)
    print("BATCHED DATASET BUILD PASSED")
    print("=" * 70)


if __name__ == "__main__":
    main()