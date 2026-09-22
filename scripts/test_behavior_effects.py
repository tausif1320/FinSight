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
from finsight.data_generation.transactions import generate_transactions
from finsight.features.monthly_profile import build_monthly_profile


SEED = 42
N_USERS = 100
N_MONTHS = 12


def build_common_inputs():
    """
    Generate all inputs that should remain identical across
    behavioral scenarios.
    """
    rng = np.random.default_rng(SEED)

    population = generate_population(
        n_users=N_USERS,
        rng=rng,
    )

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
        n_months=N_MONTHS,
        rng=rng,
    )

    obligations_df = generate_obligations(
        population,
        rng,
    )

    spending_df = generate_spending_profile(
        population,
        rng,
    )

    initial_balances_df = generate_initial_balances(
        population,
        income_df,
        rng,
    )

    behavior_df = generate_behavioral_profile(
        population,
        n_months=N_MONTHS,
        rng=rng,
    )

    return (
        population,
        income_df,
        obligations_df,
        spending_df,
        initial_balances_df,
        behavior_df,
    )


def create_controlled_behavior(
    behavior_df: pd.DataFrame,
    regime: str,
) -> pd.DataFrame:

    result = behavior_df.copy()

    regime_parameters = {
        "STABLE": {
            "spending_multiplier": 1.00,
            "discretionary_multiplier": 1.00,
            "transaction_multiplier": 1.00,
        },
        "SPENDING_EXPANSION": {
            "spending_multiplier": 1.15,
            "discretionary_multiplier": 1.35,
            "transaction_multiplier": 1.15,
        },
        "RECOVERY": {
            "spending_multiplier": 0.90,
            "discretionary_multiplier": 0.75,
            "transaction_multiplier": 0.90,
        },
    }

    params = regime_parameters[regime]

    result["regime"] = regime
    result["spending_multiplier"] = params["spending_multiplier"]
    result["discretionary_multiplier"] = params[
        "discretionary_multiplier"
    ]
    result["transaction_multiplier"] = params[
        "transaction_multiplier"
    ]

    return result


def generate_scenario(
    common_inputs,
    regime: str,
):
    (
        population,
        income_df,
        obligations_df,
        spending_df,
        initial_balances_df,
        behavior_df,
    ) = common_inputs

    controlled_behavior = create_controlled_behavior(
        behavior_df,
        regime,
    )

    # Empty shock dataframe.
    shock_df = pd.DataFrame(
        columns=[
            "user_id",
            "month",
            "shock_type",
            "shock_amount",
            "income_impact",
            "duration_months",
            "severity",
        ]
    )

    transactions = generate_transactions(
        population=population,
        income_df=income_df,
        obligations_df=obligations_df,
        spending_df=spending_df,
        initial_balance_df=initial_balances_df,
        n_months=N_MONTHS,
        behavior_df=controlled_behavior,
        shock_df=shock_df,
        rng=np.random.default_rng(SEED + 1),
    )

    monthly_profile = build_monthly_profile(
        transactions
    )

    return transactions, monthly_profile


def main():

    common_inputs = build_common_inputs()

    scenarios = {}

    for regime in [
        "STABLE",
        "SPENDING_EXPANSION",
        "RECOVERY",
    ]:
        transactions, profile = generate_scenario(
            common_inputs,
            regime,
        )

        scenarios[regime] = {
            "transactions": transactions,
            "profile": profile,
        }

    summary_records = []

    for regime, data in scenarios.items():

        profile = data["profile"]

        summary_records.append(
            {
                "regime": regime,
                "mean_variable_spending":
                    profile["variable_spending"].mean(),
                "mean_discretionary_spending":
                    profile["discretionary_spending"].mean(),
                "mean_transaction_count":
                    profile["transaction_count"].mean(),
                "mean_discretionary_ratio":
                    profile[
                        "discretionary_spending_ratio"
                    ].mean(),
                "mean_savings_rate":
                    profile["savings_rate"].mean(),
            }
        )

    summary = pd.DataFrame(summary_records)

    print("\nBehavioral effect summary:")
    print(
        summary.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    stable = summary[
        summary["regime"] == "STABLE"
    ].iloc[0]

    expansion = summary[
        summary["regime"] == "SPENDING_EXPANSION"
    ].iloc[0]

    recovery = summary[
        summary["regime"] == "RECOVERY"
    ].iloc[0]

    print("\nChecking behavioral directions...")

    # Spending expansion should increase spending.
    assert (
        expansion["mean_variable_spending"]
        > stable["mean_variable_spending"]
    ), (
        "SPENDING_EXPANSION did not increase "
        "variable spending."
    )

    assert (
        expansion["mean_discretionary_spending"]
        > stable["mean_discretionary_spending"]
    ), (
        "SPENDING_EXPANSION did not increase "
        "discretionary spending."
    )

    assert (
        expansion["mean_transaction_count"]
        > stable["mean_transaction_count"]
    ), (
        "SPENDING_EXPANSION did not increase "
        "transaction count."
    )

    # Recovery should reduce spending.
    assert (
        recovery["mean_variable_spending"]
        < stable["mean_variable_spending"]
    ), (
        "RECOVERY did not reduce "
        "variable spending."
    )

    assert (
        recovery["mean_discretionary_spending"]
        < stable["mean_discretionary_spending"]
    ), (
        "RECOVERY did not reduce "
        "discretionary spending."
    )

    assert (
        recovery["mean_transaction_count"]
        < stable["mean_transaction_count"]
    ), (
        "RECOVERY did not reduce "
        "transaction count."
    )

    # Higher spending should generally reduce savings.
    assert (
        expansion["mean_savings_rate"]
        < stable["mean_savings_rate"]
    ), (
        "SPENDING_EXPANSION did not reduce "
        "savings rate."
    )

    assert (
        recovery["mean_savings_rate"]
        > expansion["mean_savings_rate"]
    ), (
        "RECOVERY did not improve savings rate "
        "relative to SPENDING_EXPANSION."
    )

    print("Behavioral direction checks passed.")

    # Print relative changes for interpretation.
    print("\nRelative changes vs STABLE:")

    for regime in [
        "SPENDING_EXPANSION",
        "RECOVERY",
    ]:

        row = summary[
            summary["regime"] == regime
        ].iloc[0]

        variable_change = (
            row["mean_variable_spending"]
            / stable["mean_variable_spending"]
            - 1
        )

        discretionary_change = (
            row["mean_discretionary_spending"]
            / stable["mean_discretionary_spending"]
            - 1
        )

        transaction_change = (
            row["mean_transaction_count"]
            / stable["mean_transaction_count"]
            - 1
        )

        savings_change = (
            row["mean_savings_rate"]
            - stable["mean_savings_rate"]
        )

        print(f"\n{regime}")
        print(
            f"Variable spending: "
            f"{variable_change:+.2%}"
        )
        print(
            f"Discretionary spending: "
            f"{discretionary_change:+.2%}"
        )
        print(
            f"Transaction count: "
            f"{transaction_change:+.2%}"
        )
        print(
            f"Savings rate change: "
            f"{savings_change:+.4f}"
        )

    print("\nBehavior effect validation passed.")


if __name__ == "__main__":
    main()