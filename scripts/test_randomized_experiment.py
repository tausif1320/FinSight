from __future__ import annotations

from pathlib import Path

import pandas as pd

from finsight.models.dataset import (
    TARGET_COLUMN,
    chronological_split,
    get_feature_columns,
)
from finsight.models.tree_models import build_hist_gradient_boosting
from finsight.experimentation.randomized_experiment import (
    BALANCE_FEATURES,
    calculate_cohort_summary,
    calculate_continuous_effects,
    calculate_intervention_effects,
    calculate_ate,
    assign_policy_specific_cohorts,
    check_randomization_balance,
    simulate_treatment_response,
)


DATA_PATH = Path("data/processed/modeling_dataset_1000_users.csv")
OUTPUT_DIR = Path("reports/experimentation")
RANDOM_STATE = 42
MIN_COHORT_SIZE = 20
ELIGIBILITY_PERCENTILE = 0.70
RERANDOMIZATION_CANDIDATES = 3000


def print_section(title: str) -> None:
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def main() -> None:
    print_section("FinSight Policy-Specific Randomized Experiment v3")
    print("Final Cohort Design + Rerandomized Synthetic Experiment")

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Modeling dataset not found: {DATA_PATH}"
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print_section("LOADING MODELING DATASET")
    modeling_df = pd.read_csv(DATA_PATH)
    print(f"Rows: {len(modeling_df):,}")
    print(f"Users: {modeling_df['user_id'].nunique():,}")
    print(f"Stress cases: {modeling_df[TARGET_COLUMN].sum():,}")

    print_section("TEMPORAL DATA SPLIT")
    train_df, validation_df, test_df = chronological_split(modeling_df)
    print(f"Train: {len(train_df):,}")
    print(f"Validation: {len(validation_df):,}")
    print(f"Test: {len(test_df):,}")

    feature_columns = get_feature_columns(modeling_df)

    X_train = train_df[feature_columns]
    y_train = train_df[TARGET_COLUMN].astype(int)
    X_validation = validation_df[feature_columns]

    print("\nTraining HistGradientBoosting on TRAIN only...")
    model = build_hist_gradient_boosting()
    model.fit(X_train, y_train)

    validation_probability = model.predict_proba(X_validation)[:, 1]

    experiment_population = validation_df.copy().reset_index(drop=True)
    experiment_population["risk_probability"] = validation_probability

    print_section("EXPERIMENT POPULATION")
    print(f"Rows: {len(experiment_population):,}")
    print("The validation period is used as the synthetic experiment population.")
    print("The test period remains untouched for final predictive evaluation.")

    print_section("POLICY ELIGIBILITY + STRATIFIED RANDOMIZATION")
    experiment_population = assign_policy_specific_cohorts(
        experiment_population,
        random_state=RANDOM_STATE,
        treatment_probability=0.50,
        eligibility_percentile=ELIGIBILITY_PERCENTILE,
        min_cohort_size=MIN_COHORT_SIZE,
        rerandomization_candidates=RERANDOMIZATION_CANDIDATES,
    )

    eligible = experiment_population[
        experiment_population["eligible"]
    ].copy()

    print(f"Total observations: {len(experiment_population):,}")
    print(f"Eligible observations: {len(eligible):,}")
    print(
        f"Eligibility rate: "
        f"{len(eligible) / len(experiment_population):.4f}"
    )

    cohort_summary = calculate_cohort_summary(experiment_population)
    print_section("INTERVENTION COHORTS")
    print(cohort_summary.to_string(index=False))

    invalid_cohorts = cohort_summary[
        (cohort_summary["total_n"] > 0)
        & (
            (cohort_summary["treatment_n"] == 0)
            | (cohort_summary["control_n"] == 0)
        )
    ]

    if not invalid_cohorts.empty:
        raise AssertionError(
            "Every non-empty intervention cohort must contain both "
            "treatment and control observations.\n"
            f"Invalid cohorts:\n{invalid_cohorts}"
        )

    small_cohorts = cohort_summary[
        (cohort_summary["total_n"] > 0)
        & (cohort_summary["total_n"] < MIN_COHORT_SIZE)
    ]

    if not small_cohorts.empty:
        print(
            "\nWARNING: The following cohorts are smaller than "
            f"{MIN_COHORT_SIZE} observations:"
        )
        print(small_cohorts.to_string(index=False))

    print_section("RANDOMIZATION BALANCE")
    balance_frames = []

    for intervention in cohort_summary.loc[
        cohort_summary["total_n"] > 0,
        "intervention_category",
    ]:
        balance = check_randomization_balance(
            experiment_population,
            intervention_category=intervention,
            features=BALANCE_FEATURES,
            max_smd=0.20,
        )
        balance_frames.append(balance)

        print(f"\n{intervention}")
        print(balance.to_string(index=False))

    balance_df = (
        pd.concat(balance_frames, ignore_index=True)
        if balance_frames
        else pd.DataFrame()
    )

    print_section("BALANCE DIAGNOSTIC")
    if balance_df.empty:
        raise AssertionError("No randomization balance results were generated.")

    max_smd = balance_df["smd"].replace([float("inf")], pd.NA).max()
    imbalanced = balance_df[~balance_df["balanced"]].copy()

    print(f"Maximum finite SMD: {max_smd:.4f}")
    print(f"Features above SMD 0.20: {len(imbalanced)}")

    if not imbalanced.empty:
        print("\nBaseline imbalance remains in some features.")
        print("This is reported, not silently hidden or overwritten.")

    print_section("SIMULATING SYNTHETIC TREATMENT RESPONSE")
    experiment_records = simulate_treatment_response(
        experiment_population,
        random_state=RANDOM_STATE,
    )

    print("Synthetic outcomes generated for eligible observations.")

    print_section("OVERALL SYNTHETIC TREATMENT EFFECT")
    overall_effect = calculate_ate(
        experiment_records[experiment_records["eligible"]].copy(),
        outcome_column="synthetic_financial_stress_30d",
        confidence=0.95,
    )
    print(pd.Series(overall_effect).to_string())

    print_section("POLICY-SPECIFIC SYNTHETIC EFFECTS")
    intervention_effects = calculate_intervention_effects(
        experiment_records,
        outcome_column="synthetic_financial_stress_30d",
        confidence=0.95,
    )

    if intervention_effects.empty:
        raise AssertionError("No policy-specific treatment effects were generated.")

    print(intervention_effects.to_string(index=False))

    print_section("CONTINUOUS SYNTHETIC OUTCOMES")
    continuous_outcomes = [
        "synthetic_cash_buffer_change",
        "synthetic_savings_rate_change",
        "synthetic_discretionary_spending_change",
    ]

    continuous_effects = calculate_continuous_effects(
        experiment_records,
        outcome_columns=continuous_outcomes,
        confidence=0.95,
    )

    if continuous_effects.empty:
        raise AssertionError("No continuous treatment effects were generated.")

    print(continuous_effects.to_string(index=False))

    print_section("EXPERIMENT VALIDATION")

    if len(experiment_records) != len(validation_df):
        raise AssertionError("Experiment population size changed unexpectedly.")

    if experiment_records["treatment"].sum() + experiment_records["control"].sum() != len(eligible):
        raise AssertionError("Treatment/control assignment does not cover eligible observations.")

    if experiment_records.loc[
        experiment_records["eligible"], "treatment"
    ].isna().any():
        raise AssertionError("Treatment assignment contains NaN values.")

    # Users correctly appear in both validation and test because the unit of
    # observation is user-month. Leakage is checked at the temporal split level.
    validation_months = set(validation_df["month"].astype(str))
    test_months = set(test_df["month"].astype(str))
    month_overlap = validation_months.intersection(test_months)

    if month_overlap:
        raise AssertionError(
            "Test leakage detected: validation and test months overlap: "
            f"{sorted(month_overlap)}"
        )

    print("✓ Temporal separation preserved")
    print("✓ Test set remains untouched")
    print("✓ Every non-empty intervention cohort has treatment and control")
    print("✓ Stratified randomization completed")
    print("✓ Standardized mean differences calculated")
    print("✓ Synthetic binary treatment outcome generated")
    print("✓ Synthetic continuous outcomes generated")
    print("✓ Overall treatment effect calculated")
    print("✓ Policy-specific treatment effects calculated")

    experiment_records.to_csv(
        OUTPUT_DIR / "randomized_experiment_v3_records.csv",
        index=False,
    )
    cohort_summary.to_csv(
        OUTPUT_DIR / "intervention_cohort_summary_v3.csv",
        index=False,
    )
    balance_df.to_csv(
        OUTPUT_DIR / "randomization_balance_v3.csv",
        index=False,
    )
    intervention_effects.to_csv(
        OUTPUT_DIR / "intervention_effects_v3.csv",
        index=False,
    )
    continuous_effects.to_csv(
        OUTPUT_DIR / "continuous_intervention_effects_v3.csv",
        index=False,
    )
    pd.DataFrame([overall_effect]).to_csv(
        OUTPUT_DIR / "overall_treatment_effect_v3.csv",
        index=False,
    )

    print_section("FILES SAVED")
    for path in [
        "randomized_experiment_v3_records.csv",
        "intervention_cohort_summary_v3.csv",
        "randomization_balance_v3.csv",
        "intervention_effects_v3.csv",
        "continuous_intervention_effects_v3.csv",
        "overall_treatment_effect_v3.csv",
    ]:
        print(f"  {OUTPUT_DIR / path}")

    print_section("IMPORTANT METHODOLOGICAL NOTE")
    print(
        "All treatment outcomes in this experiment are synthetic simulation "
        "results. They are used to validate FinSight's experimentation and "
        "effect-estimation pipeline and do not establish real-world causal "
        "effectiveness."
    )

    print_section("RANDOMIZED EXPERIMENT V3 PASSED")


if __name__ == "__main__":
    main()
