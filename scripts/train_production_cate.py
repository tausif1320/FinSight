from __future__ import annotations

from pathlib import Path
import sys

import joblib
import pandas as pd


ROOT = (
    Path(__file__)
    .resolve()
    .parents[1]
)

sys.path.insert(
    0,
    str(ROOT / "src"),
)


from finsight.experimentation.cate import (
    CATEConfig,
    DEFAULT_FEATURES,
    fit_production_cate_models,
    save_production_cate_models,
)


EXPERIMENT_PATH = (
    ROOT
    / "reports"
    / "experimentation"
    / "randomized_experiment_v3_records.csv"
)


MODEL_DIR = (
    ROOT
    / "models"
)


MODEL_PATH = (
    MODEL_DIR
    / "cate_models.joblib"
)


# These are the intervention cohorts created by
# the finalized randomized experiment v3.
VALID_INTERVENTIONS = {
    "LIQUIDITY_PROTECTION",
    "SPENDING_CONTROL",
    "OBLIGATION_MANAGEMENT",
}


def main():

    print("=" * 80)
    print("FinSight Production CATE Training")
    print("=" * 80)

    # ========================================================================
    # 1. LOAD FINALIZED V3 EXPERIMENT
    # ========================================================================

    print(
        "\nLoading finalized v3 experiment..."
    )

    experiment_df = pd.read_csv(
        EXPERIMENT_PATH
    )

    print(
        f"Rows loaded: "
        f"{len(experiment_df):,}"
    )

    print(
        "\nAvailable columns:"
    )

    print(
        list(experiment_df.columns)
    )

    # ========================================================================
    # 2. VALIDATE REQUIRED COLUMNS
    # ========================================================================

    required_columns = {
        "intervention_category",
        "treatment",
        "synthetic_financial_stress_30d",
    }

    missing = (
        required_columns
        - set(experiment_df.columns)
    )

    if missing:

        raise ValueError(
            "Finalized experiment is missing "
            f"required columns: {sorted(missing)}"
        )

    # ========================================================================
    # 3. INSPECT ORIGINAL INTERVENTION COLUMNS
    # ========================================================================

    print(
        "\nIntervention categories in source data:"
    )

    print(
        experiment_df[
            "intervention_category"
        ]
        .value_counts(
            dropna=False
        )
        .to_string()
    )

    # ========================================================================
    # 4. KEEP ONLY RANDOMIZED INTERVENTION COHORTS
    # ========================================================================
    #
    # The v3 experiment contains 1,000 validation observations,
    # but only 300 were eligible for intervention testing:
    #
    #   LIQUIDITY_PROTECTION
    #   SPENDING_CONTROL
    #   OBLIGATION_MANAGEMENT
    #
    # Non-intervention observations must NOT be used for the
    # treatment/control CATE learners.

    cate_df = experiment_df[
        experiment_df[
            "intervention_category"
        ].isin(
            VALID_INTERVENTIONS
        )
    ].copy()

    cate_df = (
        cate_df
        .reset_index(drop=True)
    )

    print(
        "\nCATE training population:"
    )

    print(
        f"Rows: "
        f"{len(cate_df):,}"
    )

    print(
        "\nIntervention cohorts:"
    )

    print(
        cate_df[
            "intervention_category"
        ]
        .value_counts()
        .sort_index()
        .to_string()
    )

    # ========================================================================
    # 5. VERIFY EXPECTED V3 SAMPLE SIZE
    # ========================================================================

    expected_rows = 300

    if len(cate_df) != expected_rows:

        raise ValueError(
            "Unexpected CATE population size. "
            f"Expected {expected_rows}, "
            f"got {len(cate_df)}."
        )

    # ========================================================================
    # 6. CHECK TREATMENT/CONTROL BALANCE
    # ========================================================================

    print(
        "\nTreatment/control distribution:"
    )

    treatment_summary = (
        cate_df
        .groupby(
            [
                "intervention_category",
                "treatment",
            ]
        )
        .size()
        .unstack(
            fill_value=0
        )
    )

    print(
        treatment_summary.to_string()
    )

    for intervention in sorted(
        VALID_INTERVENTIONS
    ):

        cohort = cate_df[
            cate_df[
                "intervention_category"
            ]
            == intervention
        ]

        treatment_count = int(
            (
                cohort[
                    "treatment"
                ] == 1
            ).sum()
        )

        control_count = int(
            (
                cohort[
                    "treatment"
                ] == 0
            ).sum()
        )

        if treatment_count < 5:

            raise ValueError(
                f"{intervention}: only "
                f"{treatment_count} treatment "
                "observations."
            )

        if control_count < 5:

            raise ValueError(
                f"{intervention}: only "
                f"{control_count} control "
                "observations."
            )

    # ========================================================================
    # 7. CHECK CATE FEATURES
    # ========================================================================

    missing_features = [
        feature
        for feature in DEFAULT_FEATURES
        if feature not in cate_df.columns
    ]

    if missing_features:

        raise ValueError(
            "Missing CATE features: "
            f"{missing_features}"
        )

    print(
        "\nCATE features:"
    )

    print(
        f"Count: "
        f"{len(DEFAULT_FEATURES)}"
    )

    for feature in DEFAULT_FEATURES:

        print(
            f"  ✓ {feature}"
        )

    # ========================================================================
    # 8. CONFIGURATION
    # ========================================================================

    config = CATEConfig()

    print(
        "\nCATE model configuration:"
    )

    print(
        f"  Model: "
        f"T-learner RandomForestRegressor"
    )

    print(
        f"  Estimators: "
        f"{config.n_estimators}"
    )

    print(
        f"  Max depth: "
        f"{config.max_depth}"
    )

    print(
        f"  Min samples leaf: "
        f"{config.min_samples_leaf}"
    )

    print(
        f"  Random state: "
        f"{config.random_state}"
    )

    # ========================================================================
    # 9. FIT PRODUCTION CATE MODELS
    # ========================================================================

    print(
        "\nTraining treatment/control models..."
    )

    models = (
        fit_production_cate_models(
            experiment_df=cate_df,
            feature_columns=DEFAULT_FEATURES,
            config=config,
        )
    )

    # ========================================================================
    # 10. VERIFY MODEL PAIRS
    # ========================================================================

    print(
        "\nTrained intervention models:"
    )

    for intervention in sorted(
        models
    ):

        model_pair = models[
            intervention
        ]

        print(
            f"\n{intervention}"
        )

        print(
            "  ✓ Control model"
        )

        print(
            "  ✓ Treatment model"
        )

        print(
            f"  Control observations: "
            f"{model_pair['n_control']}"
        )

        print(
            f"  Treatment observations: "
            f"{model_pair['n_treatment']}"
        )

    # ========================================================================
    # 11. SAVE ARTIFACT
    # ========================================================================

    print(
        "\nSaving production CATE artifact..."
    )

    save_production_cate_models(
        models=models,
        output_path=MODEL_PATH,
        feature_columns=DEFAULT_FEATURES,
        config=config,
    )

    # ========================================================================
    # 12. RELOAD ARTIFACT
    # ========================================================================

    print(
        "\nReloading saved artifact..."
    )

    artifact = joblib.load(
        MODEL_PATH
    )

    # ========================================================================
    # 13. ARTIFACT VALIDATION
    # ========================================================================

    assert (
        artifact[
            "artifact_type"
        ]
        == "finsight_cate_t_learner"
    )

    assert (
        "models" in artifact
    )

    assert (
        "feature_columns" in artifact
    )

    assert (
        artifact[
            "feature_columns"
        ]
        == DEFAULT_FEATURES
    )

    assert (
        set(
            artifact[
                "models"
            ].keys()
        )
        == VALID_INTERVENTIONS
    )

    for intervention in (
        VALID_INTERVENTIONS
    ):

        pair = artifact[
            "models"
        ][
            intervention
        ]

        assert (
            "control_model"
            in pair
        )

        assert (
            "treatment_model"
            in pair
        )

    # ========================================================================
    # 14. FINAL VALIDATION OUTPUT
    # ========================================================================

    print(
        "\n" + "=" * 80
    )

    print(
        "CATE ARTIFACT VALIDATION"
    )

    print(
        "=" * 80
    )

    print(
        "✓ Correct intervention column used"
    )

    print(
        "✓ Non-intervention observations excluded"
    )

    print(
        "✓ Exactly 300 randomized observations used"
    )

    print(
        "✓ All three intervention cohorts present"
    )

    print(
        "✓ Treatment and control arms present"
    )

    print(
        "✓ All CATE features available"
    )

    print(
        "✓ Treatment models trained"
    )

    print(
        "✓ Control models trained"
    )

    print(
        "✓ Artifact saved successfully"
    )

    print(
        "✓ Artifact successfully reloaded"
    )

    print(
        "✓ Artifact metadata validated"
    )

    print(
        f"\nSaved:"
    )

    print(
        f"  {MODEL_PATH}"
    )

    print(
        "\n" + "=" * 80
    )

    print(
        "PRODUCTION CATE TRAINING PASSED"
    )

    print(
        "=" * 80
    )


if __name__ == "__main__":
    main()