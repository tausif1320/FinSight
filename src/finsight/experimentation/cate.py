from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline


RANDOM_STATE = 42


# ============================================================================
# DEFAULT CATE FEATURES
# ============================================================================

DEFAULT_FEATURES = [
    "cash_buffer_ratio",
    "expense_pressure",
    "spending_growth",
    "spending_baseline_deviation",
    "discretionary_pressure",
    "recurring_spending",
    "balance_growth",
    "savings_rate",
    "rolling_average_balance",
    "rolling_net_cash_flow",
    "transaction_count_change",
    "income_volatility",
]


# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass(frozen=True)
class CATEConfig:

    outcome_column: str = (
        "synthetic_financial_stress_30d"
    )

    treatment_column: str = (
        "treatment"
    )

    intervention_column: str = (
        "intervention_category"
    )

    control_value: int = 0

    treatment_value: int = 1

    n_estimators: int = 300

    min_samples_leaf: int = 10

    max_depth: int = 6

    n_splits: int = 5

    random_state: int = RANDOM_STATE


# ============================================================================
# T-LEARNER MODEL BUILDER
# ============================================================================

def build_t_learner(
    config: CATEConfig | None = None,
) -> tuple[Pipeline, Pipeline]:
    """
    Build separate control and treatment outcome models.

    Returns:
        control_model, treatment_model

    CATE is estimated as:

        E[Y | X, T=1] - E[Y | X, T=0]

    Therefore:

        negative CATE -> treatment predicts lower stress
        positive CATE -> treatment predicts higher stress
    """

    config = config or CATEConfig()

    def build_model() -> Pipeline:

        return Pipeline(
            [
                (
                    "imputer",
                    SimpleImputer(
                        strategy="median"
                    ),
                ),
                (
                    "model",
                    RandomForestRegressor(
                        n_estimators=(
                            config.n_estimators
                        ),
                        max_depth=(
                            config.max_depth
                        ),
                        min_samples_leaf=(
                            config.min_samples_leaf
                        ),
                        random_state=(
                            config.random_state
                        ),
                        n_jobs=-1,
                    ),
                ),
            ]
        )

    return (
        build_model(),
        build_model(),
    )


# ============================================================================
# INPUT VALIDATION
# ============================================================================

def validate_cate_input(
    df: pd.DataFrame,
    config: CATEConfig,
    feature_columns: Iterable[str],
) -> list[str]:
    """
    Validate the structure of a CATE experiment dataset.
    """

    features = list(
        feature_columns
    )

    required = (
        features
        + [
            config.outcome_column,
            config.treatment_column,
            config.intervention_column,
        ]
    )

    missing = [
        column
        for column in required
        if column not in df.columns
    ]

    if missing:

        raise ValueError(
            f"Missing CATE columns: {missing}"
        )

    if df.empty:

        raise ValueError(
            "CATE input is empty."
        )

    treatment_values = set(
        df[
            config.treatment_column
        ]
        .dropna()
        .unique()
    )

    if not treatment_values.issubset(
        {0, 1}
    ):

        raise ValueError(
            "Treatment must contain only 0 and 1."
        )

    return features


# ============================================================================
# STANDARD IN-SAMPLE CATE
# ============================================================================

def estimate_cate(
    df: pd.DataFrame,
    feature_columns: Iterable[str] = DEFAULT_FEATURES,
    config: CATEConfig | None = None,
) -> pd.DataFrame:
    """
    Standard in-sample T-learner.

    Models are trained on the complete intervention cohort and
    predictions are generated on the same observations.

    This is useful for exploratory analysis but can be optimistic.
    """

    config = config or CATEConfig()

    features = validate_cate_input(
        df,
        config,
        feature_columns,
    )

    results = []

    for intervention, cohort in df.groupby(
        config.intervention_column,
        sort=True,
    ):

        cohort = (
            cohort
            .reset_index(drop=True)
            .copy()
        )

        treatment = cohort[
            config.treatment_column
        ].astype(int)

        outcome = cohort[
            config.outcome_column
        ].astype(float)

        X = cohort[
            features
        ]

        treatment_mask = (
            treatment
            == config.treatment_value
        )

        control_mask = (
            treatment
            == config.control_value
        )

        if treatment_mask.sum() < 5:

            raise ValueError(
                f"{intervention} has too few "
                "treatment observations."
            )

        if control_mask.sum() < 5:

            raise ValueError(
                f"{intervention} has too few "
                "control observations."
            )

        control_model, treatment_model = (
            build_t_learner(config)
        )

        control_model.fit(
            X.loc[control_mask],
            outcome.loc[control_mask],
        )

        treatment_model.fit(
            X.loc[treatment_mask],
            outcome.loc[treatment_mask],
        )

        mu_control = (
            control_model.predict(X)
        )

        mu_treatment = (
            treatment_model.predict(X)
        )

        cate = (
            mu_treatment
            - mu_control
        )

        cohort[
            "predicted_outcome_control"
        ] = mu_control

        cohort[
            "predicted_outcome_treatment"
        ] = mu_treatment

        cohort[
            "cate"
        ] = cate

        cohort[
            "absolute_effect"
        ] = np.abs(cate)

        cohort[
            "cate_estimation_type"
        ] = "in_sample"

        results.append(
            cohort
        )

    return pd.concat(
        results,
        ignore_index=True,
    )


# ============================================================================
# CROSS-FITTED CATE
# ============================================================================

def estimate_cross_fitted_cate(
    df: pd.DataFrame,
    feature_columns: Iterable[str] = DEFAULT_FEATURES,
    config: CATEConfig | None = None,
) -> pd.DataFrame:
    """
    Cross-fitted T-learner.

    Each observation receives predictions from models that were
    NOT trained on that observation.

    This reduces optimistic in-sample CATE bias.
    """

    config = config or CATEConfig()

    features = validate_cate_input(
        df,
        config,
        feature_columns,
    )

    results = []

    for intervention, cohort in df.groupby(
        config.intervention_column,
        sort=True,
    ):

        cohort = (
            cohort
            .reset_index(drop=True)
            .copy()
        )

        treatment = cohort[
            config.treatment_column
        ].astype(int)

        outcome = cohort[
            config.outcome_column
        ].astype(float)

        X = cohort[
            features
        ]

        treatment_count = int(
            (
                treatment
                == config.treatment_value
            ).sum()
        )

        control_count = int(
            (
                treatment
                == config.control_value
            ).sum()
        )

        if treatment_count < 5:

            raise ValueError(
                f"{intervention}: insufficient "
                "treatment observations."
            )

        if control_count < 5:

            raise ValueError(
                f"{intervention}: insufficient "
                "control observations."
            )

        max_splits = min(
            config.n_splits,
            treatment_count,
            control_count,
        )

        if max_splits < 2:

            raise ValueError(
                f"{intervention}: unable to create "
                "cross-fitting folds."
            )

        splitter = StratifiedKFold(
            n_splits=max_splits,
            shuffle=True,
            random_state=config.random_state,
        )

        mu_control = np.full(
            len(cohort),
            np.nan,
            dtype=float,
        )

        mu_treatment = np.full(
            len(cohort),
            np.nan,
            dtype=float,
        )

        for train_idx, test_idx in splitter.split(
            X,
            treatment,
        ):

            train_treatment = (
                treatment.iloc[train_idx]
                == config.treatment_value
            )

            train_control = (
                treatment.iloc[train_idx]
                == config.control_value
            )

            if (
                train_treatment.sum() < 3
                or train_control.sum() < 3
            ):

                raise ValueError(
                    f"{intervention}: a cross-fitting "
                    "fold lacks both treatment arms."
                )

            control_model, treatment_model = (
                build_t_learner(config)
            )

            X_train = X.iloc[
                train_idx
            ]

            y_train = outcome.iloc[
                train_idx
            ]

            control_model.fit(
                X_train.loc[
                    train_control
                ],
                y_train.loc[
                    train_control
                ],
            )

            treatment_model.fit(
                X_train.loc[
                    train_treatment
                ],
                y_train.loc[
                    train_treatment
                ],
            )

            X_test = X.iloc[
                test_idx
            ]

            mu_control[
                test_idx
            ] = control_model.predict(
                X_test
            )

            mu_treatment[
                test_idx
            ] = treatment_model.predict(
                X_test
            )

        if np.isnan(
            mu_control
        ).any():

            raise ValueError(
                f"{intervention}: missing "
                "cross-fitted control predictions."
            )

        if np.isnan(
            mu_treatment
        ).any():

            raise ValueError(
                f"{intervention}: missing "
                "cross-fitted treatment predictions."
            )

        cate = (
            mu_treatment
            - mu_control
        )

        cohort[
            "predicted_outcome_control"
        ] = mu_control

        cohort[
            "predicted_outcome_treatment"
        ] = mu_treatment

        cohort[
            "cate"
        ] = cate

        cohort[
            "absolute_effect"
        ] = np.abs(cate)

        cohort[
            "cate_estimation_type"
        ] = "cross_fitted"

        results.append(
            cohort
        )

    return pd.concat(
        results,
        ignore_index=True,
    )


# ============================================================================
# PRODUCTION CATE MODEL TRAINING
# ============================================================================

def fit_production_cate_models(
    experiment_df: pd.DataFrame,
    feature_columns: Iterable[str] = DEFAULT_FEATURES,
    config: CATEConfig | None = None,
) -> dict:
    """
    Train final production T-learner models.

    One treatment/control model pair is trained for each
    intervention_category.

    These models are intended for inference on future observations.

    Important:
        This is a synthetic-experiment model because the training
        outcomes originate from the FinSight simulator.
    """

    config = config or CATEConfig()

    features = validate_cate_input(
        experiment_df,
        config,
        feature_columns,
    )

    models = {}

    for intervention, cohort in (
        experiment_df.groupby(
            config.intervention_column,
            sort=True,
        )
    ):

        cohort = cohort.copy()

        treatment_mask = (
            cohort[
                config.treatment_column
            ]
            == config.treatment_value
        )

        control_mask = (
            cohort[
                config.treatment_column
            ]
            == config.control_value
        )

        treatment_count = int(
            treatment_mask.sum()
        )

        control_count = int(
            control_mask.sum()
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

        X = cohort[
            features
        ]

        y = cohort[
            config.outcome_column
        ].astype(float)

        control_model, treatment_model = (
            build_t_learner(config)
        )

        control_model.fit(
            X.loc[control_mask],
            y.loc[control_mask],
        )

        treatment_model.fit(
            X.loc[treatment_mask],
            y.loc[treatment_mask],
        )

        models[
            intervention
        ] = {
            "control_model":
                control_model,

            "treatment_model":
                treatment_model,

            "n_treatment":
                treatment_count,

            "n_control":
                control_count,
        }

    return models


# ============================================================================
# PRODUCTION CATE PREDICTION
# ============================================================================

def predict_production_cate(
    df: pd.DataFrame,
    intervention: str,
    models: dict,
    feature_columns: Iterable[str] = DEFAULT_FEATURES,
) -> np.ndarray:
    """
    Generate CATE estimates for new observations.

    CATE:
        predicted treatment outcome
        -
        predicted control outcome

    Negative CATE means lower predicted synthetic stress
    under treatment.
    """

    features = list(
        feature_columns
    )

    missing = [
        column
        for column in features
        if column not in df.columns
    ]

    if missing:

        raise ValueError(
            f"Missing CATE features: {missing}"
        )

    if intervention not in models:

        raise ValueError(
            f"No CATE model exists for "
            f"intervention '{intervention}'."
        )

    model_pair = models[
        intervention
    ]

    control_model = model_pair[
        "control_model"
    ]

    treatment_model = model_pair[
        "treatment_model"
    ]

    X = df[
        features
    ]

    mu_control = (
        control_model.predict(X)
    )

    mu_treatment = (
        treatment_model.predict(X)
    )

    cate = (
        mu_treatment
        - mu_control
    )

    cate = np.asarray(
        cate,
        dtype=float,
    )

    if not np.all(
        np.isfinite(cate)
    ):

        raise ValueError(
            "CATE model produced non-finite "
            "predictions."
        )

    return cate


def predict_all_production_cate(
    df: pd.DataFrame,
    models: dict,
    feature_columns: Iterable[str] = DEFAULT_FEATURES,
) -> pd.DataFrame:
    """
    Generate policy-specific CATE estimates.

    IMPORTANT:
        These estimates represent each intervention model's
        estimated effect. They must NOT automatically be interpreted
        as a valid comparison of all interventions for the same user,
        because the randomized experiment assigned users to
        policy-specific cohorts.
    """

    result = df.copy()

    for intervention in sorted(
        models.keys()
    ):

        safe_name = (
            intervention.lower()
        )

        column_name = (
            f"cate_{safe_name}"
        )

        result[
            column_name
        ] = predict_production_cate(
            df=result,
            intervention=intervention,
            models=models,
            feature_columns=feature_columns,
        )

    return result


# ============================================================================
# MODEL PERSISTENCE
# ============================================================================

def save_production_cate_models(
    models: dict,
    output_path: str | Path,
    feature_columns: Iterable[str] = DEFAULT_FEATURES,
    config: CATEConfig | None = None,
) -> None:
    """
    Persist production CATE models and metadata.
    """

    config = config or CATEConfig()

    artifact = {
        "artifact_type":
            "finsight_cate_t_learner",

        "artifact_version":
            "1.0",

        "models":
            models,

        "feature_columns":
            list(feature_columns),

        "outcome_column":
            config.outcome_column,

        "treatment_column":
            config.treatment_column,

        "intervention_column":
            config.intervention_column,

        "control_value":
            config.control_value,

        "treatment_value":
            config.treatment_value,

        "model_type":
            "T-learner RandomForestRegressor",

        "n_estimators":
            config.n_estimators,

        "min_samples_leaf":
            config.min_samples_leaf,

        "max_depth":
            config.max_depth,

        "random_state":
            config.random_state,

        "cate_definition":
            "E[Y|T=1,X] - E[Y|T=0,X]",

        "negative_cate_definition":
            "Treatment predicts lower "
            "synthetic financial stress",

        "synthetic_data_warning":
            "CATE estimates are based on "
            "the FinSight synthetic randomized "
            "experiment and are not real-world "
            "causal effectiveness estimates.",
    }

    output_path = Path(
        output_path
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(
        artifact,
        output_path,
    )


def load_production_cate_models(
    model_path: str | Path,
) -> dict:
    """
    Load and validate a persisted CATE artifact.
    """

    model_path = Path(
        model_path
    )

    if not model_path.exists():

        raise FileNotFoundError(
            f"CATE model artifact not found: "
            f"{model_path}"
        )

    artifact = joblib.load(
        model_path
    )

    required_keys = [
        "artifact_type",
        "models",
        "feature_columns",
        "outcome_column",
        "treatment_column",
        "intervention_column",
    ]

    missing = [
        key
        for key in required_keys
        if key not in artifact
    ]

    if missing:

        raise ValueError(
            "Invalid CATE artifact. "
            f"Missing keys: {missing}"
        )

    if (
        artifact[
            "artifact_type"
        ]
        != "finsight_cate_t_learner"
    ):

        raise ValueError(
            "Invalid CATE artifact type."
        )

    if not artifact[
        "models"
    ]:

        raise ValueError(
            "CATE artifact contains no models."
        )

    return artifact


# ============================================================================
# CATE SUMMARY
# ============================================================================

def summarize_cate(
    cate_df: pd.DataFrame,
) -> pd.DataFrame:

    rows = []

    for intervention, cohort in (
        cate_df.groupby(
            "intervention_category",
            sort=True,
        )
    ):

        cate = cohort[
            "cate"
        ].astype(float)

        rows.append(
            {
                "intervention_category":
                    intervention,

                "n":
                    len(cohort),

                "mean_cate":
                    float(cate.mean()),

                "median_cate":
                    float(cate.median()),

                "std_cate":
                    float(
                        cate.std(
                            ddof=1
                        )
                    ),

                "q10_cate":
                    float(
                        cate.quantile(
                            0.10
                        )
                    ),

                "q90_cate":
                    float(
                        cate.quantile(
                            0.90
                        )
                    ),

                "benefit_rate":
                    float(
                        (
                            cate < 0
                        ).mean()
                    ),

                "strong_benefit_rate":
                    float(
                        (
                            cate <= -0.10
                        ).mean()
                    ),
            }
        )

    return pd.DataFrame(
        rows
    )


# ============================================================================
# CATE SEGMENTS
# ============================================================================

def build_cate_segments(
    cate_df: pd.DataFrame,
    n_segments: int = 4,
) -> pd.DataFrame:

    result = cate_df.copy()

    segment_frames = []

    for intervention, group in (
        result.groupby(
            "intervention_category",
            sort=False,
        )
    ):

        values = group[
            "cate"
        ]

        unique_count = (
            values.nunique()
        )

        if unique_count < 2:

            labels = pd.Series(
                ["Q1"] * len(group),
                index=group.index,
            )

        else:

            q = min(
                n_segments,
                unique_count,
            )

            labels = pd.qcut(
                values,
                q=q,
                labels=[
                    f"Q{i}"
                    for i in range(
                        1,
                        q + 1,
                    )
                ],
                duplicates="drop",
            )

            labels = pd.Series(
                labels.astype(str).values,
                index=group.index,
            )

        segment_frames.append(
            labels
        )

    result[
        "cate_segment"
    ] = pd.concat(
        segment_frames
    ).sort_index()

    rows = []

    for (
        intervention,
        segment,
    ), group in result.groupby(
        [
            "intervention_category",
            "cate_segment",
        ],
        sort=True,
    ):

        rows.append(
            {
                "intervention_category":
                    intervention,

                "cate_segment":
                    segment,

                "n":
                    len(group),

                "mean_cate":
                    float(
                        group[
                            "cate"
                        ].mean()
                    ),

                "median_cate":
                    float(
                        group[
                            "cate"
                        ].median()
                    ),

                "benefit_rate":
                    float(
                        (
                            group[
                                "cate"
                            ] < 0
                        ).mean()
                    ),
            }
        )

    return pd.DataFrame(
        rows
    )


# ============================================================================
# CATE FEATURE ASSOCIATION
# ============================================================================

def estimate_cate_feature_importance(
    cate_df: pd.DataFrame,
    feature_columns: Iterable[str],
    random_state: int = RANDOM_STATE,
) -> pd.DataFrame:
    """
    Estimate which baseline features are associated with
    CATE heterogeneity.

    This is NOT causal feature importance.
    """

    features = list(
        feature_columns
    )

    X = cate_df[
        features
    ]

    y = cate_df[
        "cate"
    ].astype(float)

    model = Pipeline(
        [
            (
                "imputer",
                SimpleImputer(
                    strategy="median"
                ),
            ),
            (
                "model",
                RandomForestRegressor(
                    n_estimators=300,
                    max_depth=5,
                    min_samples_leaf=10,
                    random_state=random_state,
                    n_jobs=-1,
                ),
            ),
        ]
    )

    model.fit(
        X,
        y,
    )

    importance = pd.Series(
        model.named_steps[
            "model"
        ].feature_importances_,
        index=features,
        name="importance",
    ).sort_values(
        ascending=False
    )

    return (
        importance
        .reset_index()
        .rename(
            columns={
                "index":
                    "feature"
            }
        )
    )


# ============================================================================
# IN-SAMPLE VS CROSS-FITTED COMPARISON
# ============================================================================

def compare_cate_estimators(
    in_sample: pd.DataFrame,
    cross_fitted: pd.DataFrame,
) -> pd.DataFrame:

    rows = []

    interventions = sorted(
        cross_fitted[
            "intervention_category"
        ].unique()
    )

    for intervention in interventions:

        in_sample_group = (
            in_sample[
                in_sample[
                    "intervention_category"
                ]
                == intervention
            ]
        )

        cross_group = (
            cross_fitted[
                cross_fitted[
                    "intervention_category"
                ]
                == intervention
            ]
        )

        rows.append(
            {
                "intervention_category":
                    intervention,

                "in_sample_mean_cate":
                    float(
                        in_sample_group[
                            "cate"
                        ].mean()
                    ),

                "cross_fitted_mean_cate":
                    float(
                        cross_group[
                            "cate"
                        ].mean()
                    ),

                "in_sample_std_cate":
                    float(
                        in_sample_group[
                            "cate"
                        ].std()
                    ),

                "cross_fitted_std_cate":
                    float(
                        cross_group[
                            "cate"
                        ].std()
                    ),

                "in_sample_benefit_rate":
                    float(
                        (
                            in_sample_group[
                                "cate"
                            ] < 0
                        ).mean()
                    ),

                "cross_fitted_benefit_rate":
                    float(
                        (
                            cross_group[
                                "cate"
                            ] < 0
                        ).mean()
                    ),
            }
        )

    return pd.DataFrame(
        rows
    )


# ============================================================================
# POLICY VALUE DIAGNOSTIC
# ============================================================================

def evaluate_policy_value(
    cate_df: pd.DataFrame,
    outcome_column: str = (
        "synthetic_financial_stress_30d"
    ),
) -> pd.DataFrame:
    """
    Descriptive simulator-based policy diagnostic.

    This is NOT a real-world causal policy evaluation.
    """

    if "cate" not in cate_df.columns:

        raise ValueError(
            "cate column is required."
        )

    if "treatment" not in cate_df.columns:

        raise ValueError(
            "treatment column is required."
        )

    if outcome_column not in cate_df.columns:

        raise ValueError(
            f"Missing outcome column: "
            f"{outcome_column}"
        )

    df = cate_df.copy()

    observed_rate = float(
        df[
            outcome_column
        ].mean()
    )

    control = df[
        df[
            "treatment"
        ] == 0
    ]

    control_rate = float(
        control[
            outcome_column
        ].mean()
    )

    personalized = df[
        df[
            "cate"
        ] < 0
    ]

    personalized_rate = (
        float(
            personalized[
                outcome_column
            ].mean()
        )
        if len(personalized)
        else np.nan
    )

    strong = df[
        df[
            "cate"
        ] <= -0.10
    ]

    strong_rate = (
        float(
            strong[
                outcome_column
            ].mean()
        )
        if len(strong)
        else np.nan
    )

    return pd.DataFrame(
        [
            {
                "policy":
                    "OBSERVED_RANDOMIZED",

                "n":
                    len(df),

                "observed_outcome_rate":
                    observed_rate,
            },
            {
                "policy":
                    "CONTROL_ONLY",

                "n":
                    len(control),

                "observed_outcome_rate":
                    control_rate,
            },
            {
                "policy":
                    "PERSONALIZED_CATE_BENEFIT",

                "n":
                    len(personalized),

                "observed_outcome_rate":
                    personalized_rate,
            },
            {
                "policy":
                    "STRONG_CATE_BENEFIT",

                "n":
                    len(strong),

                "observed_outcome_rate":
                    strong_rate,
            },
        ]
    )


# ============================================================================
# POLICY RECOMMENDATION
# ============================================================================

def build_policy_recommendations(
    cate_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Create recommendations from estimated CATE.

    IMPORTANT:
        This function does not compare different interventions
        for the same observation.

        It only determines whether the intervention assigned to
        the observation has estimated benefit.
    """

    result = cate_df.copy()

    result[
        "estimated_benefit"
    ] = -result[
        "cate"
    ]

    result[
        "recommended_action"
    ] = np.where(
        result[
            "cate"
        ] < 0,
        result[
            "intervention_category"
        ],
        "NO_SIMULATED_BENEFIT",
    )

    return result[
        [
            column
            for column in [
                "user_id",
                "month",
                "intervention_category",
                "cate",
                "estimated_benefit",
                "predicted_outcome_control",
                "predicted_outcome_treatment",
                "recommended_action",
            ]
            if column in result.columns
        ]
    ]


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    "CATEConfig",
    "DEFAULT_FEATURES",
    "build_t_learner",
    "validate_cate_input",
    "estimate_cate",
    "estimate_cross_fitted_cate",
    "fit_production_cate_models",
    "predict_production_cate",
    "predict_all_production_cate",
    "save_production_cate_models",
    "load_production_cate_models",
    "summarize_cate",
    "build_cate_segments",
    "estimate_cate_feature_importance",
    "compare_cate_estimators",
    "evaluate_policy_value",
    "build_policy_recommendations",
]