from __future__ import annotations

from pathlib import Path
import json
import sys

import joblib
import pandas as pd

from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline


ROOT = Path(
    __file__
).resolve().parents[1]

sys.path.insert(
    0,
    str(ROOT / "src"),
)


from finsight.models.dataset import (
    chronological_split,
)

from finsight.models.feature_sets import (
    get_static_plus_dynamic_features,
)

from finsight.models.tree_models import (
    build_hist_gradient_boosting,
)


DATA_PATH = (
    ROOT
    / "data"
    / "processed"
    / "modeling_dataset_1000_users.csv"
)


MODEL_DIR = (
    ROOT
    / "models"
)


TARGET = (
    "financial_stress_30d"
)


RANDOM_STATE = 42


def main():

    print("=" * 80)
    print("FinSight Final Stress Model Training")
    print("=" * 80)

    # ================================================================
    # 1. LOAD DATA
    # ================================================================

    print(
        "\nLoading modeling dataset..."
    )

    df = pd.read_csv(
        DATA_PATH
    )

    print(
        f"Rows: {len(df):,}"
    )

    print(
        f"Users: "
        f"{df['user_id'].nunique():,}"
    )

    print(
        f"Stress cases: "
        f"{df[TARGET].sum():,}"
    )

    # ================================================================
    # 2. CHRONOLOGICAL SPLIT
    # ================================================================

    train_df, validation_df, test_df = (
        chronological_split(
            df
        )
    )

    print(
        "\nChronological split:"
    )

    print(
        f"Train:      {len(train_df):,}"
    )

    print(
        f"Validation: {len(validation_df):,}"
    )

    print(
        f"Test:       {len(test_df):,}"
    )

    # ================================================================
    # 3. FEATURE SET
    # ================================================================

    feature_columns = (
        get_static_plus_dynamic_features(
            df.columns
        )
    )

    print(
        f"\nFeatures: "
        f"{len(feature_columns)}"
    )

    print(
        feature_columns
    )

    # ================================================================
    # 4. MODEL SELECTION
    # ================================================================

    print(
        "\nUsing HistGradientBoostingClassifier."
    )

    print(
        "The model configuration was selected "
        "during the earlier model comparison."
    )

    base_model = (
        build_hist_gradient_boosting()
    )

    # ================================================================
    # 5. FINAL TRAINING DATA
    # ================================================================

    # The test period remains completely untouched.
    #
    # After model selection, train + validation become
    # the final development dataset.

    development_df = pd.concat(
        [
            train_df,
            validation_df,
        ],
        axis=0,
        ignore_index=True,
    )

    X_development = (
        development_df[
            feature_columns
        ]
    )

    y_development = (
        development_df[
            TARGET
        ]
        .astype(int)
    )

    print(
        "\nFinal development rows: "
        f"{len(development_df):,}"
    )

    # ================================================================
    # 6. CALIBRATION
    # ================================================================

    print(
        "\nTraining calibrated HGB..."
    )

    calibrated_model = (
        CalibratedClassifierCV(
            estimator=base_model,
            method="sigmoid",
            cv=5,
            n_jobs=-1,
        )
    )

    calibrated_model.fit(
        X_development,
        y_development,
    )

    # ================================================================
    # 7. SAVE MODEL
    # ================================================================

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    model_path = (
        MODEL_DIR
        / "stress_model.joblib"
    )

    metadata_path = (
        MODEL_DIR
        / "feature_metadata.json"
    )

    joblib.dump(
        calibrated_model,
        model_path,
    )

    metadata = {
        "model_name":
            "Calibrated HistGradientBoostingClassifier",

        "target":
            TARGET,

        "feature_columns":
            feature_columns,

        "n_features":
            len(feature_columns),

        "development_rows":
            len(development_df),

        "train_rows":
            len(train_df),

        "validation_rows":
            len(validation_df),

        "test_rows_reserved":
            len(test_df),

        "random_state":
            RANDOM_STATE,

        "calibration_method":
            "sigmoid",

        "calibration_cv":
            5,

        "decision_threshold":
            0.60,

        "test_set_used_for_training":
            False,

        "note":
            (
                "Final production candidate trained "
                "after model selection. The held-out "
                "test period was not used for fitting."
            ),
    }

    with open(
        metadata_path,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            metadata,
            file,
            indent=2,
        )

    # ================================================================
    # 8. VERIFY RELOAD
    # ================================================================

    print(
        "\nReloading saved model..."
    )

    reloaded = joblib.load(
        model_path
    )

    probabilities = (
        reloaded.predict_proba(
            validation_df[
                feature_columns
            ]
        )[:, 1]
    )

    assert len(
        probabilities
    ) == len(
        validation_df
    )

    assert (
        probabilities >= 0
    ).all()

    assert (
        probabilities <= 1
    ).all()

    print(
        "✓ Saved model successfully reloaded"
    )

    print(
        "✓ Probability inference works"
    )

    print(
        "✓ Probabilities are within [0, 1]"
    )

    print(
        "✓ Test period was excluded from fitting"
    )

    print(
        "\nSaved:"
    )

    print(
        f"  {model_path}"
    )

    print(
        f"  {metadata_path}"
    )

    print("\n" + "=" * 80)
    print("FINAL STRESS MODEL TRAINING PASSED")
    print("=" * 80)


if __name__ == "__main__":
    main()