from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd

from finsight.inference.explanation import (
    FinSightExplanationEngine,
)


MODEL_PATH = Path(
    "models/stress_model.joblib"
)

METADATA_PATH = Path(
    "models/feature_metadata.json"
)

BACKGROUND_PATH = Path(
    "models/shap_background.csv"
)

DATA_PATH = Path(
    "data/processed/modeling_dataset_1000_users.csv"
)


def main() -> None:

    print("Loading stress model...")

    model = joblib.load(
        MODEL_PATH
    )

    print(
        f"Model: {type(model).__name__}"
    )

    with open(
        METADATA_PATH,
        "r",
        encoding="utf-8",
    ) as file:

        metadata = json.load(file)

    feature_columns = metadata[
        "feature_columns"
    ]

    print(
        f"Production features: "
        f"{len(feature_columns)}"
    )

    background = pd.read_csv(
        BACKGROUND_PATH
    )

    df = pd.read_csv(
        DATA_PATH
    )

    row = (
        df
        .iloc[[0]]
        .copy()
    )

    print(
        f"Testing user_id: "
        f"{row.iloc[0]['user_id']}"
    )

    print(
        f"Testing month: "
        f"{row.iloc[0]['month']}"
    )

    explainer = (
        FinSightExplanationEngine(
            model=model,
            feature_columns=feature_columns,
            background_data=background,
            top_n=5,
        )
    )

    drivers = explainer.explain_one(
        row.iloc[0]
    )

    print()
    print("MODEL DRIVERS")
    print("-" * 60)

    for index, driver in enumerate(
        drivers,
        start=1,
    ):

        print(
            f"{index}. {driver}"
        )

    assert len(drivers) > 0

    print()
    print(
        "EXPLANATION TEST PASSED"
    )


if __name__ == "__main__":
    main()