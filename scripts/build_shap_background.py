from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from finsight.models.dataset import (
    chronological_split,
    get_feature_columns,
)


DATA_PATH = Path(
    "data/processed/modeling_dataset_1000_users.csv"
)

OUTPUT_PATH = Path(
    "models/shap_background.csv"
)

RANDOM_STATE = 42
BACKGROUND_SIZE = 200


def main() -> None:

    df = pd.read_csv(
        DATA_PATH
    )

    df["month"] = pd.to_datetime(
        df["month"]
    )

    train_df, _, _ = chronological_split(
        df
    )

    feature_columns = get_feature_columns(
        train_df
    )

    background = (
        train_df[
            feature_columns
        ]
        .sample(
            n=min(
                BACKGROUND_SIZE,
                len(train_df),
            ),
            random_state=RANDOM_STATE,
        )
        .reset_index(drop=True)
    )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    background.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    print(
        f"Training rows: {len(train_df)}"
    )

    print(
        f"Feature count: "
        f"{len(feature_columns)}"
    )

    print(
        f"Background rows: "
        f"{len(background)}"
    )

    print(
        f"Saved: {OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()