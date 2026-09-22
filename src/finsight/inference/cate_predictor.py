from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from finsight.experimentation.cate import (
    load_production_cate_models,
    predict_production_cate,
)


class CATEPredictor:

    def __init__(
        self,
        model_path: str | Path,
    ):

        artifact = (
            load_production_cate_models(
                model_path
            )
        )

        self.models = artifact[
            "models"
        ]

        self.feature_columns = artifact[
            "feature_columns"
        ]

    def predict(
        self,
        df: pd.DataFrame,
        intervention: str,
    ) -> np.ndarray:

        return predict_production_cate(
            df=df,
            intervention=intervention,
            models=self.models,
            feature_columns=self.feature_columns,
        )

    def predict_for_interventions(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:

        result = df.copy()

        for intervention in self.models:

            column_name = (
                "cate_"
                + intervention.lower()
            )

            result[
                column_name
            ] = self.predict(
                df,
                intervention,
            )

        return result