from __future__ import annotations

from pathlib import Path
from typing import Any
import json
import joblib
import numpy as np
import pandas as pd

from finsight.decision.engine import (
    DecisionConfig,
    generate_decision,
)


class FinSightPredictor:
    """
    Production inference wrapper for the FinSight stress model.

    Responsibilities:
        1. Load the persisted model.
        2. Validate feature columns.
        3. Generate calibrated stress probability.
        4. Pass the result to the Decision Engine.

    CATE is intentionally optional because the current
    cross-fitted CATE CSV contains estimates, not a reusable
    inference model.
    """

    def __init__(
        self,
        model_path: str | Path,
        metadata_path: str | Path,
        decision_config: DecisionConfig | None = None,
    ):

        self.model_path = Path(
            model_path
        )

        self.metadata_path = Path(
            metadata_path
        )

        self.decision_config = (
            decision_config
            or DecisionConfig()
        )

        self.model = joblib.load(
            self.model_path
        )

        with open(
            self.metadata_path,
            "r",
            encoding="utf-8",
        ) as file:
            self.metadata = json.load(file)

        self.feature_columns = list(
            self.metadata[
                "feature_columns"
            ]
        )

        self.model_name = self.metadata.get(
            "model_name",
            "unknown",
        )

        self.threshold = float(
            self.metadata.get(
                "decision_threshold",
                0.50,
            )
        )

    def validate_features(
        self,
        df: pd.DataFrame,
    ) -> None:

        missing = [
            feature
            for feature in self.feature_columns
            if feature not in df.columns
        ]

        if missing:

            raise ValueError(
                "Missing inference features: "
                f"{missing}"
            )

    def predict_probability(
        self,
        df: pd.DataFrame,
    ) -> np.ndarray:

        self.validate_features(
            df
        )

        X = df[
            self.feature_columns
        ]

        if not hasattr(
            self.model,
            "predict_proba",
        ):

            raise TypeError(
                "Persisted model does not "
                "support predict_proba()."
            )

        probabilities = (
            self.model.predict_proba(X)[:, 1]
        )

        probabilities = np.asarray(
            probabilities,
            dtype=float,
        )

        if not np.all(
            np.isfinite(probabilities)
        ):

            raise ValueError(
                "Model produced non-finite "
                "probabilities."
            )

        probabilities = np.clip(
            probabilities,
            0.0,
            1.0,
        )

        return probabilities

    def predict(
        self,
        df: pd.DataFrame,
        cate_values: pd.Series | None = None,
    ) -> pd.DataFrame:

        self.validate_features(
            df
        )

        probabilities = (
            self.predict_probability(df)
        )

        working = df.copy()

        working[
            "risk_probability"
        ] = probabilities

        if cate_values is not None:

            cate_values = pd.Series(
                cate_values,
                index=working.index,
            )

            working[
                "cate"
            ] = cate_values

        else:

            working[
                "cate"
            ] = np.nan

        decisions = []

        for index, row in working.iterrows():

            cate = row.get(
                "cate"
            )

            if pd.isna(cate):

                cate = None

            else:

                cate = float(cate)

            decision = generate_decision(
                row=row,
                risk_probability=float(
                    row[
                        "risk_probability"
                    ]
                ),
                cate=cate,
                config=self.decision_config,
            )

            decisions.append(
                {
                    "user_id":
                        decision.user_id,

                    "month":
                        decision.month,

                    "risk_probability":
                        decision.risk_probability,

                    "risk_level":
                        decision.risk_level,

                    "financial_pressure":
                        decision.financial_pressure,

                    "intervention":
                        decision.intervention,

                    "intervention_priority":
                        decision.intervention_priority,

                    "cate":
                        decision.cate,

                    "recommendation_status":
                        decision.recommendation_status,

                    "explanation":
                        " | ".join(
                            decision.explanation
                        ),

                    "model_drivers":
                        " | ".join(
                            decision.model_drivers
                        ),
                }
            )

        return pd.DataFrame(
            decisions
        )

    def predict_one(
        self,
        row: pd.Series,
        cate: float | None = None,
    ) -> dict[str, Any]:

        df = pd.DataFrame(
            [row.to_dict()]
        )

        cate_values = None

        if cate is not None:

            cate_values = pd.Series(
                [cate]
            )

        result = self.predict(
            df,
            cate_values=cate_values,
        )

        return result.iloc[
            0
        ].to_dict()