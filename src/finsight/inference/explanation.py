from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV

from finsight.explainability.shap_explainer import (
    build_shap_explainer,
    calculate_shap_values,
    build_local_explanation,
)


class FinSightExplanationEngine:
    """
    Production SHAP explanation layer for FinSight.

    Explains the underlying HistGradientBoosting estimators
    inside the calibrated stress model.

    Important:
        SHAP explains model behavior.
        It does not establish causality.
    """

    FEATURE_LABELS = {
        "minimum_balance_ratio":
            "minimum balance relative to income",

        "average_balance_ratio":
            "average balance relative to income",

        "rolling_average_balance":
            "recent average balance",

        "rolling_minimum_balance":
            "recent minimum balance",

        "expense_pressure":
            "expense pressure",

        "income_growth":
            "income growth",

        "income_volatility":
            "income volatility",

        "recurring_spending":
            "recurring spending",

        "balance_growth":
            "balance trajectory",

        "rolling_net_cash_flow":
            "recent net cash flow",

        "rolling_transaction_count":
            "recent transaction activity",

        "transaction_count_change":
            "change in transaction activity",

        "spending_baseline_deviation":
            "deviation from personal spending baseline",

        "spending_baseline_ratio":
            "spending relative to personal baseline",

        "spending_growth":
            "spending growth",

        "spending_acceleration":
            "spending acceleration",

        "discretionary_pressure":
            "discretionary spending pressure",

        "savings_rate":
            "savings rate",

        "rolling_savings_rate":
            "recent savings rate",

        "cash_buffer":
            "cash buffer",

        "cash_buffer_ratio":
            "cash buffer relative to income",

        "category_shift":
            "spending category shift",

        "state_changed":
            "recent financial-state change",

        "state_duration":
            "financial-state duration",
    }

    def __init__(
        self,
        model: Any,
        feature_columns: list[str],
        background_data: pd.DataFrame,
        top_n: int = 5,
    ) -> None:

        if not isinstance(
            model,
            CalibratedClassifierCV,
        ):
            raise TypeError(
                "FinSightExplanationEngine expects "
                "a fitted CalibratedClassifierCV."
            )

        if background_data.empty:
            raise ValueError(
                "background_data cannot be empty."
            )

        self.model = model
        self.feature_columns = list(
            feature_columns
        )
        self.top_n = top_n

        self.explainers = []

        for calibrated_model in (
            model.calibrated_classifiers_
        ):

            pipeline = calibrated_model.estimator

            if "imputer" not in pipeline.named_steps:
                raise ValueError(
                    "Expected an 'imputer' step "
                    "in the stress-model pipeline."
                )

            if "model" not in pipeline.named_steps:
                raise ValueError(
                    "Expected a 'model' step "
                    "in the stress-model pipeline."
                )

            imputer = pipeline.named_steps[
                "imputer"
            ]

            tree_model = pipeline.named_steps[
                "model"
            ]

            background_transformed = (
                imputer.transform(
                    background_data[
                        self.feature_columns
                    ]
                )
            )

            background_transformed = pd.DataFrame(
                background_transformed,
                columns=self.feature_columns,
            )

            explainer = build_shap_explainer(
                tree_model,
                background_transformed,
            )

            self.explainers.append(
                (
                    imputer,
                    explainer,
                )
            )

        if not self.explainers:
            raise ValueError(
                "No calibrated estimators found."
            )

    def explain(
        self,
        df: pd.DataFrame,
    ) -> list[list[str]]:

        if df.empty:
            return []

        missing = [
            column
            for column in self.feature_columns
            if column not in df.columns
        ]

        if missing:
            raise ValueError(
                "Missing explanation features: "
                + ", ".join(missing)
            )

        X = df[
            self.feature_columns
        ].copy()

        fold_shap_values = []

        for (
            imputer,
            explainer,
        ) in self.explainers:

            X_transformed = imputer.transform(
                X
            )

            X_transformed = pd.DataFrame(
                X_transformed,
                columns=self.feature_columns,
            )

            values = calculate_shap_values(
                explainer,
                X_transformed,
            )

            fold_shap_values.append(
                values
            )

        shap_values = np.mean(
            np.stack(
                fold_shap_values,
                axis=0,
            ),
            axis=0,
        )

        results = []

        for row_index in range(
            len(X)
        ):

            local = build_local_explanation(
                feature_names=self.feature_columns,
                feature_values=X.iloc[
                    row_index
                ].to_numpy(),
                shap_values=shap_values[
                    row_index
                ],
                top_n=self.top_n,
            )

            drivers = []

            for _, row in local.iterrows():

                feature = row[
                    "feature"
                ]

                direction = row[
                    "direction"
                ]

                label = self.FEATURE_LABELS.get(
                    feature,
                    feature.replace(
                        "_",
                        " ",
                    ),
                )

                if (
                    direction
                    == "increases_stress_risk"
                ):
                    drivers.append(
                        f"{label} increases "
                        f"predicted stress risk"
                    )
                else:
                    drivers.append(
                        f"{label} reduces "
                        f"predicted stress risk"
                    )

            results.append(
                drivers
            )

        return results

    def explain_one(
        self,
        row: pd.Series | dict,
    ) -> list[str]:

        if isinstance(
            row,
            dict,
        ):
            row = pd.Series(row)

        result = self.explain(
            pd.DataFrame(
                [row]
            )
        )

        return result[0]