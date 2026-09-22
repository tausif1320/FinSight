from __future__ import annotations

from pathlib import Path

import pandas as pd

from finsight.decision.engine import (
    DecisionConfig,
    generate_decision,
)

from finsight.inference.cate_predictor import (
    CATEPredictor,
)

from finsight.inference.explanation import (
    FinSightExplanationEngine,
)

from finsight.inference.predictor import (
    FinSightPredictor,
)


class FinSightInferencePipeline:
    """
    End-to-end FinSight inference pipeline.

    Pipeline:

        Financial Features
                |
                v
        Calibrated Stress Model
                |
                v
        Risk Probability
                |
                +----------------------+
                |                      |
                v                      v
        Financial Pressure        SHAP Explanation
                |                      |
                v                      v
        Pressure -> Intervention   Model Drivers
                |
                v
        Intervention-specific CATE
                |
                v
        Final Recommendation
    """

    # ========================================================================
    # INITIALIZATION
    # ========================================================================

    def __init__(
        self,
        stress_model_path: str | Path,
        stress_metadata_path: str | Path,
        cate_model_path: str | Path,
        decision_config: DecisionConfig | None = None,
        shap_background_path: str | Path = "models/shap_background.csv",
    ) -> None:

        self.decision_config = (
            decision_config
            if decision_config is not None
            else DecisionConfig()
        )

        # --------------------------------------------------------------------
        # Stress model
        # --------------------------------------------------------------------

        self.stress_predictor = FinSightPredictor(
            model_path=stress_model_path,
            metadata_path=stress_metadata_path,
            decision_config=self.decision_config,
        )

        # --------------------------------------------------------------------
        # CATE model
        # --------------------------------------------------------------------

        self.cate_predictor = CATEPredictor(
            model_path=cate_model_path,
        )

        # --------------------------------------------------------------------
        # SHAP background
        # --------------------------------------------------------------------

        self.shap_background_path = Path(
            shap_background_path
        )

        if not self.shap_background_path.exists():
            raise FileNotFoundError(
                "SHAP background dataset not found: "
                f"{self.shap_background_path}"
            )

        self.shap_background = pd.read_csv(
            self.shap_background_path
        )

        if self.shap_background.empty:
            raise ValueError(
                "SHAP background dataset is empty."
            )

        # --------------------------------------------------------------------
        # SHAP explanation engine
        #
        # IMPORTANT:
        # Use the exact 53 production features from the
        # stress-model metadata, not all columns in the
        # original modeling dataset.
        # --------------------------------------------------------------------

        self.explanation_engine = (
            FinSightExplanationEngine(
                model=self.stress_predictor.model,
                feature_columns=(
                    self.stress_predictor.feature_columns
                ),
                background_data=self.shap_background,
                top_n=5,
            )
        )

    # ========================================================================
    # PRESSURE -> INTERVENTION MAPPING
    # ========================================================================

    @staticmethod
    def pressure_to_intervention(
        financial_pressure: str,
    ) -> str | None:
        """
        Map a financial-pressure category to its corresponding
        intervention family.

        Mapping:

            LIQUIDITY_PRESSURE
                -> LIQUIDITY_PROTECTION

            SPENDING_PRESSURE
                -> SPENDING_CONTROL

            OBLIGATION_PRESSURE
                -> OBLIGATION_MANAGEMENT

            STABLE
                -> None

        Stable users do not receive an intervention-specific
        CATE estimate because they were not part of the
        randomized intervention cohorts.
        """

        mapping = {
            "LIQUIDITY_PRESSURE": (
                "LIQUIDITY_PROTECTION"
            ),

            "SPENDING_PRESSURE": (
                "SPENDING_CONTROL"
            ),

            "OBLIGATION_PRESSURE": (
                "OBLIGATION_MANAGEMENT"
            ),

            "STABLE": None,
        }

        return mapping.get(
            financial_pressure
        )

    # ========================================================================
    # INTERVENTION PRIORITY
    # ========================================================================

    @staticmethod
    def intervention_priority(
        intervention: str,
    ) -> int:
        """
        Return the priority associated with each intervention.
        """

        priorities = {
            "LIQUIDITY_PROTECTION": 1,
            "SPENDING_CONTROL": 2,
            "OBLIGATION_MANAGEMENT": 3,
            "SAVINGS_REINFORCEMENT": 4,
        }

        return priorities.get(
            intervention,
            4,
        )

    # ========================================================================
    # RECOMMENDATION STATUS
    # ========================================================================

    @staticmethod
    def recommendation_status(
        intervention: str,
        cate: float | None,
    ) -> str:
        """
        Determine the recommendation status.

        Stable users:
            MONITOR

        Active intervention with:
            CATE <= -0.10
                -> STRONG_SIMULATED_SUPPORT

            -0.10 < CATE < 0
                -> SIMULATED_SUPPORT

            CATE >= 0
                -> RULE_BASED_RECOMMENDATION

        Important:
        CATE represents simulated treatment-effect evidence from
        the synthetic randomized experiment. It must not be
        interpreted as real-world causal effectiveness.
        """

        if intervention == "SAVINGS_REINFORCEMENT":
            return "MONITOR"

        if cate is None:
            return "RULE_BASED_RECOMMENDATION"

        if cate <= -0.10:
            return "STRONG_SIMULATED_SUPPORT"

        if cate < 0:
            return "SIMULATED_SUPPORT"

        return "RULE_BASED_RECOMMENDATION"

    # ========================================================================
    # SINGLE OBSERVATION
    # ========================================================================

    def predict_one(
        self,
        row: pd.Series,
    ) -> dict:
        """
        Run the complete FinSight inference pipeline for one
        user-month observation.
        """

        # ====================================================================
        # 1. PREPARE INPUT
        # ====================================================================

        row_df = pd.DataFrame(
            [row.to_dict()]
        )

        # ====================================================================
        # 2. STRESS MODEL
        # ====================================================================

        risk_probability = float(
            self.stress_predictor.predict_probability(
                row_df
            )[0]
        )

        # ====================================================================
        # 3. SHAP MODEL EXPLANATION
        # ====================================================================
        #
        # SHAP explains:
        #
        #     "Which features pushed the model's prediction?"
        #
        # This is deliberately separate from the Decision Engine.
        # SHAP is model attribution, not causal evidence.
        # ====================================================================

        model_drivers = (
            self.explanation_engine.explain_one(
                row
            )
        )

        # ====================================================================
        # 4. FINANCIAL PRESSURE CLASSIFICATION
        # ====================================================================
        #
        # We call the Decision Engine without CATE.
        #
        # At this stage we only need:
        #
        #     risk level
        #     financial pressure
        #     explanation
        #
        # Intervention assignment remains controlled by the
        # explicit pressure -> intervention policy below.
        # ====================================================================

        preliminary_decision = generate_decision(
            row=row,
            risk_probability=risk_probability,
            cate=None,
            config=self.decision_config,
        )

        financial_pressure = (
            preliminary_decision.financial_pressure
        )

        # ====================================================================
        # 5. DETERMINE APPLICABLE INTERVENTION
        # ====================================================================

        intervention = (
            self.pressure_to_intervention(
                financial_pressure
            )
        )

        # Stable users use the default monitoring action.

        if intervention is None:
            intervention = (
                "SAVINGS_REINFORCEMENT"
            )

        # ====================================================================
        # 6. INTERVENTION-SPECIFIC CATE
        # ====================================================================
        #
        # IMPORTANT:
        #
        # Estimate CATE ONLY for the intervention family
        # corresponding to the user's financial pressure.
        #
        # We do NOT compare CATE values across interventions.
        #
        # The experiment did not estimate all counterfactual
        # interventions for every user.
        # ====================================================================

        cate = None

        if (
            financial_pressure
            != "STABLE"
        ):

            cate = float(
                self.cate_predictor.predict(
                    row_df,
                    intervention,
                )[0]
            )

        # ====================================================================
        # 7. FINAL DECISION INFORMATION
        # ====================================================================
        #
        # We call the Decision Engine again to obtain:
        #
        #     final risk level
        #     final explanation
        #
        # We intentionally do not use its intervention output,
        # because intervention assignment is controlled by the
        # pressure -> intervention policy above.
        # ====================================================================

        final_decision = generate_decision(
            row=row,
            risk_probability=risk_probability,
            cate=cate,
            config=self.decision_config,
        )

        # ====================================================================
        # 8. RECOMMENDATION STATUS
        # ====================================================================

        status = self.recommendation_status(
            intervention=intervention,
            cate=cate,
        )

        # ====================================================================
        # 9. BUILD FINAL OUTPUT
        # ====================================================================

        result = {
            "user_id": final_decision.user_id,

            "month": final_decision.month,

            "risk_probability": (
                float(risk_probability)
            ),

            "risk_level": (
                final_decision.risk_level
            ),

            "financial_pressure": (
                financial_pressure
            ),

            "intervention": (
                intervention
            ),

            "intervention_priority": (
                self.intervention_priority(
                    intervention
                )
            ),

            "cate": (
                float(cate)
                if cate is not None
                else None
            ),

            "recommendation_status": (
                status
            ),

            "explanation": (
                " | ".join(
                    final_decision.explanation
                )
            ),

            # SHAP drivers are now actual local model
            # explanations rather than Decision Engine rules.
            "model_drivers": (
                " | ".join(
                    model_drivers
                )
            ),
        }

        return result

    # ========================================================================
    # BATCH INFERENCE
    # ========================================================================

    def predict(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Run end-to-end inference for multiple observations.
        """

        if df.empty:
            raise ValueError(
                "Inference dataframe is empty."
            )

        results = []

        for _, row in df.iterrows():

            result = self.predict_one(
                row
            )

            results.append(
                result
            )

        return pd.DataFrame(
            results
        )