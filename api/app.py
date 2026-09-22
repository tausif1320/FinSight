from __future__ import annotations

from pathlib import Path
import sys
from typing import Any

import pandas as pd

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field


# =============================================================================
# PROJECT PATHS
# =============================================================================

ROOT = Path(__file__).resolve().parents[1]

SRC_PATH = ROOT / "src"

if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))


# =============================================================================
# FINSight IMPORTS
# =============================================================================

from finsight.inference.pipeline import (  # noqa: E402
    FinSightInferencePipeline,
)


# =============================================================================
# DATASET
# =============================================================================

MODELING_DATA_PATH = (
    ROOT
    / "data"
    / "processed"
    / "modeling_dataset_1000_users.csv"
)


# =============================================================================
# MODEL ARTIFACTS
# =============================================================================

STRESS_MODEL_PATH = (
    ROOT
    / "models"
    / "stress_model.joblib"
)

STRESS_METADATA_PATH = (
    ROOT
    / "models"
    / "feature_metadata.json"
)

CATE_MODEL_PATH = (
    ROOT
    / "models"
    / "cate_models.joblib"
)


# =============================================================================
# PRODUCTION FEATURE LIST
# =============================================================================

PRODUCTION_FEATURES = [
    "salary_income",
    "bonus_income",
    "total_income",
    "total_debits",
    "variable_spending",
    "essential_spending",
    "discretionary_spending",
    "recurring_spending",
    "transaction_count",
    "average_transaction",
    "starting_balance",
    "ending_balance",
    "minimum_balance",
    "maximum_balance",
    "average_balance",
    "net_cash_flow",
    "savings_rate",
    "income_to_expense_ratio",
    "essential_spending_ratio",
    "discretionary_spending_ratio",
    "minimum_balance_ratio",
    "average_balance_ratio",
    "state_changed",
    "state_duration",
    "recent_state_change",
    "historical_spending_mean",
    "historical_spending_std",
    "spending_baseline_deviation",
    "spending_baseline_ratio",
    "transaction_count_change",
    "savings_rate_change",
    "rolling_spending_mean",
    "rolling_spending_std",
    "rolling_spending_min",
    "rolling_spending_max",
    "rolling_income_mean",
    "income_volatility",
    "rolling_average_balance",
    "rolling_minimum_balance",
    "rolling_savings_rate",
    "rolling_transaction_count",
    "rolling_net_cash_flow",
    "spending_growth",
    "discretionary_spending_growth",
    "income_growth",
    "balance_growth",
    "spending_acceleration",
    "cash_buffer",
    "cash_buffer_ratio",
    "expense_pressure",
    "discretionary_pressure",
    "negative_cash_flow",
    "category_shift",
]


# =============================================================================
# DASHBOARD DISPLAY COLUMNS
# =============================================================================

DASHBOARD_COLUMNS = [
    "user_id",
    "month",
    *PRODUCTION_FEATURES,
]


# =============================================================================
# FASTAPI APPLICATION
# =============================================================================

app = FastAPI(
    title="FinSight Financial Intelligence API",
    description=(
        "Predictive financial behavior intelligence "
        "and personalized intervention API."
    ),
    version="1.0.0",
)


# =============================================================================
# CORS
# =============================================================================
#
# The dashboard is currently served with:
#
#     python -m http.server 5500 --directory dashboard
#
# Therefore the browser is running on port 5500 while FastAPI
# runs on port 8000.
#
# This allows the frontend to call the backend safely during
# local development.
# =============================================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:5500",
        "http://localhost:5500",
        "http://127.0.0.1:8000",
        "http://localhost:8000",
    ],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =============================================================================
# LOAD DATASET
# =============================================================================

DASHBOARD_DATA_READY = False
DASHBOARD_DATA_ERROR: str | None = None

dashboard_df = pd.DataFrame()


try:

    if not MODELING_DATA_PATH.exists():
        raise FileNotFoundError(
            f"Modeling dataset not found: "
            f"{MODELING_DATA_PATH}"
        )

    dashboard_df = pd.read_csv(
        MODELING_DATA_PATH
    )

    required_columns = {
        "user_id",
        "month",
        *PRODUCTION_FEATURES,
    }

    missing_columns = sorted(
        required_columns
        - set(dashboard_df.columns)
    )

    if missing_columns:
        raise ValueError(
            "Dashboard dataset is missing required "
            f"columns: {missing_columns}"
        )

    dashboard_df["month"] = pd.to_datetime(
        dashboard_df["month"],
        errors="coerce",
    )

    if dashboard_df["month"].isna().any():
        raise ValueError(
            "Dataset contains invalid month values."
        )

    dashboard_df = (
        dashboard_df
        .sort_values(
            ["user_id", "month"]
        )
        .reset_index(drop=True)
    )

    DASHBOARD_DATA_READY = True

except Exception as exc:

    dashboard_df = pd.DataFrame()

    DASHBOARD_DATA_READY = False

    DASHBOARD_DATA_ERROR = str(exc)


# =============================================================================
# LOAD PRODUCTION INFERENCE PIPELINE
# =============================================================================

PIPELINE_READY = False
PIPELINE_ERROR: str | None = None

inference_pipeline = None


try:

    inference_pipeline = FinSightInferencePipeline(
        stress_model_path=STRESS_MODEL_PATH,
        stress_metadata_path=STRESS_METADATA_PATH,
        cate_model_path=CATE_MODEL_PATH,
    )

    PIPELINE_READY = True

except Exception as exc:

    inference_pipeline = None

    PIPELINE_READY = False

    PIPELINE_ERROR = str(exc)


# =============================================================================
# REQUEST SCHEMA
# =============================================================================

class PredictionRequest(BaseModel):
    """
    Request body for one user-month prediction.

    The model expects the same 53 production features
    used by the final calibrated stress model.
    """

    user_id: int = Field(
        ...,
        description="Unique user identifier.",
    )

    month: str = Field(
        ...,
        description="Observation month in YYYY-MM-DD format.",
    )

    salary_income: float = 0.0
    bonus_income: float = 0.0
    total_income: float = 0.0

    total_debits: float = 0.0
    variable_spending: float = 0.0
    essential_spending: float = 0.0
    discretionary_spending: float = 0.0
    recurring_spending: float = 0.0

    transaction_count: float = 0.0
    average_transaction: float = 0.0

    starting_balance: float = 0.0
    ending_balance: float = 0.0
    minimum_balance: float = 0.0
    maximum_balance: float = 0.0
    average_balance: float = 0.0

    net_cash_flow: float = 0.0
    savings_rate: float = 0.0

    income_to_expense_ratio: float = 0.0
    essential_spending_ratio: float = 0.0
    discretionary_spending_ratio: float = 0.0
    minimum_balance_ratio: float = 0.0
    average_balance_ratio: float = 0.0

    state_changed: float = 0.0
    state_duration: float = 0.0
    recent_state_change: float = 0.0

    historical_spending_mean: float = 0.0
    historical_spending_std: float = 0.0
    spending_baseline_deviation: float = 0.0
    spending_baseline_ratio: float = 0.0

    transaction_count_change: float = 0.0
    savings_rate_change: float = 0.0

    rolling_spending_mean: float = 0.0
    rolling_spending_std: float = 0.0
    rolling_spending_min: float = 0.0
    rolling_spending_max: float = 0.0

    rolling_income_mean: float = 0.0
    income_volatility: float = 0.0

    rolling_average_balance: float = 0.0
    rolling_minimum_balance: float = 0.0
    rolling_savings_rate: float = 0.0
    rolling_transaction_count: float = 0.0
    rolling_net_cash_flow: float = 0.0

    spending_growth: float = 0.0
    discretionary_spending_growth: float = 0.0
    income_growth: float = 0.0
    balance_growth: float = 0.0

    spending_acceleration: float = 0.0

    cash_buffer: float = 0.0
    cash_buffer_ratio: float = 0.0

    expense_pressure: float = 0.0
    discretionary_pressure: float = 0.0

    negative_cash_flow: float = 0.0

    category_shift: float = 0.0


# =============================================================================
# RESPONSE SCHEMA
# =============================================================================

class PredictionResponse(BaseModel):

    user_id: int

    month: str

    risk_probability: float

    risk_level: str

    financial_pressure: str

    intervention: str

    intervention_priority: int

    cate: float | None

    recommendation_status: str

    explanation: str

    model_drivers: str


# =============================================================================
# ROOT
# =============================================================================

@app.get("/")
def root() -> dict[str, str]:

    return {
        "service": "FinSight Financial Intelligence API",
        "status": "running",
        "docs": "/docs",
        "health": "/health",
    }


# =============================================================================
# HEALTH
# =============================================================================

@app.get("/health")
def health() -> dict[str, Any]:

    return {
        "status": (
            "healthy"
            if PIPELINE_READY
            else "unhealthy"
        ),
        "service": "finsight",
        "pipeline_ready": PIPELINE_READY,
        "pipeline_error": PIPELINE_ERROR,
        "dashboard_data_ready": DASHBOARD_DATA_READY,
        "dashboard_data_error": DASHBOARD_DATA_ERROR,
        "users": (
            int(
                dashboard_df["user_id"]
                .nunique()
            )
            if DASHBOARD_DATA_READY
            else 0
        ),
        "observations": (
            int(len(dashboard_df))
            if DASHBOARD_DATA_READY
            else 0
        ),
    }


# =============================================================================
# MODEL INFORMATION
# =============================================================================

@app.get("/model-info")
def model_info() -> dict[str, Any]:

    if not PIPELINE_READY:

        raise HTTPException(
            status_code=503,
            detail=(
                "Inference pipeline is not available."
            ),
        )

    return {

        "model": (
            "Calibrated HistGradientBoostingClassifier"
        ),

        "stress_model": (
            "models/stress_model.joblib"
        ),

        "cate_model": (
            "models/cate_models.joblib"
        ),

        "feature_count": len(
            PRODUCTION_FEATURES
        ),

        "prediction_horizon": "30 days",

        "historical_window": "90 days",

        "dataset_users": (
            int(
                dashboard_df["user_id"]
                .nunique()
            )
            if DASHBOARD_DATA_READY
            else 0
        ),

        "dataset_observations": (
            int(len(dashboard_df))
            if DASHBOARD_DATA_READY
            else 0
        ),

        "interventions": [
            "LIQUIDITY_PROTECTION",
            "SPENDING_CONTROL",
            "OBLIGATION_MANAGEMENT",
            "SAVINGS_REINFORCEMENT",
        ],
    }


# =============================================================================
# DASHBOARD USERS
# =============================================================================

@app.get("/api/dashboard/users")
def dashboard_users() -> dict[str, Any]:

    if not DASHBOARD_DATA_READY:

        raise HTTPException(
            status_code=503,
            detail=(
                "Dashboard dataset is not available: "
                f"{DASHBOARD_DATA_ERROR}"
            ),
        )

    users = (
        dashboard_df["user_id"]
        .drop_duplicates()
        .sort_values()
        .tolist()
    )

    return {
        "users": [
            int(user_id)
            for user_id in users
        ],
        "count": len(users),
    }


# =============================================================================
# DASHBOARD USER HISTORY
# =============================================================================

@app.get(
    "/api/dashboard/users/{user_id}/history"
)
def dashboard_user_history(
    user_id: int,
) -> dict[str, Any]:

    if not DASHBOARD_DATA_READY:

        raise HTTPException(
            status_code=503,
            detail=(
                "Dashboard dataset is not available: "
                f"{DASHBOARD_DATA_ERROR}"
            ),
        )

    user_df = dashboard_df[
        dashboard_df["user_id"] == user_id
    ].copy()

    if user_df.empty:

        raise HTTPException(
            status_code=404,
            detail=f"User {user_id} not found.",
        )

    user_df = user_df.sort_values(
        "month"
    )

    user_df["month"] = (
        user_df["month"]
        .dt.strftime("%Y-%m-%d")
    )

    records_df = user_df[
        DASHBOARD_COLUMNS
    ].copy()

    # Convert NaN / inf values into JSON-safe None.
    records_df = records_df.replace(
        [float("inf"), float("-inf")],
        pd.NA,
    )

    records_df = records_df.astype(
        object
    ).where(
        pd.notna(records_df),
        None,
    )

    records = (
        records_df
        .to_dict(orient="records")
    )

    return {
        "user_id": int(user_id),
        "count": len(records),
        "history": records,
    }


# =============================================================================
# SINGLE PREDICTION
# =============================================================================

@app.post(
    "/predict",
    response_model=PredictionResponse,
)
def predict(
    request: PredictionRequest,
) -> PredictionResponse:

    if not PIPELINE_READY:

        raise HTTPException(
            status_code=503,
            detail=(
                "FinSight inference pipeline "
                "is not available."
            ),
        )

    try:

        # -------------------------------------------------------------
        # Convert request to DataFrame
        # -------------------------------------------------------------

        input_data = request.model_dump()

        df = pd.DataFrame(
            [input_data]
        )

        # -------------------------------------------------------------
        # Ensure production features are present
        # -------------------------------------------------------------

        missing_features = [
            feature
            for feature in PRODUCTION_FEATURES
            if feature not in df.columns
        ]

        if missing_features:

            raise ValueError(
                "Missing production features: "
                f"{missing_features}"
            )

        # -------------------------------------------------------------
        # Run production inference
        # -------------------------------------------------------------

        result = (
            inference_pipeline.predict(
                df
            )
        )

        if result.empty:

            raise RuntimeError(
                "Inference returned no result."
            )

        prediction = result.iloc[0]

        # -------------------------------------------------------------
        # CATE
        # -------------------------------------------------------------

        cate = prediction["cate"]

        if pd.isna(cate):

            cate = None

        else:

            cate = float(cate)

        # -------------------------------------------------------------
        # Drivers
        # -------------------------------------------------------------

        model_drivers = prediction[
            "model_drivers"
        ]

        if pd.isna(model_drivers):

            model_drivers = ""

        else:

            model_drivers = str(
                model_drivers
            )

        # -------------------------------------------------------------
        # Response
        # -------------------------------------------------------------

        return PredictionResponse(

            user_id=int(
                prediction["user_id"]
            ),

            month=str(
                prediction["month"]
            ),

            risk_probability=float(
                prediction[
                    "risk_probability"
                ]
            ),

            risk_level=str(
                prediction[
                    "risk_level"
                ]
            ),

            financial_pressure=str(
                prediction[
                    "financial_pressure"
                ]
            ),

            intervention=str(
                prediction[
                    "intervention"
                ]
            ),

            intervention_priority=int(
                prediction[
                    "intervention_priority"
                ]
            ),

            cate=cate,

            recommendation_status=str(
                prediction[
                    "recommendation_status"
                ]
            ),

            explanation=str(
                prediction[
                    "explanation"
                ]
            ),

            model_drivers=model_drivers,
        )

    except HTTPException:

        raise

    except Exception as exc:

        raise HTTPException(
            status_code=400,
            detail=(
                f"Inference failed: {exc}"
            ),
        ) from exc


# =============================================================================
# OPTIONAL: SERVE DASHBOARD FROM FASTAPI
# =============================================================================
#
# This is intentionally included so that eventually you can run:
#
#     uvicorn api.app:app --reload --port 8000
#
# and open:
#
#     http://127.0.0.1:8000/dashboard/
#
# You can still use the separate port-5500 development server.
# =============================================================================

DASHBOARD_PATH = ROOT / "dashboard"

if DASHBOARD_PATH.exists():

    app.mount(
        "/dashboard",
        StaticFiles(
            directory=str(
                DASHBOARD_PATH
            ),
            html=True,
        ),
        name="dashboard",
    )