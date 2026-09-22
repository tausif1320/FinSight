from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st


# =============================================================================
# CONFIG
# =============================================================================

st.set_page_config(
    page_title="FinSight",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)

API_URL = "http://127.0.0.1:8000"

DATA_PATH = Path(
    "data/processed/modeling_dataset_1000_users.csv"
)

METADATA_PATH = Path(
    "models/feature_metadata.json"
)


# =============================================================================
# COLORS
# =============================================================================

BG = "#070B14"
SURFACE = "#0D1422"
SURFACE_2 = "#111B2C"
BORDER = "#1D2B42"

WHITE = "#F5F7FB"
TEXT = "#C7D0DE"
MUTED = "#7F8EA6"

BLUE = "#4F8CFF"
CYAN = "#36D9FF"
GREEN = "#35D07F"
RED = "#FF5C7A"
AMBER = "#FFB547"
PURPLE = "#9B7BFF"


# =============================================================================
# GLOBAL CSS
# =============================================================================

st.markdown(
    """
<style>

/* ============================================================
   PAGE
   ============================================================ */

.stApp {
    background:
        radial-gradient(
            circle at 85% 0%,
            rgba(79,140,255,0.10),
            transparent 30%
        ),
        radial-gradient(
            circle at 10% 20%,
            rgba(54,217,255,0.045),
            transparent 25%
        ),
        #070B14;
}

.block-container {
    max-width: 1500px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}


/* ============================================================
   TYPOGRAPHY
   ============================================================ */

html, body, [class*="css"] {
    font-family:
        Inter,
        -apple-system,
        BlinkMacSystemFont,
        "Segoe UI",
        sans-serif;
}

h1 {
    font-size: 2.4rem !important;
    font-weight: 800 !important;
    letter-spacing: -0.045em !important;
}

h2 {
    font-size: 1.45rem !important;
    font-weight: 750 !important;
}

h3 {
    font-size: 1.05rem !important;
    font-weight: 750 !important;
}


/* ============================================================
   SIDEBAR
   ============================================================ */

section[data-testid="stSidebar"] {
    background: #090F1A;
    border-right: 1px solid #172338;
}

section[data-testid="stSidebar"] * {
    font-size: 0.95rem;
}


/* ============================================================
   METRICS
   ============================================================ */

[data-testid="stMetric"] {
    background:
        linear-gradient(
            145deg,
            rgba(17,27,44,0.98),
            rgba(11,18,30,0.98)
        );

    border: 1px solid #1D2B42;
    border-radius: 16px;

    padding: 22px 24px;

    min-height: 145px;

    box-shadow:
        0 12px 35px rgba(0,0,0,0.18);
}

[data-testid="stMetricLabel"] {
    color: #8291A8 !important;

    font-size: 0.78rem !important;

    font-weight: 700 !important;

    text-transform: uppercase;

    letter-spacing: 0.08em;
}

[data-testid="stMetricValue"] {
    color: #F5F7FB !important;

    font-size: 2rem !important;

    font-weight: 800 !important;

    letter-spacing: -0.035em;
}

[data-testid="stMetricDelta"] {
    font-size: 0.78rem !important;
}


/* ============================================================
   BUTTON
   ============================================================ */

.stButton > button {
    width: 100%;

    min-height: 48px;

    border-radius: 11px;

    border: 1px solid rgba(79,140,255,0.45);

    background:
        linear-gradient(
            135deg,
            #3978E8,
            #4F8CFF
        );

    color: white;

    font-size: 0.9rem;

    font-weight: 750;

    box-shadow:
        0 8px 25px rgba(79,140,255,0.18);
}

.stButton > button:hover {
    border-color: #72A4FF;
    transform: translateY(-1px);
}


/* ============================================================
   SELECTBOX
   ============================================================ */

div[data-baseweb="select"] > div {
    background: #0D1625;
    border: 1px solid #243650;
    border-radius: 10px;
    min-height: 44px;
}


/* ============================================================
   CARDS
   ============================================================ */

.fs-card {
    background:
        linear-gradient(
            145deg,
            rgba(17,27,44,0.96),
            rgba(10,17,29,0.96)
        );

    border: 1px solid #1B2A41;

    border-radius: 16px;

    padding: 22px;

    box-shadow:
        0 12px 35px rgba(0,0,0,0.17);
}

.fs-card-title {
    font-size: 0.75rem;

    color: #7787A0;

    font-weight: 800;

    letter-spacing: 0.11em;

    text-transform: uppercase;

    margin-bottom: 15px;
}

.fs-big {
    font-size: 2.1rem;

    font-weight: 800;

    color: #F5F7FB;

    letter-spacing: -0.04em;
}

.fs-small {
    color: #8392A9;

    font-size: 0.84rem;

    line-height: 1.5;
}


/* ============================================================
   BRAND
   ============================================================ */

.brand {
    display: flex;

    align-items: center;

    gap: 10px;

    font-size: 1.55rem;

    font-weight: 850;

    color: #F5F7FB;
}

.brand-mark {
    width: 34px;
    height: 34px;

    display: flex;

    align-items: center;
    justify-content: center;

    border-radius: 10px;

    background:
        linear-gradient(
            135deg,
            #4F8CFF,
            #36D9FF
        );

    color: white;

    font-weight: 900;

    box-shadow:
        0 6px 22px rgba(54,217,255,0.20);
}

.subtitle {
    margin-top: 4px;

    color: #6F8099;

    font-size: 0.82rem;
}


/* ============================================================
   STATUS
   ============================================================ */

.live {
    display: inline-flex;

    align-items: center;

    gap: 8px;

    padding: 7px 12px;

    border-radius: 999px;

    background: rgba(53,208,127,0.08);

    border: 1px solid rgba(53,208,127,0.18);

    color: #72E7A5;

    font-size: 0.75rem;

    font-weight: 750;
}

.live-dot {
    width: 7px;
    height: 7px;

    border-radius: 50%;

    background: #35D07F;

    box-shadow:
        0 0 10px rgba(53,208,127,0.8);
}


/* ============================================================
   DRIVER ROWS
   ============================================================ */

.driver-row {
    display: flex;

    align-items: center;

    justify-content: space-between;

    padding: 13px 0;

    border-bottom:
        1px solid rgba(255,255,255,0.055);

    font-size: 0.88rem;
}

.driver-name {
    color: #C8D0DD;
}

.driver-up {
    color: #FF6A85;

    font-weight: 800;
}

.driver-down {
    color: #42D889;

    font-weight: 800;
}


/* ============================================================
   FOOTER
   ============================================================ */

.footer {
    text-align: center;

    margin-top: 35px;

    color: #44536B;

    font-size: 0.72rem;
}

</style>
""",
    unsafe_allow_html=True,
)


# =============================================================================
# DATA LOADING
# =============================================================================

@st.cache_data
def load_data():

    df = pd.read_csv(DATA_PATH)

    df["month"] = pd.to_datetime(
        df["month"]
    )

    return df


@st.cache_data
def load_metadata():

    with open(
        METADATA_PATH,
        "r",
        encoding="utf-8",
    ) as file:

        return json.load(file)


df = load_data()
metadata = load_metadata()


# =============================================================================
# API CALL
# =============================================================================

def predict_with_api(row):

    payload = {
        "user_id": int(
            row["user_id"]
        ),

        "month": row[
            "month"
        ].strftime(
            "%Y-%m-%d"
        ),
    }

    for feature in metadata[
        "feature_columns"
    ]:

        value = row[feature]

        if pd.isna(value):

            payload[feature] = None

        elif hasattr(
            value,
            "item",
        ):

            payload[
                feature
            ] = value.item()

        else:

            payload[
                feature
            ] = value

    response = requests.post(
        f"{API_URL}/predict",
        json=payload,
        timeout=90,
    )

    response.raise_for_status()

    return response.json()


# =============================================================================
# SIDEBAR
# =============================================================================

with st.sidebar:

    st.markdown(
        """
        <div class="brand">
            <span class="brand-mark">F</span>
            FinSight
        </div>

        <div class="subtitle">
            Predictive Financial Intelligence
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown(
        "**ANALYSIS CONTROLS**"
    )

    selected_user = st.selectbox(
        "User",
        sorted(
            df["user_id"].unique()
        ),
    )

    user_history = df[
        df["user_id"]
        == selected_user
    ].sort_values(
        "month"
    )

    selected_month = st.selectbox(
        "Observation Month",
        user_history[
            "month"
        ].dt.strftime(
            "%Y-%m-%d"
        ).tolist(),
    )

    st.markdown("<br>", unsafe_allow_html=True)

    analyze = st.button(
        "Analyze Financial State"
    )

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown(
        """
        <div class="fs-card">

            <div class="fs-card-title">
                System
            </div>

            <div class="live">
                <span class="live-dot"></span>
                API CONNECTED
            </div>

            <div class="fs-small"
                 style="margin-top:12px;">
                Calibrated HGB · SHAP · CATE
            </div>

        </div>
        """,
        unsafe_allow_html=True,
    )


# =============================================================================
# HEADER
# =============================================================================

left, right = st.columns(
    [5, 1]
)

with left:

    st.title(
        "Financial Intelligence"
    )

    st.markdown(
        """
        **Understand behavior. Predict financial stress.
        Personalize intervention.**
        """
    )

with right:

    st.markdown(
        """
        <div style="
            text-align:right;
            padding-top:18px;
        ">
            <span class="live">
                <span class="live-dot"></span>
                LIVE ENGINE
            </span>
        </div>
        """,
        unsafe_allow_html=True,
    )


# =============================================================================
# SELECTED ROW
# =============================================================================

selected_row = user_history[
    user_history["month"]
    == pd.Timestamp(
        selected_month
    )
].iloc[0]


# =============================================================================
# PREDICTION
# =============================================================================

if (
    analyze
    or "prediction" not in st.session_state
):

    try:

        with st.spinner(
            "Running intelligence engine..."
        ):

            st.session_state[
                "prediction"
            ] = predict_with_api(
                selected_row
            )

    except Exception as exc:

        st.error(
            f"Prediction failed: {exc}"
        )

        st.stop()


result = st.session_state[
    "prediction"
]


# =============================================================================
# KPI HEADER
# =============================================================================

st.subheader(
    "Financial Overview"
)

risk = result[
    "risk_probability"
]

risk_level = result[
    "risk_level"
]

pressure = result[
    "financial_pressure"
].replace(
    "_",
    " ",
)


k1, k2, k3, k4 = st.columns(4)


with k1:

    st.metric(
        "30-DAY STRESS RISK",
        f"{risk * 100:.2f}%",
        help="Predicted probability of financial stress within the next 30 days.",
    )


with k2:

    st.metric(
        "RISK LEVEL",
        risk_level,
    )


with k3:

    st.metric(
        "FINANCIAL PRESSURE",
        pressure,
    )


with k4:

    st.metric(
        "SAVINGS RATE",
        f"{selected_row['savings_rate'] * 100:.1f}%",
    )


st.markdown("<br>", unsafe_allow_html=True)


# =============================================================================
# SECONDARY KPI ROW
# =============================================================================

k1, k2, k3, k4 = st.columns(4)


with k1:

    st.metric(
        "MONTHLY INCOME",
        f"₹{selected_row['total_income']:,.0f}",
    )


with k2:

    st.metric(
        "TOTAL SPENDING",
        f"₹{selected_row['total_debits']:,.0f}",
    )


with k3:

    st.metric(
        "CASH BUFFER",
        f"₹{selected_row['cash_buffer']:,.0f}",
    )


with k4:

    st.metric(
        "INCOME VOLATILITY",
        f"₹{selected_row['income_volatility']:,.0f}",
    )


st.markdown("<br>", unsafe_allow_html=True)


# =============================================================================
# MAIN ANALYTICS
# =============================================================================

left, right = st.columns(
    [1.55, 1]
)


# =============================================================================
# FINANCIAL TRAJECTORY
# =============================================================================

with left:

    st.markdown(
        '<div class="fs-card-title">Financial Trajectory</div>',
        unsafe_allow_html=True,
    )

    history = user_history[
        [
            "month",
            "total_income",
            "total_debits",
            "ending_balance",
        ]
    ].copy()

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=history["month"],
            y=history["total_income"],
            name="Income",
            mode="lines+markers",
            line=dict(
                color=CYAN,
                width=3,
            ),
            marker=dict(
                size=6,
            ),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=history["month"],
            y=history["total_debits"],
            name="Spending",
            mode="lines+markers",
            line=dict(
                color=RED,
                width=3,
            ),
            marker=dict(
                size=6,
            ),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=history["month"],
            y=history["ending_balance"],
            name="Balance",
            mode="lines",
            line=dict(
                color=BLUE,
                width=3,
            ),
        )
    )

    fig.update_layout(
        height=380,

        paper_bgcolor="rgba(0,0,0,0)",

        plot_bgcolor="rgba(0,0,0,0)",

        font=dict(
            color=MUTED,
            size=12,
        ),

        margin=dict(
            l=10,
            r=10,
            t=20,
            b=10,
        ),

        legend=dict(
            orientation="h",
            y=1.08,
        ),

        xaxis=dict(
            showgrid=False,
        ),

        yaxis=dict(
            showgrid=True,
            gridcolor="rgba(255,255,255,0.05)",
        ),

        hovermode="x unified",
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
    )


# =============================================================================
# DECISION INTELLIGENCE
# =============================================================================

with right:

    st.markdown(
        '<div class="fs-card-title">Decision Intelligence</div>',
        unsafe_allow_html=True,
    )

    intervention = result[
        "intervention"
    ].replace(
        "_",
        " ",
    )

    status = result[
        "recommendation_status"
    ].replace(
        "_",
        " ",
    )

    cate = result[
        "cate"
    ]

    st.markdown(
        f"""
        <div class="fs-card">

            <div class="fs-card-title">
                Recommended Intervention
            </div>

            <div class="fs-big"
                 style="
                 font-size:1.55rem;
                 color:#4F8CFF;
                 ">
                {intervention}
            </div>

            <div class="fs-small"
                 style="margin-top:7px;">
                {status}
            </div>

            <div style="
                margin-top:25px;
                display:flex;
                justify-content:space-between;
            ">

                <div>
                    <div class="fs-card-title">
                        PRIORITY
                    </div>

                    <div class="fs-big"
                         style="
                         font-size:1.35rem;
                         ">
                        {result['intervention_priority']:02d}
                    </div>
                </div>

                <div>
                    <div class="fs-card-title">
                        CATE
                    </div>

                    <div class="fs-big"
                         style="
                         font-size:1.35rem;
                         ">
                        {
                            "N/A"
                            if cate is None
                            else f"{cate:.3f}"
                        }
                    </div>
                </div>

            </div>

        </div>
        """,
        unsafe_allow_html=True,
    )


# =============================================================================
# MODEL EXPLANATION
# =============================================================================

st.markdown("<br>", unsafe_allow_html=True)

left, right = st.columns(
    [1.1, 1]
)


with left:

    st.markdown(
        '<div class="fs-card-title">Model Drivers</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="fs-card">
        """,
        unsafe_allow_html=True,
    )

    drivers = result[
        "model_drivers"
    ].split(
        " | "
    )

    for driver in drivers:

        increases = (
            "increases"
            in driver.lower()
        )

        arrow = "↑" if increases else "↓"

        css = (
            "driver-up"
            if increases
            else "driver-down"
        )

        st.markdown(
            f"""
            <div class="driver-row">

                <span class="driver-name">
                    {driver.replace(
                        "increases predicted stress risk",
                        ""
                    ).replace(
                        "reduces predicted stress risk",
                        ""
                    ).strip()}
                </span>

                <span class="{css}">
                    {arrow}
                </span>

            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown(
        """
        <div class="fs-small"
             style="margin-top:14px;">
            Local SHAP attribution. These values explain
            model behavior and are not causal estimates.
        </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# =============================================================================
# BEHAVIORAL SIGNALS
# =============================================================================

with right:

    st.markdown(
        '<div class="fs-card-title">Behavioral Signals</div>',
        unsafe_allow_html=True,
    )

    b1, b2 = st.columns(2)

    with b1:

        st.metric(
            "Expense Pressure",
            f"{selected_row['expense_pressure']:.2f}x",
        )

        st.metric(
            "Cash Buffer Ratio",
            f"{selected_row['cash_buffer_ratio']:.2f}x",
        )

    with b2:

        st.metric(
            "Spending Growth",
            f"{selected_row['spending_growth'] * 100:+.1f}%",
        )

        st.metric(
            "Transaction Activity",
            f"{selected_row['transaction_count']:.0f}",
        )


# =============================================================================
# EXPLANATION
# =============================================================================

st.markdown("<br>", unsafe_allow_html=True)

st.markdown(
    '<div class="fs-card-title">Decision Context</div>',
    unsafe_allow_html=True,
)

st.info(
    result[
        "explanation"
    ]
)


# =============================================================================
# FOOTER
# =============================================================================

st.markdown(
    """
    <div class="footer">
        FinSight · Predictive Financial Behavior Intelligence
        <br>
        Calibrated HGB · SHAP · CATE · Synthetic experimentation
    </div>
    """,
    unsafe_allow_html=True,
)