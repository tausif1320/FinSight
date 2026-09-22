# FinSight

### Predictive Financial Behavior Intelligence Engine

> **What if a financial system could detect behavioral deterioration before it becomes financial stress, explain why, recommend an intervention, and estimate whether that intervention could help?**
>
> **FinSight was built to explore exactly that.**

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python\&logoColor=white)](#)
[![ML](https://img.shields.io/badge/ML-Calibrated%20HGB-orange)](#)
[![Explainability](https://img.shields.io/badge/Explainability-SHAP-red)](#)
[![Causal Inference](https://img.shields.io/badge/Causal%20Inference-CATE-purple)](#)
[![API](https://img.shields.io/badge/API-FastAPI-009688?logo=fastapi\&logoColor=white)](#)

---

## The Problem

Most personal-finance systems answer:

> **"Where did your money go?"**

FinSight asks:

> **"What is changing, what is likely to happen next, why is the system concerned, and what intervention should be considered?"**

It models personal finance as a **dynamic behavioral system**, combining:

**Temporal ML + behavioral modeling + calibration + SHAP + experimentation + CATE + decision intelligence**

---

## Why It Is Different

```text
Traditional finance analytics
Transactions → Categorization → Dashboard

FinSight
Transactions
     ↓
Behavioral change
     ↓
30-day stress prediction
     ↓
Pressure diagnosis
     ↓
Personalized intervention
     ↓
Experimental treatment-effect estimation
     ↓
Explainable decision
```

The goal is not simply to predict risk. It is to connect:

> **Prediction → Explanation → Intervention → Evaluation**

---

## Project at a Glance

| Metric                             |                              Result |
| ---------------------------------- | ----------------------------------: |
| Synthetic users                    |                           **1,000** |
| Generated history                  |                       **12 months** |
| Modeling observations              |                           **8,996** |
| Production features                |                              **53** |
| Historical window                  |                         **90 days** |
| Prediction horizon                 |                         **30 days** |
| Final model                        | **Calibrated HistGradientBoosting** |
| Test PR-AUC                        |                          **0.8322** |
| Test ROC-AUC                       |                          **0.9959** |
| Test Brier Score                   |                         **0.00356** |
| SHAP                               |     **Global + local explanations** |
| Intervention-eligible observations |                             **300** |
| CATE models                        |                               **3** |
| API                                |                         **FastAPI** |
| Dashboard                          |      **HTML / CSS / JS / Chart.js** |

> **Data note:** The current system uses synthetic financial data. Intervention and CATE results demonstrate the methodology and engineering pipeline, not real-world customer effectiveness.

---

# Core Pipeline

```text
Financial Data
      ↓
Temporal + Behavioral Features
      ↓
Calibrated HGB
      ↓
30-Day Stress Probability
      ↓
Financial Pressure Classification
      ↓
Intervention Selection
      ↓
CATE Estimation
      ↓
SHAP Explanation
      ↓
Decision Engine
      ↓
FastAPI
      ↓
Interactive Dashboard
```

---

# 1. Dynamic Behavioral Modeling

FinSight uses **53 production features** rather than relying only on monthly aggregates.

### Static signals

```text
income
total spending
essential spending
discretionary spending
recurring spending
average balance
net cash flow
savings rate
income-to-expense ratio
```

### Dynamic signals

```text
spending baseline deviation
spending growth
income growth
balance growth
cash-buffer trajectory
income volatility
transaction-count change
savings-rate change
spending acceleration
category shift
financial-state changes
```

The historical feature window is **90 days**, followed by a **30-day prediction horizon**.

```text
Past 90 days
     ↓
Feature construction
     ↓
Prediction date
     ↓
Next 30 days
     ↓
Outcome
```

Future outcome information is excluded from model predictors to prevent temporal leakage.

---

# 2. Financial Stress Target

`financial_stress_30d` is triggered when at least one condition occurs:

### Critical liquidity

```text
minimum_future_balance <= 0
```

### Obligation failure

```text
obligation_coverage < 1
```

### Severe liquidity drawdown

```text
minimum_future_balance / prediction_balance < 0.25
AND
future_net_cash_flow < 0
```

Daily negative cash flow alone is not treated as financial stress because normal salary-cycle behavior can temporarily produce negative daily cash flow.

---

# 3. Static vs Dynamic Feature Experiment

FinSight explicitly tested whether behavioral features add predictive information.

| Metric      |     Static | Static + Dynamic |
| ----------- | ---------: | ---------------: |
| PR-AUC      | **0.8239** |           0.7816 |
| ROC-AUC     | **0.9969** |           0.9967 |
| Precision   |     0.2857 |       **0.3529** |
| Recall      |     0.9474 |       **0.9474** |
| F1          |     0.4390 |       **0.5143** |
| Brier Score |     0.0165 |       **0.0117** |

The result is intentionally not presented as "dynamic features always win."

They improved precision, F1 and Brier score at the evaluated threshold while PR-AUC decreased.

**The feature set was evaluated empirically rather than justified after the fact.**

---

# 4. Final ML Model

Models evaluated included:

* Dummy baseline
* Logistic Regression
* Random Forest
* HistGradientBoosting

The production pipeline uses:

```text
HistGradientBoosting
        +
Sigmoid probability calibration
        +
53 production features
```

Chronological split:

```text
Training      6,000
Validation    1,000
Test          1,996
```

The final model was evaluated on an untouched chronological test period.

### Test performance

| Metric          |      Result |
| --------------- | ----------: |
| **PR-AUC**      |  **0.8322** |
| **ROC-AUC**     |  **0.9959** |
| **Precision**   |  **0.8571** |
| **Recall**      |  **0.6316** |
| **F1**          |  **0.7273** |
| **Brier Score** | **0.00356** |

Probability calibration matters because predicted risk feeds the downstream decision engine.

> **Caveat:** The test set contains only 19 positive stress cases, so threshold-dependent metrics should be interpreted cautiously.

---

# 5. Explainability with SHAP

FinSight uses SHAP to explain individual predictions and identify global model drivers.

Top global drivers by mean absolute SHAP included:

1. `minimum_balance_ratio`
2. `rolling_average_balance`
3. `average_balance_ratio`
4. `expense_pressure`
5. `income_growth`
6. `recurring_spending`
7. `income_volatility`
8. `balance_growth`
9. `rolling_transaction_count`
10. `rolling_net_cash_flow`

Example dashboard explanation:

```text
Average balance relative to income
↓ reduces risk

Discretionary spending pressure
↑ increases risk
```

SHAP explains **model attribution**, not causality.

---

# 6. Financial Pressure → Intervention

FinSight separates predicted stress from current behavioral pressure.

| Detected pressure   | Intervention                           |
| ------------------- | -------------------------------------- |
| Liquidity pressure  | **Liquidity Protection**               |
| Spending pressure   | **Spending Control**                   |
| Obligation pressure | **Obligation Management**              |
| Stable              | **Savings Reinforcement / Monitoring** |

Example:

```text
LOW predicted stress
+
SPENDING PRESSURE
```

is a valid outcome.

It means the user's current behavior is deteriorating, but existing liquidity may prevent that behavior from becoming near-term financial stress.

This separation is central to FinSight's decision architecture.

---

# 7. Experimentation + CATE

Prediction answers:

> **What might happen?**

Experimentation asks:

> **Would an intervention change the outcome?**

FinSight includes a randomized synthetic intervention experiment.

### Experiment population

```text
1,000 validation observations
300 intervention-eligible observations
149 treatment
151 control
```

| Intervention          | Total | Treatment | Control |
| --------------------- | ----: | --------: | ------: |
| Liquidity Protection  |    82 |        41 |      41 |
| Spending Control      |   141 |        70 |      71 |
| Obligation Management |    77 |        38 |      39 |

Maximum finite standardized mean difference:

```text
0.208
```

### Estimated treatment effects

| Intervention          |         ATE |
| --------------------- | ----------: |
| Liquidity Protection  |     -0.0244 |
| Spending Control      |     -0.0258 |
| Obligation Management |     -0.1748 |
| **Overall**           | **-0.0638** |

These are **synthetic experimental results** and should not be interpreted as real-world intervention effectiveness.

Production CATE models are trained separately for:

```text
LIQUIDITY_PROTECTION
SPENDING_CONTROL
OBLIGATION_MANAGEMENT
```

The system first identifies the applicable pressure family and then evaluates the corresponding intervention effect.

---

# 8. Decision Intelligence

The final decision combines:

```text
Risk probability
      +
Financial pressure
      +
Behavioral signals
      +
Applicable intervention
      +
CATE evidence
      +
SHAP explanation
```

Example:

```text
Stress risk        0.24%
Risk level         LOW
Pressure           SPENDING_PRESSURE
Spending growth    +39.7%
Expense pressure   1.16x
Cash buffer ratio  3.27x

Intervention       SPENDING_CONTROL
CATE               -0.137
Status             STRONG_SIMULATED_SUPPORT
```

The system can therefore identify behavioral deterioration even when immediate predicted stress remains low.

---

# Dashboard

The production-style dashboard provides:

* User and month selection
* 30-day stress probability
* Risk level
* Financial pressure
* Income and spending
* Cash buffer
* Financial trajectory
* Intervention recommendation
* CATE evidence
* SHAP drivers
* Behavioral signals
* Decision context
* API health status

### Stack

```text
Frontend       HTML / CSS / JavaScript / Chart.js
Backend        FastAPI / Uvicorn
ML             scikit-learn / SHAP
Data           Pandas / NumPy
Engineering    Git / pytest / YAML
```

---

# Repository Structure

```text
FinSight/
├── api/
├── configs/
├── dashboard/
├── docs/
├── models/
├── notebooks/
├── reports/
├── scripts/
├── src/
│   └── finsight/
│       ├── data_generation/
│       ├── preprocessing/
│       ├── features/
│       ├── models/
│       ├── experimentation/
│       ├── explainability/
│       ├── decision/
│       └── inference/
├── tests/
├── .env.example
├── .gitignore
├── LICENSE
├── pyproject.toml
└── requirements.txt
```

---

# Quick Start

```powershell
git clone https://github.com/YOUR_USERNAME/FinSight.git
cd FinSight

python -m venv .venv
.\.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install shap
python -m pip install -e .
```

### Start the API

```powershell
uvicorn api.app:app --reload --port 8000
```

### Start the dashboard

Open another terminal:

```powershell
python -m http.server 5500 --directory dashboard
```

Open:

```text
http://127.0.0.1:5500
```

API documentation:

```text
http://127.0.0.1:8000/docs
```

Health check:

```text
http://127.0.0.1:8000/health
```

---

# Validation

The project includes validation for:

* Sequential transaction accounting
* Missing values
* Duplicate user-month observations
* Financial-state consistency
* Temporal leakage
* Model calibration
* Treatment/control balance
* CATE artifact loading
* Production feature schema
* End-to-end inference

Useful commands:

```powershell
python .\scripts\test_final_stress_model.py
python .\scripts\test_explanation.py
python .\scripts\test_production_cate.py
python .\scripts\test_end_to_end_inference.py
```

---

# Limitations

FinSight is currently a research/engineering prototype using synthetic financial data.

Key limitations:

* Synthetic data may not reproduce real financial behavior.
* CATE results do not establish real-world intervention effectiveness.
* Only 19 positive stress cases exist in the final test period.
* CATE estimates depend on the synthetic treatment-generating process.
* Real deployment would require privacy, security, fairness, monitoring, governance and regulatory controls.
* The system does not connect to real bank accounts.

---

# Future Work

* Real-world anonymized financial datasets
* Multi-arm intervention experiments
* Uplift modeling
* Stronger heterogeneous treatment-effect estimation
* Drift detection and model monitoring
* Fairness analysis
* Real-time transaction ingestion
* Secure financial-data connectors
* Counterfactual policy simulation
* Adaptive intervention sequencing

---

# Technical Takeaway

FinSight is intentionally more than a classifier:

```text
8,996 observations
        ↓
53 behavioral + financial features
        ↓
Calibrated ML
        ↓
30-day stress prediction
        ↓
Behavioral pressure diagnosis
        ↓
Targeted intervention
        ↓
SHAP explanation
        ↓
Randomized experimentation
        ↓
CATE estimation
        ↓
Production API + dashboard
```

> **FinSight turns financial history into an explainable decision pipeline: Predict → Explain → Intervene → Evaluate.**

---

## Limitations Matter

The strongest part of an ML project is not pretending the model is perfect.

FinSight explicitly distinguishes:

* prediction from causation
* synthetic experiments from real-world evidence
* model attribution from causal explanation
* behavioral pressure from predicted financial stress

That distinction is fundamental to building responsible ML systems.
