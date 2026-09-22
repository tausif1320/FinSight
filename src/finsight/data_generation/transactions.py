import numpy as np
import pandas as pd


VARIABLE_CATEGORIES = [
    "Food",
    "Groceries",
    "Transport",
    "Shopping",
    "Entertainment",
    "Travel",
    "Healthcare",
    "Education",
    "Other",
]


PAYMENT_METHODS = [
    "UPI",
    "Debit_Card",
    "Credit_Card",
    "Bank_Transfer",
]


DISCRETIONARY_CATEGORIES = {
    "Shopping",
    "Entertainment",
    "Travel",
    "Other",
}


SHOCK_CATEGORY_MAP = {
    "MEDICAL": "Healthcare",
    "TRAVEL": "Travel",
    "LARGE_PURCHASE": "Shopping",
    "EMERGENCY": "Other",
    "UNEXPECTED_BILL": "Other",
}


# Relative spending weights used to allocate a user's
# monthly variable-spending budget across categories.
BASE_CATEGORY_WEIGHTS = {
    "Food": 1.00,
    "Groceries": 0.80,
    "Transport": 0.70,
    "Shopping": 0.60,
    "Entertainment": 0.45,
    "Travel": 0.20,
    "Healthcare": 0.15,
    "Education": 0.10,
    "Other": 0.35,
}


# ============================================================
# PAYMENT METHOD
# ============================================================


def _choose_payment_method(
    category: str,
    rng: np.random.Generator,
) -> str:
    """Select a plausible payment method."""

    if category in {
        "Rent",
        "EMI",
        "Utility",
        "Other_Recurring",
    }:
        methods = [
            "Bank_Transfer",
            "UPI",
        ]

        probabilities = [
            0.65,
            0.35,
        ]

    elif category == "Subscription":

        methods = [
            "Credit_Card",
            "Debit_Card",
            "UPI",
        ]

        probabilities = [
            0.45,
            0.30,
            0.25,
        ]

    else:

        methods = PAYMENT_METHODS

        probabilities = [
            0.65,
            0.15,
            0.15,
            0.05,
        ]

    return rng.choice(
        methods,
        p=probabilities,
    )


# ============================================================
# DAILY TRANSACTION COUNT
# ============================================================


def _generate_daily_transaction_count(
    expected_transactions: float,
    weekend_multiplier: float,
    date: pd.Timestamp,
    rng: np.random.Generator,
    transaction_multiplier: float = 1.0,
) -> int:
    """Generate the number of variable transactions for a day."""

    multiplier = (
        weekend_multiplier
        if date.weekday() >= 5
        else 1.0
    )

    expected = (
        expected_transactions
        * multiplier
        * transaction_multiplier
    )

    return int(
        rng.poisson(
            max(expected, 0.1)
        )
    )


# ============================================================
# CATEGORY WEIGHTS
# ============================================================


def _get_category_weights(
    spending_profile: pd.Series,
    discretionary_multiplier: float = 1.0,
) -> dict:
    """
    Build user-specific category weights.

    The generated spending profile influences the relative
    frequency of discretionary categories.

    Behavioral regime effects can additionally modify
    discretionary categories.
    """

    spending_propensity = float(
        spending_profile["spending_propensity"]
    )

    savings_propensity = float(
        spending_profile["savings_propensity"]
    )

    weights = BASE_CATEGORY_WEIGHTS.copy()

    # Higher spending propensity increases discretionary
    # categories.
    weights["Shopping"] *= (
        0.70
        + 1.50 * spending_propensity
    )

    weights["Entertainment"] *= (
        0.70
        + 1.50 * spending_propensity
    )

    weights["Travel"] *= (
        0.70
        + 1.30 * spending_propensity
    )

    # Higher savings propensity shifts behavior toward
    # necessities.
    weights["Groceries"] *= (
        0.80
        + 0.50 * savings_propensity
    )

    weights["Education"] *= (
        0.80
        + 0.50 * savings_propensity
    )

    # Behavioral regime effect.
    for category in DISCRETIONARY_CATEGORIES:
        weights[category] *= discretionary_multiplier

    return weights


# ============================================================
# VARIABLE CATEGORY
# ============================================================


def _choose_variable_category(
    spending_profile: pd.Series,
    rng: np.random.Generator,
    discretionary_multiplier: float = 1.0,
) -> str:
    """Choose a spending category for a variable transaction."""

    probability_columns = [
        "food_probability",
        "groceries_probability",
        "transport_probability",
        "shopping_probability",
        "entertainment_probability",
        "travel_probability",
        "healthcare_probability",
        "education_probability",
        "other_probability",
    ]

    probabilities = np.array(
        [
            spending_profile[column]
            for column in probability_columns
        ],
        dtype=float,
    )

    category_weights = _get_category_weights(
        spending_profile,
        discretionary_multiplier=(
            discretionary_multiplier
        ),
    )

    categories = VARIABLE_CATEGORIES

    for index, category in enumerate(categories):
        probabilities[index] *= (
            category_weights[category]
        )

    probability_sum = probabilities.sum()

    if probability_sum <= 0:
        raise ValueError(
            "Category probabilities must sum to a positive value."
        )

    probabilities /= probability_sum

    return rng.choice(
        categories,
        p=probabilities,
    )


# ============================================================
# MONTHLY SPENDING BUDGET
# ============================================================


def _calculate_monthly_spending_budget(
    monthly_income: float,
    recurring_obligations: float,
    spending_propensity: float,
    savings_propensity: float,
    essential_ratio: float,
    discretionary_ratio: float,
    rng: np.random.Generator,
    spending_multiplier: float = 1.0,
) -> float:
    """
    Calculate a financially coherent variable-spending budget.

    Behavioral regime effects modify the resulting budget.
    """

    disposable_income = max(
        monthly_income
        - recurring_obligations,
        0.0,
    )

    # Base spending tendency.
    base_rate = (
        0.35
        + 0.40 * spending_propensity
        - 0.15 * savings_propensity
    )

    behavioral_adjustment = (
        0.10 * essential_ratio
        + 0.15 * discretionary_ratio
    )

    spending_rate = (
        base_rate
        + behavioral_adjustment
    )

    # Small user-month variation.
    spending_rate += rng.normal(
        0.0,
        0.04,
    )

    # Apply behavioral regime.
    spending_rate *= spending_multiplier

    # Prevent pathological synthetic spending.
    spending_rate = np.clip(
        spending_rate,
        0.30,
        1.00,
    )

    budget = (
        disposable_income
        * spending_rate
    )

    if disposable_income <= 0:
        budget = 0.0

    return round(
        max(float(budget), 0.0),
        2,
    )


# ============================================================
# DAILY BUDGET
# ============================================================


def _allocate_daily_budget(
    monthly_budget: float,
    current_date: pd.Timestamp,
    days_in_month: int,
    weekend_multiplier: float,
) -> float:
    """Estimate the spending budget available for a particular day."""

    if monthly_budget <= 0:
        return 0.0

    base_daily_budget = (
        monthly_budget
        / days_in_month
    )

    if current_date.weekday() >= 5:
        return (
            base_daily_budget
            * weekend_multiplier
        )

    return base_daily_budget


# ============================================================
# TRANSACTION AMOUNT
# ============================================================


def _generate_transaction_amount(
    category: str,
    daily_budget: float,
    daily_transaction_count: int,
    rng: np.random.Generator,
) -> float:
    """
    Generate a transaction amount around the available daily
    spending budget.
    """

    if daily_budget <= 0:
        return 0.0

    if daily_transaction_count <= 0:
        return 0.0

    category_multipliers = {
        "Food": 0.90,
        "Groceries": 1.50,
        "Transport": 0.60,
        "Shopping": 1.40,
        "Entertainment": 1.00,
        "Travel": 3.00,
        "Healthcare": 2.00,
        "Education": 2.00,
        "Other": 0.80,
    }

    multiplier = category_multipliers[
        category
    ]

    target_amount = (
        daily_budget
        / daily_transaction_count
    )

    target_amount *= (
        multiplier
        / 1.25
    )

    noise = rng.lognormal(
        mean=0.0,
        sigma=0.45,
    )

    amount = (
        target_amount
        * noise
    )

    minimum_amount = 20.0

    maximum_amount = max(
        daily_budget * 2.5,
        minimum_amount,
    )

    amount = np.clip(
        amount,
        minimum_amount,
        maximum_amount,
    )

    return round(
        float(amount),
        2,
    )


# ============================================================
# SALARY
# ============================================================


def _salary_transaction(
    user: pd.Series,
    income_row: pd.Series,
    timestamp: pd.Timestamp,
    salary_multiplier: float = 1.0,
) -> dict:
    """Create a salary transaction."""

    salary = (
        float(income_row["monthly_salary"])
        * salary_multiplier
    )

    return {
        "user_id": user["user_id"],
        "timestamp": timestamp,
        "transaction_type": "CREDIT",
        "merchant_category": "SALARY",
        "amount": round(
            max(salary, 0.0),
            2,
        ),
        "payment_method": "Bank_Transfer",
    }


# ============================================================
# BONUS
# ============================================================


def _bonus_transaction(
    user: pd.Series,
    income_row: pd.Series,
    timestamp: pd.Timestamp,
) -> dict | None:
    """Create a bonus transaction when applicable."""

    bonus = float(
        income_row["bonus"]
    )

    if bonus <= 0:
        return None

    return {
        "user_id": user["user_id"],
        "timestamp": timestamp,
        "transaction_type": "CREDIT",
        "merchant_category": "BONUS",
        "amount": round(
            bonus,
            2,
        ),
        "payment_method": "Bank_Transfer",
    }


# ============================================================
# RECURRING TRANSACTION
# ============================================================


def _recurring_transaction(
    user: pd.Series,
    category: str,
    amount: float,
    timestamp: pd.Timestamp,
    rng: np.random.Generator,
) -> dict:
    """Create a recurring debit transaction."""

    return {
        "user_id": user["user_id"],
        "timestamp": timestamp,
        "transaction_type": "DEBIT",
        "merchant_category": category,
        "amount": float(
            max(amount, 0.0)
        ),
        "payment_method": _choose_payment_method(
            category,
            rng,
        ),
    }


# ============================================================
# SHOCK TRANSACTION
# ============================================================


def _shock_transaction(
    user_id: str,
    shock_type: str,
    amount: float,
    timestamp: pd.Timestamp,
    rng: np.random.Generator,
) -> dict:
    """Create a debit transaction caused by an external shock."""

    if shock_type not in SHOCK_CATEGORY_MAP:
        raise ValueError(
            "Unsupported expense shock type: "
            f"{shock_type}"
        )

    category = SHOCK_CATEGORY_MAP[
        shock_type
    ]

    return {
        "user_id": user_id,
        "timestamp": timestamp,
        "transaction_type": "DEBIT",
        "merchant_category": category,
        "amount": round(
            max(float(amount), 0.0),
            2,
        ),
        "payment_method": _choose_payment_method(
            category,
            rng,
        ),
    }


# ============================================================
# BEHAVIOR LOOKUP
# ============================================================


def _prepare_behavior_lookup(
    behavior_df: pd.DataFrame | None,
) -> dict:

    if behavior_df is None:
        return {}

    required_columns = {
        "user_id",
        "month",
        "behavioral_regime",
        "spending_multiplier",
        "discretionary_multiplier",
        "transaction_multiplier",
    }

    missing = (
        required_columns
        - set(behavior_df.columns)
    )

    if missing:
        raise ValueError(
            "behavior_df is missing columns: "
            f"{sorted(missing)}"
        )

    if behavior_df[
        ["user_id", "month"]
    ].duplicated().any():
        raise ValueError(
            "behavior_df contains duplicate "
            "user-month observations."
        )

    lookup = {}

    for _, row in behavior_df.iterrows():

        key = (
            row["user_id"],
            int(row["month"]),
        )

        lookup[key] = {
            "behavioral_regime": (
                row["behavioral_regime"]
            ),
            "spending_multiplier": float(
                row["spending_multiplier"]
            ),
            "discretionary_multiplier": float(
                row["discretionary_multiplier"]
            ),
            "transaction_multiplier": float(
                row["transaction_multiplier"]
            ),
        }

    return lookup


# ============================================================
# SHOCK LOOKUP
# ============================================================


def _prepare_shock_lookup(
    shock_df: pd.DataFrame | None,
) -> dict:

    if shock_df is None:
        return {}

    required_columns = {
        "user_id",
        "month",
        "shock_type",
        "shock_amount",
        "income_impact",
        "duration_months",
    }

    missing = (
        required_columns
        - set(shock_df.columns)
    )

    if missing:
        raise ValueError(
            "shock_df is missing columns: "
            f"{sorted(missing)}"
        )

    if shock_df[
        ["user_id", "month"]
    ].duplicated().any():
        raise ValueError(
            "shock_df contains duplicate "
            "user-month observations."
        )

    lookup = {}

    for _, row in shock_df.iterrows():

        key = (
            row["user_id"],
            int(row["month"]),
        )

        lookup[key] = {
            "shock_type": row["shock_type"],
            "shock_amount": float(
                row["shock_amount"]
            ),
            "income_impact": float(
                row["income_impact"]
            ),
            "duration_months": int(
                row["duration_months"]
            ),
        }

    return lookup


# ============================================================
# ACTIVE INCOME DISRUPTION
# ============================================================


def _get_income_disruption_multiplier(
    user_id: str,
    month: int,
    shock_lookup: dict,
) -> float:
    """
    Determine whether an income disruption from an earlier
    month is still active.
    """

    multiplier = 1.0

    for (
        key_user_id,
        shock_month,
    ), shock in shock_lookup.items():

        if key_user_id != user_id:
            continue

        if shock["shock_type"] != "INCOME_DISRUPTION":
            continue

        duration = shock["duration_months"]

        if (
            shock_month
            <= month
            <= shock_month + duration - 1
        ):

            multiplier *= (
                1.0
                - shock["income_impact"]
            )

    return float(
        np.clip(
            multiplier,
            0.0,
            1.0,
        )
    )


# ============================================================
# MAIN TRANSACTION GENERATOR
# ============================================================


def generate_transactions(
    population: pd.DataFrame,
    income_df: pd.DataFrame,
    obligations_df: pd.DataFrame,
    spending_df: pd.DataFrame,
    initial_balance_df: pd.DataFrame,
    n_months: int,
    rng: np.random.Generator,
    start_date: str = "2026-01-01",
    behavior_df: pd.DataFrame | None = None,
    shock_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Generate a transaction-level financial ledger.

    The generator combines:

    - income
    - recurring obligations
    - behavioral spending propensity
    - behavioral regimes
    - category preferences
    - calendar effects
    - external financial shocks
    - initial liquidity

    Balance is calculated sequentially and never generated
    independently.

    behavior_df and shock_df are optional so the original
    baseline generator remains compatible.
    """

    if n_months <= 0:
        raise ValueError(
            "n_months must be greater than 0"
        )

    # ========================================================
    # REQUIRED INPUT COLUMNS
    # ========================================================

    required_columns = {
        "population": {
            "user_id",
            "salary_day",
        },

        "income_df": {
            "user_id",
            "month",
            "monthly_salary",
            "bonus",
        },

        "obligations_df": {
            "user_id",
            "rent",
            "emi",
            "utilities",
            "subscriptions",
            "other_recurring",
        },

        "spending_df": {
            "user_id",
            "spending_propensity",
            "savings_propensity",
            "essential_ratio",
            "discretionary_ratio",
            "expected_daily_transactions",
            "weekend_multiplier",
        },

        "initial_balance_df": {
            "user_id",
            "initial_balance",
        },
    }

    dataframes = {
        "population": population,
        "income_df": income_df,
        "obligations_df": obligations_df,
        "spending_df": spending_df,
        "initial_balance_df": initial_balance_df,
    }

    for name, required in required_columns.items():

        missing = (
            required
            - set(dataframes[name].columns)
        )

        if missing:
            raise ValueError(
                f"{name} is missing columns: "
                f"{sorted(missing)}"
            )

    # ========================================================
    # PREPARE BEHAVIOR / SHOCK LOOKUPS
    # ========================================================

    behavior_lookup = (
        _prepare_behavior_lookup(
            behavior_df
        )
    )

    shock_lookup = (
        _prepare_shock_lookup(
            shock_df
        )
    )

    # ========================================================
    # DATE SETUP
    # ========================================================

    start = pd.Timestamp(
        start_date
    )

    all_records = []

    obligation_due_days = {
        "rent": 1,
        "emi": 5,
        "utilities": 10,
        "subscriptions": 15,
        "other_recurring": 20,
    }

    category_map = {
        "rent": "Rent",
        "emi": "EMI",
        "utilities": "Utility",
        "subscriptions": "Subscription",
        "other_recurring": "Other_Recurring",
    }

    # ========================================================
    # USER LOOP
    # ========================================================

    for _, user in population.iterrows():

        user_id = user["user_id"]

        user_income = (
            income_df[
                income_df["user_id"]
                == user_id
            ]
            .sort_values("month")
        )

        user_obligations = (
            obligations_df[
                obligations_df["user_id"]
                == user_id
            ]
            .iloc[0]
        )

        user_spending = (
            spending_df[
                spending_df["user_id"]
                == user_id
            ]
            .iloc[0]
        )

        # ====================================================
        # MONTH LOOP
        # ====================================================

        for month in range(
            1,
            n_months + 1,
        ):

            month_start = (
                start
                + pd.DateOffset(
                    months=month - 1
                )
            )

            # ------------------------------------------------
            # Income
            # ------------------------------------------------

            income_row = user_income[
                user_income["month"]
                == month
            ]

            if income_row.empty:
                raise ValueError(
                    f"Missing income for "
                    f"{user_id}, month {month}"
                )

            income_row = income_row.iloc[0]

            # ------------------------------------------------
            # Behavioral regime
            # ------------------------------------------------

            behavior = behavior_lookup.get(
                (user_id, month),
                {
                    "behavioral_regime": "STABLE",
                    "spending_multiplier": 1.0,
                    "discretionary_multiplier": 1.0,
                    "transaction_multiplier": 1.0,
                },
            )

            spending_multiplier = float(
                behavior[
                    "spending_multiplier"
                ]
            )

            discretionary_multiplier = float(
                behavior[
                    "discretionary_multiplier"
                ]
            )

            transaction_multiplier = float(
                behavior[
                    "transaction_multiplier"
                ]
            )

            # ------------------------------------------------
            # Shock information
            # ------------------------------------------------

            current_shock = shock_lookup.get(
                (user_id, month)
            )

            # ------------------------------------------------
            # Income disruption
            # ------------------------------------------------

            income_disruption_multiplier = (
                _get_income_disruption_multiplier(
                    user_id=user_id,
                    month=month,
                    shock_lookup=shock_lookup,
                )
            )

            effective_salary = (
                float(
                    income_row[
                        "monthly_salary"
                    ]
                )
                * income_disruption_multiplier
            )

            # Bonus is not automatically disrupted.
            effective_bonus = float(
                income_row["bonus"]
            )

            monthly_income = (
                effective_salary
                + effective_bonus
            )

            # ------------------------------------------------
            # Recurring obligations
            # ------------------------------------------------

            recurring_obligations = float(
                user_obligations["rent"]
                + user_obligations["emi"]
                + user_obligations["utilities"]
                + user_obligations["subscriptions"]
                + user_obligations["other_recurring"]
            )

            # ------------------------------------------------
            # Monthly variable spending budget
            # ------------------------------------------------

            monthly_budget = (
                _calculate_monthly_spending_budget(
                    monthly_income=monthly_income,
                    recurring_obligations=(
                        recurring_obligations
                    ),
                    spending_propensity=float(
                        user_spending[
                            "spending_propensity"
                        ]
                    ),
                    savings_propensity=float(
                        user_spending[
                            "savings_propensity"
                        ]
                    ),
                    essential_ratio=float(
                        user_spending[
                            "essential_ratio"
                        ]
                    ),
                    discretionary_ratio=float(
                        user_spending[
                            "discretionary_ratio"
                        ]
                    ),
                    rng=rng,
                    spending_multiplier=(
                        spending_multiplier
                    ),
                )
            )

            days_in_month = (
                month_start
                + pd.offsets.MonthEnd(0)
            ).day

            # =================================================
            # SALARY
            # =================================================

            salary_day = min(
                int(user["salary_day"]),
                days_in_month,
            )

            salary_timestamp = (
                month_start
                + pd.Timedelta(
                    days=salary_day - 1
                )
                + pd.Timedelta(
                    hours=int(
                        rng.integers(
                            8,
                            11,
                        )
                    )
                )
                + pd.Timedelta(
                    minutes=int(
                        rng.integers(
                            0,
                            60,
                        )
                    )
                )
            )

            all_records.append(
                _salary_transaction(
                    user=user,
                    income_row=income_row,
                    timestamp=salary_timestamp,
                    salary_multiplier=(
                        income_disruption_multiplier
                    ),
                )
            )

            # =================================================
            # BONUS
            # =================================================

            # Only generate the bonus if it exists.
            if effective_bonus > 0:

                bonus_record = {
                    "user_id": user_id,
                    "timestamp": (
                        salary_timestamp
                        + pd.Timedelta(
                            hours=int(
                                rng.integers(
                                    1,
                                    6,
                                )
                            )
                        )
                    ),
                    "transaction_type": "CREDIT",
                    "merchant_category": "BONUS",
                    "amount": round(
                        effective_bonus,
                        2,
                    ),
                    "payment_method": "Bank_Transfer",
                }

                all_records.append(
                    bonus_record
                )

            # =================================================
            # RECURRING OBLIGATIONS
            # =================================================

            for (
                column,
                due_day,
            ) in obligation_due_days.items():

                amount = float(
                    user_obligations[column]
                )

                if amount <= 0:
                    continue

                due_day = min(
                    due_day,
                    days_in_month,
                )

                timestamp = (
                    month_start
                    + pd.Timedelta(
                        days=due_day - 1
                    )
                    + pd.Timedelta(
                        hours=int(
                            rng.integers(
                                9,
                                18,
                            )
                        )
                    )
                    + pd.Timedelta(
                        minutes=int(
                            rng.integers(
                                0,
                                60,
                            )
                        )
                    )
                )

                all_records.append(
                    _recurring_transaction(
                        user=user,
                        category=category_map[
                            column
                        ],
                        amount=amount,
                        timestamp=timestamp,
                        rng=rng,
                    )
                )

            # =================================================
            # VARIABLE DAILY SPENDING
            # =================================================

            for day in range(
                days_in_month
            ):

                current_date = (
                    month_start
                    + pd.Timedelta(
                        days=day
                    )
                )

                daily_count = (
                    _generate_daily_transaction_count(
                        expected_transactions=float(
                            user_spending[
                                "expected_daily_transactions"
                            ]
                        ),
                        weekend_multiplier=float(
                            user_spending[
                                "weekend_multiplier"
                            ]
                        ),
                        date=current_date,
                        rng=rng,
                        transaction_multiplier=(
                            transaction_multiplier
                        ),
                    )
                )

                if daily_count <= 0:
                    continue

                daily_budget = (
                    _allocate_daily_budget(
                        monthly_budget=monthly_budget,
                        current_date=current_date,
                        days_in_month=days_in_month,
                        weekend_multiplier=float(
                            user_spending[
                                "weekend_multiplier"
                            ]
                        ),
                    )
                )

                if daily_budget <= 0:
                    continue

                # Randomly distribute daily budget between
                # transactions.
                raw_amounts = rng.lognormal(
                    mean=0.0,
                    sigma=0.45,
                    size=daily_count,
                )

                raw_amounts = (
                    raw_amounts
                    / raw_amounts.sum()
                )

                daily_amounts = (
                    raw_amounts
                    * daily_budget
                )

                for transaction_index in range(
                    daily_count
                ):

                    category = (
                        _choose_variable_category(
                            spending_profile=user_spending,
                            rng=rng,
                            discretionary_multiplier=(
                                discretionary_multiplier
                            ),
                        )
                    )

                    amount = float(
                        daily_amounts[
                            transaction_index
                        ]
                    )

                    if amount < 20:
                        amount = 20.0

                    hour = int(
                        rng.integers(
                            8,
                            23,
                        )
                    )

                    minute = int(
                        rng.integers(
                            0,
                            60,
                        )
                    )

                    timestamp = (
                        current_date
                        + pd.Timedelta(
                            hours=hour
                        )
                        + pd.Timedelta(
                            minutes=minute
                        )
                    )

                    all_records.append(
                        {
                            "user_id": user_id,
                            "timestamp": timestamp,
                            "transaction_type": "DEBIT",
                            "merchant_category": category,
                            "amount": round(
                                amount,
                                2,
                            ),
                            "payment_method": (
                                _choose_payment_method(
                                    category,
                                    rng,
                                )
                            ),
                        }
                    )

            # =================================================
            # EXTERNAL SHOCK
            # =================================================

            if (
                current_shock is not None
                and current_shock[
                    "shock_type"
                ] != "INCOME_DISRUPTION"
            ):

                shock_amount = float(
                    current_shock[
                        "shock_amount"
                    ]
                )

                if shock_amount > 0:

                    # Place shock after the middle of the
                    # month so it can affect the remaining
                    # liquidity trajectory.
                    shock_day = int(
                        rng.integers(
                            10,
                            max(
                                11,
                                days_in_month - 2,
                            ),
                        )
                    )

                    shock_timestamp = (
                        month_start
                        + pd.Timedelta(
                            days=shock_day - 1
                        )
                        + pd.Timedelta(
                            hours=int(
                                rng.integers(
                                    9,
                                    22,
                                )
                            )
                        )
                        + pd.Timedelta(
                            minutes=int(
                                rng.integers(
                                    0,
                                    60,
                                )
                            )
                        )
                    )

                    all_records.append(
                        _shock_transaction(
                            user_id=user_id,
                            shock_type=(
                                current_shock[
                                    "shock_type"
                                ]
                            ),
                            amount=shock_amount,
                            timestamp=shock_timestamp,
                            rng=rng,
                        )
                    )

    # ========================================================
    # CREATE DATAFRAME
    # ========================================================

    transactions = pd.DataFrame(
        all_records
    )

    if transactions.empty:
        raise ValueError(
            "No transactions were generated."
        )

    # ========================================================
    # SORT TRANSACTIONS
    # ========================================================

    transactions = transactions.sort_values(
        [
            "user_id",
            "timestamp",
        ],
        kind="mergesort",
    ).reset_index(drop=True)

    # ========================================================
    # TRANSACTION IDS
    # ========================================================

    transactions.insert(
        0,
        "transaction_id",
        [
            f"TX{i:08d}"
            for i in range(
                1,
                len(transactions) + 1,
            )
        ],
    )

    # Deterministic tie-breaking when multiple transactions
    # occur at the same timestamp.
    transactions = transactions.sort_values(
        [
            "user_id",
            "timestamp",
            "transaction_id",
        ],
        kind="mergesort",
    ).reset_index(drop=True)

    # ========================================================
    # CALCULATE BALANCE SEQUENTIALLY
    # ========================================================

    initial_balances = dict(
        zip(
            initial_balance_df["user_id"],
            initial_balance_df[
                "initial_balance"
            ],
        )
    )

    balances_before = []
    balances_after = []

    current_user = None
    current_balance = None

    for _, row in transactions.iterrows():

        if row["user_id"] != current_user:

            current_user = row["user_id"]

            if current_user not in initial_balances:
                raise ValueError(
                    "Missing initial balance for "
                    f"user {current_user}"
                )

            current_balance = float(
                initial_balances[
                    current_user
                ]
            )

        balance_before = current_balance

        if row["transaction_type"] == "CREDIT":

            current_balance += float(
                row["amount"]
            )

        elif row["transaction_type"] == "DEBIT":

            current_balance -= float(
                row["amount"]
            )

        else:

            raise ValueError(
                "Invalid transaction type: "
                f"{row['transaction_type']}"
            )

        balance_after = current_balance

        balances_before.append(
            round(
                balance_before,
                2,
            )
        )

        balances_after.append(
            round(
                balance_after,
                2,
            )
        )

    transactions[
        "balance_before"
    ] = balances_before

    transactions[
        "balance_after"
    ] = balances_after

    return transactions
