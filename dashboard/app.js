// ============================================================================
// FinSight Dashboard
// Production Frontend
// ============================================================================

const API_URL = "http://127.0.0.1:8000";

let userHistory = [];
let chart = null;


// ============================================================================
// DOM ELEMENTS
// ============================================================================

const userSelect =
    document.getElementById("userSelect");

const monthSelect =
    document.getElementById("monthSelect");

const analyzeButton =
    document.getElementById("analyzeButton");


// ============================================================================
// UTILITY
// ============================================================================

function safeNumber(value, fallback = 0) {

    const number = Number(value);

    if (!Number.isFinite(number)) {
        return fallback;
    }

    return number;
}


function formatCurrency(value) {

    return new Intl.NumberFormat(
        "en-IN",
        {
            style: "currency",
            currency: "INR",
            maximumFractionDigits: 0,
        }
    ).format(
        safeNumber(value)
    );
}


function formatMonth(dateString) {

    const date = new Date(
        `${dateString}T00:00:00`
    );

    if (Number.isNaN(date.getTime())) {
        return dateString;
    }

    return date.toLocaleDateString(
        "en-IN",
        {
            month: "short",
            year: "numeric",
        }
    );
}


function formatLabel(value) {

    if (
        value === null ||
        value === undefined
    ) {
        return "--";
    }

    return String(value)
        .replaceAll("_", " ")
        .replace(
            /\b\w/g,
            character =>
                character.toUpperCase()
        );
}


function formatPercentage(
    value,
    decimals = 1
) {

    return `${(
        safeNumber(value) * 100
    ).toFixed(decimals)}%`;
}


// ============================================================================
// API STATUS
// ============================================================================

async function checkAPI() {

    try {

        const response =
            await fetch(
                `${API_URL}/health`
            );

        if (!response.ok) {
            throw new Error(
                "API health check failed."
            );
        }

        const data =
            await response.json();

        updateSystemStatus(
            data.pipeline_ready &&
            data.dashboard_data_ready
        );

        return data;

    } catch (error) {

        console.error(
            "API health check failed:",
            error
        );

        updateSystemStatus(false);

        return null;
    }
}


function updateSystemStatus(
    connected
) {

    const statusElement =
        document.querySelector(
            ".system-status"
        );

    if (!statusElement) {
        return;
    }

    const dot =
        statusElement.querySelector(
            ".status-dot"
        );

    if (connected) {

        statusElement.innerHTML = `
            <span class="status-dot"></span>
            API CONNECTED
        `;

    } else {

        statusElement.innerHTML = `
            <span
                class="status-dot"
                style="background:#ff5c7a;"
            ></span>
            API OFFLINE
        `;
    }
}


// ============================================================================
// LOAD USERS
// ============================================================================

async function loadUsers() {

    try {

        const response =
            await fetch(
                `${API_URL}/api/dashboard/users`
            );

        if (!response.ok) {

            const message =
                await response.text();

            throw new Error(
                message ||
                "Unable to load users."
            );
        }

        const data =
            await response.json();

        userSelect.innerHTML = "";

        data.users.forEach(
            userId => {

                const option =
                    document.createElement(
                        "option"
                    );

                option.value =
                    String(userId);

                option.textContent =
                    `User ${userId}`;

                userSelect.appendChild(
                    option
                );
            }
        );

        if (data.users.length === 0) {

            throw new Error(
                "No users found in dataset."
            );
        }

        await loadUserHistory(
            data.users[0]
        );

    } catch (error) {

        console.error(
            "User loading failed:",
            error
        );

        showDashboardError(
            "Unable to load users from FinSight API."
        );
    }
}


// ============================================================================
// LOAD USER HISTORY
// ============================================================================

async function loadUserHistory(
    userId
) {

    try {

        monthSelect.innerHTML = `
            <option value="">
                Loading months...
            </option>
        `;

        const response =
            await fetch(
                `${API_URL}/api/dashboard/users/${userId}/history`
            );

        if (!response.ok) {

            const message =
                await response.text();

            throw new Error(
                message ||
                "Unable to load user history."
            );
        }

        const data =
            await response.json();

        userHistory =
            Array.isArray(data.history)
                ? data.history
                : [];

        if (userHistory.length === 0) {

            throw new Error(
                `No history found for user ${userId}.`
            );
        }

        populateMonths();

        const latestRow =
            userHistory[
                userHistory.length - 1
            ];

        monthSelect.value =
            latestRow.month;

        updateFinancialOverview(
            latestRow
        );

        updateBehavioralSignals(
            latestRow
        );

        updateChart();

        clearDecisionArea();

        await runPrediction(
            latestRow
        );

    } catch (error) {

        console.error(
            "User history loading failed:",
            error
        );

        showDashboardError(
            "Unable to load financial history."
        );
    }
}


// ============================================================================
// MONTH DROPDOWN
// ============================================================================

function populateMonths() {

    monthSelect.innerHTML = "";

    userHistory.forEach(
        row => {

            const option =
                document.createElement(
                    "option"
                );

            option.value =
                row.month;

            option.textContent =
                formatMonth(
                    row.month
                );

            monthSelect.appendChild(
                option
            );
        }
    );
}


// ============================================================================
// SELECTED ROW
// ============================================================================

function getSelectedRow() {

    return userHistory.find(
        row =>
            row.month ===
            monthSelect.value
    );
}


// ============================================================================
// USER CHANGE
// ============================================================================

userSelect.addEventListener(
    "change",
    async () => {

        await loadUserHistory(
            Number(
                userSelect.value
            )
        );
    }
);


// ============================================================================
// MONTH CHANGE
// ============================================================================

monthSelect.addEventListener(
    "change",
    async () => {

        const row =
            getSelectedRow();

        if (!row) {
            return;
        }

        updateFinancialOverview(
            row
        );

        updateBehavioralSignals(
            row
        );

        updateChart();

        clearDecisionArea();

        await runPrediction(
            row
        );
    }
);


// ============================================================================
// PRODUCTION FEATURES
// ============================================================================

const PRODUCTION_FEATURES = [

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
];


// ============================================================================
// BUILD PREDICTION PAYLOAD
// ============================================================================

function buildPredictionPayload(
    row
) {

    const payload = {

        user_id:
            Number(
                row.user_id
            ),

        month:
            row.month,
    };


    PRODUCTION_FEATURES.forEach(
        feature => {

            let value =
                row[feature];


            if (
                value === null ||
                value === undefined ||
                value === ""
            ) {

                payload[feature] = 0;

                return;
            }


            // Convert booleans to numeric values.
            if (
                typeof value ===
                "boolean"
            ) {

                payload[feature] =
                    value
                        ? 1
                        : 0;

                return;
            }


            const numericValue =
                Number(value);


            if (
                Number.isFinite(
                    numericValue
                )
            ) {

                payload[feature] =
                    numericValue;

            } else {

                payload[feature] = 0;
            }
        }
    );


    return payload;
}


// ============================================================================
// RUN MODEL PREDICTION
// ============================================================================

async function runPrediction(
    row
) {

    if (!row) {
        return;
    }


    setAnalyzingState(
        true
    );


    try {

        const payload =
            buildPredictionPayload(
                row
            );


        const response =
            await fetch(
                `${API_URL}/predict`,
                {
                    method: "POST",

                    headers: {
                        "Content-Type":
                            "application/json",
                    },

                    body:
                        JSON.stringify(
                            payload
                        ),
                }
            );


        if (!response.ok) {

            const errorText =
                await response.text();

            throw new Error(
                errorText ||
                "Prediction request failed."
            );
        }


        const result =
            await response.json();


        updatePrediction(
            result,
            row
        );


    } catch (error) {

        console.error(
            "Prediction failed:",
            error
        );

        showDashboardError(
            "FinSight prediction failed. Check the API terminal."
        );

    } finally {

        setAnalyzingState(
            false
        );
    }
}


// ============================================================================
// ANALYZE BUTTON
// ============================================================================

analyzeButton.addEventListener(
    "click",
    async () => {

        const row =
            getSelectedRow();

        if (!row) {

            showDashboardError(
                "Select a valid user and month."
            );

            return;
        }

        await runPrediction(
            row
        );
    }
);


// ============================================================================
// FINANCIAL OVERVIEW
// ============================================================================

function updateFinancialOverview(
    row
) {

    document.getElementById(
        "savingsRate"
    ).textContent =
        formatPercentage(
            row.savings_rate,
            1
        );


    document.getElementById(
        "income"
    ).textContent =
        formatCurrency(
            row.total_income
        );


    document.getElementById(
        "spending"
    ).textContent =
        formatCurrency(
            row.total_debits
        );


    document.getElementById(
        "cashBuffer"
    ).textContent =
        formatCurrency(
            row.cash_buffer
        );


    document.getElementById(
        "incomeVolatility"
    ).textContent =
        formatCurrency(
            row.income_volatility
        );
}


// ============================================================================
// BEHAVIORAL SIGNALS
// ============================================================================

function updateBehavioralSignals(
    row
) {

    document.getElementById(
        "expensePressure"
    ).textContent =
        `${safeNumber(
            row.expense_pressure
        ).toFixed(2)}x`;


    document.getElementById(
        "cashBufferRatio"
    ).textContent =
        `${safeNumber(
            row.cash_buffer_ratio
        ).toFixed(2)}x`;


    const growth =
        safeNumber(
            row.spending_growth
        );


    document.getElementById(
        "spendingGrowth"
    ).textContent =
        `${
            growth >= 0
                ? "+"
                : ""
        }${(
            growth * 100
        ).toFixed(1)}%`;


    document.getElementById(
        "transactionActivity"
    ).textContent =
        Math.round(
            safeNumber(
                row.transaction_count
            )
        );
}


// ============================================================================
// PREDICTION UI
// ============================================================================

function updatePrediction(
    result,
    row
) {

    const risk =
        safeNumber(
            result.risk_probability
        );


    document.getElementById(
        "riskProbability"
    ).textContent =
        `${(
            risk * 100
        ).toFixed(2)}%`;


    document.getElementById(
        "riskLevel"
    ).textContent =
        formatLabel(
            result.risk_level
        );


    document.getElementById(
        "financialPressure"
    ).textContent =
        formatLabel(
            result.financial_pressure
        );


    document.getElementById(
        "intervention"
    ).textContent =
        formatLabel(
            result.intervention
        );


    document.getElementById(
        "recommendationStatus"
    ).textContent =
        formatLabel(
            result.recommendation_status
        );


    document.getElementById(
        "priority"
    ).textContent =
        String(
            safeNumber(
                result.intervention_priority
            )
        ).padStart(
            2,
            "0"
        );


    const cateElement =
        document.getElementById(
            "cate"
        );


    if (
        result.cate === null ||
        result.cate === undefined
    ) {

        cateElement.textContent =
            "N/A";

    } else {

        cateElement.textContent =
            safeNumber(
                result.cate
            ).toFixed(3);
    }


    const cateExplanation =
        document.getElementById(
            "cateExplanation"
        );


    if (
        result.cate === null ||
        result.cate === undefined
    ) {

        cateExplanation.textContent =
            "No intervention-specific CATE is applicable to this financial state.";

    } else {

        const cate =
            safeNumber(
                result.cate
            );


        if (cate <= -0.10) {

            cateExplanation.textContent =
                "The simulated experiment provides strong support for the assigned intervention.";

        } else if (cate < 0) {

            cateExplanation.textContent =
                "The simulated experiment indicates a negative estimated treatment effect for the assigned intervention.";

        } else {

            cateExplanation.textContent =
                "The simulated experiment does not show a negative estimated treatment effect for the assigned intervention.";
        }
    }


    updateDrivers(
        result.model_drivers
    );


    document.getElementById(
        "decisionContext"
    ).textContent =
        result.explanation;
}


// ============================================================================
// SHAP / MODEL DRIVERS
// ============================================================================

function updateDrivers(
    driverText
) {

    const container =
        document.getElementById(
            "drivers"
        );


    container.innerHTML = "";


    if (
        !driverText ||
        driverText.trim() === ""
    ) {

        const empty =
            document.createElement(
                "div"
            );

        empty.className =
            "empty-state";

        empty.textContent =
            "No model drivers available.";

        container.appendChild(
            empty
        );

        return;
    }


    const drivers =
        driverText
            .split(" | ")
            .map(
                driver =>
                    driver.trim()
            )
            .filter(
                Boolean
            );


    drivers.forEach(
        driver => {

            const lower =
                driver.toLowerCase();


            const increases =
                lower.includes(
                    "increases"
                );


            const reduces =
                lower.includes(
                    "reduces"
                );


            let cleanName =
                driver;


            cleanName =
                cleanName.replace(
                    " increases predicted stress risk",
                    ""
                );


            cleanName =
                cleanName.replace(
                    " reduces predicted stress risk",
                    ""
                );


            const driverRow =
                document.createElement(
                    "div"
                );


            driverRow.className =
                "driver-row";


            const name =
                document.createElement(
                    "span"
                );


            name.className =
                "driver-name";

            name.textContent =
                cleanName;


            const direction =
                document.createElement(
                    "span"
                );


            direction.className =
                increases
                    ? "driver-direction driver-up"
                    : "driver-direction driver-down";


            direction.textContent =
                increases
                    ? "↑ increases risk"
                    : reduces
                        ? "↓ reduces risk"
                        : "• model signal";


            driverRow.appendChild(
                name
            );

            driverRow.appendChild(
                direction
            );


            container.appendChild(
                driverRow
            );
        }
    );
}


// ============================================================================
// CHART
// ============================================================================

function updateChart() {

    const canvas =
        document.getElementById(
            "financialChart"
        );


    if (!canvas) {
        return;
    }


    const labels =
        userHistory.map(
            row =>
                formatMonth(
                    row.month
                )
        );


    const income =
        userHistory.map(
            row =>
                safeNumber(
                    row.total_income
                )
        );


    const spending =
        userHistory.map(
            row =>
                safeNumber(
                    row.total_debits
                )
        );


    const balance =
        userHistory.map(
            row =>
                safeNumber(
                    row.ending_balance
                )
        );


    if (chart) {

        chart.destroy();

        chart = null;
    }


    chart =
        new Chart(
            canvas,
            {

                type: "line",

                data: {

                    labels,

                    datasets: [

                        {
                            label: "Income",

                            data: income,

                            borderColor:
                                "#36D9FF",

                            backgroundColor:
                                "rgba(54,217,255,0.08)",

                            borderWidth: 2,

                            tension: 0.35,

                            pointRadius: 2,

                            pointHoverRadius: 5,

                            fill: false,
                        },

                        {
                            label: "Spending",

                            data: spending,

                            borderColor:
                                "#FF5C7A",

                            backgroundColor:
                                "rgba(255,92,122,0.08)",

                            borderWidth: 2,

                            tension: 0.35,

                            pointRadius: 2,

                            pointHoverRadius: 5,

                            fill: false,
                        },

                        {
                            label: "Balance",

                            data: balance,

                            borderColor:
                                "#4F8CFF",

                            backgroundColor:
                                "rgba(79,140,255,0.08)",

                            borderWidth: 2,

                            tension: 0.35,

                            pointRadius: 2,

                            pointHoverRadius: 5,

                            fill: false,
                        },
                    ],
                },


                options: {

                    responsive: true,

                    maintainAspectRatio: false,

                    interaction: {
                        intersect: false,
                        mode: "index",
                    },


                    plugins: {

                        legend: {

                            labels: {

                                color:
                                    "#8291A8",

                                font: {
                                    size: 11,
                                },
                            },
                        },


                        tooltip: {

                            callbacks: {

                                label:
                                    context => {

                                        return `${
                                            context.dataset.label
                                        }: ${
                                            formatCurrency(
                                                context.parsed.y
                                            )
                                        }`;
                                    },
                            },
                        },
                    },


                    scales: {

                        x: {

                            grid: {
                                display: false,
                            },

                            ticks: {

                                color:
                                    "#56667E",

                                font: {
                                    size: 10,
                                },
                            },
                        },


                        y: {

                            grid: {

                                color:
                                    "rgba(255,255,255,0.045)",
                            },

                            ticks: {

                                color:
                                    "#56667E",

                                font: {
                                    size: 10,
                                },

                                callback:
                                    value =>
                                        formatCompactCurrency(
                                            value
                                        ),
                            },
                        },
                    },
                },
            }
        );
}


// ============================================================================
// COMPACT CURRENCY FOR CHART
// ============================================================================

function formatCompactCurrency(
    value
) {

    const number =
        safeNumber(
            value
        );


    if (
        Math.abs(number) >=
        10000000
    ) {

        return `₹${(
            number / 10000000
        ).toFixed(1)}Cr`;
    }


    if (
        Math.abs(number) >=
        100000
    ) {

        return `₹${(
            number / 100000
        ).toFixed(1)}L`;
    }


    if (
        Math.abs(number) >=
        1000
    ) {

        return `₹${(
            number / 1000
        ).toFixed(0)}K`;
    }


    return `₹${number}`;
}


// ============================================================================
// CLEAR DECISION AREA
// ============================================================================

function clearDecisionArea() {

    document.getElementById(
        "riskProbability"
    ).textContent = "--";


    document.getElementById(
        "riskLevel"
    ).textContent = "--";


    document.getElementById(
        "financialPressure"
    ).textContent = "--";


    document.getElementById(
        "intervention"
    ).textContent = "--";


    document.getElementById(
        "recommendationStatus"
    ).textContent = "--";


    document.getElementById(
        "priority"
    ).textContent = "--";


    document.getElementById(
        "cate"
    ).textContent = "--";


    document.getElementById(
        "cateExplanation"
    ).textContent = "--";


    document.getElementById(
        "decisionContext"
    ).textContent = "--";


    updateDrivers("");
}


// ============================================================================
// LOADING STATE
// ============================================================================

function setAnalyzingState(
    analyzing
) {

    analyzeButton.disabled =
        analyzing;


    analyzeButton.textContent =
        analyzing
            ? "Analyzing..."
            : "Analyze Financial State";
}


// ============================================================================
// ERROR DISPLAY
// ============================================================================

function showDashboardError(
    message
) {

    console.error(
        message
    );

    const context =
        document.getElementById(
            "decisionContext"
        );


    if (context) {

        context.textContent =
            message;
    }
}


// ============================================================================
// INITIALIZATION
// ============================================================================

async function initializeDashboard() {

    console.log(
        "Initializing FinSight dashboard..."
    );


    const health =
        await checkAPI();


    if (!health) {

        showDashboardError(
            "FinSight API is not reachable. Start FastAPI on port 8000."
        );

        return;
    }


    if (
        !health.dashboard_data_ready
    ) {

        showDashboardError(
            "FinSight dataset is not available on the API."
        );

        return;
    }


    if (
        !health.pipeline_ready
    ) {

        showDashboardError(
            "FinSight inference pipeline is not ready."
        );

        return;
    }


    await loadUsers();


    console.log(
        "FinSight dashboard initialized."
    );
}


document.addEventListener(
    "DOMContentLoaded",
    initializeDashboard
);