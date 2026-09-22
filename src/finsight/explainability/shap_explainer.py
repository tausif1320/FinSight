from __future__ import annotations

import numpy as np
import pandas as pd
import shap


def build_shap_explainer(
    model,
    background_data: pd.DataFrame,
):
    """
    Build a SHAP TreeExplainer for a fitted tree-based model.

    Parameters
    ----------
    model:
        Fitted sklearn tree-based estimator.

    background_data:
        Preprocessed numeric background dataset.

    Returns
    -------
    shap.TreeExplainer
    """

    if background_data.empty:
        raise ValueError(
            "background_data cannot be empty."
        )

    if not np.all(
        np.isfinite(
            background_data.to_numpy(dtype=float)
        )
    ):
        raise ValueError(
            "background_data contains non-finite values."
        )

    return shap.TreeExplainer(
        model,
        data=background_data,
        feature_perturbation="interventional",
    )

def calculate_shap_values(
    explainer,
    X: pd.DataFrame,
) -> np.ndarray:
    """
    Calculate SHAP values for binary classification.

    Returns
    -------
    np.ndarray
        Shape: (n_samples, n_features)
    """

    if X.empty:
        raise ValueError(
            "X cannot be empty."
        )

    shap_output = explainer.shap_values(X, check_additivity=False,)

    # SHAP versions differ in their binary
    # classification output format.
    if isinstance(shap_output, list):
        if len(shap_output) != 2:
            raise ValueError(
                "Unexpected SHAP output format."
            )

        values = np.asarray(
            shap_output[1]
        )

    else:
        values = np.asarray(
            shap_output
        )

        # Some SHAP versions return:
        # (samples, features, classes)
        if values.ndim == 3:
            values = values[:, :, 1]

    if values.ndim != 2:
        raise ValueError(
            f"Unexpected SHAP value shape: "
            f"{values.shape}"
        )

    if values.shape != X.shape:
        raise ValueError(
            f"SHAP shape {values.shape} "
            f"does not match X shape {X.shape}."
        )

    if not np.all(np.isfinite(values)):
        raise ValueError(
            "SHAP values contain non-finite values."
        )

    return values


def build_feature_importance_table(
    feature_names,
    shap_values: np.ndarray,
) -> pd.DataFrame:
    """
    Calculate global SHAP feature importance.
    """

    feature_names = list(feature_names)

    if shap_values.shape[1] != len(feature_names):
        raise ValueError(
            "Number of features does not match "
            "SHAP value columns."
        )

    importance = np.abs(
        shap_values
    ).mean(axis=0)

    table = pd.DataFrame(
        {
            "feature": feature_names,
            "mean_abs_shap": importance,
        }
    )

    return (
        table
        .sort_values(
            "mean_abs_shap",
            ascending=False,
        )
        .reset_index(drop=True)
    )


def build_local_explanation(
    feature_names,
    feature_values,
    shap_values,
    top_n: int = 10,
) -> pd.DataFrame:
    """
    Build a local explanation for one observation.

    Positive SHAP values push the prediction toward
    financial stress.

    Negative SHAP values push the prediction away
    from financial stress.
    """

    feature_names = list(feature_names)

    values = np.asarray(
        feature_values,
        dtype=float,
    )

    contributions = np.asarray(
        shap_values,
        dtype=float,
    )

    if len(feature_names) != len(values):
        raise ValueError(
            "Feature names and values have "
            "different lengths."
        )

    if len(values) != len(contributions):
        raise ValueError(
            "Feature values and SHAP values "
            "have different lengths."
        )

    explanation = pd.DataFrame(
        {
            "feature": feature_names,
            "feature_value": values,
            "shap_value": contributions,
        }
    )

    explanation["direction"] = np.where(
        explanation["shap_value"] >= 0,
        "increases_stress_risk",
        "decreases_stress_risk",
    )

    explanation["absolute_shap"] = (
        explanation["shap_value"].abs()
    )

    return (
        explanation
        .sort_values(
            "absolute_shap",
            ascending=False,
        )
        .head(top_n)
        .reset_index(drop=True)
    )