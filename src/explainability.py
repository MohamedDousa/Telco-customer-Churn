"""
Model explainability utilities using SHAP and LIME.

Provides global explanations (SHAP feature importance, dependence plots) and
local explanations (LIME per-instance) for any scikit-learn compatible
classifier.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
import shap
from lime.lime_tabular import LimeTabularExplainer
from sklearn.linear_model import LogisticRegression, RidgeClassifier, SGDClassifier

logger = logging.getLogger(__name__)

# Linear model types that should use shap.LinearExplainer instead of TreeExplainer.
_LINEAR_MODEL_TYPES = (LogisticRegression, RidgeClassifier, SGDClassifier)


def _sample_features(x: pd.DataFrame, max_samples: int = 1000) -> pd.DataFrame:
    """
    Return a random subsample of ``x`` capped at ``max_samples`` rows.

    Args:
        x: Input feature ``DataFrame``.
        max_samples: Maximum number of rows to return.

    Returns:
        Sampled copy of ``x`` (or the full frame if already small enough).
    """
    if len(x) <= max_samples:
        return x.copy()
    return x.sample(n=max_samples, random_state=42).copy()


def compute_shap_values(
    model: Any,
    x: pd.DataFrame,
    max_samples: int = 1000,
) -> tuple[Any, pd.DataFrame]:
    """
    Compute SHAP values using the appropriate explainer for the model type.

    Linear models (``LogisticRegression``, ``RidgeClassifier``,
    ``SGDClassifier``) use ``shap.LinearExplainer``; all other models fall back
    to ``shap.TreeExplainer``.

    Args:
        model: Fitted scikit-learn compatible classifier.
        x: Feature ``DataFrame`` to explain.
        max_samples: Cap on the number of rows sampled before computing SHAP
            values (for performance).

    Returns:
        Tuple of (SHAP ``Explanation`` object, sampled feature ``DataFrame``).
    """
    x_sampled = _sample_features(x, max_samples=max_samples)
    logger.info("Computing SHAP values for %d rows (%d features)", len(x_sampled), x_sampled.shape[1])

    if isinstance(model, _LINEAR_MODEL_TYPES):
        explainer = shap.LinearExplainer(model, x_sampled)
    else:
        explainer = shap.TreeExplainer(model)

    shap_values = explainer(x_sampled)
    logger.info("SHAP computation complete")
    return shap_values, x_sampled


def plot_shap_summary(shap_values: Any, x: pd.DataFrame, max_display: int = 15) -> None:
    """
    Render a SHAP beeswarm summary plot.

    Args:
        shap_values: SHAP ``Explanation`` object returned by ``compute_shap_values``.
        x: Feature ``DataFrame`` corresponding to ``shap_values``.
        max_display: Maximum number of features to display.
    """
    shap.summary_plot(shap_values, x, max_display=max_display, show=True)


def plot_shap_bar(shap_values: Any, max_display: int = 15) -> None:
    """
    Render a SHAP global mean-absolute-value bar chart.

    Args:
        shap_values: SHAP ``Explanation`` object returned by ``compute_shap_values``.
        max_display: Maximum number of features to display.
    """
    shap.plots.bar(shap_values, max_display=max_display, show=True)


def plot_shap_dependence(shap_values: Any, feature: str, x: pd.DataFrame) -> None:
    """
    Render a SHAP dependence scatter plot for a single feature.

    Args:
        shap_values: SHAP ``Explanation`` object returned by ``compute_shap_values``.
        feature: Feature name to plot on the x-axis.
        x: Feature ``DataFrame`` corresponding to ``shap_values``.
    """
    shap.dependence_plot(feature, shap_values.values, x, show=True)


def explain_with_lime(
    model: Any,
    x_train: pd.DataFrame,
    instance: pd.Series,
    feature_names: list[str],
    class_names: tuple[str, str] = ("No Churn", "Churn"),
    num_features: int = 10,
) -> Any:
    """
    Generate a LIME local explanation for a single customer instance.

    Args:
        model: Fitted classifier with a ``predict_proba`` method.
        x_train: Training feature matrix used to fit the LIME explainer
            (provides the feature distribution context).
        instance: Single-row ``Series`` to explain.
        feature_names: Ordered list of feature names matching ``x_train`` columns.
        class_names: Labels for the two output classes.
        num_features: Number of top features to include in the explanation.

    Returns:
        LIME ``Explanation`` object (call ``.as_list()`` or
        ``.as_pyplot_figure()`` to inspect results).
    """
    explainer = LimeTabularExplainer(
        training_data=x_train.to_numpy(),
        feature_names=feature_names,
        class_names=list(class_names),
        mode="classification",
        discretize_continuous=True,
        random_state=42,
    )

    explanation = explainer.explain_instance(
        data_row=instance.to_numpy(dtype=float),
        predict_fn=model.predict_proba,
        num_features=num_features,
    )
    return explanation


def get_top_shap_features(
    shap_values: Any,
    feature_names: list[str],
    top_n: int = 10,
) -> list[str]:
    """
    Return the top-N most important features by mean absolute SHAP value.

    Args:
        shap_values: SHAP ``Explanation`` object returned by ``compute_shap_values``.
        feature_names: Ordered list of feature names.
        top_n: Number of top features to return.

    Returns:
        List of feature name strings, ordered by descending importance.
    """
    values = shap_values.values
    if values.ndim == 3:
        values = values[:, :, 1]
    importance = np.abs(values).mean(axis=0)
    sorted_idx = np.argsort(importance)[::-1][:top_n]
    return [feature_names[i] for i in sorted_idx]
