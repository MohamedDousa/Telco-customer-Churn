from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import shap
from lime.lime_tabular import LimeTabularExplainer
from sklearn.linear_model import LogisticRegression


def _sample_features(x: pd.DataFrame, max_samples: int = 1000) -> pd.DataFrame:
    if len(x) <= max_samples:
        return x.copy()
    return x.sample(n=max_samples, random_state=42).copy()


def compute_shap_values(model: Any, x: pd.DataFrame, max_samples: int = 1000) -> tuple[Any, pd.DataFrame]:
    """
    Compute SHAP values with a model-specific explainer.

    Returns:
        shap_values: output from shap explainer
        x_sampled: sampled feature frame used for SHAP
    """
    x_sampled = _sample_features(x, max_samples=max_samples)

    if isinstance(model, LogisticRegression):
        explainer = shap.LinearExplainer(model, x_sampled)
        shap_values = explainer(x_sampled)
    else:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer(x_sampled)

    return shap_values, x_sampled


def plot_shap_summary(shap_values: Any, x: pd.DataFrame, max_display: int = 15) -> None:
    shap.summary_plot(shap_values, x, max_display=max_display, show=True)


def plot_shap_bar(shap_values: Any, max_display: int = 15) -> None:
    shap.plots.bar(shap_values, max_display=max_display, show=True)


def plot_shap_dependence(shap_values: Any, feature: str, x: pd.DataFrame) -> None:
    shap.dependence_plot(feature, shap_values.values, x, show=True)


def explain_with_lime(
    model: Any,
    x_train: pd.DataFrame,
    instance: pd.Series,
    feature_names: list[str],
    class_names: tuple[str, str] = ("No Churn", "Churn"),
    num_features: int = 10,
) -> Any:
    """Run LIME for one row; useful for spot-checking model behavior."""
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


def get_top_shap_features(shap_values: Any, feature_names: list[str], top_n: int = 10) -> list[str]:
    # Quick global ranking from mean abs SHAP values.
    values = shap_values.values
    if values.ndim == 3:
        values = values[:, :, 1]
    importance = np.abs(values).mean(axis=0)
    sorted_idx = np.argsort(importance)[::-1][:top_n]
    return [feature_names[i] for i in sorted_idx]
