"""
Model training, cross-validation, threshold tuning, and evaluation utilities.

Supports logistic regression, random forest, and LightGBM classifiers with
four imbalance-handling strategies: baseline, SMOTE, SMOTE+Tomek, and
class-weight balancing.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from imblearn.combine import SMOTETomek
from imblearn.over_sampling import SMOTE
from lightgbm import LGBMClassifier
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate

logger = logging.getLogger(__name__)

# Small epsilon to prevent division by zero when computing F1 from P/R arrays.
_F1_EPSILON: float = 1e-12


def load_processed_data(
    processed_dir: str | Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Load the preprocessed train/test splits from disk.

    Args:
        processed_dir: Directory containing ``X_train.csv``, ``X_test.csv``,
            ``y_train.csv``, and ``y_test.csv``.

    Returns:
        Tuple of (X_train, X_test, y_train, y_test).

    Raises:
        FileNotFoundError: If any of the expected CSV files are missing.
    """
    processed_path = Path(processed_dir)
    required = ["X_train.csv", "X_test.csv", "y_train.csv", "y_test.csv"]
    for fname in required:
        fpath = processed_path / fname
        if not fpath.exists():
            raise FileNotFoundError(
                f"Processed data file not found: {fpath}. "
                "Run notebook 02 (preprocessing) first."
            )

    x_train = pd.read_csv(processed_path / "X_train.csv")
    x_test = pd.read_csv(processed_path / "X_test.csv")
    y_train = pd.read_csv(processed_path / "y_train.csv")["Churn"]
    y_test = pd.read_csv(processed_path / "y_test.csv")["Churn"]

    logger.info(
        "Loaded processed data — x_train: %s, x_test: %s",
        x_train.shape,
        x_test.shape,
    )
    return x_train, x_test, y_train, y_test


def get_resampling_strategies() -> dict[str, Any]:
    """Return the set of imbalance-handling options used in model comparison."""
    return {
        "baseline": None,
        "smote": SMOTE(random_state=42),
        "smote_tomek": SMOTETomek(random_state=42),
    }


def get_model_configs() -> dict[str, Any]:
    """Return base model configurations with sensible reproducible defaults."""
    return {
        "logistic_regression": LogisticRegression(max_iter=1000, random_state=42, n_jobs=-1),
        "logistic_regression_balanced": LogisticRegression(
            max_iter=1000,
            random_state=42,
            n_jobs=-1,
            class_weight="balanced",
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=200,
            random_state=42,
            n_jobs=-1,
        ),
        "random_forest_balanced": RandomForestClassifier(
            n_estimators=200,
            random_state=42,
            n_jobs=-1,
            class_weight="balanced",
        ),
        "lightgbm": LGBMClassifier(
            n_estimators=300,
            random_state=42,
            n_jobs=-1,
            verbose=-1,
        ),
        "lightgbm_balanced": LGBMClassifier(
            n_estimators=300,
            random_state=42,
            n_jobs=-1,
            verbose=-1,
            class_weight="balanced",
        ),
    }


def cross_validate_model(
    model: Any,
    x: pd.DataFrame,
    y: pd.Series,
    cv: int = 5,
    scoring: tuple[str, ...] = ("f1", "roc_auc", "precision", "recall"),
) -> dict[str, float]:
    """
    Run stratified cross-validation and return mean/std summaries per metric.

    Args:
        model: Scikit-learn compatible estimator (may be a Pipeline).
        x: Feature matrix.
        y: Binary target series.
        cv: Number of stratified folds.
        scoring: Metric names to evaluate.

    Returns:
        Dict with keys like ``"f1_mean"``, ``"f1_std"``, etc.
    """
    splitter = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    cv_results = cross_validate(
        estimator=model,
        X=x,
        y=y,
        scoring=scoring,
        cv=splitter,
        n_jobs=-1,
        return_train_score=False,
    )

    summary: dict[str, float] = {}
    for metric in scoring:
        metric_key = f"test_{metric}"
        summary[f"{metric}_mean"] = float(np.mean(cv_results[metric_key]))
        summary[f"{metric}_std"] = float(np.std(cv_results[metric_key]))

    logger.debug("CV summary: %s", {k: round(v, 4) for k, v in summary.items()})
    return summary


def evaluate_on_test(
    model: Any,
    x_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict[str, Any]:
    """
    Evaluate a fitted model on the held-out test set.

    Args:
        model: Fitted estimator supporting ``predict`` and ``predict_proba``.
        x_test: Test feature matrix.
        y_test: True binary labels.

    Returns:
        Dict containing confusion matrix, classification report, scalar metrics
        (ROC-AUC, PR-AUC, F1, precision, recall), and raw prediction arrays
        (``y_pred``, ``y_proba``).

    Raises:
        AttributeError: If the model does not support ``predict_proba``.
    """
    if not hasattr(model, "predict_proba"):
        raise AttributeError(
            f"Model {type(model).__name__} does not support predict_proba, "
            "which is required for ROC-AUC and PR-AUC computation."
        )

    y_pred = model.predict(x_test)
    y_proba = model.predict_proba(x_test)[:, 1]

    metrics: dict[str, Any] = {
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "classification_report": classification_report(y_test, y_pred, output_dict=True),
        "roc_auc": float(roc_auc_score(y_test, y_proba)),
        "pr_auc": float(average_precision_score(y_test, y_proba)),
        "f1": float(f1_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "y_pred": y_pred.tolist(),
        "y_proba": y_proba.tolist(),
    }

    logger.info(
        "Test metrics — ROC-AUC: %.4f  PR-AUC: %.4f  F1: %.4f",
        metrics["roc_auc"],
        metrics["pr_auc"],
        metrics["f1"],
    )
    return metrics


def find_optimal_threshold_from_proba(y_true: pd.Series, y_proba: np.ndarray) -> float:
    """
    Pick the decision threshold that maximises F1 given pre-computed probabilities.

    Args:
        y_true: True binary labels.
        y_proba: Predicted probabilities for the positive class.

    Returns:
        Optimal threshold as a float in (0, 1).
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
    # ``thresholds`` has length n-1; align with precision/recall by excluding
    # the final sentinel point before computing F1.
    f1_scores = (
        2 * (precision[:-1] * recall[:-1]) / (precision[:-1] + recall[:-1] + _F1_EPSILON)
    )
    best_idx = int(np.argmax(f1_scores))
    optimal = float(thresholds[best_idx])
    logger.debug("Optimal threshold: %.4f (F1=%.4f)", optimal, f1_scores[best_idx])
    return optimal


def find_optimal_threshold(model: Any, x_test: pd.DataFrame, y_test: pd.Series) -> float:
    """
    Pick the decision threshold that gives best F1 on the provided data.

    Args:
        model: Fitted estimator supporting ``predict_proba``.
        x_test: Feature matrix.
        y_test: True binary labels.

    Returns:
        Optimal threshold as a float in (0, 1).
    """
    y_proba = model.predict_proba(x_test)[:, 1]
    return find_optimal_threshold_from_proba(y_test, y_proba)


def get_hyperparam_grid(model_name: str) -> dict[str, Any]:
    """
    Return the randomized-search parameter grid for a given model.

    The ``_balanced`` suffix is stripped before lookup so that balanced
    variants share the same grid as their base counterparts.

    Args:
        model_name: One of ``"logistic_regression"``, ``"random_forest"``,
            ``"lightgbm"`` (with or without ``"_balanced"`` suffix).

    Returns:
        Parameter distribution dict suitable for ``RandomizedSearchCV``.

    Raises:
        ValueError: If no grid is configured for ``model_name``.
    """
    grids: dict[str, dict[str, Any]] = {
        "logistic_regression": {
            "C": np.logspace(-3, 2, 30),
            "solver": ["liblinear", "lbfgs"],
        },
        "random_forest": {
            "n_estimators": [150, 250, 350, 500],
            "max_depth": [None, 5, 10, 20],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
            "max_features": ["sqrt", "log2", None],
        },
        "lightgbm": {
            "n_estimators": [150, 250, 400, 600],
            "learning_rate": [0.01, 0.03, 0.05, 0.1],
            "num_leaves": [15, 31, 63, 127],
            "max_depth": [-1, 5, 10, 20],
            "subsample": [0.7, 0.8, 0.9, 1.0],
            "colsample_bytree": [0.6, 0.8, 1.0],
            "min_child_samples": [10, 20, 40],
        },
    }

    key = model_name.replace("_balanced", "")
    if key not in grids:
        raise ValueError(f"No hyperparameter grid configured for model '{model_name}'")
    return grids[key]


def clone_for_resampling(model: Any, strategy_name: str) -> Any:
    """
    Clone a model and neutralise ``class_weight`` when synthetic oversampling is used.

    Combining ``class_weight="balanced"`` with SMOTE/SMOTE+Tomek double-counts
    the minority class; this helper prevents that by resetting ``class_weight``
    to ``None`` when a resampling strategy is active.

    Args:
        model: Scikit-learn estimator to clone.
        strategy_name: One of ``"baseline"``, ``"smote"``, ``"smote_tomek"``.

    Returns:
        Cloned estimator with ``class_weight=None`` if applicable.
    """
    cloned = clone(model)
    if strategy_name in {"smote", "smote_tomek"} and hasattr(cloned, "class_weight"):
        cloned.set_params(class_weight=None)
    return cloned
