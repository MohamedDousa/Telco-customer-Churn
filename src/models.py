from __future__ import annotations

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


def load_processed_data(processed_dir: str | Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Load processed train/test datasets from phase 2 outputs."""
    processed_path = Path(processed_dir)
    x_train = pd.read_csv(processed_path / "X_train.csv")
    x_test = pd.read_csv(processed_path / "X_test.csv")
    y_train = pd.read_csv(processed_path / "y_train.csv")["Churn"]
    y_test = pd.read_csv(processed_path / "y_test.csv")["Churn"]
    return x_train, x_test, y_train, y_test


def get_resampling_strategies() -> dict[str, Any]:
    """Return resampling strategies used in phase 3."""
    return {
        "baseline": None,
        "smote": SMOTE(random_state=42),
        "smote_tomek": SMOTETomek(random_state=42),
    }


def get_model_configs() -> dict[str, Any]:
    """Return base model configurations with sensible defaults."""
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
    """Run stratified CV and return metric mean/std summaries."""
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
    return summary


def evaluate_on_test(model: Any, x_test: pd.DataFrame, y_test: pd.Series) -> dict[str, Any]:
    """Evaluate fitted model on test data and return structured metrics."""
    y_pred = model.predict(x_test)
    y_proba = model.predict_proba(x_test)[:, 1]

    metrics: dict[str, Any] = {
        "confusion_matrix": confusion_matrix(y_test, y_pred),
        "classification_report": classification_report(y_test, y_pred, output_dict=True),
        "roc_auc": float(roc_auc_score(y_test, y_proba)),
        "pr_auc": float(average_precision_score(y_test, y_proba)),
        "f1": float(f1_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "y_pred": y_pred,
        "y_proba": y_proba,
    }
    return metrics


def find_optimal_threshold(model: Any, x_test: pd.DataFrame, y_test: pd.Series) -> float:
    """Find probability threshold that maximizes F1 score on test set."""
    y_proba = model.predict_proba(x_test)[:, 1]
    precision, recall, thresholds = precision_recall_curve(y_test, y_proba)

    # thresholds has len n-1; align with precision/recall excluding final point.
    f1_scores = 2 * (precision[:-1] * recall[:-1]) / (precision[:-1] + recall[:-1] + 1e-12)
    best_idx = int(np.argmax(f1_scores))
    return float(thresholds[best_idx])


def get_hyperparam_grid(model_name: str) -> dict[str, Any]:
    """Return RandomizedSearchCV parameter distributions for each model."""
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
    Clone model and neutralize class_weight when synthetic oversampling is used.
    """
    cloned = clone(model)
    if strategy_name in {"smote", "smote_tomek"} and hasattr(cloned, "class_weight"):
        cloned.set_params(class_weight=None)
    return cloned
