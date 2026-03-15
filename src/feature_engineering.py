"""
Domain-specific feature engineering for the Telco Churn dataset.

All public functions accept a raw (pre-encoding) ``DataFrame`` and return an
augmented copy.  ``engineer_all_features`` is the convenience orchestrator that
applies every transformation in the correct order.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Columns representing add-on services (string "Yes"/"No" before encoding).
SERVICE_COLUMNS = [
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
]

# Tenure segmentation boundaries (months).
TENURE_BINS: list[float] = [-1, 12, 24, 48, 60, np.inf]
TENURE_LABELS: list[str] = ["0-12", "13-24", "25-48", "49-60", "61+"]

# Tenure threshold (months) below which a customer is considered "new".
NEW_CUSTOMER_TENURE_THRESHOLD: int = 6

# Risk weight maps used by add_contract_risk_score.
CONTRACT_RISK_MAP: dict[str, int] = {"Month-to-month": 2, "One year": 1, "Two year": 0}
PAYMENT_RISK_MAP: dict[str, int] = {"Electronic check": 2, "Mailed check": 1}


def _assert_string_dtype(df: pd.DataFrame, columns: list[str], fn_name: str) -> None:
    """Guard against calling feature functions on already-encoded (integer) columns."""
    for col in columns:
        if col in df.columns and not pd.api.types.is_object_dtype(df[col]):
            raise ValueError(
                f"{fn_name}: column '{col}' has dtype {df[col].dtype!r} but expects string. "
                "Feature engineering must be applied before binary/OHE encoding."
            )


def add_tenure_group(df: pd.DataFrame) -> pd.DataFrame:
    """
    Bin ``tenure`` (months) into five labelled groups.

    Groups: ``0-12``, ``13-24``, ``25-48``, ``49-60``, ``61+``.

    Args:
        df: Input ``DataFrame`` containing a numeric ``tenure`` column.

    Returns:
        Copy of ``df`` with a new ``tenure_group`` categorical column.
    """
    engineered = df.copy()
    engineered["tenure_group"] = pd.cut(
        engineered["tenure"],
        bins=TENURE_BINS,
        labels=TENURE_LABELS,
    )
    return engineered


def add_avg_monthly_charge(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute average monthly charge as ``TotalCharges / tenure``.

    Falls back to ``MonthlyCharges`` when ``tenure`` is zero to avoid
    division by zero.

    Args:
        df: Input ``DataFrame`` with ``TotalCharges``, ``tenure``, and
            ``MonthlyCharges`` columns.

    Returns:
        Copy of ``df`` with a new ``avg_monthly_charge`` float column.
    """
    engineered = df.copy()
    avg = np.where(
        engineered["tenure"] > 0,
        engineered["TotalCharges"] / engineered["tenure"],
        engineered["MonthlyCharges"],
    )
    engineered["avg_monthly_charge"] = avg
    return engineered


def add_service_count(df: pd.DataFrame) -> pd.DataFrame:
    """
    Count how many add-on services a customer has subscribed to.

    Counts "Yes" values across the six service columns defined in
    ``SERVICE_COLUMNS``.

    Args:
        df: Input ``DataFrame`` with string-typed service columns
            (must not yet be binary-encoded).

    Returns:
        Copy of ``df`` with a new integer ``service_count`` column.

    Raises:
        ValueError: If any service column has already been integer-encoded.
    """
    _assert_string_dtype(df, SERVICE_COLUMNS, "add_service_count")
    engineered = df.copy()
    engineered["service_count"] = (engineered[SERVICE_COLUMNS] == "Yes").sum(axis=1)
    return engineered


def add_charge_per_service(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute monthly charge per active service.

    Clips ``service_count`` to a minimum of 1 to prevent division by zero for
    customers with no add-on services.

    Args:
        df: Input ``DataFrame`` with ``MonthlyCharges`` and ``service_count``
            columns.  Call ``add_service_count`` first.

    Returns:
        Copy of ``df`` with a new ``charge_per_service`` float column.
    """
    engineered = df.copy()
    denominator = engineered["service_count"].clip(lower=1)
    engineered["charge_per_service"] = engineered["MonthlyCharges"] / denominator
    return engineered


def add_is_new_customer(df: pd.DataFrame) -> pd.DataFrame:
    """
    Flag customers whose tenure is below ``NEW_CUSTOMER_TENURE_THRESHOLD`` months.

    Args:
        df: Input ``DataFrame`` with a numeric ``tenure`` column.

    Returns:
        Copy of ``df`` with a new binary integer ``is_new_customer`` column
        (1 = new, 0 = established).
    """
    engineered = df.copy()
    engineered["is_new_customer"] = (
        engineered["tenure"] < NEW_CUSTOMER_TENURE_THRESHOLD
    ).astype(int)
    return engineered


def add_has_premium_support(df: pd.DataFrame) -> pd.DataFrame:
    """
    Flag customers who have both TechSupport and OnlineSecurity active.

    Args:
        df: Input ``DataFrame`` with string-typed ``TechSupport`` and
            ``OnlineSecurity`` columns (must not yet be binary-encoded).

    Returns:
        Copy of ``df`` with a new binary integer ``has_premium_support`` column.

    Raises:
        ValueError: If the relevant columns have already been integer-encoded.
    """
    _assert_string_dtype(df, ["TechSupport", "OnlineSecurity"], "add_has_premium_support")
    engineered = df.copy()
    engineered["has_premium_support"] = (
        (engineered["TechSupport"] == "Yes") & (engineered["OnlineSecurity"] == "Yes")
    ).astype(int)
    return engineered


def add_contract_risk_score(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute an integer risk score from contract type and payment method.

    Higher scores indicate a higher likelihood of churn based on domain
    knowledge: month-to-month contracts and electronic-check payments are
    associated with higher churn rates.

    Score = ``CONTRACT_RISK_MAP[Contract]`` + ``PAYMENT_RISK_MAP[PaymentMethod]``
    (unknown values default to 0).

    Args:
        df: Input ``DataFrame`` with string-typed ``Contract`` and
            ``PaymentMethod`` columns (must not yet be encoded).

    Returns:
        Copy of ``df`` with a new integer ``contract_risk_score`` column
        ranging from 0 to 4.

    Raises:
        ValueError: If the relevant columns have already been integer-encoded.
    """
    _assert_string_dtype(df, ["Contract", "PaymentMethod"], "add_contract_risk_score")
    engineered = df.copy()

    contract_component = engineered["Contract"].map(CONTRACT_RISK_MAP).fillna(0)
    payment_component = engineered["PaymentMethod"].map(PAYMENT_RISK_MAP).fillna(0)

    engineered["contract_risk_score"] = (contract_component + payment_component).astype(int)
    return engineered


def engineer_all_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply all feature engineering steps in the correct order.

    This is the primary entry point for feature engineering.  The input
    ``DataFrame`` must contain raw (string-typed, pre-encoded) columns.

    Args:
        df: Raw ``DataFrame`` after ``load_and_clean`` but before any encoding.

    Returns:
        Augmented copy of ``df`` with all engineered columns added.
    """
    engineered = df.copy()
    engineered = add_tenure_group(engineered)
    engineered = add_avg_monthly_charge(engineered)
    engineered = add_service_count(engineered)
    engineered = add_charge_per_service(engineered)
    engineered = add_is_new_customer(engineered)
    engineered = add_has_premium_support(engineered)
    engineered = add_contract_risk_score(engineered)

    n_new = engineered.shape[1] - df.shape[1]
    logger.debug("Engineered %d features; output shape: %s", n_new, engineered.shape)
    return engineered
