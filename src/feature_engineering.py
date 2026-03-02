from __future__ import annotations

import numpy as np
import pandas as pd


SERVICE_COLUMNS = [
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
]


def add_tenure_group(df: pd.DataFrame) -> pd.DataFrame:
    engineered = df.copy()
    bins = [-1, 12, 24, 48, 60, np.inf]
    labels = ["0-12", "13-24", "25-48", "49-60", "61+"]
    engineered["tenure_group"] = pd.cut(engineered["tenure"], bins=bins, labels=labels)
    return engineered


def add_avg_monthly_charge(df: pd.DataFrame) -> pd.DataFrame:
    engineered = df.copy()
    avg = np.where(
        engineered["tenure"] > 0,
        engineered["TotalCharges"] / engineered["tenure"],
        engineered["MonthlyCharges"],
    )
    engineered["avg_monthly_charge"] = avg
    return engineered


def add_service_count(df: pd.DataFrame) -> pd.DataFrame:
    engineered = df.copy()
    engineered["service_count"] = (engineered[SERVICE_COLUMNS] == "Yes").sum(axis=1)
    return engineered


def add_charge_per_service(df: pd.DataFrame) -> pd.DataFrame:
    engineered = df.copy()
    denominator = engineered["service_count"].clip(lower=1)
    engineered["charge_per_service"] = engineered["MonthlyCharges"] / denominator
    return engineered


def add_is_new_customer(df: pd.DataFrame) -> pd.DataFrame:
    engineered = df.copy()
    engineered["is_new_customer"] = (engineered["tenure"] < 6).astype(int)
    return engineered


def add_has_premium_support(df: pd.DataFrame) -> pd.DataFrame:
    engineered = df.copy()
    engineered["has_premium_support"] = (
        (engineered["TechSupport"] == "Yes") & (engineered["OnlineSecurity"] == "Yes")
    ).astype(int)
    return engineered


def add_contract_risk_score(df: pd.DataFrame) -> pd.DataFrame:
    engineered = df.copy()

    contract_map = {"Month-to-month": 2, "One year": 1, "Two year": 0}
    payment_map = {"Electronic check": 2, "Mailed check": 1}

    contract_component = engineered["Contract"].map(contract_map).fillna(0)
    payment_component = engineered["PaymentMethod"].map(payment_map).fillna(0)

    engineered["contract_risk_score"] = (contract_component + payment_component).astype(int)
    return engineered


def engineer_all_features(df: pd.DataFrame) -> pd.DataFrame:
    engineered = df.copy()
    engineered = add_tenure_group(engineered)
    engineered = add_avg_monthly_charge(engineered)
    engineered = add_service_count(engineered)
    engineered = add_charge_per_service(engineered)
    engineered = add_is_new_customer(engineered)
    engineered = add_has_premium_support(engineered)
    engineered = add_contract_risk_score(engineered)
    return engineered
