from __future__ import annotations

from typing import Any

import pandas as pd
from sklearn.preprocessing import OneHotEncoder, RobustScaler


def load_and_clean(csv_path: str) -> pd.DataFrame:
    """Load raw telco churn data and apply phase-2 base cleaning."""
    df = pd.read_csv(csv_path)

    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(0.0)

    if "customerID" in df.columns:
        df = df.drop(columns=["customerID"])

    df["Churn"] = df["Churn"].map({"No": 0, "Yes": 1}).astype(int)
    return df


def encode_binary_columns(
    df: pd.DataFrame,
    columns: list[str],
    mapping: dict[str, dict[Any, int]] | None = None,
) -> tuple[pd.DataFrame, dict[str, dict[Any, int]]]:
    """
    Encode binary columns to 0/1.

    If mapping is not provided, infer from observed values.
    """
    encoded = df.copy()
    learned_mapping: dict[str, dict[Any, int]] = {}

    for col in columns:
        if col not in encoded.columns:
            continue

        if mapping and col in mapping:
            col_map = mapping[col]
        else:
            unique_values = [val for val in encoded[col].dropna().unique().tolist()]
            unique_set = set(unique_values)

            if unique_set == {"Yes", "No"}:
                col_map = {"No": 0, "Yes": 1}
            elif unique_set == {"Female", "Male"}:
                col_map = {"Female": 0, "Male": 1}
            elif unique_set == {0, 1}:
                col_map = {0: 0, 1: 1}
            elif len(unique_set) == 2:
                ordered_values = sorted(unique_values, key=lambda x: str(x))
                col_map = {ordered_values[0]: 0, ordered_values[1]: 1}
            else:
                raise ValueError(f"Column '{col}' is not binary and cannot be encoded safely.")

        encoded[col] = encoded[col].map(col_map)

        if encoded[col].isna().any():
            missing_values = df.loc[encoded[col].isna(), col].dropna().unique().tolist()
            raise ValueError(
                f"Column '{col}' has unmapped values after encoding: {missing_values}"
            )

        encoded[col] = encoded[col].astype(int)
        learned_mapping[col] = col_map

    return encoded, learned_mapping


def encode_multiclass_columns(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    columns: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, OneHotEncoder]:
    """One-hot encode multiclass columns with train-only fit."""
    train_encoded = df_train.copy()
    test_encoded = df_test.copy()

    available_cols = [col for col in columns if col in train_encoded.columns]
    encoder = OneHotEncoder(
        sparse_output=False,
        handle_unknown="ignore",
        drop="first",
    )

    train_matrix = encoder.fit_transform(train_encoded[available_cols])
    test_matrix = encoder.transform(test_encoded[available_cols])

    feature_names = encoder.get_feature_names_out(available_cols)
    train_ohe = pd.DataFrame(train_matrix, columns=feature_names, index=train_encoded.index)
    test_ohe = pd.DataFrame(test_matrix, columns=feature_names, index=test_encoded.index)

    train_encoded = pd.concat([train_encoded.drop(columns=available_cols), train_ohe], axis=1)
    test_encoded = pd.concat([test_encoded.drop(columns=available_cols), test_ohe], axis=1)

    return train_encoded, test_encoded, encoder


def scale_numeric_columns(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    columns: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, RobustScaler]:
    """Scale numeric columns with RobustScaler using train-only fit."""
    train_scaled = df_train.copy()
    test_scaled = df_test.copy()

    available_cols = [col for col in columns if col in train_scaled.columns]
    scaler = RobustScaler()

    train_scaled[available_cols] = scaler.fit_transform(train_scaled[available_cols])
    test_scaled[available_cols] = scaler.transform(test_scaled[available_cols])

    return train_scaled, test_scaled, scaler
