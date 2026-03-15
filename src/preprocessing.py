"""
Data cleaning, encoding, and scaling utilities for the Telco Churn pipeline.

Functions follow a leakage-safe convention: encoders and scalers are always
fitted on the training split and then applied to the test split.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.preprocessing import OneHotEncoder, RobustScaler

logger = logging.getLogger(__name__)


def load_and_clean(csv_path: str | Path) -> pd.DataFrame:
    """
    Load raw Telco CSV and apply initial cleaning steps.

    Steps performed:
    - Coerce ``TotalCharges`` to numeric (blank strings → 0.0).
    - Drop the ``customerID`` column (non-predictive identifier).
    - Encode the ``Churn`` target column as 0 / 1.

    Args:
        csv_path: Path to the raw ``WA_Fn-UseC_-Telco-Customer-Churn.csv`` file.

    Returns:
        Cleaned ``DataFrame`` ready for train/test splitting.

    Raises:
        FileNotFoundError: If ``csv_path`` does not exist.
        ValueError: If ``Churn`` column contains unexpected values.
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)

    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    imputed_count = int(df["TotalCharges"].isna().sum())
    if imputed_count > 0:
        logger.info(
            "Imputed %d row(s) where TotalCharges was blank/non-numeric → 0.0",
            imputed_count,
        )
    df["TotalCharges"] = df["TotalCharges"].fillna(0.0)

    if "customerID" in df.columns:
        df = df.drop(columns=["customerID"])

    churn_mapped = df["Churn"].map({"No": 0, "Yes": 1})
    if churn_mapped.isna().any():
        bad = df.loc[churn_mapped.isna(), "Churn"].unique().tolist()
        raise ValueError(f"Unexpected Churn values (expected 'No'/'Yes'): {bad}")
    df["Churn"] = churn_mapped.astype(int)

    logger.info("Loaded %d rows from %s", len(df), csv_path)
    return df


def encode_binary_columns(
    df: pd.DataFrame,
    columns: list[str],
    mapping: dict[str, dict[Any, int]] | None = None,
) -> tuple[pd.DataFrame, dict[str, dict[Any, int]]]:
    """
    Encode binary columns to 0/1.

    If ``mapping`` is provided (e.g. from a previous training call), it is
    applied directly.  Otherwise the mapping is inferred from observed values.

    Args:
        df: Input ``DataFrame``.
        columns: Column names to encode.
        mapping: Optional pre-fitted mapping ``{column: {value: int}}``.

    Returns:
        Tuple of (encoded DataFrame, learned mapping dict).

    Raises:
        ValueError: If a column is not binary or contains unmapped values.
    """
    encoded = df.copy()
    learned_mapping: dict[str, dict[Any, int]] = {}

    for col in columns:
        if col not in encoded.columns:
            continue

        if mapping is not None and col in mapping:
            col_map = mapping[col]
        else:
            unique_values = encoded[col].dropna().unique().tolist()
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

    logger.debug("Binary-encoded %d column(s): %s", len(learned_mapping), list(learned_mapping))
    return encoded, learned_mapping


def encode_multiclass_columns(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    columns: list[str],
    provided_encoder: OneHotEncoder | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, OneHotEncoder]:
    """
    One-hot encode multiclass columns, fitting only on the training split.

    Args:
        df_train: Training ``DataFrame``.
        df_test: Test ``DataFrame``.
        columns: Column names to one-hot encode.
        provided_encoder: Optional pre-fitted ``OneHotEncoder`` to reuse
            (e.g. when transforming a new batch without re-fitting).

    Returns:
        Tuple of (encoded train DataFrame, encoded test DataFrame, fitted encoder).
    """
    train_encoded = df_train.copy()
    test_encoded = df_test.copy()

    available_cols = [col for col in columns if col in train_encoded.columns]

    if provided_encoder is not None:
        encoder = provided_encoder
    else:
        encoder = OneHotEncoder(
            sparse_output=False,
            handle_unknown="ignore",
            drop="first",
        )
        encoder.fit(train_encoded[available_cols])

    train_matrix = encoder.transform(train_encoded[available_cols])
    test_matrix = encoder.transform(test_encoded[available_cols])

    feature_names = encoder.get_feature_names_out(available_cols)
    train_ohe = pd.DataFrame(train_matrix, columns=feature_names, index=train_encoded.index)
    test_ohe = pd.DataFrame(test_matrix, columns=feature_names, index=test_encoded.index)

    train_encoded = pd.concat([train_encoded.drop(columns=available_cols), train_ohe], axis=1)
    test_encoded = pd.concat([test_encoded.drop(columns=available_cols), test_ohe], axis=1)

    logger.debug(
        "OHE-encoded %d column(s), producing %d features",
        len(available_cols),
        len(feature_names),
    )
    return train_encoded, test_encoded, encoder


def scale_numeric_columns(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    columns: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, RobustScaler]:
    """
    Scale numeric columns with ``RobustScaler``, fitting only on the training split.

    Args:
        df_train: Training ``DataFrame``.
        df_test: Test ``DataFrame``.
        columns: Numeric column names to scale.

    Returns:
        Tuple of (scaled train DataFrame, scaled test DataFrame, fitted scaler).
    """
    train_scaled = df_train.copy()
    test_scaled = df_test.copy()

    available_cols = [col for col in columns if col in train_scaled.columns]
    scaler = RobustScaler()

    if not available_cols:
        logger.warning("scale_numeric_columns: none of the requested columns are present; skipping.")
        return train_scaled, test_scaled, scaler

    train_scaled[available_cols] = scaler.fit_transform(train_scaled[available_cols])
    test_scaled[available_cols] = scaler.transform(test_scaled[available_cols])

    logger.debug("Scaled %d numeric column(s) with RobustScaler", len(available_cols))
    return train_scaled, test_scaled, scaler
