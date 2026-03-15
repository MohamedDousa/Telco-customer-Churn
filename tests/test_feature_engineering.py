from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.feature_engineering import (
    add_avg_monthly_charge,
    add_charge_per_service,
    add_contract_risk_score,
    add_has_premium_support,
    add_is_new_customer,
    add_service_count,
    add_tenure_group,
    engineer_all_features,
)

# ---------------------------------------------------------------------------
# Minimal fixture representing pre-encoding (string) data
# ---------------------------------------------------------------------------

def _base_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "tenure": [0, 5, 13, 30, 62],
            "MonthlyCharges": [30.0, 50.0, 70.0, 90.0, 110.0],
            "TotalCharges": [0.0, 250.0, 910.0, 2700.0, 6820.0],
            "OnlineSecurity": ["No", "Yes", "No", "Yes", "Yes"],
            "OnlineBackup": ["No", "No", "Yes", "Yes", "Yes"],
            "DeviceProtection": ["No", "No", "No", "Yes", "Yes"],
            "TechSupport": ["No", "No", "No", "No", "Yes"],
            "StreamingTV": ["No", "No", "No", "No", "Yes"],
            "StreamingMovies": ["No", "No", "No", "No", "Yes"],
            "Contract": ["Month-to-month", "Month-to-month", "One year", "Two year", "Two year"],
            "PaymentMethod": [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)",
                "Electronic check",
            ],
        }
    )


# ---------------------------------------------------------------------------
# add_tenure_group
# ---------------------------------------------------------------------------

class TestAddTenureGroup:
    def test_bins_assigned_correctly(self):
        df = _base_df()
        out = add_tenure_group(df)
        expected = ["0-12", "0-12", "13-24", "25-48", "61+"]
        assert list(out["tenure_group"].astype(str)) == expected

    def test_does_not_mutate_input(self):
        df = _base_df()
        _ = add_tenure_group(df)
        assert "tenure_group" not in df.columns


# ---------------------------------------------------------------------------
# add_avg_monthly_charge
# ---------------------------------------------------------------------------

class TestAddAvgMonthlyCharge:
    def test_zero_tenure_falls_back_to_monthly_charges(self):
        df = _base_df()
        out = add_avg_monthly_charge(df)
        assert out.loc[0, "avg_monthly_charge"] == pytest.approx(30.0)

    def test_nonzero_tenure_is_total_over_tenure(self):
        df = _base_df()
        out = add_avg_monthly_charge(df)
        assert out.loc[1, "avg_monthly_charge"] == pytest.approx(250.0 / 5)


# ---------------------------------------------------------------------------
# add_service_count
# ---------------------------------------------------------------------------

class TestAddServiceCount:
    def test_counts_yes_values(self):
        df = _base_df()
        out = add_service_count(df)
        assert list(out["service_count"]) == [0, 1, 1, 3, 6]

    def test_raises_on_encoded_column(self):
        df = _base_df()
        df["OnlineSecurity"] = df["OnlineSecurity"].map({"No": 0, "Yes": 1})
        with pytest.raises(ValueError, match="add_service_count"):
            add_service_count(df)


# ---------------------------------------------------------------------------
# add_charge_per_service
# ---------------------------------------------------------------------------

class TestAddChargePerService:
    def test_charge_per_service_no_zero_division(self):
        df = _base_df()
        df = add_service_count(df)
        out = add_charge_per_service(df)
        # row 0 has service_count=0, clip(lower=1) prevents division by zero
        assert np.isfinite(out.loc[0, "charge_per_service"])


# ---------------------------------------------------------------------------
# add_is_new_customer
# ---------------------------------------------------------------------------

class TestAddIsNewCustomer:
    def test_tenure_below_6_is_flagged(self):
        df = _base_df()
        out = add_is_new_customer(df)
        assert list(out["is_new_customer"]) == [1, 1, 0, 0, 0]


# ---------------------------------------------------------------------------
# add_has_premium_support
# ---------------------------------------------------------------------------

class TestAddHasPremiumSupport:
    def test_both_yes_gives_1(self):
        df = _base_df()
        out = add_has_premium_support(df)
        # only row 4 has both TechSupport=Yes and OnlineSecurity=Yes
        assert list(out["has_premium_support"]) == [0, 0, 0, 0, 1]

    def test_raises_on_encoded_column(self):
        df = _base_df()
        df["TechSupport"] = df["TechSupport"].map({"No": 0, "Yes": 1})
        with pytest.raises(ValueError, match="add_has_premium_support"):
            add_has_premium_support(df)


# ---------------------------------------------------------------------------
# add_contract_risk_score
# ---------------------------------------------------------------------------

class TestAddContractRiskScore:
    def test_month_to_month_electronic_check_is_highest(self):
        df = _base_df()
        out = add_contract_risk_score(df)
        # row 0: Month-to-month(2) + Electronic check(2) = 4
        assert out.loc[0, "contract_risk_score"] == 4

    def test_two_year_credit_card_is_zero(self):
        df = _base_df()
        out = add_contract_risk_score(df)
        # row 3: Two year(0) + Credit card(0) = 0
        assert out.loc[3, "contract_risk_score"] == 0

    def test_raises_on_encoded_column(self):
        df = _base_df()
        df["Contract"] = df["Contract"].map(
            {"Month-to-month": 2, "One year": 1, "Two year": 0}
        )
        with pytest.raises(ValueError, match="add_contract_risk_score"):
            add_contract_risk_score(df)


# ---------------------------------------------------------------------------
# engineer_all_features (integration)
# ---------------------------------------------------------------------------

class TestEngineerAllFeatures:
    def test_all_features_present(self):
        df = _base_df()
        out = engineer_all_features(df)
        expected_cols = [
            "tenure_group",
            "avg_monthly_charge",
            "service_count",
            "charge_per_service",
            "is_new_customer",
            "has_premium_support",
            "contract_risk_score",
        ]
        for col in expected_cols:
            assert col in out.columns, f"Missing column: {col}"

    def test_no_nans_introduced(self):
        df = _base_df()
        out = engineer_all_features(df)
        numeric_cols = out.select_dtypes(include="number").columns
        assert not out[numeric_cols].isna().any().any()

    def test_input_not_mutated(self):
        df = _base_df()
        original_cols = set(df.columns)
        _ = engineer_all_features(df)
        assert set(df.columns) == original_cols
