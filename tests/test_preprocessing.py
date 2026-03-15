from __future__ import annotations

import io

import pandas as pd
import pytest

from src.preprocessing import (
    encode_binary_columns,
    encode_multiclass_columns,
    load_and_clean,
    scale_numeric_columns,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_RAW_CSV = """\
customerID,gender,SeniorCitizen,Partner,Dependents,tenure,PhoneService,MultipleLines,InternetService,OnlineSecurity,OnlineBackup,DeviceProtection,TechSupport,StreamingTV,StreamingMovies,Contract,PaperlessBilling,PaymentMethod,MonthlyCharges,TotalCharges,Churn
A1,Male,0,Yes,No,1,Yes,No,DSL,No,Yes,No,No,No,No,Month-to-month,Yes,Electronic check,29.85, ,No
A2,Female,0,No,No,34,Yes,No,DSL,Yes,No,Yes,No,No,No,One year,No,Mailed check,56.95,1889.50,No
A3,Male,0,No,No,2,Yes,No,DSL,Yes,Yes,No,No,No,No,Month-to-month,Yes,Mailed check,53.85,108.15,Yes
"""


def _make_df() -> pd.DataFrame:
    return pd.read_csv(io.StringIO(_RAW_CSV))


# ---------------------------------------------------------------------------
# load_and_clean
# ---------------------------------------------------------------------------

class TestLoadAndClean:
    def test_total_charges_blank_becomes_zero(self, tmp_path):
        csv = tmp_path / "data.csv"
        csv.write_text(_RAW_CSV)
        df = load_and_clean(str(csv))
        assert df.loc[0, "TotalCharges"] == 0.0

    def test_total_charges_dtype_is_float(self, tmp_path):
        csv = tmp_path / "data.csv"
        csv.write_text(_RAW_CSV)
        df = load_and_clean(str(csv))
        assert pd.api.types.is_float_dtype(df["TotalCharges"])

    def test_customer_id_dropped(self, tmp_path):
        csv = tmp_path / "data.csv"
        csv.write_text(_RAW_CSV)
        df = load_and_clean(str(csv))
        assert "customerID" not in df.columns

    def test_churn_is_integer(self, tmp_path):
        csv = tmp_path / "data.csv"
        csv.write_text(_RAW_CSV)
        df = load_and_clean(str(csv))
        assert df["Churn"].dtype == int
        assert set(df["Churn"].unique()).issubset({0, 1})


# ---------------------------------------------------------------------------
# encode_binary_columns
# ---------------------------------------------------------------------------

class TestEncodeBinaryColumns:
    def _base_df(self) -> pd.DataFrame:
        return pd.DataFrame({"gender": ["Male", "Female", "Male"], "Partner": ["Yes", "No", "Yes"]})

    def test_yes_no_encoded_to_0_1(self):
        df = self._base_df()
        encoded, mapping = encode_binary_columns(df, ["Partner"])
        assert list(encoded["Partner"]) == [1, 0, 1]
        assert mapping["Partner"] == {"No": 0, "Yes": 1}

    def test_male_female_encoded_to_0_1(self):
        df = self._base_df()
        encoded, mapping = encode_binary_columns(df, ["gender"])
        assert mapping["gender"] == {"Female": 0, "Male": 1}

    def test_provided_mapping_is_respected(self):
        df = self._base_df()
        _, learned = encode_binary_columns(df, ["gender"])
        df2 = pd.DataFrame({"gender": ["Female", "Male"]})
        encoded, _ = encode_binary_columns(df2, ["gender"], mapping=learned)
        assert list(encoded["gender"]) == [0, 1]

    def test_non_binary_column_raises(self):
        df = pd.DataFrame({"color": ["red", "green", "blue"]})
        with pytest.raises(ValueError, match="not binary"):
            encode_binary_columns(df, ["color"])

    def test_unmapped_value_raises(self):
        df_train = pd.DataFrame({"flag": ["Yes", "No"]})
        _, mapping = encode_binary_columns(df_train, ["flag"])
        df_test = pd.DataFrame({"flag": ["Yes", "Maybe"]})
        with pytest.raises(ValueError, match="unmapped values"):
            encode_binary_columns(df_test, ["flag"], mapping=mapping)

    def test_missing_column_is_skipped(self):
        df = pd.DataFrame({"other": [1, 2]})
        encoded, mapping = encode_binary_columns(df, ["nonexistent"])
        assert list(encoded.columns) == ["other"]
        assert mapping == {}


# ---------------------------------------------------------------------------
# encode_multiclass_columns
# ---------------------------------------------------------------------------

class TestEncodeMulticlassColumns:
    def _make_train_test(self):
        train = pd.DataFrame({"Contract": ["Month-to-month", "One year", "Two year", "One year"]})
        test = pd.DataFrame({"Contract": ["Two year", "Month-to-month"]})
        return train, test

    def test_output_shapes_correct(self):
        train, test = self._make_train_test()
        enc_train, enc_test, _ = encode_multiclass_columns(train, test, ["Contract"])
        # drop="first" drops one category, so 2 columns from 3 unique values
        assert enc_train.shape[1] == 2
        assert enc_test.shape[1] == 2

    def test_encoder_fit_on_train_only(self):
        train, test = self._make_train_test()
        _, _, encoder = encode_multiclass_columns(train, test, ["Contract"])
        categories = encoder.categories_[0].tolist()
        assert "Month-to-month" in categories

    def test_unknown_test_category_does_not_raise(self):
        train = pd.DataFrame({"Contract": ["Month-to-month", "One year"]})
        test = pd.DataFrame({"Contract": ["Unknown-plan"]})
        enc_train, enc_test, _ = encode_multiclass_columns(train, test, ["Contract"])
        # handle_unknown="ignore" means unknown rows become all zeros
        assert enc_test.shape[1] == enc_train.shape[1]


# ---------------------------------------------------------------------------
# scale_numeric_columns
# ---------------------------------------------------------------------------

class TestScaleNumericColumns:
    def test_scaler_fit_on_train_not_test(self):
        train = pd.DataFrame({"tenure": [0.0, 12.0, 24.0, 36.0]})
        # test has an extreme outlier that should not affect scaling
        test = pd.DataFrame({"tenure": [9999.0]})
        scaled_train, scaled_test, scaler = scale_numeric_columns(train, test, ["tenure"])
        # median should be ~18 → scaled median ≈ 0 in train
        median_scaled = scaled_train["tenure"].median()
        assert abs(median_scaled) < 0.2

    def test_missing_column_is_skipped(self):
        train = pd.DataFrame({"a": [1.0, 2.0]})
        test = pd.DataFrame({"a": [3.0]})
        scaled_train, scaled_test, _ = scale_numeric_columns(train, test, ["nonexistent"])
        assert list(scaled_train["a"]) == [1.0, 2.0]
