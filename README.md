# Customer Intelligence Engine - Telco Churn

This project predicts telecom customer churn and turns model outputs into explainable, business-focused retention recommendations.

## Setup

1. Create and activate the virtual environment:
   - `python3 -m venv .venv`
   - `source .venv/bin/activate`
2. Install dependencies:
   - `pip install -r requirements.txt`
3. Register Jupyter kernel:
   - `python -m ipykernel install --user --name=churn-env --display-name="Churn Project (Python 3.11)"`

## Dataset

- Source: https://www.kaggle.com/datasets/blastchar/telco-customer-churn
- Download helper: `src/data_loader.py`
- Expected local CSV: `data/WA_Fn-UseC_-Telco-Customer-Churn.csv`

## Pipeline Overview

1. **Phase 0** - Project setup and data acquisition
2. **Phase 1** - EDA and churn pattern discovery
3. **Phase 2** - Preprocessing and feature engineering
4. **Phase 3-4** - Imbalance handling, model training, tuning, and evaluation
5. **Phase 5** - Explainability with SHAP and LIME
6. **Phase 6** - Business value analysis, segmentation, and recommendations

## Run Order

Execute notebooks in this order:

1. `notebooks/01_eda.ipynb`
2. `notebooks/02_preprocessing_feature_engineering.ipynb`
3. `notebooks/03_modeling.ipynb`
4. `notebooks/04_explainability.ipynb`
5. `notebooks/05_business_value.ipynb`

Or execute from terminal:

- `jupyter nbconvert --to notebook --execute --inplace "notebooks/02_preprocessing_feature_engineering.ipynb"`
- `jupyter nbconvert --to notebook --execute --inplace "notebooks/03_modeling.ipynb"`
- `jupyter nbconvert --to notebook --execute --inplace "notebooks/04_explainability.ipynb"`
- `jupyter nbconvert --to notebook --execute --inplace "notebooks/05_business_value.ipynb"`

## Artifacts

- Processed data:
  - `data/processed/X_train.csv`
  - `data/processed/X_test.csv`
  - `data/processed/y_train.csv`
  - `data/processed/y_test.csv`
- Preprocessing artifacts:
  - `models/binary_mapping.joblib`
  - `models/multiclass_encoder.joblib`
  - `models/numeric_scaler.joblib`
- Modeling and explainability artifacts:
  - `models/best_model.joblib`
  - `models/best_threshold.joblib`
  - `models/model_cv_results.csv`
  - `models/model_summary.json`
  - `models/explainability_summary.json`
- Business outputs:
  - `models/business_value_summary.json`
  - `models/profit_curve.csv`
  - `models/segment_recommendations.csv`

## Project Structure

- `data/`
  - raw dataset and processed train/test data
- `models/`
  - preprocessing, model, explainability, and business summary artifacts
- `notebooks/`
  - `01_eda.ipynb`
  - `02_preprocessing_feature_engineering.ipynb`
  - `03_modeling.ipynb`
  - `04_explainability.ipynb`
  - `05_business_value.ipynb`
- `src/`
  - `data_loader.py`
  - `preprocessing.py`
  - `feature_engineering.py`
  - `models.py`
  - `explainability.py`
- `requirements.txt`
