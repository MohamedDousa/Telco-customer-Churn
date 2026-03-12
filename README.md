# Telco Customer Churn Prediction

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
- License: CC0: Public Domain

## Pipeline Overview

1. Project setup and data acquisition
2. EDA and churn pattern discovery
3. Preprocessing and feature engineering
4. Imbalance handling, model training, tuning, and evaluation
5. Explainability with SHAP and LIME
6. Business value analysis, segmentation, and recommendations

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

Processed train/test datasets are written to `data/processed/`, and `models/` stores preprocessing artifacts, trained model files, evaluation summaries, explainability outputs, and business analysis exports.

For safe public sharing, serialized model binaries (`models/*.joblib`) are ignored in git. Recreate them by running the pipeline notebooks in order.

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

## License

This project is released under the MIT License. See `LICENSE`.
