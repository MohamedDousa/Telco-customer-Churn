# Customer Intelligence Engine - Telco Churn

This project predicts whether a telecom customer is likely to churn using the Telco Churn dataset.

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

## Project Structure

- `data/` - Local dataset files
- `notebooks/` - Analysis notebooks
- `src/` - Reusable Python modules
- `requirements.txt` - Pinned dependencies
