#!/usr/bin/env bash
# Executes the full Telco Churn pipeline from raw data to business summary.
# Run from the project root with the virtual environment active:
#   source .venv/bin/activate && pip install -e .
#   bash run_pipeline.sh
#
# Notebook 01 (EDA) is intentionally excluded — it is exploratory only and
# does not produce any pipeline artifacts.

set -euo pipefail

NB_DIR="notebooks"
KERNEL="${KERNEL:-churn-env}"
TIMEOUT="${TIMEOUT:-600}"

# Verify the virtual environment is active.
if [[ -z "${VIRTUAL_ENV:-}" ]]; then
    echo "ERROR: No virtual environment is active." >&2
    echo "Run 'source .venv/bin/activate' before executing this script." >&2
    exit 1
fi

_run_notebook() {
    local nb="$1"
    local label="$2"
    echo "==> ${label}"
    echo "    Notebook : ${nb}"
    echo "    Kernel   : ${KERNEL}"
    echo "    Timeout  : ${TIMEOUT}s"
    echo "    Started  : $(date '+%Y-%m-%d %H:%M:%S')"
    jupyter nbconvert \
        --to notebook \
        --execute \
        --inplace \
        --ExecutePreprocessor.kernel_name="${KERNEL}" \
        --ExecutePreprocessor.timeout="${TIMEOUT}" \
        "${nb}"
    echo "    Finished : $(date '+%Y-%m-%d %H:%M:%S')"
    echo ""
}

_run_notebook "${NB_DIR}/02_preprocessing_feature_engineering.ipynb" "[1/4] Preprocessing and feature engineering"
_run_notebook "${NB_DIR}/03_modeling.ipynb"                           "[2/4] Modelling and evaluation"
_run_notebook "${NB_DIR}/04_explainability.ipynb"                     "[3/4] Explainability (SHAP / LIME)"
_run_notebook "${NB_DIR}/05_business_value.ipynb"                     "[4/4] Business value analysis"

echo "Pipeline complete. Artifacts written to data/processed/ and models/."
