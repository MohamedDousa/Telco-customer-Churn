"""Download the Telco Customer Churn dataset from Kaggle via kagglehub."""

import logging
import shutil
from pathlib import Path

import kagglehub

logger = logging.getLogger(__name__)

KAGGLE_DATASET = "blastchar/telco-customer-churn"


def download_dataset(target_dir: str = "data") -> Path:
    """
    Download the Telco Customer Churn dataset from Kaggle and copy it locally.

    Requires Kaggle API credentials configured via ``~/.kaggle/kaggle.json``
    or the ``KAGGLE_USERNAME`` / ``KAGGLE_KEY`` environment variables.

    Args:
        target_dir: Directory where the raw CSV will be placed.

    Returns:
        Path to the directory containing the downloaded file(s).

    Raises:
        RuntimeError: If the Kaggle download or local copy fails.
    """
    target = Path(target_dir)
    target.mkdir(exist_ok=True)

    try:
        cache_path = kagglehub.dataset_download(KAGGLE_DATASET)
        logger.info("Dataset downloaded to kaggle cache: %s", cache_path)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to download dataset '{KAGGLE_DATASET}' from Kaggle: {exc}"
        ) from exc

    try:
        for file_path in Path(cache_path).iterdir():
            dest = target / file_path.name
            shutil.copy2(file_path, dest)
            logger.info("Copied %s → %s", file_path.name, dest)
    except OSError as exc:
        raise RuntimeError(
            f"Failed to copy dataset files to '{target}': {exc}"
        ) from exc

    return target


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    output_path = download_dataset()
    print(f"Dataset copied to: {output_path.resolve()}")
