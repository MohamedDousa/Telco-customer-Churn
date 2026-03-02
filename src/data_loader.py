import shutil
from pathlib import Path

import kagglehub


def download_dataset(target_dir: str = "data") -> Path:
    target = Path(target_dir)
    target.mkdir(exist_ok=True)
    cache_path = kagglehub.dataset_download("blastchar/telco-customer-churn")

    for file_path in Path(cache_path).iterdir():
        shutil.copy2(file_path, target / file_path.name)

    return target


if __name__ == "__main__":
    output_path = download_dataset()
    print(f"Dataset copied to: {output_path.resolve()}")
