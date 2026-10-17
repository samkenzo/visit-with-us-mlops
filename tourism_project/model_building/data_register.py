"""Step 1 - Data registration: upload the raw dataset to a Hugging Face *dataset* repo."""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402
from pipeline_utils import retry  # noqa: E402

from huggingface_hub import HfApi  # noqa: E402


def main():
    token = os.getenv("HF_TOKEN")
    if not token:
        sys.exit("HF_TOKEN is not set. Add it as an environment variable (or a GitHub secret).")
    if config.HF_USERNAME.startswith("YOUR_"):
        sys.exit("Set HF_USERNAME in the notebook config cell first (it writes config.py).")
    if not config.RAW_DATA_PATH.exists():
        sys.exit(f"Raw data not found: {config.RAW_DATA_PATH}")

    api = HfApi(token=token)
    retry(
        lambda: api.create_repo(repo_id=config.DATASET_REPO, repo_type="dataset", private=False, exist_ok=True),
        what="create dataset repo",
    )
    retry(
        lambda: api.upload_file(
            path_or_fileobj=str(config.RAW_DATA_PATH),
            path_in_repo="tourism.csv",
            repo_id=config.DATASET_REPO,
            repo_type="dataset",
            commit_message="Register raw tourism dataset",
        ),
        what="upload tourism.csv",
    )
    if not api.file_exists(repo_id=config.DATASET_REPO, filename="tourism.csv", repo_type="dataset"):
        sys.exit("Upload finished but tourism.csv is not visible in the dataset repo - check the token's permissions.")
    print("Dataset registered:", f"https://huggingface.co/datasets/{config.DATASET_REPO}")


if __name__ == "__main__":
    main()
