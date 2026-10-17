"""Step 2 - Data preparation: load from the HF dataset repo, clean, split, save and re-upload."""
import json
import os
import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402
from pipeline_utils import retry  # noqa: E402


def validate_raw(df: pd.DataFrame) -> None:
    """Fail with a readable message if the raw file does not look like tourism.csv."""
    missing = [c for c in [config.TARGET, *config.FEATURE_COLUMNS] if c not in df.columns]
    if missing:
        sys.exit(f"Raw data is missing expected columns: {missing}")
    if len(df) < 100:
        sys.exit(f"Raw data has only {len(df)} rows - looks truncated.")


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Fix labels, drop identifiers and duplicates. Keeps post-contact columns (for EDA)."""
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]
    df = df.drop(columns=[c for c in config.ID_COLUMNS if c in df.columns])

    for col in df.select_dtypes(exclude="number").columns:
        df[col] = df[col].str.strip()

    # 'Fe Male' is a typo of 'Female'
    gender = df["Gender"].str.replace(" ", "", regex=False).str.lower()
    df["Gender"] = gender.map({"male": "Male", "female": "Female"})
    # 'Unmarried' and 'Single' describe the same status
    df["MaritalStatus"] = df["MaritalStatus"].replace({"Unmarried": "Single"})

    df = df.drop_duplicates()
    df = df[df[config.TARGET].isin([0, 1])].copy()
    df[config.TARGET] = df[config.TARGET].astype(int)
    return df.reset_index(drop=True)


def select_pre_contact_features(df: pd.DataFrame) -> pd.DataFrame:
    """Drop columns that only exist after the customer has been contacted."""
    return df.drop(columns=[c for c in config.POST_CONTACT_COLUMNS if c in df.columns])


def load_raw() -> pd.DataFrame:
    token = os.getenv("HF_TOKEN")
    if token:
        try:
            from huggingface_hub import hf_hub_download

            path = hf_hub_download(
                repo_id=config.DATASET_REPO, filename="tourism.csv", repo_type="dataset", token=token
            )
            print("Loaded raw data from Hugging Face dataset repo:", config.DATASET_REPO)
            return pd.read_csv(path)
        except Exception as exc:  # fall back so the notebook still works offline
            print("Could not read from the Hub (", type(exc).__name__, ") - using local copy.")
    else:
        print("HF_TOKEN not set - using the local copy of tourism.csv")
    return pd.read_csv(config.RAW_DATA_PATH)


def main():
    raw = load_raw()
    validate_raw(raw)
    cleaned = clean_data(raw)
    data = select_pre_contact_features(cleaned)

    X = data.drop(columns=[config.TARGET])
    y = data[config.TARGET]
    Xtrain, Xtest, ytrain, ytest = train_test_split(
        X, y, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE, stratify=y
    )

    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    outputs = {
        "Xtrain.csv": Xtrain,
        "Xtest.csv": Xtest,
        "ytrain.csv": ytrain.to_frame(),
        "ytest.csv": ytest.to_frame(),
    }
    for name, frame in outputs.items():
        frame.to_csv(config.DATA_DIR / name, index=False)

    summary = {
        "raw_rows": int(len(raw)),
        "raw_columns": int(raw.shape[1]),
        "rows_after_cleaning": int(len(cleaned)),
        "duplicates_removed": int(len(raw) - len(cleaned)),
        "pre_contact_features_kept": X.columns.tolist(),
        "post_contact_features_dropped": config.POST_CONTACT_COLUMNS,
        "identifier_columns_dropped": config.ID_COLUMNS,
        "train_rows": int(len(Xtrain)),
        "test_rows": int(len(Xtest)),
        "train_positive_rate": round(float(ytrain.mean()), 4),
        "test_positive_rate": round(float(ytest.mean()), 4),
        "missing_values_remaining": int(X.isna().sum().sum()),
    }
    (config.DATA_DIR / "prep_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))

    token = os.getenv("HF_TOKEN")
    if token:
        from huggingface_hub import HfApi

        api = HfApi(token=token)
        retry(
            lambda: api.create_repo(repo_id=config.DATASET_REPO, repo_type="dataset", private=False, exist_ok=True),
            what="create dataset repo",
        )
        for name in outputs:
            retry(
                lambda name=name: api.upload_file(
                    path_or_fileobj=str(config.DATA_DIR / name),
                    path_in_repo=name,
                    repo_id=config.DATASET_REPO,
                    repo_type="dataset",
                    commit_message=f"Add prepared split: {name}",
                ),
                what=f"upload {name}",
            )
        print("Train/test files uploaded to", f"https://huggingface.co/datasets/{config.DATASET_REPO}")
    else:
        print("HF_TOKEN not set - split files were saved locally only.")


if __name__ == "__main__":
    main()
