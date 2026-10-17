"""Step 0 - Preflight: stop early, with a plain-English message, on the usual first-run problems.

    python preflight.py                 # strict: Hugging Face token is required (used by CI)
    python preflight.py --allow-no-token  # notebook/local use: skip the Hub checks if no token is set
"""
import os
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402

problems, warnings_ = [], []


def check_config_and_data():
    if config.HF_USERNAME.startswith("YOUR_") or not config.HF_USERNAME.strip():
        problems.append(
            "HF_USERNAME in tourism_project/config.py is still the placeholder. Run the notebook's "
            "configuration cell (it writes config.py) or edit the file, then commit and push it."
        )
    if not config.RAW_DATA_PATH.exists():
        problems.append(
            f"Raw data file missing: {config.RAW_DATA_PATH}. It must be committed to the repo "
            "(tourism_project/data/tourism.csv)."
        )
        return
    header = pd.read_csv(config.RAW_DATA_PATH, nrows=5)
    missing = [c for c in [config.TARGET, *config.FEATURE_COLUMNS] if c not in header.columns]
    if missing:
        problems.append(f"tourism.csv is missing expected columns: {missing}")


def check_hub(allow_no_token: bool):
    token = os.getenv("HF_TOKEN", "").strip()
    if not token:
        msg = (
            "HF_TOKEN is not set. In GitHub: repo > Settings > Secrets and variables > Actions > "
            "New repository secret, name it exactly HF_TOKEN and paste a Hugging Face WRITE token."
        )
        (warnings_ if allow_no_token else problems).append(msg)
        return
    try:
        from huggingface_hub import HfApi

        info = HfApi(token=token).whoami()
    except Exception as exc:  # noqa: BLE001
        problems.append(
            f"The Hugging Face token was rejected ({type(exc).__name__}). Create a new token at "
            "https://huggingface.co/settings/tokens with the 'Write' role and update the secret. "
            "Check there is no stray space or newline when pasting."
        )
        return

    name = info.get("name", "")
    orgs = [o.get("name", "") for o in info.get("orgs", [])]
    print(f"Hugging Face token OK - signed in as '{name}'")
    if config.HF_USERNAME.lower() not in [n.lower() for n in [name, *orgs]]:
        problems.append(
            f"HF_USERNAME is '{config.HF_USERNAME}' but this token belongs to '{name}'"
            f"{' (orgs: ' + ', '.join(orgs) + ')' if orgs else ''}. Repos would be created under the "
            "wrong namespace and uploads would fail. Fix HF_USERNAME in tourism_project/config.py."
        )
    token_info = info.get("auth", {}).get("accessToken", {})
    role = token_info.get("role")
    if role == "read":
        problems.append("This Hugging Face token has the 'read' role. Create a token with the 'Write' role.")
    elif role == "fineGrained":
        warnings_.append(
            "Fine-grained token detected: make sure it can create and write repos under "
            f"'{config.HF_USERNAME}' (type: model, dataset, space). A classic 'Write' token is simpler."
        )


def main():
    allow_no_token = "--allow-no-token" in sys.argv
    check_config_and_data()
    check_hub(allow_no_token)
    for w in warnings_:
        print("WARNING:", w)
    if problems:
        print("\nPREFLIGHT FAILED:")
        for i, p in enumerate(problems, 1):
            print(f"  {i}. {p}")
        sys.exit(1)
    print("Preflight passed.")


if __name__ == "__main__":
    main()
