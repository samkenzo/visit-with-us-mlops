"""Step 4 - Hosting: push the deployment folder to a Docker-based Hugging Face Space."""
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402
from pipeline_utils import retry  # noqa: E402

from huggingface_hub import HfApi  # noqa: E402

REQUIRED_FILES = ["Dockerfile", "app.py", "requirements.txt", "README.md"]
BAD_STAGES = {"BUILD_ERROR", "RUNTIME_ERROR", "CONFIG_ERROR", "NO_APP_FILE"}
WAIT_SECONDS = int(os.getenv("SPACE_WAIT_SECONDS", "720"))


def wait_for_space(api: HfApi) -> None:
    """Poll the Space until it is RUNNING (or fails). A timeout is reported but not fatal."""
    deadline = time.time() + WAIT_SECONDS
    stage = "UNKNOWN"
    while time.time() < deadline:
        try:
            stage = str(api.get_space_runtime(config.SPACE_REPO).stage)
        except Exception as exc:  # noqa: BLE001
            print(f"  could not read Space status ({type(exc).__name__}); retrying")
            stage = "UNKNOWN"
        print("  Space stage:", stage)
        if stage == "RUNNING":
            print("Space is RUNNING.")
            return
        if stage in BAD_STAGES:
            sys.exit(
                f"Space ended in stage {stage}. Open https://huggingface.co/spaces/{config.SPACE_REPO} "
                "and read the 'Logs' tab (Build / Container) for the exact error."
            )
        time.sleep(15)
    print(f"Space still '{stage}' after {WAIT_SECONDS}s - it may just be building. Check the Space page.")


def main():
    token = os.getenv("HF_TOKEN")
    if not token:
        sys.exit("HF_TOKEN is not set.")
    if not config.DEPLOY_DIR.exists():
        sys.exit(f"Deployment folder not found: {config.DEPLOY_DIR}")
    missing = [f for f in REQUIRED_FILES if not (config.DEPLOY_DIR / f).exists()]
    if missing:
        sys.exit(f"Deployment folder is missing: {missing}")
    if "sdk: docker" not in (config.DEPLOY_DIR / "README.md").read_text():
        sys.exit("deployment/README.md must contain 'sdk: docker' in its YAML header.")

    # tells app.py which model repo to download from
    (config.DEPLOY_DIR / "space_config.json").write_text(json.dumps({"model_repo": config.MODEL_REPO}, indent=2))

    api = HfApi(token=token)
    retry(
        lambda: api.create_repo(
            repo_id=config.SPACE_REPO, repo_type="space", space_sdk="docker", private=False, exist_ok=True
        ),
        what="create Space",
    )
    retry(
        lambda: api.upload_folder(
            folder_path=str(config.DEPLOY_DIR),
            repo_id=config.SPACE_REPO,
            repo_type="space",
            commit_message="Deploy Streamlit app",
            ignore_patterns=["__pycache__/*", "*.pyc"],
        ),
        what="upload Space files",
    )
    print("Space files uploaded:", f"https://huggingface.co/spaces/{config.SPACE_REPO}")

    # The app downloads the model once at start-up. If only the model changed (app files identical),
    # Hugging Face sees no new commit and would keep serving the old model - so restart explicitly.
    try:
        api.restart_space(config.SPACE_REPO)
        print("Space restart requested (so it picks up the newest model).")
    except Exception as exc:  # noqa: BLE001
        print(f"Could not restart the Space ({type(exc).__name__}: {exc}); it will rebuild if files changed.")

    wait_for_space(api)
    print("Space deployed:", f"https://huggingface.co/spaces/{config.SPACE_REPO}")


if __name__ == "__main__":
    main()
