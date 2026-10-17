"""Fast checks that run in CI before anything touches Hugging Face (no network, no token needed)."""
import json
import os
import sys
from unittest import mock

import joblib
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

import config
import prep
import train
from pipeline_utils import retry


@pytest.fixture(autouse=True)
def _fresh_streamlit_cache():
    """st.cache_resource is process-wide; clear it so app tests cannot see each other's model."""
    import streamlit as st

    st.cache_resource.clear()
    yield


@pytest.fixture(scope="module")
def raw():
    return pd.read_csv(config.RAW_DATA_PATH)


def test_raw_file_has_expected_columns(raw):
    for col in [config.TARGET, *config.FEATURE_COLUMNS, *config.POST_CONTACT_COLUMNS]:
        assert col in raw.columns


def test_cleaning_fixes_labels_and_duplicates(raw):
    df = prep.clean_data(raw)
    assert set(df["Gender"]) == {"Male", "Female"}
    assert "Unmarried" not in set(df["MaritalStatus"])
    assert not df.duplicated().any()
    assert len(df) < len(raw)  # the hidden duplicate customers are removed
    assert set(df[config.TARGET].unique()) == {0, 1}


def test_no_post_contact_columns_reach_the_model(raw):
    data = prep.select_pre_contact_features(prep.clean_data(raw))
    assert not set(config.POST_CONTACT_COLUMNS) & set(data.columns)
    assert sorted(data.drop(columns=[config.TARGET]).columns) == sorted(config.FEATURE_COLUMNS)


def test_preprocessor_and_model_fit_and_predict(raw):
    data = prep.select_pre_contact_features(prep.clean_data(raw))
    X, y = data.drop(columns=[config.TARGET]), data[config.TARGET]
    pipe = Pipeline([("preprocess", train.build_preprocessor(X)), ("model", LogisticRegression(max_iter=2000))])
    pipe.fit(X, y)
    proba = pipe.predict_proba(X.head(5))[:, 1]
    assert ((proba >= 0) & (proba <= 1)).all()
    # unseen category must not crash (handle_unknown="ignore")
    odd = X.head(1).copy()
    odd["Occupation"] = "Astronaut"
    pipe.predict_proba(odd)


def test_candidate_grids_use_model_prefix():
    for name, (_, grid) in train.get_candidates().items():
        assert all(k.startswith("model__") for k in grid), name


def test_retry_retries_then_succeeds_and_gives_up():
    calls = {"n": 0}

    def flaky():
        calls["n"] += 1
        if calls["n"] < 3:
            raise RuntimeError("boom")
        return "ok"

    assert retry(flaky, tries=3, delay=0) == "ok"
    with pytest.raises(RuntimeError):
        retry(lambda: (_ for _ in ()).throw(RuntimeError("x")), tries=2, delay=0)


def test_preflight_rejects_placeholder_username(monkeypatch):
    import preflight

    preflight.problems.clear()
    preflight.warnings_.clear()
    monkeypatch.setattr(config, "HF_USERNAME", "YOUR_HF_USERNAME")
    preflight.check_config_and_data()
    assert any("HF_USERNAME" in p for p in preflight.problems)
    preflight.problems.clear()


def test_preflight_username_mismatch_and_read_token(monkeypatch):
    import preflight

    preflight.problems.clear()
    preflight.warnings_.clear()
    monkeypatch.setenv("HF_TOKEN", "hf_fake")
    monkeypatch.setattr(config, "HF_USERNAME", "someone")
    fake_info = {"name": "other", "orgs": [], "auth": {"accessToken": {"role": "read"}}}
    with mock.patch("huggingface_hub.HfApi.whoami", return_value=fake_info):
        preflight.check_hub(allow_no_token=False)
    text = " ".join(preflight.problems)
    assert "belongs to 'other'" in text and "'read' role" in text
    preflight.problems.clear()


def test_app_loads_a_bundle_and_predicts(tmp_path, raw, monkeypatch):
    """Run the real Streamlit app headlessly against a small model trained on the real data."""
    from streamlit.testing.v1 import AppTest

    data = prep.select_pre_contact_features(prep.clean_data(raw))
    X, y = data.drop(columns=[config.TARGET]), data[config.TARGET]
    pipe = Pipeline([("preprocess", train.build_preprocessor(X)), ("model", LogisticRegression(max_iter=2000))])
    pipe.fit(X, y)
    import sklearn

    bundle = {
        "pipeline": pipe, "threshold": 0.3, "features": X.columns.tolist(), "model_name": "test_lr",
        "metrics_test_tuned_threshold": {"roc_auc": 0.8, "recall": 0.7, "precision": 0.4, "f1": 0.5},
        "sklearn_version": sklearn.__version__,
    }
    path = tmp_path / config.MODEL_FILE
    joblib.dump(bundle, path)
    monkeypatch.setenv("MODEL_PATH", str(path))

    at = AppTest.from_file(str(config.DEPLOY_DIR / "app.py"), default_timeout=60).run()
    assert not at.exception
    at.button[0].click().run()
    assert not at.exception
    assert len(at.metric) == 1 and "%" in at.metric[0].value
    assert not at.error


def test_app_shows_friendly_error_when_no_model(tmp_path, monkeypatch):
    from streamlit.testing.v1 import AppTest

    monkeypatch.delenv("MODEL_PATH", raising=False)
    monkeypatch.delenv("MODEL_REPO", raising=False)
    # copy app alone into an empty folder: no space_config.json and no artifacts fallback
    src = (config.DEPLOY_DIR / "app.py").read_text()
    app = tmp_path / "deployment" / "app.py"
    app.parent.mkdir()
    app.write_text(src)
    at = AppTest.from_file(str(app), default_timeout=60).run()
    assert not at.exception
    assert any("could not be loaded" in e.value for e in at.error)


def test_scripts_call_huggingface_with_valid_arguments(tmp_path, monkeypatch):
    """Autospec the real HfApi so any wrong keyword argument fails here instead of in CI."""
    from huggingface_hub import HfApi

    import data_register
    import hosting

    monkeypatch.setenv("HF_TOKEN", "hf_fake")
    monkeypatch.setattr(config, "HF_USERNAME", "someone")
    for attr, suffix in [("DATASET_REPO", "visit-with-us-tourism-data"), ("SPACE_REPO", "visit-with-us-tourism-app")]:
        monkeypatch.setattr(config, attr, f"someone/{suffix}")
    api = mock.create_autospec(HfApi, instance=True)
    api.file_exists.return_value = True
    api.get_space_runtime.return_value = mock.Mock(stage="RUNNING")

    # data_register
    monkeypatch.setattr(data_register, "HfApi", lambda token=None: api)
    data_register.main()
    assert api.upload_file.called

    # hosting (work on a copy of the deployment folder so space_config.json is not written into the repo)
    import shutil

    deploy = tmp_path / "deployment"
    shutil.copytree(config.DEPLOY_DIR, deploy, ignore=shutil.ignore_patterns("space_config.json", "__pycache__"))
    monkeypatch.setattr(config, "DEPLOY_DIR", deploy)
    monkeypatch.setattr(hosting, "HfApi", lambda token=None: api)
    hosting.main()
    assert api.upload_folder.called and api.restart_space.called
    assert json.loads((deploy / "space_config.json").read_text())["model_repo"] == config.MODEL_REPO


def test_preflight_flags_non_ascii_token(monkeypatch):
    import preflight

    preflight.problems.clear()
    preflight.warnings_.clear()
    monkeypatch.setenv("HF_TOKEN", "hf_abcd\u2021efgh")
    preflight.check_hub(allow_no_token=False)
    assert any("non-ASCII" in p for p in preflight.problems)
    preflight.problems.clear()
