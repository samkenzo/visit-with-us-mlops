"""Step 3 - Model training with hyper-parameter tuning, MLflow tracking and HF model registration."""
import json
import os
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import mlflow  # noqa: E402
import mlflow.sklearn  # noqa: E402,F401
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import sklearn  # noqa: E402
from sklearn.base import clone  # noqa: E402
from sklearn.compose import ColumnTransformer  # noqa: E402
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier  # noqa: E402
from sklearn.impute import SimpleImputer  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_predict  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.preprocessing import OneHotEncoder, StandardScaler  # noqa: E402
from sklearn.tree import DecisionTreeClassifier  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402
from pipeline_utils import retry  # noqa: E402

warnings.filterwarnings("ignore")
os.environ.setdefault("MLFLOW_HTTP_REQUEST_MAX_RETRIES", "2")
os.environ.setdefault("MLFLOW_HTTP_REQUEST_BACKOFF_FACTOR", "1")

RS = config.RANDOM_STATE


# ----------------------------------------------------------------------------- data
def load_split():
    token = os.getenv("HF_TOKEN")
    frames = {}
    for name in ["Xtrain.csv", "Xtest.csv", "ytrain.csv", "ytest.csv"]:
        path = config.DATA_DIR / name
        if token:
            try:
                from huggingface_hub import hf_hub_download

                path = hf_hub_download(
                    repo_id=config.DATASET_REPO, filename=name, repo_type="dataset", token=token
                )
            except Exception as exc:
                print(f"Could not download {name} from the Hub ({type(exc).__name__}); using local copy.")
        if not Path(path).exists():
            sys.exit(f"{name} not found. Run prep.py first.")
        frames[name] = pd.read_csv(path)
    return (
        frames["Xtrain.csv"],
        frames["Xtest.csv"],
        frames["ytrain.csv"].iloc[:, 0],
        frames["ytest.csv"].iloc[:, 0],
    )


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    numeric = X.select_dtypes(include="number").columns.tolist()
    categorical = X.select_dtypes(exclude="number").columns.tolist()
    return ColumnTransformer(
        [
            ("numeric", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), numeric),
            (
                "categorical",
                Pipeline(
                    [
                        ("imp", SimpleImputer(strategy="most_frequent")),
                        ("ohe", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical,
            ),
        ]
    )


# ------------------------------------------------------------------------- helpers
def evaluate(y_true, proba, threshold=0.5) -> dict:
    pred = (proba >= threshold).astype(int)
    return {
        "accuracy": float(accuracy_score(y_true, pred)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, proba)),
        "average_precision": float(average_precision_score(y_true, proba)),
    }


def setup_mlflow() -> str:
    uri = os.getenv("MLFLOW_TRACKING_URI", config.MLFLOW_DEFAULT_URI)
    try:
        mlflow.set_tracking_uri(uri)
        mlflow.set_experiment(config.EXPERIMENT_NAME)
    except Exception as exc:
        print(f"MLflow server at {uri} not reachable ({type(exc).__name__}); using local SQLite store.")
        uri = config.MLFLOW_DEFAULT_URI
        mlflow.set_tracking_uri(uri)
        mlflow.set_experiment(config.EXPERIMENT_NAME)
    print("MLflow tracking URI:", uri)
    return uri


def safe_log_artifact(path):
    try:
        mlflow.log_artifact(str(path))
    except Exception as exc:  # artifact store problems must not kill the pipeline
        print(f"  (could not log artifact {Path(path).name}: {type(exc).__name__})")


def clean_names(names):
    return [n.split("__", 1)[-1] for n in names]


def get_candidates():
    return {
        "logistic_regression": (
            LogisticRegression(max_iter=3000, class_weight="balanced", random_state=RS),
            {"model__C": [0.05, 0.1, 0.5, 1, 5, 10]},
        ),
        "decision_tree": (
            DecisionTreeClassifier(class_weight="balanced", random_state=RS),
            {"model__max_depth": [4, 6, 8, 12], "model__min_samples_leaf": [1, 5, 10, 20]},
        ),
        "random_forest": (
            RandomForestClassifier(class_weight="balanced_subsample", random_state=RS, n_jobs=1),
            {
                "model__n_estimators": [150, 300],
                "model__max_depth": [8, 12, None],
                "model__min_samples_leaf": [1, 3],
            },
        ),
        "gradient_boosting": (
            GradientBoostingClassifier(random_state=RS),
            {
                "model__n_estimators": [100, 200],
                "model__learning_rate": [0.05, 0.1],
                "model__max_depth": [2, 3, 4],
            },
        ),
    }


def track_family(name, gs, scoring, m_test, m_train):
    """Log every grid-search trial plus the family summary. Tracking problems never stop the pipeline."""
    try:
        with mlflow.start_run(run_name=name):
            mlflow.set_tag("model_family", name)
            for i, params in enumerate(gs.cv_results_["params"]):
                with mlflow.start_run(run_name=f"{name}-trial-{i:02d}", nested=True):
                    mlflow.log_params({k: str(v) for k, v in params.items()})
                    mlflow.log_metrics(
                        {f"cv_{m}": float(gs.cv_results_[f"mean_test_{m}"][i]) for m in scoring}
                    )
            mlflow.log_params({f"best_{k}": str(v) for k, v in gs.best_params_.items()})
            mlflow.log_metrics({f"cv_best_{m}": float(gs.cv_results_[f"mean_test_{m}"][gs.best_index_]) for m in scoring})
            mlflow.log_metrics({f"test_{k}": v for k, v in m_test.items()})
            mlflow.log_metrics({f"train_{k}": v for k, v in m_train.items()})
    except Exception as exc:  # noqa: BLE001
        print(f"  (MLflow logging for {name} failed: {type(exc).__name__}: {exc}) - continuing")
        while mlflow.active_run():  # make sure no half-open run blocks the next family
            mlflow.end_run()


# ------------------------------------------------------------------------- main
def main():
    config.ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    Xtrain, Xtest, ytrain, ytest = load_split()
    print(f"Train: {Xtrain.shape}  Test: {Xtest.shape}  Positive rate (train): {ytrain.mean():.3f}")
    print("Features:", Xtrain.columns.tolist())
    setup_mlflow()

    scoring = {
        "roc_auc": "roc_auc",
        "average_precision": "average_precision",
        "f1": "f1",
        "recall": "recall",
        "precision": "precision",
    }
    cv = StratifiedKFold(n_splits=config.CV_FOLDS, shuffle=True, random_state=RS)

    rows, tuned = [], {}
    for name, (estimator, grid) in get_candidates().items():
        print(f"\nTuning {name} ({np.prod([len(v) for v in grid.values()])} combinations x {config.CV_FOLDS} folds)")
        pipe = Pipeline([("preprocess", build_preprocessor(Xtrain)), ("model", estimator)])
        gs = GridSearchCV(pipe, grid, scoring=scoring, refit="average_precision", cv=cv, n_jobs=-1)
        gs.fit(Xtrain, ytrain)
        best = gs.best_estimator_
        tuned[name] = gs

        train_proba = best.predict_proba(Xtrain)[:, 1]
        test_proba = best.predict_proba(Xtest)[:, 1]
        m_train, m_test = evaluate(ytrain, train_proba), evaluate(ytest, test_proba)

        track_family(name, gs, scoring, m_test, m_train)

        rows.append(
            {
                "model": name,
                "cv_average_precision": gs.best_score_,
                "cv_roc_auc": float(gs.cv_results_["mean_test_roc_auc"][gs.best_index_]),
                "cv_f1": float(gs.cv_results_["mean_test_f1"][gs.best_index_]),
                "test_accuracy": m_test["accuracy"],
                "test_precision": m_test["precision"],
                "test_recall": m_test["recall"],
                "test_f1": m_test["f1"],
                "test_roc_auc": m_test["roc_auc"],
                "train_roc_auc": m_train["roc_auc"],
                "best_params": json.dumps({k: str(v) for k, v in gs.best_params_.items()}),
            }
        )
        print(f"  best CV average-precision: {gs.best_score_:.4f}  | test ROC-AUC: {m_test['roc_auc']:.4f}")

    comparison = pd.DataFrame(rows).sort_values("cv_average_precision", ascending=False).reset_index(drop=True)
    comparison.to_csv(config.ARTIFACT_DIR / "model_comparison.csv", index=False)

    # ---- model selection uses cross-validation on TRAIN only (the test set is not used to choose)
    best_name = comparison.loc[0, "model"]
    best_gs = tuned[best_name]
    best_model = best_gs.best_estimator_
    print("\nSelected model:", best_name, best_gs.best_params_)

    # ---- decision threshold chosen from out-of-fold predictions on TRAIN (maximise F1)
    oof = cross_val_predict(clone(best_model), Xtrain, ytrain, cv=cv, method="predict_proba", n_jobs=-1)[:, 1]
    grid_t = np.round(np.arange(0.10, 0.91, 0.01), 2)
    f1s = [f1_score(ytrain, (oof >= t).astype(int), zero_division=0) for t in grid_t]
    threshold = float(grid_t[int(np.argmax(f1s))])
    print(f"Decision threshold (max OOF F1 on train): {threshold:.2f}")

    test_proba = best_model.predict_proba(Xtest)[:, 1]
    test_pred = (test_proba >= threshold).astype(int)
    metrics_tuned = evaluate(ytest, test_proba, threshold)
    metrics_default = evaluate(ytest, test_proba, 0.5)
    cm = confusion_matrix(ytest, test_pred)
    print("\nTest metrics @ tuned threshold:", {k: round(v, 4) for k, v in metrics_tuned.items()})
    print("Test metrics @ 0.50 threshold :", {k: round(v, 4) for k, v in metrics_default.items()})
    print("Confusion matrix [[TN, FP], [FN, TP]]:", cm.tolist())
    print(classification_report(ytest, test_pred, zero_division=0))

    # ---- plots
    plot_paths = []
    try:
        fig, ax = plt.subplots(figsize=(4.5, 4))
        ax.imshow(cm, cmap="Blues")
        for (i, j), v in np.ndenumerate(cm):
            ax.text(j, i, int(v), ha="center", va="center", fontsize=14, color="black")
        ax.set_xticks([0, 1], ["No purchase", "Purchase"])
        ax.set_yticks([0, 1], ["No purchase", "Purchase"])
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title(f"Confusion matrix (threshold {threshold:.2f})")
        p = config.ARTIFACT_DIR / "confusion_matrix.png"
        fig.tight_layout()
        fig.savefig(p, dpi=130)
        plt.close(fig)
        plot_paths.append(p)

        fig, ax = plt.subplots(1, 2, figsize=(10, 4))
        fpr, tpr, _ = roc_curve(ytest, test_proba)
        ax[0].plot(fpr, tpr, label=f"AUC = {metrics_tuned['roc_auc']:.3f}")
        ax[0].plot([0, 1], [0, 1], "--", color="grey")
        ax[0].set(title="ROC curve (test)", xlabel="False positive rate", ylabel="True positive rate")
        ax[0].legend()
        ax[1].plot(grid_t, f1s, color="tab:green")
        ax[1].axvline(threshold, ls="--", color="tab:red", label=f"chosen = {threshold:.2f}")
        ax[1].set(title="F1 vs threshold (out-of-fold, train)", xlabel="Threshold", ylabel="F1")
        ax[1].legend()
        p = config.ARTIFACT_DIR / "roc_and_threshold.png"
        fig.tight_layout()
        fig.savefig(p, dpi=130)
        plt.close(fig)
        plot_paths.append(p)

        names = clean_names(best_model.named_steps["preprocess"].get_feature_names_out())
        mdl = best_model.named_steps["model"]
        if hasattr(mdl, "feature_importances_"):
            vals = mdl.feature_importances_
        elif hasattr(mdl, "coef_"):
            vals = np.abs(mdl.coef_.ravel())
        else:
            vals = None
        if vals is not None:
            imp = pd.Series(vals, index=names).sort_values(ascending=False)
            imp.to_csv(config.ARTIFACT_DIR / "feature_importance.csv", header=["importance"])
            top = imp.head(15)[::-1]
            fig, ax = plt.subplots(figsize=(7, 5))
            ax.barh(top.index, top.values, color="tab:blue")
            ax.set_title(f"Top 15 features - {best_name}")
            p = config.ARTIFACT_DIR / "feature_importance.png"
            fig.tight_layout()
            fig.savefig(p, dpi=130)
            plt.close(fig)
            plot_paths.append(p)
    except Exception as exc:
        print("Plotting skipped:", type(exc).__name__, exc)

    # ---- persist model bundle + evaluation report
    bundle = {
        "pipeline": best_model,
        "threshold": threshold,
        "features": Xtrain.columns.tolist(),
        "model_name": best_name,
        "best_params": {k: str(v) for k, v in best_gs.best_params_.items()},
        "metrics_test_tuned_threshold": metrics_tuned,
        "sklearn_version": sklearn.__version__,
        "trained_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    model_path = config.ARTIFACT_DIR / config.MODEL_FILE
    joblib.dump(bundle, model_path)

    report = {
        "selected_model": best_name,
        "best_params": bundle["best_params"],
        "threshold": threshold,
        "features": bundle["features"],
        "metrics_test_tuned_threshold": metrics_tuned,
        "metrics_test_default_threshold_0.5": metrics_default,
        "confusion_matrix_[[TN,FP],[FN,TP]]": cm.tolist(),
        "train_rows": int(len(Xtrain)),
        "test_rows": int(len(Xtest)),
        "positive_rate_train": float(ytrain.mean()),
        "sklearn_version": sklearn.__version__,
    }
    eval_path = config.ARTIFACT_DIR / "evaluation.json"
    eval_path.write_text(json.dumps(report, indent=2))

    # ---- final MLflow run for the registered model
    try:
        with mlflow.start_run(run_name="best-model-final"):
            mlflow.set_tag("selected_model", best_name)
            mlflow.log_params({"model": best_name, "threshold": threshold, **{f"best_{k}": v for k, v in bundle["best_params"].items()}})
            mlflow.log_metrics({f"final_test_{k}": v for k, v in metrics_tuned.items()})
            for p in [model_path, eval_path, config.ARTIFACT_DIR / "model_comparison.csv", *plot_paths]:
                safe_log_artifact(p)
            try:
                mlflow.sklearn.log_model(best_model, "model")
            except Exception as exc:
                print(f"  (mlflow model flavour not logged: {type(exc).__name__})")
    except Exception as exc:  # noqa: BLE001
        print(f"  (final MLflow run failed: {type(exc).__name__}: {exc}) - continuing to model registration")
        while mlflow.active_run():
            mlflow.end_run()

    # ---- register the model on the Hugging Face Model Hub
    token = os.getenv("HF_TOKEN")
    if token:
        from huggingface_hub import HfApi

        api = HfApi(token=token)
        retry(
            lambda: api.create_repo(repo_id=config.MODEL_REPO, repo_type="model", private=False, exist_ok=True),
            what="create model repo",
        )
        card = (
            "---\nlibrary_name: sklearn\ntags:\n- tabular-classification\n- tourism\n---\n"
            f"# Wellness Tourism purchase model\n\nSelected model: **{best_name}**  \n"
            f"Decision threshold: **{threshold:.2f}**  \n"
            f"Test ROC-AUC: **{metrics_tuned['roc_auc']:.3f}**, recall: **{metrics_tuned['recall']:.3f}**, "
            f"precision: **{metrics_tuned['precision']:.3f}**, F1: **{metrics_tuned['f1']:.3f}**\n\n"
            "Uses only pre-contact customer attributes. Load with `joblib.load` "
            f"(scikit-learn {sklearn.__version__}).\n"
        )
        retry(
            lambda: api.upload_file(
                path_or_fileobj=card.encode(), path_in_repo="README.md", repo_id=config.MODEL_REPO, repo_type="model"
            ),
            what="upload model card",
        )
        for path in [model_path, eval_path]:
            retry(
                lambda path=path: api.upload_file(
                    path_or_fileobj=str(path),
                    path_in_repo=path.name,
                    repo_id=config.MODEL_REPO,
                    repo_type="model",
                    commit_message=f"Register {best_name} model",
                ),
                what=f"upload {path.name}",
            )
        if not api.file_exists(repo_id=config.MODEL_REPO, filename=config.MODEL_FILE, repo_type="model"):
            sys.exit("Upload finished but the model file is not visible in the model repo - check token permissions.")
        print("Model registered:", f"https://huggingface.co/{config.MODEL_REPO}")
    else:
        print("HF_TOKEN not set - model saved locally only:", model_path)


if __name__ == "__main__":
    main()
