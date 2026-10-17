"""Streamlit front-end for the Visit with Us Wellness Tourism purchase predictor."""
import json
import os
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

MODEL_FILE = "tourism_model_v1.joblib"
HERE = Path(__file__).resolve().parent

st.set_page_config(page_title="Visit with Us | Wellness Tourism", page_icon="🌿", layout="wide")


@st.cache_resource(show_spinner="Loading model...")
def load_bundle():
    """Return (bundle, source, notes). Order: local override -> Hugging Face Model Hub -> local artifacts folder.

    Never calls st.* inside the cache; problems are returned as text and shown by the caller.
    """
    notes = []
    local = os.getenv("MODEL_PATH")
    if local and Path(local).exists():
        return joblib.load(local), f"local file {local}", notes

    repo = os.getenv("MODEL_REPO")
    cfg = HERE / "space_config.json"
    if not repo and cfg.exists():
        repo = json.loads(cfg.read_text()).get("model_repo")
    if repo:
        try:
            from huggingface_hub import hf_hub_download

            path = hf_hub_download(repo_id=repo, filename=MODEL_FILE)
            return joblib.load(path), f"Hugging Face Hub: {repo}", notes
        except Exception as exc:  # noqa: BLE001
            notes.append(f"Could not load the model from the Hub ({type(exc).__name__}: {exc}). Trying a local copy.")

    fallback = HERE.parent / "model_building" / "artifacts" / MODEL_FILE
    if fallback.exists():
        return joblib.load(fallback), f"local file {fallback.name}", notes
    return None, "", notes


bundle, source, load_notes = load_bundle()
for note in load_notes:
    st.warning(note)
if bundle is None:
    st.error(
        "The model could not be loaded. Make sure the training step finished and registered "
        "the model on the Hugging Face Hub, then restart this Space."
    )
    st.stop()

pipeline, threshold, features = bundle["pipeline"], float(bundle["threshold"]), bundle["features"]

try:
    import sklearn

    trained_with = bundle.get("sklearn_version")
    if trained_with and trained_with != sklearn.__version__:
        st.warning(
            f"The model was trained with scikit-learn {trained_with} but this app runs {sklearn.__version__}. "
            "Predictions may fail or differ - pin the same version in requirements.txt."
        )
except Exception:  # noqa: BLE001
    pass

st.title("🌿 Wellness Tourism Package - Purchase Predictor")
st.write(
    "Enter what is known about a customer **before any sales contact** to estimate how likely they are to buy "
    "the Wellness Tourism Package. Use the score to prioritise outreach, not to exclude anyone."
)

with st.form("customer_form"):
    st.subheader("Customer profile")
    c1, c2, c3 = st.columns(3)
    with c1:
        age = st.number_input("Age", 18, 100, 35)
        gender = st.selectbox("Gender", ["Male", "Female"])
        marital = st.selectbox("Marital status", ["Single", "Married", "Divorced"])
        occupation = st.selectbox("Occupation", ["Salaried", "Small Business", "Large Business", "Free Lancer"])
        designation = st.selectbox("Designation", ["Executive", "Manager", "Senior Manager", "AVP", "VP"])
    with c2:
        income = st.number_input("Gross monthly income", 0, 1_000_000, 22000, step=500)
        city = st.selectbox("City tier (1 = most developed)", [1, 2, 3])
        contact = st.selectbox("Type of contact", ["Self Enquiry", "Company Invited"])
        passport = st.selectbox("Holds a valid passport?", ["No", "Yes"])
        car = st.selectbox("Owns a car?", ["Yes", "No"])
    with c3:
        people = st.number_input("People visiting (total)", 1, 10, 3)
        children = st.number_input("Children visiting (below age 5)", 0, 5, 1)
        stars = st.selectbox("Preferred property star rating", [3, 4, 5])
        trips = st.number_input("Trips per year", 0, 30, 3)
    submitted = st.form_submit_button("Predict purchase likelihood", type="primary")

if submitted:
    record = pd.DataFrame(
        [
            {
                "Age": age,
                "TypeofContact": contact,
                "CityTier": city,
                "Occupation": occupation,
                "Gender": gender,
                "NumberOfPersonVisiting": people,
                "PreferredPropertyStar": stars,
                "MaritalStatus": marital,
                "NumberOfTrips": trips,
                "Passport": 1 if passport == "Yes" else 0,
                "OwnCar": 1 if car == "Yes" else 0,
                "NumberOfChildrenVisiting": children,
                "Designation": designation,
                "MonthlyIncome": income,
            }
        ]
    )
    missing = [c for c in features if c not in record.columns]
    if missing:
        st.error(f"The form is missing fields the model expects: {missing}")
        st.stop()
    record = record[features]
    try:
        prob = float(pipeline.predict_proba(record)[0, 1])
    except Exception as exc:  # noqa: BLE001
        st.error(f"Prediction failed: {exc}")
        st.stop()

    left, right = st.columns([1, 2])
    left.metric("Purchase probability", f"{prob:.1%}")
    right.progress(min(max(prob, 0.0), 1.0))
    if prob >= threshold:
        st.success(f"HIGH priority - score is at or above the decision threshold ({threshold:.0%}). Contact this customer first.")
    elif prob >= threshold / 2:
        st.warning("MEDIUM priority - worth a low-cost touch such as email or a self-service offer.")
    else:
        st.info("LOW priority - unlikely to buy right now; keep in the general mailing list.")
    st.caption("This is a statistical estimate, not a guarantee.")

with st.expander("About this model"):
    st.write(f"**Model:** {bundle.get('model_name')}  |  **Decision threshold:** {threshold:.2f}")
    st.write(f"**Loaded from:** {source}")
    metrics = bundle.get("metrics_test_tuned_threshold", {})
    if metrics:
        st.write(
            f"Held-out test performance - ROC-AUC {metrics['roc_auc']:.3f}, recall {metrics['recall']:.3f}, "
            f"precision {metrics['precision']:.3f}, F1 {metrics['f1']:.3f}."
        )
    st.write("Only attributes known before contact are used (no pitch duration, follow-ups or pitch satisfaction).")
