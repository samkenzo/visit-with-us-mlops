"""Central configuration shared by every pipeline script (written by the notebook)."""
from pathlib import Path

HF_USERNAME = "CasioThief"

PROJECT_DIR = Path(__file__).resolve().parent
REPO_ROOT = PROJECT_DIR.parent
DATA_DIR = PROJECT_DIR / "data"
ARTIFACT_DIR = PROJECT_DIR / "model_building" / "artifacts"
DEPLOY_DIR = PROJECT_DIR / "deployment"
RAW_DATA_PATH = DATA_DIR / "tourism.csv"

# Hugging Face Hub repositories (created automatically by the pipeline scripts)
DATASET_REPO = f"{HF_USERNAME}/visit-with-us-tourism-data"
MODEL_REPO = f"{HF_USERNAME}/visit-with-us-tourism-model"
SPACE_REPO = f"{HF_USERNAME}/visit-with-us-tourism-app"

MODEL_FILE = "tourism_model_v1.joblib"
TARGET = "ProdTaken"
ID_COLUMNS = ["Unnamed: 0", "CustomerID"]

# Only known AFTER the salesperson contacts the customer -> excluded, because the
# model must score a customer BEFORE any contact is made.
POST_CONTACT_COLUMNS = [
    "DurationOfPitch",
    "NumberOfFollowups",
    "PitchSatisfactionScore",
    "ProductPitched",
]

# Columns the raw file must contain (checked by preflight.py and prep.py)
FEATURE_COLUMNS = [
    "Age", "TypeofContact", "CityTier", "Occupation", "Gender", "NumberOfPersonVisiting",
    "PreferredPropertyStar", "MaritalStatus", "NumberOfTrips", "Passport", "OwnCar",
    "NumberOfChildrenVisiting", "Designation", "MonthlyIncome",
]

TEST_SIZE = 0.2
RANDOM_STATE = 42
CV_FOLDS = 5

EXPERIMENT_NAME = "visit-with-us-wellness-tourism"
MLFLOW_DEFAULT_URI = "sqlite:///" + str(REPO_ROOT / "mlflow.db")
