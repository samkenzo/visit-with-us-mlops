# Visit with Us — Wellness Tourism Package Purchase Prediction (MLOps)

Predicts whether a customer will buy the Wellness Tourism Package **before** they are contacted
(only pre-contact attributes are used), and automates the whole workflow with GitHub Actions.

| Stage | File | What it does |
|---|---|---|
| Preflight | `tourism_project/model_building/preflight.py` | Checks the HF token, username and data before anything runs (also first job in CI) |
| Data registration | `tourism_project/model_building/data_register.py` | Uploads `tourism.csv` to a Hugging Face dataset repo |
| Data preparation | `tourism_project/model_building/prep.py` | Cleans, de-duplicates, splits (stratified 80/20), re-uploads |
| Model training | `tourism_project/model_building/train.py` | GridSearchCV on 4 model families, MLflow tracking, threshold tuning, registers best model on the HF Model Hub |
| Deployment | `tourism_project/deployment/` | Dockerfile + Streamlit `app.py` + pinned `requirements.txt` |
| Hosting | `tourism_project/hosting/hosting.py` | Pushes the app to a public Docker Hugging Face Space |
| Tests | `tests/` | Unit tests for cleaning, leakage, the Streamlit app and Hugging Face calls (`pytest -q`) |
| CI/CD | `.github/workflows/pipeline.yml` | `preflight → register-dataset → data-prep → model-training → deploy-hosting` on every push to `main` |

Full walkthrough with EDA, results and insights: `Visit_with_Us_MLOps_Project.ipynb`.

## Setup
1. Use Python 3.11 and `pip install -r requirements.txt` (versions are pinned on purpose).
2. Set `HF_USERNAME` in `tourism_project/config.py` (the notebook's configuration cell does this for you).
3. Add a repository secret named exactly `HF_TOKEN` (Hugging Face **Write** token).
4. Push to `main`; the workflow runs automatically. Run `pytest -q` locally any time.

## Links
- GitHub repository: _add link_
- Hugging Face Space: _add link_
