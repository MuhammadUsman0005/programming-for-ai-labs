---
title: Adult Income ML Engineering Demo
sdk: docker
app_port: 7860
---

# Q3 — Adult Income: baseline, AutoML, API, and Streamlit

## Dataset

Use the professor's Kaggle link:
https://www.kaggle.com/datasets/mosapabdelghany/adult-income-prediction-dataset

Copy the labelled CSV to `data/adult.csv`. The target must contain income classes equivalent to `<=50K` and `>50K` (or binary 0/1). See `data/README_DATA.md`.

## Recommended environment

Use a dedicated Python environment. Current TPOT documentation lists Python >=3.10 and <3.14, which includes 3.13; however, AutoML dependency resolution can still vary by operating system. If installation fails on Python 3.13, use a fresh Python 3.12 environment and reinstall from `requirements.txt`.

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## First run: baseline only (quick local check)

```powershell
python -m src.train_adult --baseline-only
```

This runs EDA, preprocessing, the Logistic Regression baseline, creates initial results, and saves a working model bundle. It does not complete the AutoML requirement.

## Full run: baseline + TPOT AutoML

```powershell
python -m src.train_adult --automl
```

The TPOT run is configured for 30 minutes, matching the brief's approximate search-time guidance. It also limits each candidate pipeline evaluation to 2 minutes. AutoML can use significant CPU and memory. Do not start it during a time when you need the computer for other work.

## Run the API

In terminal 1:

```powershell
uvicorn api.main:app --reload
```

Open `http://127.0.0.1:8000/docs` to inspect and test the endpoints.

- `GET /health`
- `GET /model-info`
- `POST /predict`

## Run the Streamlit interface

In terminal 2, same environment and project folder:

```powershell
streamlit run streamlit_app.py
```

The UI calls the local FastAPI endpoint at `http://127.0.0.1:8000`. You can override it with the environment variable `API_URL`.

## Generated outputs

- `outputs/class_balance.png`
- `outputs/age_distribution.png`
- `outputs/numeric_feature_distributions.png`
- `outputs/categorical_feature_distributions.png`
- `outputs/numeric_correlation.png`
- `outputs/confusion_matrix_baseline.png`
- `outputs/confusion_matrix_automl.png` (full run)
- `outputs/model_comparison.csv`
- `outputs/feature_importance_top5.csv`
- `outputs/feature_importance_top5.png`
- `outputs/run_summary.json`
- `models/model_bundle.joblib`

## Important methodological note

The script reports baseline and AutoML test-set metrics because the assignment requests a test-set comparison. For rigorous production ML, do not repeatedly select/tune models using the test set; use cross-validation/validation data for model selection and leave the test set for final evaluation.

## Fairness / responsible-use note

The Adult dataset includes sensitive demographic attributes such as sex and race and reflects historical census data. This app is a classroom demonstration only. It must not be used for real employment, lending, eligibility, or other decisions about people.

See `docs/DEPLOYMENT_GUIDE.md` for publishing the Streamlit application to Hugging Face Spaces using the included Dockerfile. You still need to create your own account/Space and publish the files; no live deployment URL is fabricated in this package.
