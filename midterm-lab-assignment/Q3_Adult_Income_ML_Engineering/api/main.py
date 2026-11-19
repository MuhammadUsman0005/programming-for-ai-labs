"""FastAPI service that serves the saved Adult Income classifier."""
from pathlib import Path
import sys
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

PROJECT_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_DIR / "models" / "model_bundle.joblib"

app = FastAPI(title="Adult Income Educational Prediction API", version="1.0.0")


class AdultInput(BaseModel):
    age: int = Field(default=35, ge=17, le=100)
    workclass: str = "Private"
    fnlwgt: int = Field(default=150000, ge=1)
    education: str = "HS-grad"
    education_num: int = Field(default=9, ge=1, le=16)
    marital_status: str = "Never-married"
    occupation: str = "Sales"
    relationship: str = "Not-in-family"
    race: str = "White"
    sex: str = "Male"
    capital_gain: int = Field(default=0, ge=0)
    capital_loss: int = Field(default=0, ge=0)
    hours_per_week: int = Field(default=40, ge=1, le=100)
    native_country: str = "United-States"


_model_bundle = None


def get_bundle():
    global _model_bundle
    if _model_bundle is None:
        if not MODEL_PATH.exists():
            raise HTTPException(status_code=503, detail="Model file not found. Run the training script first.")
        _model_bundle = joblib.load(MODEL_PATH)
    return _model_bundle


@app.get("/health")
def health():
    return {"status": "ok", "model_file_exists": MODEL_PATH.exists()}


@app.get("/model-info")
def model_info():
    bundle = get_bundle()
    return {
        "model_name": bundle.get("model_name"),
        "test_metrics": bundle.get("test_metrics", {}),
        "top_features": bundle.get("top_features", []),
        "warning": bundle.get("warning"),
    }


@app.post("/predict")
def predict(input_data: AdultInput):
    bundle = get_bundle()
    model = bundle["model"]
    row = input_data.model_dump()
    X = pd.DataFrame([row])
    # Add the same non-target engineered features used during training.
    from src.data_utils import add_derived_features
    X = add_derived_features(X)
    feature_columns = bundle["feature_columns"]
    for col in feature_columns:
        if col not in X.columns:
            X[col] = None
    X = X[feature_columns]
    try:
        prediction = int(model.predict(X)[0])
        label = ">50K" if prediction == 1 else "<=50K"
        probability = None
        if hasattr(model, "predict_proba"):
            probability = float(model.predict_proba(X)[0][1])
        return {
            "prediction": prediction,
            "predicted_income_class": label,
            "probability_over_50k": probability,
            "model_name": bundle.get("model_name"),
            "notice": "Educational demonstration only; not suitable for real employment or other high-impact decisions.",
        }
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Prediction failed: {exc}") from exc
