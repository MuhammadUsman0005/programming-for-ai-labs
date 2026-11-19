"""Shared Adult Income data cleaning and feature preparation helpers."""
import re
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

BASE_NUMERIC = ["age", "fnlwgt", "education_num", "capital_gain", "capital_loss", "hours_per_week"]
BASE_CATEGORICAL = ["workclass", "education", "marital_status", "occupation", "relationship", "race", "sex", "native_country"]


def normalize_column_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(name).strip().lower()).strip("_")


def clean_adult_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize column names, trim strings, and turn '?' into NaN."""
    df = df.copy()
    df.columns = [normalize_column_name(c) for c in df.columns]
    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = df[col].astype(str).str.strip()
        df[col] = df[col].replace({"?": np.nan, "": np.nan, "nan": np.nan})
    # Standardize several common spelling aliases.
    aliases = {
        "education_num": ["education_num", "education_number", "educationnum"],
        "marital_status": ["marital_status", "maritalstatus"],
        "capital_gain": ["capital_gain", "capitalgain"],
        "capital_loss": ["capital_loss", "capitalloss"],
        "hours_per_week": ["hours_per_week", "hoursperweek"],
        "native_country": ["native_country", "nativecountry"],
    }
    rename = {}
    for canonical, options in aliases.items():
        for option in options:
            if option in df.columns:
                rename[option] = canonical
                break
    return df.rename(columns=rename)


def add_derived_features(X: pd.DataFrame) -> pd.DataFrame:
    """Add small, understandable features without using the target label."""
    X = clean_adult_dataframe(X)
    for column in ["capital_gain", "capital_loss", "age", "hours_per_week", "fnlwgt", "education_num"]:
        if column in X.columns:
            X[column] = pd.to_numeric(X[column], errors="coerce")
    if "capital_gain" in X.columns and "capital_loss" in X.columns:
        X["capital_net"] = X["capital_gain"].fillna(0) - X["capital_loss"].fillna(0)
        X["has_capital_change"] = ((X["capital_gain"].fillna(0) + X["capital_loss"].fillna(0)) > 0).astype(int)
    if "age" in X.columns:
        X["age_group"] = pd.cut(
            X["age"], bins=[0, 25, 40, 60, 100],
            labels=["18-25", "26-40", "41-60", "61+"], include_lowest=True
        ).astype("object")
    return X


def make_preprocessor(X: pd.DataFrame):
    """Build preprocessing that is fitted only on training data."""
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = [c for c in X.columns if c not in numeric_cols]

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    transformers = []
    if numeric_cols:
        transformers.append(("numeric", numeric_pipeline, numeric_cols))
    if categorical_cols:
        transformers.append(("categorical", categorical_pipeline, categorical_cols))
    return ColumnTransformer(transformers=transformers, remainder="drop")
