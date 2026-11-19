"""Question 3: Adult Income baseline + TPOT AutoML training script."""
from pathlib import Path
import argparse
import json
import time
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, confusion_matrix, ConfusionMatrixDisplay,
                             classification_report)
from sklearn.inspection import permutation_importance
from src.data_utils import clean_adult_dataframe, add_derived_features, make_preprocessor

PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_FILE = PROJECT_DIR / "data" / "adult.csv"
OUTPUT_DIR = PROJECT_DIR / "outputs"
MODEL_DIR = PROJECT_DIR / "models"
RANDOM_STATE = 42
TEST_SIZE = 0.20
AUTOML_TIME_MINS = 30
AUTOML_MAX_EVAL_MINS = 2
TARGET_COLUMN = None  # Set to a normalized CSV header if automatic detection does not find the label.


def load_adult_csv():
    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Adult dataset not found: {DATA_FILE}\nDownload the professor-linked Kaggle CSV, rename it adult.csv, "
            "and place it in data/. See data/README_DATA.md."
        )
    df = pd.read_csv(DATA_FILE)
    df = clean_adult_dataframe(df)
    target_candidates = [TARGET_COLUMN, "income", "class", "salary", "income_bracket", "income_class"]
    target_col = next((c for c in target_candidates if c and c in df.columns), None)
    if target_col is None:
        raise ValueError(f"Target column not found. CSV columns are: {list(df.columns)}. Set TARGET_COLUMN in this script.")

    y_raw = df[target_col].copy()
    X_raw = df.drop(columns=[target_col]).copy()
    # Adult labels sometimes contain a trailing period in test-style CSV files.
    y_text = y_raw.astype(str).str.strip().str.replace(r"\.$", "", regex=True).str.lower()
    if set(y_text.dropna().unique()).issubset({"0", "1"}):
        y = pd.to_numeric(y_text, errors="coerce")
    else:
        # Values like '>50K' are positive; values like '<=50K' are negative.
        y = y_text.str.contains(r">\s*50k", regex=True).astype(float)
    valid = y.notna()
    X = add_derived_features(X_raw.loc[valid].reset_index(drop=True))
    y = y.loc[valid].astype(int).reset_index(drop=True)
    if y.nunique() != 2:
        raise ValueError("Income target must have exactly two classes after cleaning. Check the CSV target values.")
    return X, y, target_col


def calculate_metrics(model, X_test, y_test):
    prediction = model.predict(X_test)
    result = {
        "accuracy": float(accuracy_score(y_test, prediction)),
        "precision": float(precision_score(y_test, prediction, zero_division=0)),
        "recall": float(recall_score(y_test, prediction, zero_division=0)),
        "f1": float(f1_score(y_test, prediction, zero_division=0)),
    }
    if hasattr(model, "predict_proba"):
        probability = model.predict_proba(X_test)[:, 1]
        result["roc_auc"] = float(roc_auc_score(y_test, probability))
    elif hasattr(model, "decision_function"):
        decision_values = model.decision_function(X_test)
        result["roc_auc"] = float(roc_auc_score(y_test, decision_values))
    else:
        result["roc_auc"] = None
    return result, prediction


def save_eda(X, y):
    OUTPUT_DIR.mkdir(exist_ok=True)
    plt.figure(figsize=(6, 4))
    y.value_counts().sort_index().plot(kind="bar")
    plt.title("Adult Income Class Balance")
    plt.xlabel("Income class (0 = <=50K, 1 = >50K)")
    plt.ylabel("Number of rows")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "class_balance.png", dpi=160)
    plt.close()

    if "age" in X.columns:
        plt.figure(figsize=(7, 4))
        pd.to_numeric(X["age"], errors="coerce").dropna().hist(bins=25)
        plt.title("Age Distribution")
        plt.xlabel("Age")
        plt.ylabel("Number of rows")
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "age_distribution.png", dpi=160)
        plt.close()

    # Plot several numeric feature distributions, not just one example column.
    numeric_candidates = [c for c in ["age", "education_num", "hours_per_week", "capital_gain", "capital_loss", "fnlwgt"] if c in X.columns]
    if numeric_candidates:
        fig, axes = plt.subplots(2, 3, figsize=(12, 7))
        axes = axes.flatten()
        for ax, col in zip(axes, numeric_candidates):
            pd.to_numeric(X[col], errors="coerce").dropna().hist(ax=ax, bins=20)
            ax.set_title(col.replace("_", " ").title())
            ax.set_xlabel(col.replace("_", " "))
            ax.set_ylabel("Rows")
        for ax in axes[len(numeric_candidates):]:
            ax.set_visible(False)
        fig.suptitle("Selected Numeric Feature Distributions")
        fig.tight_layout()
        fig.savefig(OUTPUT_DIR / "numeric_feature_distributions.png", dpi=160)
        plt.close(fig)

    # Plot common categorical feature distributions using the most frequent categories.
    categorical_candidates = [c for c in ["workclass", "education", "occupation", "marital_status"] if c in X.columns]
    if categorical_candidates:
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        axes = axes.flatten()
        for ax, col in zip(axes, categorical_candidates):
            X[col].fillna("Missing").astype(str).value_counts().head(8).sort_values().plot(kind="barh", ax=ax)
            ax.set_title(col.replace("_", " ").title())
            ax.set_xlabel("Rows")
            ax.set_ylabel("Category")
        for ax in axes[len(categorical_candidates):]:
            ax.set_visible(False)
        fig.suptitle("Selected Categorical Feature Distributions (Top 8 Values)")
        fig.tight_layout()
        fig.savefig(OUTPUT_DIR / "categorical_feature_distributions.png", dpi=160)
        plt.close(fig)

    numeric = X.select_dtypes(include=[np.number])
    if numeric.shape[1] >= 2:
        corr = numeric.corr(numeric_only=True)
        plt.figure(figsize=(9, 7))
        plt.imshow(corr, aspect="auto", interpolation="nearest")
        plt.colorbar(label="Correlation")
        plt.xticks(range(len(corr.columns)), corr.columns, rotation=45, ha="right")
        plt.yticks(range(len(corr.columns)), corr.columns)
        plt.title("Numeric Feature Correlation")
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "numeric_correlation.png", dpi=160)
        plt.close()


def save_confusion_matrix(y_test, predictions, filename, title):
    cm = confusion_matrix(y_test, predictions, labels=[0, 1])
    disp = ConfusionMatrixDisplay(cm, display_labels=["<=50K", ">50K"])
    fig, ax = plt.subplots(figsize=(6, 5))
    disp.plot(ax=ax, colorbar=False)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / filename, dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Train Adult Income baseline and optional TPOT AutoML.")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--baseline-only", action="store_true", help="Run only the scikit-learn baseline for a quick setup test.")
    group.add_argument("--automl", action="store_true", help="Run baseline plus the 30-minute TPOT AutoML search.")
    args = parser.parse_args()
    run_automl = args.automl

    OUTPUT_DIR.mkdir(exist_ok=True)
    MODEL_DIR.mkdir(exist_ok=True)
    X, y, target_col = load_adult_csv()
    save_eda(X, y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    preprocessor = make_preprocessor(X_train)

    baseline = Pipeline([
        ("preprocess", preprocessor),
        ("model", LogisticRegression(max_iter=1500, random_state=RANDOM_STATE)),
    ])
    start = time.perf_counter()
    baseline.fit(X_train, y_train)
    baseline_time = time.perf_counter() - start
    baseline_metrics, baseline_predictions = calculate_metrics(baseline, X_test, y_test)
    baseline_metrics["training_time_seconds"] = float(baseline_time)
    save_confusion_matrix(y_test, baseline_predictions, "confusion_matrix_baseline.png", "Baseline Logistic Regression")
    (OUTPUT_DIR / "baseline_classification_report.txt").write_text(
        classification_report(y_test, baseline_predictions, target_names=["<=50K", ">50K"], zero_division=0),
        encoding="utf-8"
    )

    model_comparisons = [{
        "model": "Baseline Logistic Regression",
        "model_complexity": "ColumnTransformer preprocessing + LogisticRegression (two main pipeline stages)",
        "preprocessed_feature_count": int(baseline.named_steps["preprocess"].transform(X_train.iloc[:1]).shape[1]),
        **baseline_metrics,
    }]
    best_model = baseline
    best_metrics = baseline_metrics
    best_name = "Baseline Logistic Regression"
    automl_time = None

    if run_automl:
        print(f"Starting TPOT search for up to {AUTOML_TIME_MINS} minutes. This can take a while.")
        try:
            from tpot import TPOTClassifier
        except ImportError as exc:
            raise ImportError("TPOT is not installed. Run: pip install -r requirements.txt") from exc

        # Fit the same preprocessing only on training data, then let TPOT search models on numeric features.
        automl_preprocessor = make_preprocessor(X_train)
        X_train_ready = automl_preprocessor.fit_transform(X_train)
        X_test_ready = automl_preprocessor.transform(X_test)
        automl = TPOTClassifier(
            search_space="linear-light",
            scorers=["roc_auc"],
            cv=3,
            max_time_mins=AUTOML_TIME_MINS,
            max_eval_time_mins=AUTOML_MAX_EVAL_MINS,
            n_jobs=-1,
            random_state=RANDOM_STATE,
            verbose=2,
            preprocessing=False,
            allow_inner_classifiers=False,
        )
        start = time.perf_counter()
        automl.fit(X_train_ready, y_train)
        automl_time = time.perf_counter() - start
        # Keep the preprocessing step with the selected TPOT pipeline so API input remains raw columns.
        automl_pipeline = Pipeline([
            ("preprocess", automl_preprocessor),
            ("automl_model", automl.fitted_pipeline_),
        ])
        automl_metrics, automl_predictions = calculate_metrics(automl_pipeline, X_test, y_test)
        automl_metrics["training_time_seconds"] = float(automl_time)
        save_confusion_matrix(y_test, automl_predictions, "confusion_matrix_automl.png", "TPOT AutoML")
        (OUTPUT_DIR / "automl_classification_report.txt").write_text(
            classification_report(y_test, automl_predictions, target_names=["<=50K", ">50K"], zero_division=0),
            encoding="utf-8"
        )
        pipeline_text = str(automl.fitted_pipeline_)
        (OUTPUT_DIR / "automl_pipeline.txt").write_text(pipeline_text, encoding="utf-8")
        model_comparisons.append({
            "model": "TPOT AutoML",
            "model_complexity": "TPOT-selected graph/pipeline; see outputs/automl_pipeline.txt",
            "pipeline_description_characters": len(pipeline_text),
            "preprocessed_feature_count": int(X_train_ready.shape[1]),
            **automl_metrics,
        })
        # Choose the model with higher test ROC-AUC for this classroom comparison.
        # For a production workflow, choose using validation/CV, not the final test set.
        if automl_metrics.get("roc_auc", -1) > best_metrics.get("roc_auc", -1):
            best_model = automl_pipeline
            best_metrics = automl_metrics
            best_name = "TPOT AutoML"

    comparisons_df = pd.DataFrame(model_comparisons)
    comparisons_df.to_csv(OUTPUT_DIR / "model_comparison.csv", index=False)

    # Permutation importance is computed on the original, named input columns.
    # This works for either the baseline or selected AutoML pipeline.
    print("Calculating top feature importance. This may take a short while.")
    importance = permutation_importance(
        best_model, X_test, y_test, n_repeats=2, random_state=RANDOM_STATE,
        scoring="accuracy", n_jobs=-1
    )
    importance_df = pd.DataFrame({
        "feature": X_test.columns,
        "importance_mean": importance.importances_mean,
        "importance_std": importance.importances_std,
    }).sort_values("importance_mean", ascending=False)
    top5 = importance_df.head(5).copy()
    top5.to_csv(OUTPUT_DIR / "feature_importance_top5.csv", index=False)
    plt.figure(figsize=(8, 5))
    plt.barh(top5["feature"][::-1], top5["importance_mean"][::-1])
    plt.title("Top 5 Feature Importances (Permutation Importance)")
    plt.xlabel("Mean decrease in accuracy when feature is shuffled")
    plt.ylabel("Original input feature")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "feature_importance_top5.png", dpi=160)
    plt.close()

    bundle = {
        "model": best_model,
        "model_name": best_name,
        "test_metrics": best_metrics,
        "all_model_metrics": model_comparisons,
        "feature_columns": X.columns.tolist(),
        "top_features": top5.to_dict(orient="records"),
        "positive_label": ">50K",
        "negative_label": "<=50K",
        "warning": "Educational dataset only. Do not use for real employment or other high-impact decisions.",
    }
    joblib.dump(bundle, MODEL_DIR / "model_bundle.joblib")

    summary = {
        "target_column": target_col,
        "rows_after_cleaning": int(len(X)),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "feature_count_after_engineering": int(X.shape[1]),
        "class_counts": {str(k): int(v) for k, v in y.value_counts().sort_index().items()},
        "random_state": RANDOM_STATE,
        "test_size": TEST_SIZE,
        "automl_run": bool(run_automl),
        "automl_time_limit_minutes": AUTOML_TIME_MINS if run_automl else None,
        "automl_measured_training_time_seconds": automl_time,
        "best_model": best_name,
        "model_comparison": model_comparisons,
        "top_features": top5.to_dict(orient="records"),
        "model_path": str(MODEL_DIR / "model_bundle.joblib"),
        "note": "Test metrics are descriptive for the assignment. Use validation/CV for model selection in production.",
    }
    (OUTPUT_DIR / "run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\nQ3 training finished.")
    print("Best model:", best_name)
    print("Saved model:", MODEL_DIR / "model_bundle.joblib")
    print(comparisons_df.to_string(index=False))
    print("Top 5 features:")
    print(top5.to_string(index=False))
    if not run_automl:
        print("NOTE: This was baseline-only. Run `python -m src.train_adult --automl` to complete the required TPOT experiment.")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print("ERROR:", exc, file=sys.stderr)
        raise
