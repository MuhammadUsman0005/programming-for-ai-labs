
"""Beginner-friendly Ames House Price regression project.

Run from the project directory:
    python src/house_price_project.py
"""
from pathlib import Path
import argparse
import json

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_squared_log_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

RANDOM_STATE = 42

# These columns match the meaningful example features named in the assignment.
NUMERIC_FEATURES = [
    "OverallQual", "GrLivArea", "TotalBsmtSF", "YearBuilt",
    "GarageCars", "FullBath", "HalfBath", "LotArea"
]
CATEGORICAL_FEATURES = ["Neighborhood", "KitchenQual", "BsmtQual"]
ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def make_preprocessor():
    """Build simple numeric and categorical preprocessing steps."""
    numeric_steps = Pipeline(steps=[
        # Median is less affected by unusually large/small values than the mean.
        ("imputer", SimpleImputer(strategy="median"))
    ])

    categorical_steps = Pipeline(steps=[
        # Fill missing text categories before converting categories to columns.
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("one_hot", OneHotEncoder(handle_unknown="ignore"))
    ])

    # Each column group gets the correct preprocessing method.
    return ColumnTransformer(transformers=[
        ("numeric", numeric_steps, NUMERIC_FEATURES),
        ("categorical", categorical_steps, CATEGORICAL_FEATURES)
    ])


def create_eda_plots(train, output_dir):
    """Create the required EDA plots; each plot has a title and labeled axes."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Plot the columns with the most missing entries.
    missing = train.isnull().sum().sort_values(ascending=False).head(15)
    plt.figure(figsize=(10, 5))
    missing.sort_values().plot(kind="barh")
    plt.title("Top 15 Columns by Missing Values")
    plt.xlabel("Number of Missing Values")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.savefig(output_dir / "missing_values.png", dpi=150)
    plt.close()

    # SalePrice is usually right-skewed; inspect the raw distribution.
    plt.figure(figsize=(8, 5))
    plt.hist(train["SalePrice"].dropna(), bins=30, edgecolor="black")
    plt.title("Distribution of House Sale Prices")
    plt.xlabel("Sale Price")
    plt.ylabel("Number of Houses")
    plt.tight_layout()
    plt.savefig(output_dir / "sale_price_distribution.png", dpi=150)
    plt.close()

    # Living area is a useful numeric feature to compare with the target.
    plt.figure(figsize=(8, 5))
    plt.scatter(train["GrLivArea"], train["SalePrice"], alpha=0.55)
    plt.title("Above-Ground Living Area vs Sale Price")
    plt.xlabel("Above-Ground Living Area (square feet)")
    plt.ylabel("Sale Price")
    plt.tight_layout()
    plt.savefig(output_dir / "living_area_vs_price.png", dpi=150)
    plt.close()


def load_and_check_data(data_dir):
    """Load CSV files and fail with a helpful message when files are absent."""
    train_path = data_dir / "train.csv"
    test_path = data_dir / "test.csv"
    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError(
            "Dataset files are missing. Read data/README_DATA.md and place "
            "Kaggle train.csv and test.csv in the data folder."
        )

    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    required = ALL_FEATURES + ["SalePrice", "Id"]
    missing_columns = [column for column in required if column not in train.columns]
    missing_test_columns = [column for column in ALL_FEATURES + ["Id"] if column not in test.columns]
    if missing_columns or missing_test_columns:
        raise ValueError(
            f"Missing required columns. Train missing: {missing_columns}; "
            f"test missing: {missing_test_columns}"
        )
    if (train["SalePrice"] < 0).any():
        raise ValueError("SalePrice contains negative values; log1p cannot be used for negative targets.")
    return train, test


def evaluate_models(X_train, X_valid, y_train, y_valid):
    """Fit two simple models and calculate validation metrics."""
    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(
            n_estimators=150,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            min_samples_leaf=1
        )
    }

    rows = []
    fitted_models = {}
    for model_name, estimator in models.items():
        # The model learns log prices. This helps with a strongly skewed target.
        pipeline = Pipeline(steps=[
            ("preprocessing", make_preprocessor()),
            ("model", estimator)
        ])
        pipeline.fit(X_train, np.log1p(y_train))

        # Convert predicted log prices back to the original price scale.
        predicted_prices = np.expm1(pipeline.predict(X_valid))
        predicted_prices = np.maximum(predicted_prices, 0)

        mse = mean_squared_error(y_valid, predicted_prices)
        rmsle = np.sqrt(mean_squared_log_error(y_valid, predicted_prices))
        rows.append({
            "Model": model_name,
            "MSE": float(mse),
            "RMSLE": float(rmsle)
        })
        fitted_models[model_name] = pipeline
        print(f"{model_name}: MSE={mse:,.2f} | RMSLE={rmsle:.5f}")

    results = pd.DataFrame(rows).sort_values("RMSLE", ascending=True)
    return results, fitted_models


def run_project(data_dir, output_dir):
    """Run the full data-analysis, model-comparison, and prediction workflow."""
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train, test = load_and_check_data(data_dir)
    print("Training shape:", train.shape)
    print("Testing shape:", test.shape)
    print("Training data preview:\n", train[ALL_FEATURES + ["SalePrice"]].head())
    print("\nMissing values in selected columns:\n", train[ALL_FEATURES].isnull().sum())

    create_eda_plots(train, output_dir)

    # Keep the original SalePrice in rupees/dollars for evaluation.
    X = train[ALL_FEATURES]
    y = train["SalePrice"].astype(float)
    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE
    )

    results, fitted_models = evaluate_models(X_train, X_valid, y_train, y_valid)
    results.to_csv(output_dir / "model_comparison.csv", index=False)

    # Plot metric comparison; separate metric magnitudes are kept in separate bars.
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].bar(results["Model"], results["MSE"])
    axes[0].set_title("Validation Mean Squared Error")
    axes[0].set_xlabel("Model")
    axes[0].set_ylabel("MSE (price units squared)")
    axes[0].tick_params(axis="x", rotation=15)
    axes[1].bar(results["Model"], results["RMSLE"])
    axes[1].set_title("Validation Root Mean Squared Log Error")
    axes[1].set_xlabel("Model")
    axes[1].set_ylabel("RMSLE")
    axes[1].tick_params(axis="x", rotation=15)
    plt.tight_layout()
    plt.savefig(output_dir / "model_comparison.png", dpi=150)
    plt.close()

    best_name = str(results.iloc[0]["Model"])
    best_validation_model = fitted_models[best_name]
    best_rmsle = float(results.iloc[0]["RMSLE"])

    # Refit the selected pipeline on every labeled row before predicting Kaggle test rows.
    best_validation_model.fit(X, np.log1p(y))
    model_path = output_dir / "best_house_model.joblib"
    joblib.dump(best_validation_model, model_path)

    test_predictions = np.expm1(best_validation_model.predict(test[ALL_FEATURES]))
    test_predictions = np.maximum(test_predictions, 0)
    prediction_file = pd.DataFrame({"Id": test["Id"], "SalePrice": test_predictions})
    prediction_file.to_csv(output_dir / "test_predictions.csv", index=False)

    summary = {
        "best_model": best_name,
        "best_validation_rmsle": best_rmsle,
        "validation_metrics_csv": "model_comparison.csv",
        "model_file": model_path.name,
        "prediction_file": "test_predictions.csv",
        "note": "Metrics are from a validation split of train.csv. Kaggle test.csv has no public SalePrice labels."
    }
    (output_dir / "run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (output_dir / "run_summary.txt").write_text(
        f"Best model: {best_name}\nBest validation RMSLE: {best_rmsle:.5f}\n"
        "Predictions saved to test_predictions.csv\n"
        "Validation metrics come from train.csv; test.csv has no SalePrice labels.\n",
        encoding="utf-8"
    )

    print("\nModel comparison:\n", results.to_string(index=False))
    print("Best model saved to:", model_path)
    print("Test predictions saved to:", output_dir / "test_predictions.csv")
    return results, summary


def main():
    parser = argparse.ArgumentParser(description="Ames house-price regression project")
    parser.add_argument("--data-dir", default="data", help="Folder containing train.csv and test.csv")
    parser.add_argument("--output-dir", default="outputs", help="Folder for graphs, model, and predictions")
    args = parser.parse_args()
    run_project(args.data_dir, args.output_dir)


if __name__ == "__main__":
    main()
