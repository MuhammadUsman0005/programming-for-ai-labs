from pathlib import Path
import json
import re
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_FILE = PROJECT_DIR / "data" / "multiple_linear_regression_dataset.csv"
OUTPUT_DIR = PROJECT_DIR / "outputs"
RANDOM_STATE = 42
TEST_SIZE = 0.20
LEARNING_RATE = 0.05
ITERATIONS = 2000
MAIN_LAMBDA = 0.1
TARGET_DIVISOR = 1000.0  # optimize income in thousands to keep gradients well-scaled

# If your Kaggle CSV uses another header, add its normalized form here.
COLUMN_ALIASES = {
    "age": ["age", "employee_age"],
    "experience": ["experience", "years_experience", "years_of_experience", "exp"],
    "income": ["income", "salary", "annual_income"],
}


def normalize_name(name: str) -> str:
    """Turn headers such as 'Years of Experience' into 'years_of_experience'."""
    name = str(name).strip().lower()
    name = re.sub(r"[^a-z0-9]+", "_", name)
    return name.strip("_")


def load_dataset() -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Load the professor-linked CSV and return X, y, and feature names."""
    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_FILE}\nDownload the professor's Kaggle dataset, "
            "extract it, and save the CSV at this exact path. See data/README_DATA.md."
        )

    df = pd.read_csv(DATA_FILE)
    df.columns = [normalize_name(c) for c in df.columns]

    selected = {}
    for desired, aliases in COLUMN_ALIASES.items():
        found = next((alias for alias in aliases if alias in df.columns), None)
        if found is None:
            raise ValueError(
                f"Could not find a column for '{desired}'. Found columns: {list(df.columns)}. "
                "Edit COLUMN_ALIASES in this script to match the CSV headers."
            )
        selected[desired] = found

    clean = df[[selected["age"], selected["experience"], selected["income"]]].copy()
    clean.columns = ["age", "experience", "income"]
    for col in clean.columns:
        clean[col] = pd.to_numeric(clean[col], errors="coerce")
    clean = clean.dropna()
    if len(clean) < 10:
        raise ValueError("At least 10 valid rows are recommended for a meaningful train/test experiment.")

    X = clean[["age", "experience"]].to_numpy(dtype=float)
    y = clean["income"].to_numpy(dtype=float)
    return X, y, ["age", "experience"]


def feature_scaling_from_scratch(X_train: np.ndarray, X_other: np.ndarray):
    """Z-score scale features using training-set statistics only."""
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0)
    std[std == 0] = 1.0  # protect against division by zero for constant columns
    X_train_scaled = (X_train - mean) / std
    X_other_scaled = (X_other - mean) / std
    return X_train_scaled, X_other_scaled, mean, std


def add_intercept(X: np.ndarray) -> np.ndarray:
    """Add the leading column of ones for theta_0 (the intercept)."""
    ones = np.ones((X.shape[0], 1))
    return np.hstack((ones, X))


def hypothesis(X_with_bias: np.ndarray, theta: np.ndarray) -> np.ndarray:
    """Compute h_theta(X) = X theta."""
    return X_with_bias @ theta


def regularized_cost(X_with_bias: np.ndarray, y: np.ndarray, theta: np.ndarray, lam: float) -> float:
    """Assignment cost: squared error plus L2 penalty; theta_0 is excluded."""
    m = len(y)
    errors = hypothesis(X_with_bias, theta) - y
    # Do not regularize theta[0], because the assignment penalty starts at j=1.
    penalty = lam * np.sum(theta[1:] ** 2)
    return float((np.sum(errors ** 2) + penalty) / (2 * m))


def compute_gradient(X_with_bias: np.ndarray, y: np.ndarray, theta: np.ndarray, lam: float) -> np.ndarray:
    """Compute the vector gradient for all parameters at once."""
    m = len(y)
    errors = hypothesis(X_with_bias, theta) - y
    gradient = (X_with_bias.T @ errors) / m
    # Add L2 penalty only to feature coefficients, never the intercept.
    regularization = np.zeros_like(theta)
    regularization[1:] = (lam / m) * theta[1:]
    return gradient + regularization


def gradient_descent(X_with_bias: np.ndarray, y: np.ndarray, lam: float,
                     alpha: float = LEARNING_RATE, iterations: int = ITERATIONS):
    """Optimize theta using simultaneous vectorized updates."""
    theta = np.zeros(X_with_bias.shape[1], dtype=float)
    cost_history = []
    for iteration in range(iterations):
        cost = regularized_cost(X_with_bias, y, theta, lam)
        cost_history.append(cost)
        gradient = compute_gradient(X_with_bias, y, theta, lam)
        # All parameters are updated together using the old theta values.
        theta = theta - alpha * gradient
        if not np.all(np.isfinite(theta)):
            raise FloatingPointError(
                "Gradient descent diverged. Reduce LEARNING_RATE in this file and run again."
            )
    cost_history.append(regularized_cost(X_with_bias, y, theta, lam))
    return theta, np.asarray(cost_history)


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    X, y_dollars, feature_names = load_dataset()

    X_train, X_test, y_train_dollars, y_test_dollars = train_test_split(
        X, y_dollars, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    # The target is divided by 1,000 only during optimization; MSE is later calculated in dollars.
    y_train = y_train_dollars / TARGET_DIVISOR
    y_test = y_test_dollars / TARGET_DIVISOR

    X_train_scaled, X_test_scaled, x_mean, x_std = feature_scaling_from_scratch(X_train, X_test)
    X_train_bias = add_intercept(X_train_scaled)
    X_test_bias = add_intercept(X_test_scaled)

    # Main regularized model: demonstrates the given lambda term.
    theta_regularized, cost_history = gradient_descent(
        X_train_bias, y_train, lam=MAIN_LAMBDA
    )
    regularized_predictions_dollars = hypothesis(X_test_bias, theta_regularized) * TARGET_DIVISOR
    regularized_mse = mean_squared_error(y_test_dollars, regularized_predictions_dollars)

    # Fair comparison: lambda=0 matches ordinary least-squares LinearRegression.
    theta_unregularized, _ = gradient_descent(X_train_bias, y_train, lam=0.0)
    gd_predictions_dollars = hypothesis(X_test_bias, theta_unregularized) * TARGET_DIVISOR
    gd_mse = mean_squared_error(y_test_dollars, gd_predictions_dollars)

    sklearn_model = LinearRegression()
    sklearn_model.fit(X_train_scaled, y_train)
    sklearn_predictions_dollars = sklearn_model.predict(X_test_scaled) * TARGET_DIVISOR
    sklearn_mse = mean_squared_error(y_test_dollars, sklearn_predictions_dollars)

    # Training MSE is recorded as an aid to the underfit/overfit discussion.
    train_predictions_dollars = hypothesis(X_train_bias, theta_regularized) * TARGET_DIVISOR
    train_mse = mean_squared_error(y_train_dollars, train_predictions_dollars)

    # Save the cost-vs-iteration convergence plot.
    plt.figure(figsize=(8, 5))
    plt.plot(np.arange(len(cost_history)), cost_history)
    plt.title(f"Gradient Descent Convergence (lambda={MAIN_LAMBDA})")
    plt.xlabel("Iteration")
    plt.ylabel("Regularized Cost J(theta)")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "convergence_curve.png", dpi=160)
    plt.close()

    # Show the fair comparison on the same split (lambda=0 gradient descent vs sklearn OLS).
    comparison = pd.DataFrame({
        "Model": ["Gradient Descent (lambda=0)", "scikit-learn LinearRegression", f"Gradient Descent (lambda={MAIN_LAMBDA})"],
        "Test_MSE_USD_squared": [gd_mse, sklearn_mse, regularized_mse],
    })
    comparison.to_csv(OUTPUT_DIR / "metrics.csv", index=False)
    plt.figure(figsize=(9, 5))
    plt.bar(comparison["Model"], comparison["Test_MSE_USD_squared"])
    plt.title("Test MSE Comparison")
    plt.xlabel("Model")
    plt.ylabel("Mean Squared Error (USD squared)")
    plt.xticks(rotation=12, ha="right")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "mse_comparison.png", dpi=160)
    plt.close()

    # Small lambda experiment; this is useful when discussing the strength of regularization.
    lambda_rows = []
    existing_thetas = {0.0: theta_unregularized, MAIN_LAMBDA: theta_regularized}
    for lam in [0.0, 0.01, MAIN_LAMBDA, 1.0, 10.0]:
        # Reuse the already-fitted runs for lambda=0 and the main lambda.
        theta_try = existing_thetas.get(lam)
        if theta_try is None:
            theta_try, _ = gradient_descent(X_train_bias, y_train, lam=lam)
        train_pred = hypothesis(X_train_bias, theta_try) * TARGET_DIVISOR
        test_pred = hypothesis(X_test_bias, theta_try) * TARGET_DIVISOR
        lambda_rows.append({
            "lambda": lam,
            "train_mse": mean_squared_error(y_train_dollars, train_pred),
            "test_mse": mean_squared_error(y_test_dollars, test_pred),
            "coefficient_norm": float(np.linalg.norm(theta_try[1:])),
        })
    lambda_df = pd.DataFrame(lambda_rows)
    lambda_df.to_csv(OUTPUT_DIR / "lambda_comparison.csv", index=False)
    plt.figure(figsize=(8, 5))
    plt.plot(lambda_df["lambda"], lambda_df["test_mse"], marker="o", label="Test MSE")
    plt.plot(lambda_df["lambda"], lambda_df["train_mse"], marker="s", label="Train MSE")
    plt.xscale("symlog", linthresh=0.01)
    plt.title("Effect of Regularization Lambda")
    plt.xlabel("Lambda (log-like scale)")
    plt.ylabel("Mean Squared Error (USD squared)")
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "lambda_comparison.png", dpi=160)
    plt.close()

    # Learning-rate experiment for the report discussion. A very small rate learns slowly;
    # a larger stable rate usually reduces the cost faster. All runs use the same data/split.
    learning_rate_rows = []
    plt.figure(figsize=(8, 5))
    for alpha_try in [0.001, 0.01, 0.05, 0.1]:
        theta_lr, cost_lr = gradient_descent(
            X_train_bias, y_train, lam=MAIN_LAMBDA, alpha=alpha_try, iterations=500
        )
        learning_rate_rows.append({
            "learning_rate": alpha_try,
            "iterations": 500,
            "final_cost": float(cost_lr[-1]),
            "test_mse_usd_squared": float(mean_squared_error(
                y_test_dollars, hypothesis(X_test_bias, theta_lr) * TARGET_DIVISOR
            )),
        })
        plt.plot(np.arange(len(cost_lr)), cost_lr, label=f"alpha={alpha_try}")
    plt.title("Effect of Learning Rate on Cost Convergence")
    plt.xlabel("Iteration")
    plt.ylabel("Regularized Cost J(theta)")
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "learning_rate_comparison.png", dpi=160)
    plt.close()
    pd.DataFrame(learning_rate_rows).to_csv(OUTPUT_DIR / "learning_rate_comparison.csv", index=False)

if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print("ERROR:", exc, file=sys.stderr)
        raise
