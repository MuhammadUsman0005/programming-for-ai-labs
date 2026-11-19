# Q1 — Multiple Linear Regression from scratch

## Dataset

Use the exact Kaggle dataset linked by the professor:
https://www.kaggle.com/datasets/hussainnasirkhan/multiple-linear-regression-dataset

Put the CSV here:
`data/multiple_linear_regression_dataset.csv`

The script expects columns equivalent to `age`, `experience`, and `income`. It normalizes capitalization and spaces, but if your downloaded CSV uses different column names, edit the `COLUMN_ALIASES` dictionary near the top of the script.

## Install

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Run

From the Q1 folder:

```powershell
python src/multiple_linear_regression.py
```

## What the script does

1. Loads and inspects the downloaded CSV.
2. Selects age and experience as input features and income as the target.
3. Splits train/test data with a fixed random seed.
4. Performs z-score feature scaling using training-set means and standard deviations only.
5. Implements the hypothesis, regularized cost, gradient, and gradient-descent updates with NumPy.
6. Trains an unregularized gradient-descent model for a fair comparison with ordinary `LinearRegression`.
7. Trains a regularized gradient-descent model to demonstrate the lambda penalty.
8. Plots convergence, MSE comparison, and the lambda experiment.
9. Saves final theta values, metrics, model parameters, and a run summary under `outputs/`.

## Important comparison note

`sklearn.linear_model.LinearRegression` is ordinary least squares and does not apply the L2 penalty in the assignment's cost function. Therefore, the fairest comparison is **gradient descent with lambda = 0** versus `LinearRegression` on the same split and scaled features. The separate lambda > 0 run demonstrates regularization and may have a different MSE. Explain this difference in your report.

## Generated outputs

- `outputs/convergence_curve.png`
- `outputs/mse_comparison.png`
- `outputs/lambda_comparison.png`
- `outputs/learning_rate_comparison.png`
- `outputs/learning_rate_comparison.csv`
- `outputs/theta_values.csv`
- `outputs/metrics.csv`
- `outputs/run_summary.json`
- `outputs/gradient_descent_model.joblib`

## Study files

- `docs/DERIVATION_GUIDE.md`
- `docs/REPORT_TEMPLATE.md`
- `notebooks/Q1_Multiple_Linear_Regression.ipynb`

**Do not submit the report with placeholder results.** Run the real dataset first and add your own handwritten derivation scan.
