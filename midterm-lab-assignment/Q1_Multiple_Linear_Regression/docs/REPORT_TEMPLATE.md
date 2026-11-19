# Q1 Report — Multiple Linear Regression

**Student:** Muhammad Usman 
**Course:** Machine Learning  
**Dataset:** Multiple Linear Regression Dataset (professor's Kaggle link)  
**Run date:** 16/11/2026

## 1. Objective

The objective was to implement multiple linear regression from scratch, derive the regularized gradients, compare gradient descent with scikit-learn LinearRegression, and evaluate both models using MSE.

## 2. Dataset description

- CSV file: multiple_linear_regression_dataset
- Number of rows: 20
- Input features: age, experience 
- Target: income 
- Train/test split: 80/20, random_state=42

## 3. Mathematical derivation
Explore pdf from docs/ folder

## 4. Preprocessing and design choices

- Feature scaling: z-score standardization using training data only.
- Target scale in the optimizer: income expressed in thousands of dollars for numerical stability; predictions are converted back to dollars before MSE is calculated.
- Learning rate alpha: 0.05
- Iterations: 2000
- Regularization lambda: 0.1

## 5. Results

Paste actual values from `outputs/metrics.csv` and `outputs/theta_values.csv`.

| Model | Lambda | Test MSE |
|---|---:|---:|
| From-scratch Gradient Descent | 0 | 753796.7693733985 |
| scikit-learn LinearRegression | Not regularized | 753796.7693734514 |
| From-scratch Gradient Descent | 0.1 | 772533.8775818327 |

Insert `convergence_curve.png`, `mse_comparison.png`, `lambda_comparison.png`, and `learning_rate_comparison.png`.

