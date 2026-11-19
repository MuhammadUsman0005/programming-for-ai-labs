
# Project 1 — House Price Prediction

## Objective
Build a regression system with the Ames Housing dataset. Compare Linear Regression and Random Forest Regressor, evaluate using MSE and RMSLE, save the selected pipeline with Joblib, and generate predictions for the Kaggle test file.

## Data setup
Read `data/README_DATA.md`. Put Kaggle `train.csv` and `test.csv` into `data/`. The assignment PDF names the Kaggle *House Prices: Advanced Regression Techniques* dataset but does not include the CSV files.

## Environment setup (Windows PowerShell)
```powershell
py -3 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PowerShell blocks activation, you can run the Python executable inside `.venv` directly or use Command Prompt activation: `.venv\Scripts\activate.bat`.

## Run
From this project directory:
```bash
python src/house_price_project.py
```

Optional custom paths:
```bash
python src/house_price_project.py --data-dir data --output-dir outputs
```

To try the saved model interactively:
```bash
python src/predict_house.py --model outputs/best_house_model.joblib
```

Open `notebooks/Project_1_House_Price_Prediction.ipynb` in Jupyter and run all cells for a notebook-based demonstration.

## Main implementation steps
1. Load and inspect the training/testing CSVs.
2. Save missing-value, price-distribution, and living-area plots.
3. Use the assignment's meaningful features: overall quality, living area, basement area, year built, neighborhood, garage capacity, kitchen/basement quality, bathrooms, and lot area.
4. Fill numeric missing values with the median; fill categorical missing values with the most frequent category.
5. One-hot encode categorical features.
6. Train both regressors on `log1p(SalePrice)` and reverse predictions with `expm1`.
7. Report MSE and RMSLE on the held-out validation set.
8. Select the lower-RMSLE model, retrain it on all labeled training rows, save it, and create test predictions.

## Outputs
The script creates figures, `model_comparison.csv`, the Joblib model, `test_predictions.csv`, and `run_summary.txt` in `outputs/`.

## Important interpretation
`test.csv` does not have known target values. The script cannot calculate test-set MSE/RMSLE for it. The reported metrics are from a hold-out split of `train.csv`, not from hidden Kaggle labels.

## Report
Fill in `reports/REPORT_TEMPLATE.md` after running the program. Do not invent or copy metrics from another person's run.

## References
- Kaggle dataset: https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/data
- `train_test_split`: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
- Regression metrics: https://scikit-learn.org/stable/modules/model_evaluation.html
- `RandomForestRegressor`: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html
- `OneHotEncoder`: https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html
