
# Project 1 Report — House Price Prediction

**Student:** ____________________  
**Course/Semester:** ____________________  
**Date:** ____________________

## 1. Objective
State the goal of predicting house sale prices using the Ames Housing dataset.

## 2. Dataset
- Source/link: Kaggle House Prices: Advanced Regression Techniques
- Training rows/columns: __________
- Test rows/columns: __________
- Selected features: OverallQual, GrLivArea, TotalBsmtSF, YearBuilt, Neighborhood, GarageCars, KitchenQual, BsmtQual, FullBath, HalfBath, LotArea.

## 3. Preprocessing
Explain median imputation, most-frequent categorical imputation, one-hot encoding, and the `log1p` target transformation.

## 4. EDA observations
Describe the missing-value plot, SalePrice distribution, and living-area vs price plot. Include at least one observation from each.

## 5. Models and evaluation
Paste the actual contents of `outputs/model_comparison.csv`.

| Model | Validation MSE | Validation RMSLE |
|---|---:|---:|
| Linear Regression | ______ | ______ |
| Random Forest | ______ | ______ |

Which model had the lower RMSLE? Why did you select it? ____________________

## 6. Prediction output
- Saved model file: `outputs/best_house_model.joblib`
- Test prediction file: `outputs/test_predictions.csv`
- Number of test predictions: __________

## 7. Limitations
Mention the limited selected features, possible overfitting, data quality, and that test.csv has no public SalePrice labels.

## 8. Possible improvements
Examples: cross-validation, careful feature engineering, tuning Random Forest parameters, trying additional approved models, and better error analysis.

## 9. Conclusion
Summarize what you learned and the validation result from your own run.
