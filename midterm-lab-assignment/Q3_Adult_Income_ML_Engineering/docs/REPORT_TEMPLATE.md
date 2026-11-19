# Q3 Report — Adult Income End-to-End ML Engineering

**Student:** [ADD NAME]  
**Course:** Programming for AI  
**Dataset:** Adult Income Classification (professor's Kaggle link)  
**Run date:** [ADD]

## 1. Objective

Build a complete binary-classification workflow that predicts whether an Adult Income record is `>50K` or `<=50K`, from data loading and EDA through model training, API, interface, and deployment.

## 2. Dataset and EDA

- Source link: [ADD]
- CSV file: [ADD]
- Rows after cleaning: [ADD]
- Input features: [ADD]
- Target: `income` (or actual detected target header)
- Class counts: [ADD from run_summary.json]
- EDA observations: [ADD from plots]

Insert `class_balance.png`, `age_distribution.png`, `numeric_feature_distributions.png`, `categorical_feature_distributions.png`, and `numeric_correlation.png`.

## 3. Preprocessing and feature engineering

Describe missing-value handling, categorical encoding, numeric scaling, and derived features (`capital_net`, `has_capital_change`, `age_group`). Mention that the preprocessing pipeline is fitted on training data only.

## 4. Baseline model

- Algorithm: Logistic Regression pipeline
- Test Accuracy: [ADD]
- Precision: [ADD]
- Recall: [ADD]
- F1: [ADD]
- ROC-AUC: [ADD]
- Training time: [ADD]

Insert baseline confusion matrix and classification report.

## 5. TPOT AutoML

- TPOT version: [ADD]
- Search duration requested: 30 minutes (or actual changed value)
- Actual search time: [ADD]
- Search space: linear-light
- CV folds: 3
- Best pipeline/model: [ADD]
- Test metrics: [ADD]

Insert the AutoML confusion matrix and comparison table.

## 6. Model comparison

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | Training time (s) |
|---|---:|---:|---:|---:|---:|---:|
| Baseline Logistic Regression | [ADD] | [ADD] | [ADD] | [ADD] | [ADD] | [ADD] |
| TPOT AutoML | [ADD] | [ADD] | [ADD] | [ADD] | [ADD] | [ADD] |

Discuss performance, runtime, pipeline complexity, interpretability, and limitations.

## 7. Model saving and API

Explain that the chosen fitted preprocessing/model pipeline was saved with joblib. Describe the FastAPI `/predict` endpoint, its JSON input, and JSON output. Insert a screenshot from `/docs` or a successful API request.

## 8. Streamlit user interface

Explain how Streamlit collects input values, sends them to the API, displays a prediction, shows test metrics, and shows the top five permutation-importance features. Insert a screenshot of your local UI.

## 9. Deployment

- Platform: Hugging Face Spaces (Docker) or Streamlit Community Cloud
- Live URL: [ADD YOUR REAL LIVE URL]
- Date tested: [ADD]
- Deployment issues/resolutions: [ADD]

Do not invent a live link. Add it only after your Space is publicly reachable and tested.

## 10. Responsible-use limitations

Discuss the historical nature of the dataset, demographic attributes, dataset bias, and why this educational model must not be used for real employment or other high-impact decisions.

## 11. Conclusion

[Write a short conclusion using your actual results.]
