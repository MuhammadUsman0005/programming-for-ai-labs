# Q3 Adult Income data folder

Professor's dataset link:
https://www.kaggle.com/datasets/mosapabdelghany/adult-income-prediction-dataset

Download and extract the labelled CSV. Copy it to this folder as:
`adult.csv`

The training script detects common target-column names (`income`, `class`, `salary`, `income_bracket`) and normalizes spaces/hyphens in field names. It expects the usual Adult features such as age, workclass, education, marital-status, occupation, relationship, race, sex, capital-gain, capital-loss, hours-per-week, and native-country (some datasets also include `fnlwgt` and `education-num`).

If the Kaggle file uses unusual column headers, inspect the first row/header and adjust `TARGET_COLUMN` / aliases in `src/train_adult.py`.
