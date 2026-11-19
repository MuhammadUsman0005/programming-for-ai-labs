# Project 1 dataset setup

1. Open the official Kaggle competition data page: https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/data
2. Sign in to Kaggle if asked and accept the competition rules if required.
3. Download at least `train.csv` and `test.csv`.
4. Put both files directly in this `data/` directory:

```text
data/
  train.csv
  test.csv
```

The training file must include `SalePrice`; the test file normally does not. The code uses a validation split from `train.csv` for metrics and generates predictions for `test.csv` separately.
