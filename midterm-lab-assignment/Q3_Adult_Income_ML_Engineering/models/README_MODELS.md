# Model files

After running `python -m src.train_adult --baseline-only` or `--automl`, the script creates:

`models/model_bundle.joblib`

This file contains the fitted preprocessing/model pipeline, final test metrics, original input feature names, and top-five permutation importance results. The API needs this file to serve predictions. Do not delete it before running FastAPI or Streamlit.

Model binaries are generated from your local dataset and therefore are not included in the starter ZIP.
