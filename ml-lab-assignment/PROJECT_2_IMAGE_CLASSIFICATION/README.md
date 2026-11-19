
# Project 2 — Image Classification Using Feature Engineering

## Objective
Classify images into Cars, Cricket Balls, and Ice-Cream Cones. Compare raw-pixel and HOG features using Logistic Regression and a linear SVM.

## Dataset — required before training
Read `data/README_DATA.md`. The assignment brief does not provide a dataset URL or actual image files. Put your professor's images into the three class folders. The script will stop with a clear error until those images are present.

## Environment setup (Windows PowerShell)
```powershell
py -3 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Run
From this project directory:
```bash
python src/image_classification.py
```

Predict one new image after training:
```bash
python src/predict_image.py path/to/new_image.jpg
```

Open `notebooks/Project_2_Image_Classification.ipynb` and run all cells for a notebook demonstration.

## What is compared
- Logistic Regression + raw pixels
- Logistic Regression + HOG
- Linear SVM + raw pixels
- Linear SVM + HOG

All images are converted to grayscale and resized to 64 × 64. Raw-pixel inputs are flattened into vectors. HOG features describe local edge directions. Each classifier uses a StandardScaler fitted only on its training split.

## Outputs
`outputs/` will contain the representative class-image montage, accuracy comparison CSV/plot, four confusion matrices, sample predictions, and the best model saved with Joblib. The program chooses the model with the highest held-out test accuracy (training accuracy is shown as a comparison).

## Report
Fill in `reports/REPORT_TEMPLATE.md` with measurements from your actual dataset run. Do not fill it with invented numbers.

## References
- HOG: https://scikit-image.org/docs/stable/api/skimage.feature.html#skimage.feature.hog
- Logistic Regression: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
- SVM: https://scikit-learn.org/stable/modules/svm.html
- Confusion matrix: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html
- Train/test split: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
