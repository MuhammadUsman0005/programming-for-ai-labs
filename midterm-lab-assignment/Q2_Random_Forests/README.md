# Q2 — Random Forest experiments on Heart Disease and Intel images

## Required datasets

Use the exact two Kaggle links from the assignment:

- Heart Disease Cleveland UCI: https://www.kaggle.com/datasets/cherngs/heart-disease-cleveland-uci
- Intel Image Classification: https://www.kaggle.com/datasets/puneet6060/intel-image-classification

See `data/README_DATA.md` for paths and accepted folder layout.

## Install

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Run

Place both datasets first, then from this folder:

```powershell
python src/run_experiments.py
```

The script stops with a clear error if a dataset is missing. It does not invent data or fake experiment scores.

## Experiments performed for each dataset

1. Stratified 80/20 train/test split with random_state=42.
2. Preprocess missing/nonnumeric/categorical tabular data.
3. Vary Random Forest `n_estimators` over 1, 10, 50, 100, and 300; save test-accuracy plot.
4. Compare a default Decision Tree, default Random Forest (100 trees), and a linear SVM (LinearSVC); record test accuracy and training time.
5. Save confusion matrices and classification reports.

For Intel images, each image is converted to grayscale, resized to 24x24 pixels, and flattened into 576 pixel features. This is a deliberately simple traditional-ML representation, not a CNN. The script randomly caps the number of images per class at 1,200 by default to keep the experiment practical on a student computer. Change `MAX_IMAGES_PER_CLASS = None` near the top of the script to use every image.

## Outputs

- `outputs/heart_tree_count_accuracy.png`
- `outputs/intel_tree_count_accuracy.png`
- `outputs/heart_model_comparison.csv`
- `outputs/intel_model_comparison.csv`
- `outputs/heart_confusion_matrix_*.png`
- `outputs/intel_confusion_matrix_*.png`
- `outputs/heart_classification_report_*.txt`
- `outputs/intel_classification_report_*.txt`
- `outputs/run_summary.json`

## Interpretation notes

- Training time depends on your CPU, thread count, and data size. Report your own recorded seconds.
- Increasing trees often reduces prediction variability and makes accuracy settle toward a plateau; it does not guarantee a higher score every time.
- A single tree is usually faster but may be less stable and more prone to overfitting.
- Random Forest and Decision Tree are both tree-based. LinearSVC is a margin-based linear classifier and needs scaled features in this implementation.
- This dataset is for educational classification experiments, not medical diagnosis.

See `docs/BREIMAN_PAPER_NOTES.md` and `docs/REPORT_TEMPLATE.md` before writing your report.
