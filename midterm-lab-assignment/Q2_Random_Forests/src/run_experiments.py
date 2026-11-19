"""Question 2: Random Forest experiments on Heart Disease and Intel images."""
from pathlib import Path
import json
import re
import time
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, classification_report

PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_DIR / "data"
OUTPUT_DIR = PROJECT_DIR / "outputs"
RANDOM_STATE = 42
TEST_SIZE = 0.20
TREE_COUNTS = [1, 10, 50, 100, 300]
IMAGE_SIZE = (24, 24)
MAX_IMAGES_PER_CLASS = 1200  # Set to None to use every available image in each class.
IMAGE_CLASSES = ["buildings", "forest", "glacier", "mountain", "sea", "street"]
VALID_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def normalize_column(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(name).strip().lower()).strip("_")


def find_heart_csv() -> Path:
    candidates = [DATA_DIR / "heart.csv", DATA_DIR / "heart_cleveland_upload.csv", DATA_DIR / "heart_disease.csv"]
    for path in candidates:
        if path.exists():
            return path
    csvs = list(DATA_DIR.glob("*.csv"))
    if csvs:
        return csvs[0]
    raise FileNotFoundError(
        "Heart Disease CSV not found. Download the professor-linked Kaggle dataset and place its CSV at data/heart.csv."
    )


def load_heart_data():
    """Load and lightly clean a Heart Disease Cleveland CSV."""
    path = find_heart_csv()
    df = pd.read_csv(path)
    df.columns = [normalize_column(c) for c in df.columns]
    target_candidates = ["target", "num", "condition", "output", "heart_disease", "heartdisease"]
    target_col = next((c for c in target_candidates if c in df.columns), None)
    if target_col is None:
        # Some source files do not include headers: fail with a helpful message instead of guessing silently.
        raise ValueError(f"Could not identify target column in {path.name}. Columns: {list(df.columns)}")

    y_raw = df[target_col]
    X = df.drop(columns=[target_col]).copy()
    # Convert string question marks to missing values; coerce numeric-looking columns.
    X = X.replace({"?": np.nan, " ?": np.nan, "": np.nan})
    for col in X.columns:
        converted = pd.to_numeric(X[col], errors="coerce")
        # If nearly all non-missing entries parse numerically, treat it as numeric.
        original_nonmissing = X[col].notna().sum()
        numeric_nonmissing = converted.notna().sum()
        if original_nonmissing == 0 or numeric_nonmissing / max(original_nonmissing, 1) >= 0.90:
            X[col] = converted
        else:
            X[col] = X[col].astype("object")

    y_text = y_raw.astype(str).str.strip().str.lower().str.rstrip(".")
    numeric_y = pd.to_numeric(y_text, errors="coerce")
    if numeric_y.notna().all():
        # In the classic Cleveland `num` field, 0 means absent; 1-4 mean disease present.
        y = (numeric_y > 0).astype(int)
    else:
        positive = {"yes", "true", "1", "presence", "present", "disease", "heart disease"}
        y = y_text.isin(positive).astype(int)
    valid = y.notna()
    X = X.loc[valid].reset_index(drop=True)
    y = y.loc[valid].astype(int).reset_index(drop=True)
    if y.nunique() != 2:
        raise ValueError("Heart target did not become two classes. Check the target column and label values.")
    return X, y, path.name


def make_tabular_preprocessor(X: pd.DataFrame, scale_numeric: bool = False):
    """Impute numeric/categorical values; one-hot encode categoricals."""
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = [c for c in X.columns if c not in numeric_cols]

    numeric_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    transformers = []
    if numeric_cols:
        transformers.append(("numeric", Pipeline(numeric_steps), numeric_cols))
    if categorical_cols:
        transformers.append(("categorical", categorical_pipeline, categorical_cols))
    return ColumnTransformer(transformers=transformers, remainder="drop")


def find_intel_class_root():
    """Search common Intel Kaggle nested folder structures for six class directories."""
    search_roots = [DATA_DIR / "intel", DATA_DIR]
    class_set = set(IMAGE_CLASSES)
    for search_root in search_roots:
        if not search_root.exists():
            continue
        for candidate in [search_root] + [p for p in search_root.rglob("*") if p.is_dir()]:
            try:
                names = {p.name.lower() for p in candidate.iterdir() if p.is_dir()}
            except (PermissionError, OSError):
                continue
            if class_set.issubset(names):
                return candidate
    raise FileNotFoundError(
        "Intel training class folders not found. Extract the Kaggle archive so data/intel/seg_train/ contains "
        "buildings, forest, glacier, mountain, sea, and street folders. See data/README_DATA.md."
    )


def load_intel_images():
    """Load grayscale, resized images and flatten each into a small pixel feature vector."""
    class_root = find_intel_class_root()
    features, labels = [], []
    rng = np.random.default_rng(RANDOM_STATE)
    class_counts = {}
    class_folder_lookup = {p.name.lower(): p for p in class_root.iterdir() if p.is_dir()}
    for label, class_name in enumerate(IMAGE_CLASSES):
        folder = class_folder_lookup[class_name.lower()]
        paths = [p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in VALID_IMAGE_EXTENSIONS]
        if MAX_IMAGES_PER_CLASS is not None and len(paths) > MAX_IMAGES_PER_CLASS:
            selected_indices = rng.choice(len(paths), size=MAX_IMAGES_PER_CLASS, replace=False)
            paths = [paths[i] for i in selected_indices]
        class_counts[class_name] = len(paths)
        for path in paths:
            try:
                with Image.open(path) as im:
                    # Grayscale conversion keeps the feature vector small and consistent.
                    im = im.convert("L").resize(IMAGE_SIZE)
                    arr = np.asarray(im, dtype=np.float32) / 255.0
                    features.append(arr.flatten())
                    labels.append(label)
            except Exception as exc:
                print(f"Skipping unreadable image {path.name}: {exc}")
    if not features:
        raise ValueError("No readable images found in the Intel class folders.")
    return np.vstack(features), np.asarray(labels, dtype=int), class_counts, str(class_root)


def fit_and_measure(name, estimator, preprocessor, X_train, X_test, y_train, y_test, output_prefix, class_names=None):
    """Fit a pipeline, record training time/accuracy, and save a confusion matrix."""
    pipeline = Pipeline([("preprocess", preprocessor), ("model", estimator)])
    start = time.perf_counter()
    pipeline.fit(X_train, y_train)
    train_seconds = time.perf_counter() - start
    predictions = pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    labels = sorted(np.unique(y_test).tolist())
    display_names = class_names if class_names is not None else [str(v) for v in labels]
    # When class_names are provided (Intel), ensure label order matches numeric labels.
    disp = ConfusionMatrixDisplay(confusion_matrix(y_test, predictions, labels=labels), display_labels=display_names)
    fig, ax = plt.subplots(figsize=(7, 6))
    disp.plot(ax=ax, xticks_rotation=45, colorbar=False)
    ax.set_title(f"{output_prefix}: {name} Confusion Matrix")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / f"{output_prefix.lower()}_confusion_matrix_{normalize_column(name)}.png", dpi=150)
    plt.close(fig)

    report = classification_report(y_test, predictions, zero_division=0)
    (OUTPUT_DIR / f"{output_prefix.lower()}_classification_report_{normalize_column(name)}.txt").write_text(report, encoding="utf-8")
    return pipeline, {
        "Model": name,
        "Test_Accuracy": float(accuracy),
        "Training_Time_Seconds": float(train_seconds),
    }


def run_tree_count_experiment(X_train, X_test, y_train, y_test, prefix, preprocessor, labels_for_plot):
    rows = []
    for n_trees in TREE_COUNTS:
        model = RandomForestClassifier(n_estimators=n_trees, random_state=RANDOM_STATE, n_jobs=-1)
        pipeline = Pipeline([("preprocess", preprocessor), ("model", model)])
        pipeline.fit(X_train, y_train)
        acc = accuracy_score(y_test, pipeline.predict(X_test))
        rows.append({"n_estimators": n_trees, "test_accuracy": float(acc)})
        print(f"{prefix}: n_estimators={n_trees}, test accuracy={acc:.4f}")
    pd.DataFrame(rows).to_csv(OUTPUT_DIR / f"{prefix.lower()}_tree_count_results.csv", index=False)
    plt.figure(figsize=(7, 5))
    plt.plot([r["n_estimators"] for r in rows], [r["test_accuracy"] for r in rows], marker="o")
    plt.title(f"{prefix}: Test Accuracy vs Number of Trees")
    plt.xlabel("Number of Trees (n_estimators)")
    plt.ylabel("Test Accuracy")
    plt.xticks(TREE_COUNTS)
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"{prefix.lower()}_tree_count_accuracy.png", dpi=160)
    plt.close()
    return rows


def run_heart_experiments():
    print("\n=== Heart Disease experiments ===")
    X, y, file_name = load_heart_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    print(f"Loaded {file_name}: {len(X)} rows; class counts={y.value_counts().to_dict()}")
    tree_preprocessor = make_tabular_preprocessor(X_train, scale_numeric=False)
    svm_preprocessor = make_tabular_preprocessor(X_train, scale_numeric=True)

    rows = []
    _, row = fit_and_measure("Decision Tree", DecisionTreeClassifier(random_state=RANDOM_STATE),
                             tree_preprocessor, X_train, X_test, y_train, y_test, "heart")
    rows.append(row)
    _, row = fit_and_measure("Random Forest (100 trees)", RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1),
                             tree_preprocessor, X_train, X_test, y_train, y_test, "heart")
    rows.append(row)
    _, row = fit_and_measure("Linear SVM (LinearSVC)", LinearSVC(random_state=RANDOM_STATE, max_iter=10000),
                             svm_preprocessor, X_train, X_test, y_train, y_test, "heart")
    rows.append(row)
    pd.DataFrame(rows).to_csv(OUTPUT_DIR / "heart_model_comparison.csv", index=False)
    curve = run_tree_count_experiment(X_train, X_test, y_train, y_test, "Heart", tree_preprocessor, None)
    return {"dataset_file": file_name, "rows": len(X), "features": X.shape[1], "class_counts": y.value_counts().to_dict(),
            "model_comparison": rows, "tree_count": curve}


def run_intel_experiments():
    print("\n=== Intel Image Classification experiments ===")
    X, y, class_counts, class_root = load_intel_images()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    print(f"Image root: {class_root}; samples={len(y)}; class counts={class_counts}; features={X.shape[1]}")
    # Pixel values are already numeric and scaled to 0..1. Imputation is not required.
    from sklearn.preprocessing import FunctionTransformer
    identity = FunctionTransformer(validate=False)
    plain_preprocessor = ColumnTransformer([("pixels", "passthrough", list(range(X.shape[1])))], remainder="drop")
    # Use LinearSVC for a tractable SVM baseline over image pixel vectors.

    rows = []
    _, row = fit_and_measure("Decision Tree", DecisionTreeClassifier(random_state=RANDOM_STATE),
                             plain_preprocessor, X_train, X_test, y_train, y_test, "intel", IMAGE_CLASSES)
    rows.append(row)
    _, row = fit_and_measure("Random Forest (100 trees)", RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1),
                             plain_preprocessor, X_train, X_test, y_train, y_test, "intel", IMAGE_CLASSES)
    rows.append(row)
    svm_pipeline_preprocessor = Pipeline([("pixels", plain_preprocessor), ("scale", StandardScaler())])
    # fit_and_measure expects a preprocessor. A Pipeline is a valid transformer and is used here.
    _, row = fit_and_measure("Linear SVM (LinearSVC)", LinearSVC(random_state=RANDOM_STATE, max_iter=10000),
                             svm_pipeline_preprocessor, X_train, X_test, y_train, y_test, "intel", IMAGE_CLASSES)
    rows.append(row)
    pd.DataFrame(rows).to_csv(OUTPUT_DIR / "intel_model_comparison.csv", index=False)
    curve = run_tree_count_experiment(X_train, X_test, y_train, y_test, "Intel", plain_preprocessor, IMAGE_CLASSES)
    return {"image_root": class_root, "samples": int(len(y)), "feature_count": int(X.shape[1]),
            "image_size": IMAGE_SIZE, "class_counts": class_counts, "model_comparison": rows, "tree_count": curve}


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    summary = {
        "random_state": RANDOM_STATE,
        "test_size": TEST_SIZE,
        "tree_counts": TREE_COUNTS,
        "max_images_per_class": MAX_IMAGES_PER_CLASS,
        "image_size": IMAGE_SIZE,
    }
    summary["heart"] = run_heart_experiments()
    summary["intel"] = run_intel_experiments()
    (OUTPUT_DIR / "run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("\nQ2 finished. All outputs are saved under:", OUTPUT_DIR)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print("ERROR:", exc, file=sys.stderr)
        raise
