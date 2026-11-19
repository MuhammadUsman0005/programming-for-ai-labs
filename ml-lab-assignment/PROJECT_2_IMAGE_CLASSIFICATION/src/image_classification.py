
"""Image classification: raw pixels/HOG x Logistic Regression/linear SVM.

Expected folders:
  data/images/Cars/
  data/images/Cricket Balls/
  data/images/Ice-Cream Cones/

Run from the project directory: python src/image_classification.py
"""
from pathlib import Path
import json

import cv2
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from skimage.feature import hog
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

RANDOM_STATE = 42
IMAGE_SIZE = (64, 64)  # width, height: every image must have the same dimensions.
CLASS_NAMES = ["Cars", "Cricket Balls", "Ice-Cream Cones"]
VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def load_images(data_dir):
    """Load images from the three class folders, grayscale and resize them."""
    images = []
    labels = []
    counts = {}

    for class_name in CLASS_NAMES:
        class_folder = data_dir / "images" / class_name
        if not class_folder.exists():
            raise FileNotFoundError(
                f"Missing class folder: {class_folder}. Read data/README_DATA.md first."
            )

        class_count = 0
        # Search inside nested folders too, so a dataset split into subfolders can be used.
        for image_path in sorted(class_folder.rglob("*")):
            if image_path.suffix.lower() not in VALID_EXTENSIONS:
                continue

            # Load directly as grayscale; this is the first required preprocessing step.
            image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
            if image is None:
                print(f"Warning: could not read image, skipping: {image_path}")
                continue

            # Resize to one fixed shape, so every raw-pixel vector has the same length.
            image = cv2.resize(image, IMAGE_SIZE)
            images.append(image)
            labels.append(class_name)
            class_count += 1
        counts[class_name] = class_count

    too_small = {name: count for name, count in counts.items() if count < 2}
    if too_small:
        raise ValueError(
            f"Need at least two readable images in every class for a stratified split. "
            f"Current counts: {counts}"
        )
    print("Images loaded per class:", counts)
    return np.array(images), np.array(labels), counts


def make_raw_pixel_features(images):
    """Flatten each 64x64 grayscale image into one raw-pixel feature vector."""
    # 64 x 64 pixels become 4,096 numeric input features per image.
    return images.reshape(len(images), -1).astype("float32")


def make_hog_features(images):
    """Extract a HOG descriptor from each grayscale image."""
    feature_rows = []
    for image in images:
        features = hog(
            image,
            orientations=9,
            pixels_per_cell=(8, 8),
            cells_per_block=(2, 2),
            block_norm="L2-Hys",
            feature_vector=True
        )
        feature_rows.append(features)
    return np.asarray(feature_rows, dtype="float32")



def save_representative_images(images, labels, output_path):
    """Save one example image from each class for the required visual inspection."""
    figure, axes = plt.subplots(1, len(CLASS_NAMES), figsize=(10, 4))
    for ax, class_name in zip(axes, CLASS_NAMES):
        # Select the first readable example belonging to this class.
        matching_indices = np.where(labels == class_name)[0]
        if len(matching_indices) > 0:
            ax.imshow(images[matching_indices[0]], cmap="gray")
            ax.set_title(class_name)
        else:
            ax.set_title(f"{class_name} (missing)")
        ax.axis("off")
    figure.suptitle("Representative Image from Each Class")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

def create_confusion_matrix_plot(y_true, y_pred, model_name, output_path):
    """Save a labeled confusion matrix for one model/feature combination."""
    matrix = confusion_matrix(y_true, y_pred, labels=CLASS_NAMES)
    display = ConfusionMatrixDisplay(confusion_matrix=matrix, display_labels=CLASS_NAMES)
    fig, ax = plt.subplots(figsize=(6, 5))
    display.plot(ax=ax, cmap="Blues", colorbar=False, values_format="d")
    ax.set_title(f"Confusion Matrix — {model_name}")
    ax.set_xlabel("Predicted Class")
    ax.set_ylabel("True Class")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def make_sample_prediction_plot(images, true_labels, predicted_labels, output_path):
    """Show up to 12 test images with true and predicted labels."""
    count = min(12, len(images))
    if count == 0:
        return
    figure, axes = plt.subplots(3, 4, figsize=(12, 8))
    axes = axes.ravel()
    for index, ax in enumerate(axes):
        ax.axis("off")
        if index < count:
            ax.imshow(images[index], cmap="gray")
            ax.set_title(
                f"True: {true_labels[index]}\nPred: {predicted_labels[index]}",
                fontsize=8
            )
    figure.suptitle("Sample Test Predictions")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def run_project(data_dir, output_dir):
    """Train/evaluate all four combinations and save the best complete pipeline."""
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    images, labels, counts = load_images(data_dir)
    save_representative_images(images, labels, output_dir / "representative_images.png")
    raw_features = make_raw_pixel_features(images)
    hog_features = make_hog_features(images)

    # One split is reused for a fair comparison across feature/model combinations.
    indices = np.arange(len(labels))
    train_indices, test_indices = train_test_split(
        indices,
        test_size=0.25,
        random_state=RANDOM_STATE,
        stratify=labels
    )
    y_train = labels[train_indices]
    y_test = labels[test_indices]

    feature_sets = {
        "Raw Pixels": raw_features,
        "HOG": hog_features
    }
    model_factories = {
        "Logistic Regression": lambda: LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        "Linear SVM": lambda: SVC(kernel="linear")
    }

    rows = []
    fitted = {}
    predictions_for_best = None
    best_test_images = images[test_indices]

    for feature_name, features in feature_sets.items():
        X_train = features[train_indices]
        X_test = features[test_indices]

        for model_name, model_factory in model_factories.items():
            # Scaling is inside the pipeline, so it is fitted only on training samples.
            pipeline = Pipeline(steps=[
                ("scaler", StandardScaler()),
                ("classifier", model_factory())
            ])
            pipeline.fit(X_train, y_train)

            train_predictions = pipeline.predict(X_train)
            test_predictions = pipeline.predict(X_test)
            train_accuracy = accuracy_score(y_train, train_predictions)
            test_accuracy = accuracy_score(y_test, test_predictions)
            combined_name = f"{model_name} + {feature_name}"

            rows.append({
                "Model": model_name,
                "Features": feature_name,
                "Training Accuracy": float(train_accuracy),
                "Testing Accuracy": float(test_accuracy)
            })
            fitted[combined_name] = {
                "pipeline": pipeline,
                "feature_type": "raw" if feature_name == "Raw Pixels" else "hog"
            }
            create_confusion_matrix_plot(
                y_test, test_predictions, combined_name,
                output_dir / f"confusion_matrix_{model_name.lower().replace(' ', '_')}_{feature_name.lower().replace(' ', '_')}.png"
            )
            print(
                f"{combined_name}: train accuracy={train_accuracy:.4f}, "
                f"test accuracy={test_accuracy:.4f}"
            )

    results = pd.DataFrame(rows).sort_values(
        ["Testing Accuracy", "Training Accuracy"], ascending=[False, False]
    )
    results.to_csv(output_dir / "model_comparison.csv", index=False)

    # A test-accuracy plot directly compares the four required combinations.
    labels_for_plot = [f"{r['Model']}\n{r['Features']}" for _, r in results.iterrows()]
    plt.figure(figsize=(10, 5))
    plt.bar(labels_for_plot, results["Testing Accuracy"])
    plt.title("Testing Accuracy — Four Model/Feature Combinations")
    plt.xlabel("Model and Feature Representation")
    plt.ylabel("Testing Accuracy")
    plt.ylim(0, 1.05)
    plt.xticks(rotation=10, ha="right")
    plt.tight_layout()
    plt.savefig(output_dir / "testing_accuracy_comparison.png", dpi=150)
    plt.close()

    best_row = results.iloc[0]
    best_name = f"{best_row['Model']} + {best_row['Features']}"
    best_entry = fitted[best_name]
    best_feature_type = best_entry["feature_type"]
    best_pipeline = best_entry["pipeline"]

    # Store both the trained estimator and the exact preprocessing details for prediction.
    model_bundle = {
        "pipeline": best_pipeline,
        "feature_type": best_feature_type,
        "image_size": IMAGE_SIZE,
        "class_names": CLASS_NAMES,
        "model_name": best_name
    }
    joblib.dump(model_bundle, output_dir / "best_image_classifier.joblib")

    # Recreate predictions for the chosen model so the sample image chart matches it.
    selected_features = raw_features if best_feature_type == "raw" else hog_features
    best_test_predictions = best_pipeline.predict(selected_features[test_indices])
    make_sample_prediction_plot(
        best_test_images, y_test, best_test_predictions,
        output_dir / "sample_predictions.png"
    )
    pd.DataFrame({
        "True Class": y_test,
        "Predicted Class": best_test_predictions
    }).to_csv(output_dir / "sample_predictions.csv", index=False)

    summary = {
        "best_model": best_name,
        "feature_type": best_feature_type,
        "testing_accuracy": float(best_row["Testing Accuracy"]),
        "training_accuracy": float(best_row["Training Accuracy"]),
        "class_counts": counts,
        "random_state": RANDOM_STATE,
        "image_size": list(IMAGE_SIZE)
    }
    (output_dir / "run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("\nModel comparison:\n", results.to_string(index=False))
    print("Best model saved to:", output_dir / "best_image_classifier.joblib")
    return results, summary


def main():
    project_dir = Path(__file__).resolve().parents[1]
    run_project(project_dir / "data", project_dir / "outputs")


if __name__ == "__main__":
    main()
