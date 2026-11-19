
"""Predict the class of one new image using the saved Project 2 model."""
from pathlib import Path
import argparse

import cv2
import joblib
import numpy as np
from skimage.feature import hog


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("image", help="Path to the image you want to classify")
    parser.add_argument("--model", default="outputs/best_image_classifier.joblib")
    args = parser.parse_args()

    model_path = Path(args.model)
    image_path = Path(args.image)
    if not model_path.exists():
        raise FileNotFoundError("Saved classifier not found. Train the model first.")
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    bundle = joblib.load(model_path)
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError("OpenCV could not read the supplied image.")

    image = cv2.resize(image, tuple(bundle["image_size"]))
    if bundle["feature_type"] == "raw":
        # Raw-pixel mode: flatten all grayscale values into one row.
        features = image.reshape(1, -1).astype("float32")
    else:
        # HOG mode: use exactly the same settings that were used in training.
        hog_features = hog(
            image,
            orientations=9,
            pixels_per_cell=(8, 8),
            cells_per_block=(2, 2),
            block_norm="L2-Hys",
            feature_vector=True
        )
        features = hog_features.reshape(1, -1).astype("float32")

    predicted_class = bundle["pipeline"].predict(features)[0]
    print("Selected trained model:", bundle["model_name"])
    print("Predicted class:", predicted_class)
    print("Note: prediction quality depends on how representative the training images are.")


if __name__ == "__main__":
    main()
