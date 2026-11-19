
# Project 2 Report — Image Classification Using Feature Engineering

**Student:** ____________________  
**Course/Semester:** ____________________  
**Date:** ____________________

## 1. Objective
Identify Cars, Cricket Balls, and Ice-Cream Cones from images.

## 2. Dataset
- Dataset source/provider: ____________________
- Cars images: ______
- Cricket Balls images: ______
- Ice-Cream Cones images: ______
- Total readable images: ______
- Train/test split: 75% / 25% (stratified; fixed seed 42)

## 3. Preprocessing
Explain grayscale conversion and resizing to 64 × 64.

## 4. Feature representations
Explain raw-pixel flattening and HOG (9 orientations, 8 × 8 pixels per cell, 2 × 2 cells per block, L2-Hys normalization).

## 5. Models and measured results
Copy the actual `outputs/model_comparison.csv` values.

| Model | Feature type | Training accuracy | Testing accuracy |
|---|---|---:|---:|
| Logistic Regression | Raw Pixels | ______ | ______ |
| Logistic Regression | HOG | ______ | ______ |
| Linear SVM | Raw Pixels | ______ | ______ |
| Linear SVM | HOG | ______ | ______ |

Best combination: ____________________  
Evidence from test accuracy/confusion matrix: ____________________

## 6. Confusion matrix observations
For each class, identify correct predictions and the most common confusion. Include at least one example error.

## 7. Limitations
Discuss dataset size, class imbalance, lighting/background variation, image quality, and generalization to images unlike the training set.

## 8. Possible improvements
Examples: more representative images, cross-validation, image augmentation, trying additional feature settings, or carefully tuning the linear SVM/Logistic Regression.

## 9. Conclusion
Summarize the best model and what HOG changed in your experiment.
