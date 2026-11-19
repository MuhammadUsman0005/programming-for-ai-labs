
# Project 3 Report — Computer Vision Detection and Feature Matching

**Student:** ____________________  
**Course/Semester:** ____________________  
**Date:** ____________________

## 1. Objective
Describe face/eye detection, local feature matching, and line detection.

## 2. Input images
List the files and explain whether each is a bundled sample or your own image.

## 3. Haar face and eye detection
- Cascade XML files used: `haarcascade_frontalface_default.xml`, `haarcascade_eye.xml`
- Parameters tested: the actual `scaleFactor`, `minNeighbors`, and `minSize` rows from `outputs/face/face_parameter_experiments.csv`.
- Face/eye detections for the single-face input: __________
- Face/eye detections for the multiple-face input: __________
- One false positive/missed detection or limitation: __________

Explain how raising `minNeighbors` and changing `scaleFactor`/`minSize` affected your output.

## 4. ORB vs SIFT
Copy actual values from `outputs/matching/orb_vs_sift_comparison.csv`.

| Method | Keypoints image 1 | Keypoints image 2 | Accepted matches | Accepted/candidate ratio | Time (seconds) |
|---|---:|---:|---:|---:|---:|
| ORB | ______ | ______ | ______ | ______ | ______ |
| SIFT | ______ | ______ | ______ | ______ | ______ |

Explain Hamming distance for ORB, L2 distance for SIFT, and the ratio test. Do not compare raw descriptor distances across ORB and SIFT as though they were the same scale.

## 5. Canny and Hough line detection
- Canny thresholds: 50 and 150
- Standard Hough vote thresholds tested: 150 and 530
- Probabilistic Hough settings tested: minimum segment lengths 40 and 540; maximum gap 10
- Number of standard lines/segments in each experiment: copy from `outputs/hough/hough_parameter_results.json`.

Explain how the Hough threshold and minimum segment length changed the output.

## 6. Limitations
Discuss false/missed Haar detections, low-resolution montage quality, synthetic feature-matching scene, and line-detection sensitivity to edge quality/parameter settings.

## 7. Possible improvements
Use more varied real face images, more challenging same-scene images, tune detector parameters for the real image, and test additional edge/Hough thresholds.

## 8. Conclusion
Summarize the method that worked best for your supplied images and the evidence from the outputs.
