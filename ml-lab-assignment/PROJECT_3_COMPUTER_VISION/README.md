
# Project 3 — Computer Vision Detection and Feature Matching

## Objective
Use OpenCV to detect faces and eyes with Haar Cascades, detect/match keypoints using ORB and SIFT, and detect Sudoku-grid lines using Canny and the standard/probabilistic Hough transforms.

## Included input images
`data/input_images/` contains sample inputs. See `data/SOURCE_INFO.md` for their origin and limitations. The sample set makes the project runnable without requiring the professor's local files.

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
python src/computer_vision_project.py
```

Open `notebooks/Project_3_Computer_Vision.ipynb` and run all cells for a notebook demonstration.

## What the script does
1. Loads single-face and multiple-face samples, converts to grayscale, detects faces and eyes, and draws rectangles.
2. Varies `scaleFactor`, `minNeighbors`, and `minSize`; records detection counts so you can discuss false/missed detections.
3. Detects and matches ORB features using Hamming distance and SIFT features using L2 distance; applies a nearest/second-nearest ratio test.
4. Saves matched keypoints/lines and a CSV comparison including keypoint count, accepted matches, match ratio, and processing time.
5. Runs Canny edge detection, `HoughLines`, and `HoughLinesP`, then changes Hough thresholds/line-length parameters and saves the outputs.

## Outputs
- Face and eye annotated images: `outputs/face/`
- ORB/SIFT matches and comparison: `outputs/matching/`
- Canny and Hough results/experiments: `outputs/hough/`

## Report
Fill in `reports/REPORT_TEMPLATE.md` after running the project. Include actual detection counts and timings from your computer. Processing speed depends on hardware and should not be copied from another machine.

## References
- OpenCV Haar Cascade tutorial: https://docs.opencv.org/4.x/db/d28/tutorial_cascade_classifier.html
- OpenCV feature matching: https://docs.opencv.org/4.x/dc/dc3/tutorial_py_matcher.html
- ORB class reference: https://docs.opencv.org/4.x/db/d95/classcv_1_1ORB.html
- SIFT class reference: https://docs.opencv.org/4.x/d7/d60/classcv_1_1SIFT.html
- Hough Line Transform: https://docs.opencv.org/4.x/d9/db0/tutorial_hough_lines.html
- scikit-image sample data source: https://scikit-image.org/docs/stable/api/skimage.data.html
