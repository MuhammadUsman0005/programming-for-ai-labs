
"""OpenCV lab: Haar face/eye detection, ORB/SIFT matching, and Hough lines."""
from pathlib import Path
import time
import json

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def load_gray_image(path):
    """Read an image in grayscale and show a useful error if it cannot be read."""
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return image


def detect_faces_and_eyes(image_path, output_path, scale_factor=1.1, min_neighbors=4, min_size=(30, 30)):
    """Detect face boxes and then eye boxes inside each detected face region."""
    gray = load_gray_image(image_path)

    # OpenCV ships pre-trained Haar XML files with its Python package.
    face_file = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    eye_file = cv2.data.haarcascades + "haarcascade_eye.xml"
    face_cascade = cv2.CascadeClassifier(face_file)
    eye_cascade = cv2.CascadeClassifier(eye_file)
    if face_cascade.empty() or eye_cascade.empty():
        raise RuntimeError("Could not load OpenCV Haar cascade XML files.")

    # scaleFactor, minNeighbors, and minSize are the parameters we experiment with.
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=scale_factor,
        minNeighbors=min_neighbors,
        minSize=min_size
    )

    # Convert grayscale to BGR so colored rectangles can be drawn and saved.
    output = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    total_eyes = 0
    for (x, y, width, height) in faces:
        cv2.rectangle(output, (x, y), (x + width, y + height), (0, 255, 0), 2)
        face_region = gray[y:y + height, x:x + width]

        # Search for eyes only inside a detected face region to reduce irrelevant boxes.
        eyes = eye_cascade.detectMultiScale(
            face_region,
            scaleFactor=1.1,
            minNeighbors=4,
            minSize=(10, 10)
        )
        total_eyes += len(eyes)
        for (eye_x, eye_y, eye_width, eye_height) in eyes:
            cv2.rectangle(
                output,
                (x + eye_x, y + eye_y),
                (x + eye_x + eye_width, y + eye_y + eye_height),
                (255, 0, 0), 2
            )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), output)
    return len(faces), total_eyes


def run_face_experiments(input_dir, output_dir):
    """Run two sample images with a few parameter combinations."""
    face_inputs = [
        input_dir / "face_single_astronaut.jpg",
        input_dir / "face_multiple_lfw_montage.png"
    ]
    # These small parameter changes illustrate the effect of the detector settings.
    parameter_sets = [
        (1.05, 3, (25, 25)),
        (1.10, 4, (30, 30)),
        (1.20, 6, (40, 40))
    ]
    rows = []
    for image_path in face_inputs:
        for scale_factor, min_neighbors, min_size in parameter_sets:
            output_name = (
                f"{image_path.stem}_sf{scale_factor}_mn{min_neighbors}_"
                f"ms{min_size[0]}.jpg"
            )
            face_count, eye_count = detect_faces_and_eyes(
                image_path,
                output_dir / output_name,
                scale_factor,
                min_neighbors,
                min_size
            )
            rows.append({
                "Input Image": image_path.name,
                "scaleFactor": scale_factor,
                "minNeighbors": min_neighbors,
                "minSize": str(min_size),
                "Face Detections": face_count,
                "Eye Detections": eye_count
            })
            print(f"{image_path.name}: {face_count} face(s), {eye_count} eye box(es)")

    results = pd.DataFrame(rows)
    results.to_csv(output_dir / "face_parameter_experiments.csv", index=False)
    return results


def ratio_test_matches(descriptor1, descriptor2, norm_type, ratio=0.75):
    """Find descriptor pairs and keep matches passing Lowe's ratio test."""
    matcher = cv2.BFMatcher(norm_type)
    # k=2 returns the best and second-best match for each descriptor.
    candidate_pairs = matcher.knnMatch(descriptor1, descriptor2, k=2)
    good_matches = []
    for pair in candidate_pairs:
        if len(pair) == 2:
            best, second_best = pair
            if best.distance < ratio * second_best.distance:
                good_matches.append(best)
    return good_matches, len(candidate_pairs)


def compare_feature_methods(image1_path, image2_path, output_dir):
    """Compare ORB and SIFT keypoints, filtered matches, and elapsed time."""
    image1 = load_gray_image(image1_path)
    image2 = load_gray_image(image2_path)
    rows = []

    # ORB creates binary descriptors, therefore use Hamming distance.
    orb = cv2.ORB_create(nfeatures=1000)
    start_time = time.perf_counter()
    orb_keypoints1, orb_descriptors1 = orb.detectAndCompute(image1, None)
    orb_keypoints2, orb_descriptors2 = orb.detectAndCompute(image2, None)
    orb_elapsed = time.perf_counter() - start_time

    orb_good = []
    orb_candidates = 0
    if orb_descriptors1 is not None and orb_descriptors2 is not None:
        match_start = time.perf_counter()
        orb_good, orb_candidates = ratio_test_matches(
            orb_descriptors1, orb_descriptors2, cv2.NORM_HAMMING
        )
        orb_elapsed += time.perf_counter() - match_start
        orb_visual = cv2.drawMatches(
            image1, orb_keypoints1, image2, orb_keypoints2,
            sorted(orb_good, key=lambda match: match.distance)[:40],
            None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
        )
        cv2.imwrite(str(output_dir / "orb_feature_matches.png"), orb_visual)

    # SIFT descriptors are floating-point vectors, therefore use Euclidean/L2 distance.
    if not hasattr(cv2, "SIFT_create"):
        raise RuntimeError(
            "This OpenCV build does not expose SIFT_create. Install a current "
            "opencv-python package (not opencv-python-headless from an old environment)."
        )
    sift = cv2.SIFT_create(nfeatures=1000)
    start_time = time.perf_counter()
    sift_keypoints1, sift_descriptors1 = sift.detectAndCompute(image1, None)
    sift_keypoints2, sift_descriptors2 = sift.detectAndCompute(image2, None)
    sift_elapsed = time.perf_counter() - start_time

    sift_good = []
    sift_candidates = 0
    if sift_descriptors1 is not None and sift_descriptors2 is not None:
        match_start = time.perf_counter()
        sift_good, sift_candidates = ratio_test_matches(
            sift_descriptors1, sift_descriptors2, cv2.NORM_L2
        )
        sift_elapsed += time.perf_counter() - match_start
        sift_visual = cv2.drawMatches(
            image1, sift_keypoints1, image2, sift_keypoints2,
            sorted(sift_good, key=lambda match: match.distance)[:40],
            None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
        )
        cv2.imwrite(str(output_dir / "sift_feature_matches.png"), sift_visual)

    rows.append({
        "Method": "ORB",
        "Keypoints Image 1": len(orb_keypoints1),
        "Keypoints Image 2": len(orb_keypoints2),
        "Candidate Pairs": orb_candidates,
        "Accepted Matches": len(orb_good),
        "Accepted/Candidate Ratio": len(orb_good) / max(orb_candidates, 1),
        "Processing Time (seconds)": orb_elapsed
    })
    rows.append({
        "Method": "SIFT",
        "Keypoints Image 1": len(sift_keypoints1),
        "Keypoints Image 2": len(sift_keypoints2),
        "Candidate Pairs": sift_candidates,
        "Accepted Matches": len(sift_good),
        "Accepted/Candidate Ratio": len(sift_good) / max(sift_candidates, 1),
        "Processing Time (seconds)": sift_elapsed
    })

    comparison = pd.DataFrame(rows)
    comparison.to_csv(output_dir / "orb_vs_sift_comparison.csv", index=False)
    return comparison


def create_sudoku_grid(output_path):
    """Create a simple Sudoku-style grid so the line-detection task has an input."""
    image = np.full((600, 600), 255, dtype=np.uint8)
    start, step = 30, 60
    for index in range(10):
        thickness = 4 if index % 3 == 0 else 1
        position = start + index * step
        cv2.line(image, (position, start), (position, 570), 0, thickness)
        cv2.line(image, (start, position), (570, position), 0, thickness)
    for row, col, value in [(0,0,'8'),(0,4,'3'),(1,2,'7'),(2,6,'1'),(3,3,'5'),(5,1,'2'),(6,8,'9'),(8,4,'6')]:
        cv2.putText(image, value, (start + col*step + 18, start + row*step + 44),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, 0, 2, cv2.LINE_AA)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), image)
    return image


def run_hough_experiments(input_path, output_dir):
    """Apply Canny, HoughLines, HoughLinesP, and simple parameter variations."""
    gray = load_gray_image(input_path)
    edges = cv2.Canny(gray, 50, 150)
    cv2.imwrite(str(output_dir / "canny_edges.png"), edges)

    # Standard Hough returns (rho, theta) values that represent infinite lines.
    standard_output = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    standard_lines = cv2.HoughLines(edges, 1, np.pi / 180, 150)
    standard_count = 0
    if standard_lines is not None:
        for line in standard_lines:
            rho, theta = line[0]
            cos_theta = np.cos(theta)
            sin_theta = np.sin(theta)
            x0 = cos_theta * rho
            y0 = sin_theta * rho
            point1 = (int(x0 + 1000 * (-sin_theta)), int(y0 + 1000 * cos_theta))
            point2 = (int(x0 - 1000 * (-sin_theta)), int(y0 - 1000 * cos_theta))
            cv2.line(standard_output, point1, point2, (0, 0, 255), 2)
            standard_count += 1
    cv2.imwrite(str(output_dir / "hough_standard.png"), standard_output)

    # Probabilistic Hough returns finite line segments (x1, y1, x2, y2).
    probabilistic_output = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    probabilistic_lines = cv2.HoughLinesP(
        edges, 1, np.pi / 180, threshold=80,
        minLineLength=40, maxLineGap=10
    )
    probabilistic_count = 0
    if probabilistic_lines is not None:
        for line in probabilistic_lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(probabilistic_output, (x1, y1), (x2, y2), (0, 0, 255), 2)
            probabilistic_count += 1
    cv2.imwrite(str(output_dir / "hough_probabilistic.png"), probabilistic_output)

    # Experiment 1: higher vote threshold should usually keep only more-supported lines.
    high_threshold_output = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    high_threshold_lines = cv2.HoughLines(edges, 1, np.pi / 180, 530)
    high_threshold_count = 0
    if high_threshold_lines is not None:
        for line in high_threshold_lines:
            rho, theta = line[0]
            a, b = np.cos(theta), np.sin(theta)
            x0, y0 = a * rho, b * rho
            p1 = (int(x0 + 1000 * (-b)), int(y0 + 1000 * a))
            p2 = (int(x0 - 1000 * (-b)), int(y0 - 1000 * a))
            cv2.line(high_threshold_output, p1, p2, (0, 0, 255), 2)
            high_threshold_count += 1
    cv2.imwrite(str(output_dir / "hough_standard_threshold_530.png"), high_threshold_output)

    # Experiment 2: a longer minimum segment length removes short fragments.
    long_segment_output = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    long_lines = cv2.HoughLinesP(
        edges, 1, np.pi / 180, threshold=80,
        minLineLength=540, maxLineGap=10
    )
    long_segment_count = 0
    if long_lines is not None:
        for line in long_lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(long_segment_output, (x1, y1), (x2, y2), (0, 0, 255), 2)
            long_segment_count += 1
    cv2.imwrite(str(output_dir / "hough_probabilistic_min_length_540.png"), long_segment_output)

    result = {
        "standard_hough_lines_threshold_150": standard_count,
        "standard_hough_lines_threshold_530": high_threshold_count,
        "probabilistic_segments_min_length_40": probabilistic_count,
        "probabilistic_segments_min_length_540": long_segment_count,
        "canny_low_threshold": 50,
        "canny_high_threshold": 150
    }
    (output_dir / "hough_parameter_results.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def run_project(input_dir, output_dir):
    """Run face/eye, feature matching, and Hough line experiments."""
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    face_output = output_dir / "face"
    matching_output = output_dir / "matching"
    hough_output = output_dir / "hough"
    for folder in [face_output, matching_output, hough_output]:
        folder.mkdir(parents=True, exist_ok=True)

    face_results = run_face_experiments(input_dir, face_output)
    matching_results = compare_feature_methods(
        input_dir / "feature_scene_1.png",
        input_dir / "feature_scene_2.png",
        matching_output
    )
    hough_results = run_hough_experiments(input_dir / "sudoku_grid.png", hough_output)

    summary = {
        "face_experiment_rows": len(face_results),
        "feature_matching_methods": matching_results["Method"].tolist(),
        "hough_results": hough_results
    }
    (output_dir / "run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("\nORB vs SIFT comparison:\n", matching_results.to_string(index=False))
    print("\nHough results:\n", hough_results)
    return face_results, matching_results, hough_results


def main():
    project_dir = Path(__file__).resolve().parents[1]
    run_project(project_dir / "data" / "input_images", project_dir / "outputs")


if __name__ == "__main__":
    main()
