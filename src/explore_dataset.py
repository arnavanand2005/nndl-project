import os
import csv
from pathlib import Path

import cv2
import mediapipe as mp

from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# --------------------------------------------------
# PATHS
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATASET_DIR = PROJECT_ROOT / "dataset"
MODEL_PATH = PROJECT_ROOT / "models" / "hand_landmarker.task"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

OUTPUT_DIR.mkdir(exist_ok=True)


# --------------------------------------------------
# MEDIAPIPE
# --------------------------------------------------

def create_detector():

    base_options = python.BaseOptions(
        model_asset_path=str(MODEL_PATH)
    )

    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.IMAGE,
        num_hands=1,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5
    )

    return vision.HandLandmarker.create_from_options(options)


# --------------------------------------------------
# MAIN
# --------------------------------------------------

def main():

    print("=" * 60)
    print("ASL DATASET EXPLORATION")
    print("=" * 60)

    print(f"\nDataset: {DATASET_DIR}")
    print(f"Model:   {MODEL_PATH}")

    if not DATASET_DIR.exists():
        print("\nERROR: Dataset directory does not exist.")
        return

    if not MODEL_PATH.exists():
        print("\nERROR: MediaPipe model does not exist.")
        return

    # --------------------------------------------------
    # FIND CLASSES
    # --------------------------------------------------

    classes = sorted(
        [
            folder
            for folder in DATASET_DIR.iterdir()
            if folder.is_dir() and folder.name.endswith("-samples")
        ]
    )

    if not classes:
        print("\nERROR: No class folders found.")
        return

    print(f"\nFound {len(classes)} classes:")

    for folder in classes:
        print(f"  {folder.name}")

    # --------------------------------------------------
    # CREATE MEDIAPIPE DETECTOR
    # --------------------------------------------------

    detector = create_detector()

    rows = []

    total_images = 0
    total_detected = 0
    total_failed = 0

    print("\n" + "=" * 60)
    print("ANALYZING IMAGES")
    print("=" * 60)

    # --------------------------------------------------
    # PROCESS EACH CLASS
    # --------------------------------------------------

    for class_folder in classes:

        label = class_folder.name.replace("-samples", "")

        image_files = sorted(
            [
                f for f in class_folder.iterdir()
                if f.suffix.lower() in [".jpg", ".jpeg", ".png"]
            ]
        )

        detected = 0
        failed = 0

        for image_path in image_files:

            image = cv2.imread(str(image_path))

            if image is None:
                failed += 1

                rows.append({
                    "label": label,
                    "file": image_path.name,
                    "width": "",
                    "height": "",
                    "hand_detected": 0,
                    "landmarks": 0
                })

                continue

            height, width = image.shape[:2]

            rgb = cv2.cvtColor(
                image,
                cv2.COLOR_BGR2RGB
            )

            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb
            )

            result = detector.detect(mp_image)

            if result.hand_landmarks:

                detected += 1
                landmark_count = len(
                    result.hand_landmarks[0]
                )

            else:

                failed += 1
                landmark_count = 0

            rows.append({
                "label": label,
                "file": image_path.name,
                "width": width,
                "height": height,
                "hand_detected": int(bool(result.hand_landmarks)),
                "landmarks": landmark_count
            })

        total_images += len(image_files)
        total_detected += detected
        total_failed += failed

        detection_rate = (
            detected / len(image_files) * 100
            if image_files else 0
        )

        print(
            f"{label:>2} | "
            f"images: {len(image_files):3d} | "
            f"detected: {detected:3d} | "
            f"failed: {failed:3d} | "
            f"rate: {detection_rate:6.2f}%"
        )

    detector.close()

    # --------------------------------------------------
    # SAVE RESULTS
    # --------------------------------------------------

    csv_path = OUTPUT_DIR / "dataset_analysis.csv"

    with open(
        csv_path,
        "w",
        newline=""
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=[
                "label",
                "file",
                "width",
                "height",
                "hand_detected",
                "landmarks"
            ]
        )

        writer.writeheader()
        writer.writerows(rows)

    # --------------------------------------------------
    # SUMMARY
    # --------------------------------------------------

    overall_rate = (
        total_detected / total_images * 100
        if total_images else 0
    )

    print("\n" + "=" * 60)
    print("DATASET SUMMARY")
    print("=" * 60)

    print(f"Classes:          {len(classes)}")
    print(f"Total images:     {total_images}")
    print(f"Hands detected:   {total_detected}")
    print(f"Detection failed: {total_failed}")
    print(f"Detection rate:   {overall_rate:.2f}%")

    print("\nAnalysis saved to:")
    print(csv_path)

    print("\nNext step:")
    print("Review the detection rates before training anything.")


if __name__ == "__main__":
    main()