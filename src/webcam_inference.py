import cv2
import json
import numpy as np
import mediapipe as mp

from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from tensorflow import keras


# ============================================================
# PATHS
# ============================================================

MODEL_PATH = "models/hand_landmarker.task"
CLASS_NAMES_PATH = "models/class_names.json"
CLASSIFIER_PATH = "models/sign_language_landmark.keras"


# ============================================================
# LOAD CLASS NAMES
# ============================================================

with open(CLASS_NAMES_PATH, "r") as f:
    class_names = json.load(f)

print("Classes:", class_names)


# ============================================================
# LOAD TRAINED CLASSIFIER
# ============================================================

print("Loading trained classifier...")

classifier = keras.models.load_model(CLASSIFIER_PATH)

print("Classifier loaded successfully.")


# ============================================================
# CREATE MEDIAPIPE HAND LANDMARKER
# ============================================================

print("Loading MediaPipe Hand Landmarker...")

base_options = python.BaseOptions(
    model_asset_path=MODEL_PATH
)

options = vision.HandLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.IMAGE,
    num_hands=1,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5
)

detector = vision.HandLandmarker.create_from_options(options)

print("MediaPipe loaded successfully.")


# ============================================================
# EXTRACT NORMALIZED 63 FEATURES
# ============================================================

def extract_landmarks(result):

    if not result.hand_landmarks:
        return None

    hand = result.hand_landmarks[0]

    # --------------------------------------------------------
    # IMPORTANT:
    # The model was trained using wrist-relative coordinates.
    # MediaPipe landmark 0 = wrist.
    # --------------------------------------------------------

    wrist = hand[0]

    features = []

    for landmark in hand:

        # Normalize relative to wrist
        x = landmark.x - wrist.x
        y = landmark.y - wrist.y
        z = landmark.z - wrist.z

        features.extend([x, y, z])

    return np.array(features, dtype=np.float32)


# ============================================================
# START WEBCAM
# ============================================================

print("Starting webcam...")

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ERROR: Could not open webcam.")
    print("Try changing VideoCapture(0) to VideoCapture(1).")
    detector.close()
    exit()

print("Webcam started.")
print("Show an ASL hand sign.")
print("Press Q to quit.")


# ============================================================
# WEBCAM LOOP
# ============================================================

while True:

    ret, frame = cap.read()

    if not ret:
        print("ERROR: Could not read frame.")
        break

    # Mirror webcam for natural interaction
    frame = cv2.flip(frame, 1)

    # --------------------------------------------------------
    # BGR -> RGB
    # --------------------------------------------------------

    rgb_frame = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb_frame
    )

    # --------------------------------------------------------
    # Detect hand
    # --------------------------------------------------------

    result = detector.detect(mp_image)

    # --------------------------------------------------------
    # Extract SAME features used during training
    # --------------------------------------------------------

    features = extract_landmarks(result)

    prediction_text = "No hand detected"
    confidence_text = ""

    # ========================================================
    # PREDICT SIGN
    # ========================================================

    if features is not None:

        input_data = features.reshape(1, -1)

        probabilities = classifier.predict(
            input_data,
            verbose=0
        )[0]

        predicted_index = int(np.argmax(probabilities))

        predicted_class = class_names[predicted_index]

        confidence = float(probabilities[predicted_index]) * 100

        prediction_text = f"Sign: {predicted_class}"
        confidence_text = f"Confidence: {confidence:.2f}%"


        # ====================================================
        # DRAW LANDMARKS
        # ====================================================

        hand = result.hand_landmarks[0]

        h, w, _ = frame.shape

        # Hand connections
        connections = [
            (0, 1), (1, 2), (2, 3), (3, 4),
            (0, 5), (5, 6), (6, 7), (7, 8),
            (0, 9), (9, 10), (10, 11), (11, 12),
            (0, 13), (13, 14), (14, 15), (15, 16),
            (0, 17), (17, 18), (18, 19), (19, 20),
            (5, 9), (9, 13), (13, 17)
        ]

        points = []

        for landmark in hand:

            x = int(landmark.x * w)
            y = int(landmark.y * h)

            points.append((x, y))

            cv2.circle(
                frame,
                (x, y),
                4,
                (0, 255, 0),
                -1
            )

        for start, end in connections:

            cv2.line(
                frame,
                points[start],
                points[end],
                (255, 0, 0),
                2
            )


    # ========================================================
    # DISPLAY RESULT
    # ========================================================

    cv2.rectangle(
        frame,
        (10, 10),
        (420, 110),
        (0, 0, 0),
        -1
    )

    cv2.putText(
        frame,
        prediction_text,
        (25, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 255, 0),
        2
    )

    cv2.putText(
        frame,
        confidence_text,
        (25, 85),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    cv2.imshow(
        "ASL Sign Language Recognition",
        frame
    )

    # Press Q to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


# ============================================================
# CLEANUP
# ============================================================

cap.release()
cv2.destroyAllWindows()
detector.close()

print("Webcam closed.")