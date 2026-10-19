import cv2
import numpy as np
import tensorflow as tf
from collections import deque

# ============================================================
# ASL WEBCAM TEST - MobileNetV2
# ============================================================

MODEL_PATH = "sign_language_mobilenetv2.keras"

IMG_SIZE = 160

CAMERA_INDEX = 0
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720

# Smoothing
SMOOTHING_FRAMES = 8

# Confidence rules
MIN_CONFIDENCE = 0.50
MIN_MARGIN = 0.12

CLASS_NAMES = [
    "A-samples", "B-samples", "C-samples", "D-samples",
    "E-samples", "F-samples", "G-samples", "H-samples",
    "I-samples", "K-samples", "L-samples", "M-samples",
    "N-samples", "O-samples", "P-samples", "Q-samples",
    "R-samples", "S-samples", "T-samples", "U-samples",
    "V-samples", "W-samples", "X-samples", "Y-samples"
]

# ============================================================
# LOAD MODEL
# ============================================================

print("\nLoading MobileNetV2 model...\n")

model = tf.keras.models.load_model(MODEL_PATH)

print("Model loaded successfully.")
print(f"Loaded {len(CLASS_NAMES)} classes.")

# ============================================================
# PREPROCESSING
# IMPORTANT:
# - BGR -> RGB
# - SQUARE crop
# - 160x160
# - MobileNetV2 preprocessing
# ============================================================

def preprocess_roi(roi):

    rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)

    rgb = cv2.resize(
        rgb,
        (IMG_SIZE, IMG_SIZE),
        interpolation=cv2.INTER_AREA
    )

    image = rgb.astype(np.float32)

    image = tf.keras.applications.mobilenet_v2.preprocess_input(
        image
    )

    return np.expand_dims(image, axis=0)


# ============================================================
# CAMERA
# ============================================================

print("\nStarting webcam...")
print("Press Q to quit.")

cap = cv2.VideoCapture(
    CAMERA_INDEX,
    cv2.CAP_AVFOUNDATION
)

if not cap.isOpened():
    cap = cv2.VideoCapture(CAMERA_INDEX)

if not cap.isOpened():
    print("\nERROR: Could not open webcam.")
    raise SystemExit

cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

# ============================================================
# PREDICTION HISTORY
# ============================================================

probability_history = deque(
    maxlen=SMOOTHING_FRAMES
)

# ============================================================
# MAIN LOOP
# ============================================================

while True:

    ret, frame = cap.read()

    if not ret:
        print("Failed to read frame.")
        break

    frame = cv2.flip(frame, 1)

    h, w = frame.shape[:2]

    # ========================================================
    # SQUARE ROI
    #
    # This is IMPORTANT.
    #
    # The previous version used a rectangular ROI and then
    # resized it to 160x160, which can distort the hand.
    # MobileNetV2 now receives a square crop.
    # ========================================================

    box_size = int(min(w, h) * 0.82)

    x1 = (w - box_size) // 2
    y1 = (h - box_size) // 2

    x2 = x1 + box_size
    y2 = y1 + box_size

    roi = frame[y1:y2, x1:x2]

    if roi.size == 0:
        continue

    # ========================================================
    # PREDICT
    # ========================================================

    input_image = preprocess_roi(roi)

    predictions = model.predict(
        input_image,
        verbose=0
    )[0]

    predictions = np.asarray(
        predictions,
        dtype=np.float32
    )

    # Softmax safety
    if not np.isclose(predictions.sum(), 1.0):
        predictions = tf.nn.softmax(predictions).numpy()

    probability_history.append(predictions)

    smoothed = np.mean(
        np.stack(probability_history),
        axis=0
    )

    top_indices = np.argsort(smoothed)[::-1]

    top1 = int(top_indices[0])
    top2 = int(top_indices[1])

    confidence = float(smoothed[top1])
    second_confidence = float(smoothed[top2])

    margin = confidence - second_confidence

    # ========================================================
    # DECISION
    # ========================================================

    if (
        confidence >= MIN_CONFIDENCE
        and margin >= MIN_MARGIN
    ):
        prediction_text = CLASS_NAMES[top1].replace(
            "-samples", ""
        )

        status_text = "Stable prediction"
        status_color = (0, 220, 0)

    else:
        prediction_text = "No confident prediction"
        status_text = "Move hand / improve lighting"
        status_color = (0, 165, 255)

    # ========================================================
    # DRAW ROI
    # ========================================================

    cv2.rectangle(
        frame,
        (x1, y1),
        (x2, y2),
        (255, 255, 255),
        3
    )

    # ========================================================
    # TOP 3
    # ========================================================

    top3 = []

    for rank in range(3):

        idx = int(top_indices[rank])

        name = CLASS_NAMES[idx].replace(
            "-samples", ""
        )

        percentage = float(
            smoothed[idx] * 100
        )

        top3.append(
            f"{name}: {percentage:.1f}%"
        )

    # ========================================================
    # HEADER
    # ========================================================

    overlay = frame.copy()

    cv2.rectangle(
        overlay,
        (0, 0),
        (w, 120),
        (0, 0, 0),
        -1
    )

    frame = cv2.addWeighted(
        overlay,
        0.78,
        frame,
        0.22,
        0
    )

    cv2.putText(
        frame,
        f"Prediction: {prediction_text}",
        (25, 45),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (255, 255, 255),
        2,
        cv2.LINE_AA
    )

    cv2.putText(
        frame,
        f"Confidence: {confidence * 100:.1f}%",
        (25, 82),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.72,
        status_color,
        2,
        cv2.LINE_AA
    )

    cv2.putText(
        frame,
        status_text,
        (25, 108),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        status_color,
        2,
        cv2.LINE_AA
    )

    # ========================================================
    # TOP 3 PANEL
    # ========================================================

    panel_x = 25
    panel_y = h - 115

    cv2.rectangle(
        frame,
        (panel_x - 10, panel_y - 28),
        (330, h - 15),
        (0, 0, 0),
        -1
    )

    cv2.putText(
        frame,
        "Top predictions:",
        (panel_x, panel_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
        cv2.LINE_AA
    )

    for i, text in enumerate(top3):

        cv2.putText(
            frame,
            text,
            (panel_x, panel_y + 28 + i * 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            (230, 230, 230),
            1,
            cv2.LINE_AA
        )

    # ========================================================
    # INSTRUCTIONS
    # ========================================================

    cv2.putText(
        frame,
        "Place one hand inside the SQUARE | Q = quit",
        (25, h - 18),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        (255, 255, 255),
        2,
        cv2.LINE_AA
    )

    cv2.imshow(
        "ASL Sign Language Recognition",
        frame
    )

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q") or key == ord("Q"):
        break

cap.release()
cv2.destroyAllWindows()

print("\nWebcam closed.")
