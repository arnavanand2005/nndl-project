import cv2
import numpy as np
import tensorflow as tf
from collections import deque

# ============================================================
# CONFIGURATION
# ============================================================

MODEL_PATH = "sign_language_mobilenetv2.keras"

IMG_HEIGHT = 160
IMG_WIDTH = 160

# Webcam
CAMERA_INDEX = 0
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720

# Prediction stability
SMOOTHING_FRAMES = 12
MIN_CONFIDENCE = 0.45
MIN_MARGIN = 0.08

# ============================================================
# CLASS NAMES
# IMPORTANT: Must match image_dataset_from_directory order
# ============================================================

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
# ============================================================

def preprocess_roi(roi):
    """
    Keep webcam preprocessing consistent with MobileNetV2 training.
    """

    # Convert BGR -> RGB
    rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)

    # Resize exactly as training
    rgb = cv2.resize(
        rgb,
        (IMG_WIDTH, IMG_HEIGHT),
        interpolation=cv2.INTER_AREA
    )

    # Convert to float32
    image = rgb.astype(np.float32)

    # MobileNetV2 preprocessing:
    # [0, 255] -> [-1, 1]
    image = tf.keras.applications.mobilenet_v2.preprocess_input(image)

    # Add batch dimension
    image = np.expand_dims(image, axis=0)

    return image


# ============================================================
# OPEN CAMERA
# ============================================================

print("\nStarting webcam...")
print("Press Q to quit.")

cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_AVFOUNDATION)

if not cap.isOpened():
    # Fallback
    cap = cv2.VideoCapture(CAMERA_INDEX)

if not cap.isOpened():
    print("\nERROR: Could not open webcam.")
    print("Check System Settings > Privacy & Security > Camera.")
    raise SystemExit

cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

# ============================================================
# TEMPORAL SMOOTHING
# ============================================================

probability_history = deque(maxlen=SMOOTHING_FRAMES)

last_label = "No confident prediction"
last_confidence = 0.0

# ============================================================
# MAIN LOOP
# ============================================================

while True:

    ret, frame = cap.read()

    if not ret:
        print("Failed to read frame.")
        break

    # Mirror webcam
    frame = cv2.flip(frame, 1)

    h, w = frame.shape[:2]

    # ========================================================
    # LARGE ROI
    # ========================================================
    #
    # The old box was too small. This box occupies most of
    # the useful camera area so the hand can be larger and
    # easier for the model to recognize.
    #

    box_width = int(w * 0.68)
    box_height = int(h * 0.78)

    x1 = (w - box_width) // 2
    y1 = int(h * 0.14)

    x2 = x1 + box_width
    y2 = y1 + box_height

    roi = frame[y1:y2, x1:x2]

    if roi.size == 0:
        continue

    # ========================================================
    # PREDICTION
    # ========================================================

    input_image = preprocess_roi(roi)

    predictions = model.predict(
        input_image,
        verbose=0
    )[0]

    # Safety normalization
    predictions = np.asarray(predictions, dtype=np.float32)

    if predictions.sum() > 0:
        predictions = predictions / predictions.sum()

    probability_history.append(predictions)

    # Average predictions over recent frames.
    # This removes rapid jumping between classes.
    smoothed = np.mean(
        np.stack(probability_history),
        axis=0
    )

    # Get top two classes
    top_indices = np.argsort(smoothed)[::-1]

    top1 = int(top_indices[0])
    top2 = int(top_indices[1])

    confidence = float(smoothed[top1])
    second_confidence = float(smoothed[top2])

    margin = confidence - second_confidence

    # ========================================================
    # CONFIDENCE DECISION
    # ========================================================

    if (
        confidence >= MIN_CONFIDENCE
        and margin >= MIN_MARGIN
    ):
        last_label = CLASS_NAMES[top1]
        last_confidence = confidence

        status = "Stable prediction"
        status_color = (0, 220, 0)

    else:
        last_label = "No confident prediction"
        last_confidence = confidence

        status = "Predictions too close"
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
    # TOP PREDICTIONS
    # ========================================================

    top3_text = []

    for rank in range(3):

        idx = int(top_indices[rank])
        prob = float(smoothed[idx]) * 100

        clean_name = CLASS_NAMES[idx].replace(
            "-samples", ""
        )

        top3_text.append(
            f"{clean_name}: {prob:.1f}%"
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

    # Main prediction
    if last_label == "No confident prediction":
        display_label = last_label
    else:
        display_label = last_label.replace(
            "-samples", ""
        )

    cv2.putText(
        frame,
        f"Prediction: {display_label}",
        (25, 45),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.05,
        (255, 255, 255),
        2,
        cv2.LINE_AA
    )

    cv2.putText(
        frame,
        f"Confidence: {last_confidence * 100:.1f}%",
        (25, 82),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.72,
        status_color,
        2,
        cv2.LINE_AA
    )

    cv2.putText(
        frame,
        status,
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
        (390, h - 15),
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

    for i, text in enumerate(top3_text):

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
        "Place ONE hand inside the box | Q = quit",
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

    # Q to quit
    key = cv2.waitKey(1) & 0xFF

    if key == ord("q") or key == ord("Q"):
        break

# ============================================================
# CLEANUP
# ============================================================

cap.release()
cv2.destroyAllWindows()

print("\nWebcam closed.")
