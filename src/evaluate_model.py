import os
import json
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

TEST_PATH = os.path.join(
    BASE_DIR,
    "outputs",
    "test.csv"
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "sign_language_landmark.keras"
)

CLASS_NAMES_PATH = os.path.join(
    BASE_DIR,
    "models",
    "class_names.json"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "outputs"
)

CONFUSION_MATRIX_PATH = os.path.join(
    OUTPUT_DIR,
    "test_confusion_matrix.png"
)

REPORT_PATH = os.path.join(
    OUTPUT_DIR,
    "test_classification_report.txt"
)


# ============================================================
# HEADER
# ============================================================

print("=" * 60)
print("SIGN LANGUAGE LANDMARK MODEL EVALUATION")
print("=" * 60)


# ============================================================
# LOAD TEST DATA
# ============================================================

print("\nLoading test dataset...")

test_df = pd.read_csv(TEST_PATH)

print(f"Test samples: {len(test_df)}")


# ============================================================
# SEPARATE FEATURES AND LABELS
# ============================================================

# ============================================================
# SEPARATE FEATURES AND LABELS
# ============================================================

# The landmark dataset contains metadata columns such as:
# class, filename, and the 63 landmark coordinates.
#
# We only want the numeric landmark columns.

EXCLUDED_COLUMNS = {
    "label",
    "class",
    "filename",
    "image",
    "image_path",
    "path"
}

feature_columns = [
    column
    for column in test_df.columns
    if column not in EXCLUDED_COLUMNS
]

feature_columns = [
    column
    for column in feature_columns
    if pd.api.types.is_numeric_dtype(test_df[column])
]

X_test = test_df[feature_columns].values.astype(
    np.float32
)

y_test_labels = test_df["label"].values

print(f"Feature columns selected: {len(feature_columns)}")


print(f"Features: {X_test.shape[1]}")


# ============================================================
# LOAD CLASS NAMES
# ============================================================

print("\nLoading class names...")

with open(CLASS_NAMES_PATH, "r") as file:
    class_names = json.load(file)

print(f"Number of classes: {len(class_names)}")

print("\nClasses:")

for i, class_name in enumerate(class_names):
    print(f"{i}: {class_name}")


# ============================================================
# CONVERT LABELS TO NUMBERS
# ============================================================

class_to_index = {
    class_name: i
    for i, class_name in enumerate(class_names)
}

y_test = np.array([
    class_to_index[label]
    for label in y_test_labels
])


# ============================================================
# LOAD MODEL
# ============================================================

print("\nLoading trained model...")

model = tf.keras.models.load_model(
    MODEL_PATH
)

print("Model loaded successfully.")


# ============================================================
# GENERATE PREDICTIONS
# ============================================================

print("\nGenerating predictions...\n")

probabilities = model.predict(
    X_test,
    verbose=1
)

predicted_labels = np.argmax(
    probabilities,
    axis=1
)


# ============================================================
# OVERALL ACCURACY
# ============================================================

accuracy = accuracy_score(
    y_test,
    predicted_labels
)

print("\n")
print("=" * 60)
print("TEST SET ACCURACY")
print("=" * 60)

print(
    f"\nTest Accuracy: {accuracy * 100:.2f}%"
)


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

print("\n")
print("=" * 60)
print("CLASSIFICATION REPORT")
print("=" * 60)

report = classification_report(
    y_test,
    predicted_labels,
    labels=np.arange(len(class_names)),
    target_names=class_names,
    digits=4,
    zero_division=0
)

print("\n")
print(report)


# ============================================================
# SAVE REPORT
# ============================================================

with open(
    REPORT_PATH,
    "w"
) as file:

    file.write(
        "SIGN LANGUAGE RECOGNITION\n"
    )

    file.write(
        "Landmark Model - Test Set Evaluation\n"
    )

    file.write(
        "=" * 60
    )

    file.write("\n\n")

    file.write(
        f"Test Accuracy: {accuracy * 100:.2f}%\n\n"
    )

    file.write(
        report
    )

print(
    f"Classification report saved to:\n{REPORT_PATH}"
)


# ============================================================
# CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_test,
    predicted_labels,
    labels=np.arange(len(class_names))
)


# ============================================================
# PRINT CONFUSION MATRIX
# ============================================================

print("\n")
print("=" * 60)
print("CONFUSION MATRIX")
print("=" * 60)

print("\n")
print(cm)


# ============================================================
# PLOT CONFUSION MATRIX
# ============================================================

plt.figure(
    figsize=(14, 12)
)

plt.imshow(
    cm,
    interpolation="nearest"
)

plt.title(
    "ASL Landmark Model - Test Confusion Matrix"
)

plt.colorbar()

tick_marks = np.arange(
    len(class_names)
)

plt.xticks(
    tick_marks,
    class_names,
    rotation=90
)

plt.yticks(
    tick_marks,
    class_names
)

plt.xlabel(
    "Predicted Label"
)

plt.ylabel(
    "True Label"
)


# Add values inside cells

threshold = cm.max() / 2.0

for i in range(cm.shape[0]):

    for j in range(cm.shape[1]):

        plt.text(
            j,
            i,
            str(cm[i, j]),
            horizontalalignment="center",
            verticalalignment="center",
            color="white"
            if cm[i, j] > threshold
            else "black"
        )


plt.tight_layout()

plt.savefig(
    CONFUSION_MATRIX_PATH,
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ============================================================
# FIND INCORRECT PREDICTIONS
# ============================================================

incorrect = np.where(
    y_test != predicted_labels
)[0]


print("\n")
print("=" * 60)
print("INCORRECT PREDICTIONS")
print("=" * 60)

print(
    f"\nIncorrect predictions: {len(incorrect)}"
)

print(
    f"Correct predictions: "
    f"{len(y_test) - len(incorrect)}"
)


if len(incorrect) > 0:

    print("\nMistakes:")

    for index in incorrect:

        actual = class_names[
            y_test[index]
        ]

        predicted = class_names[
            predicted_labels[index]
        ]

        confidence = (
            probabilities[index][
                predicted_labels[index]
            ] * 100
        )

        print(
            f"Actual: {actual:2s} "
            f"→ Predicted: {predicted:2s} "
            f"({confidence:.2f}%)"
        )


# ============================================================
# COMPLETE
# ============================================================

print("\n")
print("=" * 60)
print("EVALUATION COMPLETE")
print("=" * 60)

print("\nGenerated files:")

print(
    "1. outputs/test_classification_report.txt"
)

print(
    "2. outputs/test_confusion_matrix.png"
)