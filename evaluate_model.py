import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score
)


# ============================================================
# CONFIGURATION
# ============================================================

VAL_DIR = "dataset_split/validation"

MODEL_PATH = "sign_language_mobilenetv2.keras"

IMG_HEIGHT = 160
IMG_WIDTH = 160

BATCH_SIZE = 32


# ============================================================
# LOAD MODEL
# ============================================================

print("\nLoading MobileNetV2 model...\n")

model = tf.keras.models.load_model(
    MODEL_PATH
)

print("Model loaded successfully.")


# ============================================================
# LOAD VALIDATION DATASET
# ============================================================

print("\nLoading validation dataset...\n")

val_ds = tf.keras.utils.image_dataset_from_directory(
    VAL_DIR,
    image_size=(
        IMG_HEIGHT,
        IMG_WIDTH
    ),
    batch_size=BATCH_SIZE,
    label_mode="int",
    shuffle=False
)


# ============================================================
# CLASS NAMES
# ============================================================

class_names = val_ds.class_names

num_classes = len(class_names)

print("Number of classes:", num_classes)

print("\nClasses:")

for i, class_name in enumerate(class_names):
    print(f"{i}: {class_name}")


# ============================================================
# GENERATE PREDICTIONS
# ============================================================

print("\nGenerating predictions...\n")

true_labels = []
predicted_labels = []

for images, labels in val_ds:

    predictions = model.predict(
        images,
        verbose=0
    )

    predicted = np.argmax(
        predictions,
        axis=1
    )

    true_labels.extend(
        labels.numpy()
    )

    predicted_labels.extend(
        predicted
    )


true_labels = np.array(true_labels)
predicted_labels = np.array(predicted_labels)


# ============================================================
# OVERALL ACCURACY
# ============================================================

accuracy = accuracy_score(
    true_labels,
    predicted_labels
)

print("\n")
print("=" * 60)
print("OVERALL ACCURACY")
print("=" * 60)

print(
    f"\nAccuracy: {accuracy * 100:.2f}%"
)


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

print("\n")
print("=" * 60)
print("CLASSIFICATION REPORT")
print("=" * 60)

report = classification_report(
    true_labels,
    predicted_labels,
    labels=np.arange(num_classes),
    target_names=class_names,
    digits=4,
    zero_division=0
)

print("\n")
print(report)


# ============================================================
# SAVE CLASSIFICATION REPORT
# ============================================================

with open(
    "classification_report.txt",
    "w"
) as file:

    file.write(
        "SIGN LANGUAGE RECOGNITION\n"
    )

    file.write(
        "MobileNetV2 Classification Report\n"
    )

    file.write(
        "=" * 60
    )

    file.write("\n\n")

    file.write(
        f"Overall Accuracy: {accuracy * 100:.2f}%\n\n"
    )

    file.write(
        report
    )

print(
    "\nClassification report saved as:"
)

print(
    "classification_report.txt"
)


# ============================================================
# CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    true_labels,
    predicted_labels,
    labels=np.arange(num_classes)
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
    "ASL Sign Recognition - Confusion Matrix"
)

plt.colorbar()

tick_marks = np.arange(
    num_classes
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


# ============================================================
# ADD VALUES TO CELLS
# ============================================================

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
    "confusion_matrix.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ============================================================
# PER-CLASS ACCURACY
# ============================================================

print("\n")
print("=" * 60)
print("PER-CLASS ACCURACY")
print("=" * 60)

print()

for i, class_name in enumerate(class_names):

    total = np.sum(
        true_labels == i
    )

    correct = np.sum(
        (true_labels == i)
        &
        (predicted_labels == i)
    )

    if total > 0:

        class_accuracy = (
            correct / total
        ) * 100

    else:

        class_accuracy = 0

    print(
        f"{class_name:12s}: "
        f"{class_accuracy:.2f}%"
    )


# ============================================================
# COMPLETE
# ============================================================

print("\n")
print("=" * 60)
print("EVALUATION COMPLETE")
print("=" * 60)

print("\nGenerated files:")

print("1. classification_report.txt")
print("2. confusion_matrix.png")
