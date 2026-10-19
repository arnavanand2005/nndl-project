import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2


# ============================================================
# CONFIGURATION
# ============================================================

TRAIN_DIR = "dataset_split/train"
VAL_DIR = "dataset_split/validation"

IMG_HEIGHT = 160
IMG_WIDTH = 160

BATCH_SIZE = 32
SEED = 42

INITIAL_EPOCHS = 10
FINE_TUNE_EPOCHS = 15

MODEL_PATH = "sign_language_mobilenetv2.keras"


# ============================================================
# LOAD TRAINING DATASET
# ============================================================

print("\nLoading training dataset...\n")

train_ds = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    label_mode="int",
    shuffle=True,
    seed=SEED
)


# ============================================================
# LOAD VALIDATION DATASET
# ============================================================

print("\nLoading validation dataset...\n")

val_ds = tf.keras.utils.image_dataset_from_directory(
    VAL_DIR,
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    label_mode="int",
    shuffle=False
)


# ============================================================
# CLASSES
# ============================================================

class_names = train_ds.class_names
num_classes = len(class_names)

print("\nClasses:")

for i, class_name in enumerate(class_names):
    print(f"{i}: {class_name}")

print("\nNumber of classes:", num_classes)


# ============================================================
# PERFORMANCE
# ============================================================

AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.prefetch(
    buffer_size=AUTOTUNE
)

val_ds = val_ds.prefetch(
    buffer_size=AUTOTUNE
)


# ============================================================
# DATA AUGMENTATION
# ============================================================

data_augmentation = tf.keras.Sequential([
    layers.RandomRotation(0.08),
    layers.RandomZoom(0.10),
    layers.RandomTranslation(
        height_factor=0.05,
        width_factor=0.05
    )
], name="data_augmentation")


# ============================================================
# MOBILE NET V2
# ============================================================

print("\nLoading MobileNetV2...")

base_model = MobileNetV2(
    input_shape=(
        IMG_HEIGHT,
        IMG_WIDTH,
        3
    ),
    include_top=False,
    weights="imagenet"
)


# ============================================================
# FREEZE BASE MODEL
# ============================================================

base_model.trainable = False


# ============================================================
# BUILD MODEL
# ============================================================

inputs = layers.Input(
    shape=(
        IMG_HEIGHT,
        IMG_WIDTH,
        3
    )
)

x = data_augmentation(inputs)

# MobileNetV2 expects pixels approximately in [-1, 1]
x = layers.Rescaling(
    scale=1.0 / 127.5,
    offset=-1
)(x)

x = base_model(
    x,
    training=False
)

x = layers.GlobalAveragePooling2D()(x)

x = layers.Dense(
    128,
    activation="relu"
)(x)

x = layers.Dropout(
    0.4
)(x)

outputs = layers.Dense(
    num_classes,
    activation="softmax"
)(x)


model = models.Model(
    inputs,
    outputs
)


# ============================================================
# MODEL SUMMARY
# ============================================================

print("\n")
print("=" * 60)
print("MOBILENETV2 TRANSFER LEARNING MODEL")
print("=" * 60)

model.summary()


# ============================================================
# COMPILE - STAGE 1
# ============================================================

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.001
    ),
    loss="sparse_categorical_crossentropy",
    metrics=[
        "accuracy"
    ]
)


# ============================================================
# CALLBACKS - STAGE 1
# ============================================================

callbacks_stage1 = [

    tf.keras.callbacks.ModelCheckpoint(
        MODEL_PATH,
        monitor="val_accuracy",
        mode="max",
        save_best_only=True,
        verbose=1
    ),

    tf.keras.callbacks.EarlyStopping(
        monitor="val_accuracy",
        mode="max",
        patience=4,
        restore_best_weights=True,
        verbose=1
    ),

    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=2,
        min_lr=0.00001,
        verbose=1
    )
]


# ============================================================
# STAGE 1: FEATURE EXTRACTION
# ============================================================

print("\n")
print("=" * 60)
print("STAGE 1: FEATURE EXTRACTION")
print("=" * 60)
print("\n")

history1 = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=INITIAL_EPOCHS,
    callbacks=callbacks_stage1
)


# ============================================================
# STAGE 2: FINE-TUNING
# ============================================================

print("\n")
print("=" * 60)
print("STAGE 2: FINE-TUNING")
print("=" * 60)
print("\n")


# Unfreeze MobileNetV2
base_model.trainable = True


# Freeze the earlier layers.
# Only the last 30 layers will be fine-tuned.

for layer in base_model.layers[:-30]:
    layer.trainable = False


# ============================================================
# RECOMPILE WITH SMALL LEARNING RATE
# ============================================================

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.00001
    ),
    loss="sparse_categorical_crossentropy",
    metrics=[
        "accuracy"
    ]
)


# ============================================================
# CALLBACKS - STAGE 2
# ============================================================

callbacks_stage2 = [

    tf.keras.callbacks.ModelCheckpoint(
        MODEL_PATH,
        monitor="val_accuracy",
        mode="max",
        save_best_only=True,
        verbose=1
    ),

    tf.keras.callbacks.EarlyStopping(
        monitor="val_accuracy",
        mode="max",
        patience=5,
        restore_best_weights=True,
        verbose=1
    ),

    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=2,
        min_lr=0.000001,
        verbose=1
    )
]


# ============================================================
# TRAIN FINE-TUNED MODEL
# ============================================================

history2 = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=FINE_TUNE_EPOCHS,
    callbacks=callbacks_stage2
)


# ============================================================
# LOAD BEST MODEL
# ============================================================

print("\nLoading best model...")

model = tf.keras.models.load_model(
    MODEL_PATH
)


# ============================================================
# FINAL EVALUATION
# ============================================================

print("\n")
print("=" * 60)
print("FINAL MODEL EVALUATION")
print("=" * 60)

val_loss, val_accuracy = model.evaluate(
    val_ds,
    verbose=1
)

print("\nValidation Loss:")
print(val_loss)

print("\nValidation Accuracy:")
print(
    f"{val_accuracy * 100:.2f}%"
)


# ============================================================
# SAVE CLASS NAMES
# ============================================================

with open(
    "mobilenet_class_names.txt",
    "w"
) as file:

    for class_name in class_names:
        file.write(
            class_name + "\n"
        )


# ============================================================
# COMBINE TRAINING HISTORY
# ============================================================

accuracy = (
    history1.history["accuracy"]
    +
    history2.history["accuracy"]
)

val_accuracy_history = (
    history1.history["val_accuracy"]
    +
    history2.history["val_accuracy"]
)

loss = (
    history1.history["loss"]
    +
    history2.history["loss"]
)

val_loss_history = (
    history1.history["val_loss"]
    +
    history2.history["val_loss"]
)


# ============================================================
# ACCURACY GRAPH
# ============================================================

plt.figure(
    figsize=(10, 6)
)

plt.plot(
    accuracy,
    label="Training Accuracy"
)

plt.plot(
    val_accuracy_history,
    label="Validation Accuracy"
)

plt.axvline(
    x=INITIAL_EPOCHS - 1,
    linestyle="--",
    label="Fine-Tuning Starts"
)

plt.title(
    "MobileNetV2 Training and Validation Accuracy"
)

plt.xlabel(
    "Epoch"
)

plt.ylabel(
    "Accuracy"
)

plt.legend()

plt.grid(
    True,
    alpha=0.3
)

plt.tight_layout()

plt.savefig(
    "mobilenet_accuracy.png",
    dpi=300
)

plt.show()


# ============================================================
# LOSS GRAPH
# ============================================================

plt.figure(
    figsize=(10, 6)
)

plt.plot(
    loss,
    label="Training Loss"
)

plt.plot(
    val_loss_history,
    label="Validation Loss"
)

plt.axvline(
    x=INITIAL_EPOCHS - 1,
    linestyle="--",
    label="Fine-Tuning Starts"
)

plt.title(
    "MobileNetV2 Training and Validation Loss"
)

plt.xlabel(
    "Epoch"
)

plt.ylabel(
    "Loss"
)

plt.legend()

plt.grid(
    True,
    alpha=0.3
)

plt.tight_layout()

plt.savefig(
    "mobilenet_loss.png",
    dpi=300
)

plt.show()


# ============================================================
# COMPLETE
# ============================================================

print("\n")
print("=" * 60)
print("TRANSFER LEARNING COMPLETE")
print("=" * 60)

print("\nModel saved:")
print(MODEL_PATH)

print("\nClass names saved:")
print("mobilenet_class_names.txt")

print("\nGraphs saved:")
print("mobilenet_accuracy.png")
print("mobilenet_loss.png")