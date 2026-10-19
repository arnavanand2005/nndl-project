import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models


# ============================================================
# CONFIGURATION
# ============================================================

DATASET_DIR = "dataset"

IMG_HEIGHT = 128
IMG_WIDTH = 128

BATCH_SIZE = 32
SEED = 42
EPOCHS = 30

MODEL_PATH = "sign_language_model.keras"


# ============================================================
# LOAD DATASET
# ============================================================

print("\nLoading dataset...\n")

train_ds = tf.keras.utils.image_dataset_from_directory(
    DATASET_DIR,
    validation_split=0.2,
    subset="training",
    seed=SEED,
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    label_mode="int",
    shuffle=True
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    DATASET_DIR,
    validation_split=0.2,
    subset="validation",
    seed=SEED,
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    label_mode="int",
    shuffle=False
)


# ============================================================
# CLASS INFORMATION
# ============================================================

class_names = train_ds.class_names
num_classes = len(class_names)

print("\nClasses:")

for i, class_name in enumerate(class_names):
    print(f"{i}: {class_name}")

print("\nNumber of classes:", num_classes)


# ============================================================
# CHECK DATA
# ============================================================

print("\nChecking dataset...")

for images, labels in train_ds.take(1):

    print("Image batch shape:", images.shape)
    print("Label batch shape:", labels.shape)

    print(
        "Pixel range:",
        tf.reduce_min(images).numpy(),
        "to",
        tf.reduce_max(images).numpy()
    )


# ============================================================
# PERFORMANCE OPTIMIZATION
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

# We intentionally DO NOT use horizontal flipping.
# Some ASL hand signs can change meaning/orientation when flipped.

data_augmentation = tf.keras.Sequential([

    layers.RandomRotation(
        0.08
    ),

    layers.RandomZoom(
        0.10
    ),

    layers.RandomTranslation(
        height_factor=0.05,
        width_factor=0.05
    )

], name="data_augmentation")


# ============================================================
# CNN MODEL
# ============================================================

model = models.Sequential([

    # --------------------------------------------------------
    # INPUT
    # --------------------------------------------------------

    layers.Input(
        shape=(IMG_HEIGHT, IMG_WIDTH, 3)
    ),

    # --------------------------------------------------------
    # NORMALIZATION
    # --------------------------------------------------------

    layers.Rescaling(
        1.0 / 255
    ),

    # --------------------------------------------------------
    # DATA AUGMENTATION
    # --------------------------------------------------------

    data_augmentation,

    # --------------------------------------------------------
    # CNN BLOCK 1
    # --------------------------------------------------------

    layers.Conv2D(
        32,
        (3, 3),
        padding="same",
        activation="relu"
    ),

    layers.BatchNormalization(),

    layers.MaxPooling2D(
        (2, 2)
    ),

    # --------------------------------------------------------
    # CNN BLOCK 2
    # --------------------------------------------------------

    layers.Conv2D(
        64,
        (3, 3),
        padding="same",
        activation="relu"
    ),

    layers.BatchNormalization(),

    layers.MaxPooling2D(
        (2, 2)
    ),

    # --------------------------------------------------------
    # CNN BLOCK 3
    # --------------------------------------------------------

    layers.Conv2D(
        128,
        (3, 3),
        padding="same",
        activation="relu"
    ),

    layers.BatchNormalization(),

    layers.MaxPooling2D(
        (2, 2)
    ),

    # --------------------------------------------------------
    # CNN BLOCK 4
    # --------------------------------------------------------

    layers.Conv2D(
        256,
        (3, 3),
        padding="same",
        activation="relu"
    ),

    layers.BatchNormalization(),

    layers.MaxPooling2D(
        (2, 2)
    ),

    # --------------------------------------------------------
    # FEATURE AGGREGATION
    # --------------------------------------------------------

    layers.GlobalAveragePooling2D(),

    # --------------------------------------------------------
    # FULLY CONNECTED LAYER
    # --------------------------------------------------------

    layers.Dense(
        128,
        activation="relu"
    ),

    layers.BatchNormalization(),

    layers.Dropout(
        0.5
    ),

    # --------------------------------------------------------
    # OUTPUT
    # --------------------------------------------------------

    layers.Dense(
        num_classes,
        activation="softmax"
    )

])


# ============================================================
# MODEL SUMMARY
# ============================================================

print("\n")
print("=" * 60)
print("CNN MODEL")
print("=" * 60)

model.summary()


# ============================================================
# COMPILE MODEL
# ============================================================

model.compile(

    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.0001
    ),

    loss="sparse_categorical_crossentropy",

    metrics=[
        "accuracy"
    ]
)


# ============================================================
# CALLBACKS
# ============================================================

callbacks = [

    # Stop if validation accuracy stops improving
    tf.keras.callbacks.EarlyStopping(

        monitor="val_accuracy",

        patience=7,

        mode="max",

        restore_best_weights=True,

        verbose=1
    ),

    # Save the best model
    tf.keras.callbacks.ModelCheckpoint(

        MODEL_PATH,

        monitor="val_accuracy",

        mode="max",

        save_best_only=True,

        verbose=1
    ),

    # Reduce learning rate when validation loss stops improving
    tf.keras.callbacks.ReduceLROnPlateau(

        monitor="val_loss",

        factor=0.5,

        patience=3,

        min_lr=0.000001,

        verbose=1
    )

]


# ============================================================
# TRAIN MODEL
# ============================================================

print("\n")
print("=" * 60)
print("STARTING CNN TRAINING")
print("=" * 60)
print("\n")

history = model.fit(

    train_ds,

    validation_data=val_ds,

    epochs=EPOCHS,

    callbacks=callbacks
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

print("\nValidation Loss:", val_loss)

print(
    "Validation Accuracy:",
    f"{val_accuracy * 100:.2f}%"
)


# ============================================================
# SAVE CLASS NAMES
# ============================================================

with open("class_names.txt", "w") as file:

    for class_name in class_names:

        file.write(
            class_name + "\n"
        )

print("\nClass names saved to: class_names.txt")


# ============================================================
# PLOT ACCURACY
# ============================================================

plt.figure(
    figsize=(10, 6)
)

plt.plot(
    history.history["accuracy"],
    label="Training Accuracy"
)

plt.plot(
    history.history["val_accuracy"],
    label="Validation Accuracy"
)

plt.title(
    "CNN Training and Validation Accuracy"
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
    "accuracy.png",
    dpi=300
)

plt.show()


# ============================================================
# PLOT LOSS
# ============================================================

plt.figure(
    figsize=(10, 6)
)

plt.plot(
    history.history["loss"],
    label="Training Loss"
)

plt.plot(
    history.history["val_loss"],
    label="Validation Loss"
)

plt.title(
    "CNN Training and Validation Loss"
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
    "loss.png",
    dpi=300
)

plt.show()


# ============================================================
# COMPLETE
# ============================================================

print("\n")
print("=" * 60)
print("TRAINING COMPLETE")
print("=" * 60)

print("\nModel:")
print(MODEL_PATH)

print("\nClass names:")
print("class_names.txt")

print("\nGraphs:")
print("accuracy.png")
print("loss.png")