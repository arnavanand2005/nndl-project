import json
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.preprocessing import LabelEncoder


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

OUTPUT_DIR = PROJECT_ROOT / "outputs"
MODEL_DIR = PROJECT_ROOT / "models"

MODEL_DIR.mkdir(exist_ok=True)


TRAIN_FILE = OUTPUT_DIR / "train.csv"
VALIDATION_FILE = OUTPUT_DIR / "validation.csv"


MODEL_FILE = MODEL_DIR / "sign_language_landmark.keras"
CLASS_NAMES_FILE = MODEL_DIR / "class_names.json"


# ============================================================
# CONFIGURATION
# ============================================================

RANDOM_SEED = 42

BATCH_SIZE = 32
EPOCHS = 50

LEARNING_RATE = 0.001


# ============================================================
# REPRODUCIBILITY
# ============================================================

np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)


# ============================================================
# LOAD DATA
# ============================================================

def load_dataset(file_path):

    df = pd.read_csv(file_path)

    feature_columns = [
        column
        for column in df.columns
        if column.startswith(("x", "y", "z"))
    ]

    X = df[feature_columns].values.astype(
        np.float32
    )

    y = df["label"].values

    return X, y


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("SIGN LANGUAGE LANDMARK MODEL TRAINING")
    print("=" * 60)

    # --------------------------------------------------------
    # Load training data
    # --------------------------------------------------------

    print("\nLoading training dataset...")

    X_train, y_train_text = load_dataset(
        TRAIN_FILE
    )

    print(
        f"Training samples: {len(X_train)}"
    )

    # --------------------------------------------------------
    # Load validation data
    # --------------------------------------------------------

    print("\nLoading validation dataset...")

    X_validation, y_validation_text = load_dataset(
        VALIDATION_FILE
    )

    print(
        f"Validation samples: {len(X_validation)}"
    )

    # --------------------------------------------------------
    # Encode labels
    # --------------------------------------------------------

    label_encoder = LabelEncoder()

    label_encoder.fit(
        np.concatenate([
            y_train_text,
            y_validation_text
        ])
    )

    y_train = label_encoder.transform(
        y_train_text
    )

    y_validation = label_encoder.transform(
        y_validation_text
    )

    class_names = label_encoder.classes_.tolist()

    num_classes = len(class_names)

    print(
        f"\nNumber of classes: {num_classes}"
    )

    print("\nClasses:")

    for index, class_name in enumerate(
        class_names
    ):

        print(
            f"{index}: {class_name}"
        )

    # --------------------------------------------------------
    # Build model
    # --------------------------------------------------------

    print("\nBuilding model...")

    model = tf.keras.Sequential([

        tf.keras.layers.Input(
            shape=(63,)
        ),

        tf.keras.layers.Dense(
            128,
            activation="relu"
        ),

        tf.keras.layers.BatchNormalization(),

        tf.keras.layers.Dropout(
            0.25
        ),

        tf.keras.layers.Dense(
            64,
            activation="relu"
        ),

        tf.keras.layers.BatchNormalization(),

        tf.keras.layers.Dropout(
            0.20
        ),

        tf.keras.layers.Dense(
            32,
            activation="relu"
        ),

        tf.keras.layers.Dense(
            num_classes,
            activation="softmax"
        )
    ])

    # --------------------------------------------------------
    # Compile
    # --------------------------------------------------------

    model.compile(

        optimizer=tf.keras.optimizers.Adam(
            learning_rate=LEARNING_RATE
        ),

        loss="sparse_categorical_crossentropy",

        metrics=[
            "accuracy"
        ]
    )

    print("\nModel architecture:")

    model.summary()

    # --------------------------------------------------------
    # Callbacks
    # --------------------------------------------------------

    callbacks = [

        tf.keras.callbacks.EarlyStopping(

            monitor="val_accuracy",

            patience=8,

            mode="max",

            restore_best_weights=True,

            verbose=1
        ),

        tf.keras.callbacks.ReduceLROnPlateau(

            monitor="val_loss",

            factor=0.5,

            patience=4,

            min_lr=1e-6,

            verbose=1
        )
    ]

    # --------------------------------------------------------
    # Train
    # --------------------------------------------------------

    print("\n")
    print("=" * 60)
    print("STARTING TRAINING")
    print("=" * 60)

    history = model.fit(

        X_train,

        y_train,

        validation_data=(
            X_validation,
            y_validation
        ),

        epochs=EPOCHS,

        batch_size=BATCH_SIZE,

        callbacks=callbacks,

        shuffle=True,

        verbose=1
    )

    # --------------------------------------------------------
    # Save model
    # --------------------------------------------------------

    print("\n")
    print("=" * 60)
    print("SAVING MODEL")
    print("=" * 60)

    model.save(
        MODEL_FILE
    )

    print(
        f"\nModel saved to:"
    )

    print(
        MODEL_FILE
    )

    # --------------------------------------------------------
    # Save class names
    # --------------------------------------------------------

    with open(
        CLASS_NAMES_FILE,
        "w"
    ) as file:

        json.dump(
            class_names,
            file,
            indent=4
        )

    print(
        f"\nClass names saved to:"
    )

    print(
        CLASS_NAMES_FILE
    )

    # --------------------------------------------------------
    # Best validation accuracy
    # --------------------------------------------------------

    best_val_accuracy = max(
        history.history["val_accuracy"]
    )

    best_epoch = (
        np.argmax(
            history.history["val_accuracy"]
        ) + 1
    )

    print("\n")
    print("=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)

    print(
        f"\nBest validation accuracy: "
        f"{best_val_accuracy * 100:.2f}%"
    )

    print(
        f"Best epoch: {best_epoch}"
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()