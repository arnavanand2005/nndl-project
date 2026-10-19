import os
import shutil
import random


# ============================================================
# CONFIGURATION
# ============================================================

SOURCE_DIR = "dataset"
OUTPUT_DIR = "dataset_split"

TRAIN_DIR = os.path.join(
    OUTPUT_DIR,
    "train"
)

VAL_DIR = os.path.join(
    OUTPUT_DIR,
    "validation"
)

TRAIN_RATIO = 0.8

SEED = 42

random.seed(SEED)


# ============================================================
# CREATE OUTPUT DIRECTORIES
# ============================================================

os.makedirs(
    TRAIN_DIR,
    exist_ok=True
)

os.makedirs(
    VAL_DIR,
    exist_ok=True
)


# ============================================================
# GET CLASSES
# ============================================================

classes = sorted([

    folder

    for folder in os.listdir(SOURCE_DIR)

    if os.path.isdir(
        os.path.join(
            SOURCE_DIR,
            folder
        )
    )

])


print("\nClasses found:")
print(classes)

print(
    "\nNumber of classes:",
    len(classes)
)


# ============================================================
# SPLIT EACH CLASS
# ============================================================

total_train = 0
total_val = 0


for class_name in classes:

    source_class_dir = os.path.join(
        SOURCE_DIR,
        class_name
    )

    train_class_dir = os.path.join(
        TRAIN_DIR,
        class_name
    )

    val_class_dir = os.path.join(
        VAL_DIR,
        class_name
    )

    os.makedirs(
        train_class_dir,
        exist_ok=True
    )

    os.makedirs(
        val_class_dir,
        exist_ok=True
    )


    # --------------------------------------------------------
    # GET IMAGES
    # --------------------------------------------------------

    images = [

        file

        for file in os.listdir(
            source_class_dir
        )

        if file.lower().endswith(
            (
                ".jpg",
                ".jpeg",
                ".png"
            )
        )

    ]


    # --------------------------------------------------------
    # SHUFFLE
    # --------------------------------------------------------

    random.shuffle(
        images
    )


    # --------------------------------------------------------
    # CALCULATE SPLIT
    # --------------------------------------------------------

    split_index = int(
        len(images) * TRAIN_RATIO
    )

    train_images = images[
        :split_index
    ]

    val_images = images[
        split_index:
    ]


    # --------------------------------------------------------
    # COPY TRAINING IMAGES
    # --------------------------------------------------------

    for image in train_images:

        source = os.path.join(
            source_class_dir,
            image
        )

        destination = os.path.join(
            train_class_dir,
            image
        )

        shutil.copy2(
            source,
            destination
        )


    # --------------------------------------------------------
    # COPY VALIDATION IMAGES
    # --------------------------------------------------------

    for image in val_images:

        source = os.path.join(
            source_class_dir,
            image
        )

        destination = os.path.join(
            val_class_dir,
            image
        )

        shutil.copy2(
            source,
            destination
        )


    total_train += len(
        train_images
    )

    total_val += len(
        val_images
    )


    print(
        f"{class_name:12s} "
        f"Train: {len(train_images):3d} "
        f"Validation: {len(val_images):3d}"
    )


# ============================================================
# SUMMARY
# ============================================================

print("\n")
print("=" * 60)
print("DATASET SPLIT COMPLETE")
print("=" * 60)

print(
    "\nTotal training images:",
    total_train
)

print(
    "Total validation images:",
    total_val
)

print(
    "Total images:",
    total_train + total_val
)

print(
    "\nDataset created at:",
    OUTPUT_DIR
)