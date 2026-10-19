import tensorflow as tf



DATASET_DIR = "dataset"
IMG_SIZE = (128, 128)
BATCH_SIZE = 32
SEED = 42


train_ds = tf.keras.utils.image_dataset_from_directory(
    DATASET_DIR,
    validation_split=0.2,
    subset="training",
    seed=SEED,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="int"
)



val_ds = tf.keras.utils.image_dataset_from_directory(
    DATASET_DIR,
    validation_split=0.2,
    subset="validation",
    seed=SEED,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="int"
)


class_names = train_ds.class_names

print("\nClasses:")
print(class_names)

print("\nNumber of classes:", len(class_names))



normalization_layer = tf.keras.layers.Rescaling(1.0 / 255)

train_ds = train_ds.map(
    lambda x, y: (normalization_layer(x), y)
)

val_ds = val_ds.map(
    lambda x, y: (normalization_layer(x), y)
)


AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.cache().shuffle(1000).prefetch(
    buffer_size=AUTOTUNE
)

val_ds = val_ds.cache().prefetch(
    buffer_size=AUTOTUNE
)



for images, labels in train_ds.take(1):

    print("\nImage batch shape:", images.shape)
    print("Label batch shape:", labels.shape)
    print("Pixel value range:",
          tf.reduce_min(images).numpy(),
          "to",
          tf.reduce_max(images).numpy())
    
# =========================
# Check Dataset
# =========================

for images, labels in train_ds.take(1):

    print("\nImage batch shape:", images.shape)
    print("Label batch shape:", labels.shape)
    print(
        "Pixel value range:",
        tf.reduce_min(images).numpy(),
        "to",
        tf.reduce_max(images).numpy()
    )

from collections import Counter

label_counts = Counter()

for _, labels in train_ds.unbatch():
    label_counts[int(labels.numpy())] += 1

print("\nTraining samples per class:")

for i, class_name in enumerate(class_names):
    print(f"{class_name}: {label_counts[i]}")