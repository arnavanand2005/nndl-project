import tensorflow as tf
import matplotlib.pyplot as plt


DATASET_DIR = "dataset"

IMG_SIZE = (128, 128)
BATCH_SIZE = 16
SEED = 42


dataset = tf.keras.utils.image_dataset_from_directory(
    DATASET_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=True,
    seed=SEED
)



class_names = dataset.class_names

print("\nClasses:")

for i, class_name in enumerate(class_names):
    print(f"{i}: {class_name}")



images, labels = next(iter(dataset))


print("\nImage batch shape:")
print(images.shape)

print("\nLabel batch shape:")
print(labels.shape)



plt.figure(figsize=(12, 8))

for i in range(12):

    plt.subplot(3, 4, i + 1)

    plt.imshow(
        images[i].numpy().astype("uint8")
    )

    plt.title(
        class_names[labels[i]]
    )

    plt.axis("off")


plt.tight_layout()

plt.savefig(
    "dataset_preview.png",
    dpi=200
)

print("\nDataset preview saved as:")
print("dataset_preview.png")


plt.show()