import tensorflow as tf
import os
import shutil
import matplotlib.pyplot as plt

# from tensorflow.keras import layers, models, optimizers

# import numpy as np
# import sklearn


def load_dataset():
    dataset_url = "https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip"  # noqa: E501

    data_dir = tf.keras.utils.get_file(
        'cats_and_dogs.zip',
        origin=dataset_url,
        extract=True
    )
    # Path where the zip file is extracted
    extracted_dir_path = os.path.join(
        os.path.dirname(data_dir), "cats_and_dogs_filtered"
    )
    # Define target directory for dataset within project
    target_dir = os.path.join("data", "cats_and_dogs_filtered")
    # Copy dataset to target directory if it's not already there
    if not os.path.exists(target_dir):
        shutil.copytree(extracted_dir_path, target_dir)
    return target_dir


def main():
    data_dir = load_dataset()
    print(f"data_dir: {data_dir}")
    # Define the image size and batch size
    img_size = (160, 160)
    batch_size = 32
    # Load the training and validation datasets
    train_dataset = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=img_size,
        batch_size=batch_size,
    )
    validation_dataset = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=img_size,
        batch_size=batch_size,
    )
    # print(train_dataset)
    # print(validation_dataset)

    # Configure the Dataset for Performance
    AUTOTUNE = tf.data.AUTOTUNE
    train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)
    validation_dataset = validation_dataset.prefetch(buffer_size=AUTOTUNE)

    # Load the Pre-trained Model
    pretrained_model = tf.keras.applications.MobileNetV2(
        input_shape=(160, 160, 3), include_top=False, weights="imagenet"
    )

    # Freeze the Convolutional Base
    pretrained_model.trainable = False

    # Add a Classification Head
    model = tf.keras.models.Sequential(
        [
            pretrained_model,
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(1)
        ]
    )

    # Compile the Model
    model.compile(
        optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=0.0001),
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )

    # Train the Model
    history = model.fit(
        train_dataset,
        epochs=10,
        validation_data=validation_dataset
    )

    # Evaluate the Model
    acc = history.history["accuracy"]
    val_acc = history.history["val_accuracy"]
    loss = history.history["loss"]
    val_loss = history.history["val_loss"]

    plt.figure(figsize=(8, 8))
    plt.subplot(2, 1, 1)
    plt.plot(acc, label="Training Accuracy")
    plt.plot(val_acc, label="Validation Accuracy")
    plt.legend(loc="lower right")
    plt.ylabel("Accuracy")
    plt.ylim([min(plt.ylim()), 1])
    plt.title("Training and Validation Accuracy")

    plt.subplot(2, 1, 2)
    plt.plot(loss, label="Training Loss")
    plt.plot(val_loss, label="Validation Loss")
    plt.legend(loc="upper right")
    plt.ylabel("Cross Entropy")
    plt.ylim([0, 1.0])
    plt.title("Training and Validation Loss")
    plt.xlabel("epoch")
    plt.show()


# project_dir = os.path.dirname(os.path.abspath(__file__))
# print(f"project_dir: {project_dir}")
# print(np.__version__)
# print(tf.__version__)
# print(sklearn.__version__)


if __name__ == "__main__":
    main()
