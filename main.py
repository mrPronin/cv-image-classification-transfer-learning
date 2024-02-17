import tensorflow as tf
import os
import shutil

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
    # project_dir = os.path.dirname(os.path.abspath(__file__))
    # print(f"project_dir: {project_dir}")
    # print(np.__version__)
    # print(tf.__version__)
    # print(sklearn.__version__)


if __name__ == "__main__":
    main()
