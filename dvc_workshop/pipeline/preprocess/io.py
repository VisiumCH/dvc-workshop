"""IO module for the preprocessing step."""
import glob
import os
from pathlib import Path

import cv2
import pandas as pd

from dvc_workshop.pipeline.preprocess.utils import perform_stratification


def read_images(source_directory: str) -> dict:
    """Simple function to showcase reading data from io.
    Args:
        source_directory (str): source iamges directory
    Returns:
        dict: dict where the key is the path and the value is the image
    """
    # pylint: disable=no-member
    image_paths = glob.glob(source_directory + "/**/*.png", recursive=True) + glob.glob(
        source_directory + "/**/*.jpg", recursive=True
    )
    images = {}
    for image_path in image_paths:
        images[image_path] = cv2.imread(image_path)
    return images


def save_images(images: dict, target_directory: Path) -> None:
    """Save the images to disk."""
    # pylint: disable=no-member, use-maxsplit-arg
    for image_path, image in images.items():
        cv2.imwrite(filename=str(Path(target_directory) / str(image_path).split("/")[-1]), img=image)


def generate_dataset(source_file: str, images_directory: str, target_directory: str) -> None:
    """Generates stratified  train test and validation csv files from the dataset."""
    path_labels_df = pd.read_csv(
        os.path.join(source_file, "train.csv"),
        usecols=["Labels", "Paths"],
    )

    path_labels_df["Paths"] = images_directory + "/" + (path_labels_df["Paths"]).str.split("/", expand=True).iloc[:, -1]
    print(path_labels_df.head())

    path_labels_df["Labels"] = path_labels_df["Labels"].apply(lambda x: [x])
    print(path_labels_df.info())
    train, test, valid = perform_stratification(path_labels_df, 0.3, 0.8, 1)

    train.to_csv(os.path.join(target_directory, "train.csv"), index=False)
    test.to_csv(os.path.join(target_directory, "test.csv"), index=False)
    valid.to_csv(os.path.join(target_directory, "valid.csv"), index=False)
