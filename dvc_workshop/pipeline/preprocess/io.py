"""IO module for the preprocessing step."""
import ast
import glob
import os
import shutil
from typing import Iterable, List

import pandas as pd

from dvc_workshop.params import GlobalParams
from dvc_workshop.pipeline.preprocess.utils import create_path, perform_stratification


def read_images(source_directory: str) -> List[str]:
    """Simple function to showcase reading data from io.

    Args:
        source_directory (str): source iamges directory

    Returns:
        List[str]: list of images
    """
    return glob.glob(source_directory + "/**/*.png", recursive=True)


def save_images(images: Iterable[str], target_directory: str) -> None:
    """Simple function to create directory and save the images there.

    Args:
        images (Iterable[str]): Image paths
        target_directory (str): Destination directory
    """
    create_path(target_directory)

    # copy selected images
    for img in images:
        if not os.path.isfile(os.path.join(target_directory, img)):
            shutil.copy(img, target_directory)


def generate_dataset(source_file: str, images_directory: str, target_directory: str) -> None:
    """Generates stratified  train test and validation csv files from the dataset."""
    path_labels_df = pd.read_csv(
        os.path.join(source_file, "train.csv"),
        usecols=["Labels", "Paths"],
    )
    # path_labels_df = path_labels_df[["Id", "Genre"]]
    # path_labels_df.columns = ["Paths", "Labels"]
    path_labels_df["Paths"] = images_directory + "/" + (path_labels_df["Paths"]).str.split("/", expand=True).iloc[:, -1]
    print(path_labels_df.head())
    # if GlobalParams.DEBUG:
    #     path_labels_df = path_labels_df.sample(n=2000, random_state=42)
    path_labels_df["Labels"] = path_labels_df["Labels"].apply(lambda x: [x])
    print(path_labels_df.info())
    train, test, valid = perform_stratification(path_labels_df, 0.3, 0.8, 1)

    train.to_csv(os.path.join(target_directory, "train.csv"), index=False)
    test.to_csv(os.path.join(target_directory, "test.csv"), index=False)
    valid.to_csv(os.path.join(target_directory, "valid.csv"), index=False)
