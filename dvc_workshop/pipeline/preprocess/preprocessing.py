"""Preprocessing module."""
import logging

import cv2
import numpy as np
from tqdm import tqdm

from dvc_workshop.params import ModelParams
from dvc_workshop.pipeline.preprocess.constants import (
    PREPROCESS_DIRECTORY,
    RAW_DIRECTORY,
    SOURCE_DIRECTORY,
    TARGET_DIRECTORY,
)
from dvc_workshop.pipeline.preprocess.io import generate_dataset, read_images, save_images
from dvc_workshop.pipeline.preprocess.utils import create_path


def main() -> None:
    """Load the dataset, preprocess it, split it into train test and eval in stratified fasion,then save to csv."""
    create_path(target_directory=TARGET_DIRECTORY)
    # read image data
    images = read_images(SOURCE_DIRECTORY)

    # Resize images
    images = resize_image(images, img_size=(ModelParams.IMAGE_HEIGHT, ModelParams.IMAGE_WIDTH))

    # Resize the images
    images = standardize(images)

    # Save the images
    save_images(images, target_directory=TARGET_DIRECTORY)

    # generate train, val, test labels
    generate_dataset(RAW_DIRECTORY, TARGET_DIRECTORY, PREPROCESS_DIRECTORY)


def resize_image(images: dict, img_size: tuple[int, int]) -> None:
    """Resize all of the images to desired output size."""
    # pylint: disable=no-member
    for image_path, image in tqdm(images.items()):
        images[image_path] = cv2.resize(image, dsize=img_size)

    return images


def standardize(images: dict, tolerance: float = 1e-5) -> dict:
    """Standardize the input images."""
    # pylint: disable=no-member
    stacked_images = np.stack(images.values(), axis=0)
    std = stacked_images.std()
    mean = stacked_images.mean()

    if std > tolerance:
        for image_path, image in tqdm(images.items()):
            images[image_path] = (image - mean) / std
    else:
        logging.warning("Standard deviation too small, we will only substract the mean.")
        for image_path, image in tqdm(images.items()):
            images[image_path] = image - mean

    return images


if __name__ == "__main__":
    main()
