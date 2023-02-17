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

    # standardize: insert code here

    # Rotate and crop the images
    images = rotate_and_crop_images(images, angle_interval=(-5, 5))

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
    """Standardize the input images.

    images is a dictionnary, keys being str paths to image and value the image array

    - Standardize stacked images if the standard deviation is above threshold
    - if not, substract the mean


    """
    # pylint: disable=no-member
    stacked_images = np.stack(images.values(), axis=0)
    std = stacked_images.std()
    mean = stacked_images.mean()

    ############# CODE HERE #############
    raise NotImplementedError("Code standardization function here")


def rotate_and_crop_images(images: dict, angle_interval: tuple[float, float]) -> dict:
    """Rotate and center crop an image."""

    # pylint: disable=no-member
    def _rotate_image(image: np.array, angle: float) -> np.array:
        """Rotate an image."""
        image_center = tuple(np.array(image.shape[1::-1]) / 2)
        rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
        result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
        return result

    def _crop_image(image: np.array, width: int, height: int) -> np.array:
        """Center crop an image to the desired dimension."""
        center = image.shape
        x = center[1] / 2 - width / 2
        y = center[0] / 2 - height / 2

        cropped_image = image[int(y) : int(y + height), int(x) : int(x + width)]
        return cropped_image

    for image_path, image in tqdm(images.items()):
        image_shape = image.shape
        (original_width, original_height) = image_shape[0], image_shape[1]

        assert (original_width % 2 == 0) and (original_height % 2 == 0), "Image dimensions are not even!"

        # rotate with random angle in angle_interval
        random_angle = float(np.random.uniform(low=angle_interval[0], high=angle_interval[1], size=1))
        rotated_image = _rotate_image(image, angle=random_angle)

        # center crop
        cropped_image = _crop_image(rotated_image, original_width, original_height)
        cropped_image_shape = cropped_image.shape

        assert (original_width == cropped_image_shape[0]) and (
            original_height == cropped_image_shape[1]
        ), "Image dimension changes after resize and crop!"

        images[image_path] = cropped_image

    return images


if __name__ == "__main__":
    main()
