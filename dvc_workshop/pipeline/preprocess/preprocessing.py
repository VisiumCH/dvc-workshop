"""Preprocessing module."""
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from dvc_workshop.params import PreprocessParams
from dvc_workshop.pipeline.preprocess.constants import (
    PREPROCESS_DIRECTORY,
    RAW_DIRECTORY,
    SOURCE_DIRECTORY,
    TARGET_DIRECTORY,
)
from dvc_workshop.pipeline.preprocess.io import generate_dataset, read_images
from dvc_workshop.pipeline.preprocess.utils import create_path


def main() -> None:
    """Load the dataset, preprocess it, split it into train test and eval in stratified fasion,then save to csv."""
    create_path(target_directory=TARGET_DIRECTORY)
    # read image data
    images = read_images(SOURCE_DIRECTORY)

    # filter on color content
    # color_filtered = filter(color_detector, images)

    # Resize images
    def _resize_image(images: list[str], img_size: tuple[int, int], target_directory: Path) -> None:
        """Resize all of the images to desired output size."""
        # pylint: disable=no-member
        print(images)
        for image_path in tqdm(images):
            image = cv2.imread(image_path)
            image_resized = cv2.resize(image, dsize=img_size)

            # print("PATH:", Path(target_directory) / str(image_path).split("/")[-1])
            cv2.imwrite(filename=str(Path(target_directory) / str(image_path).split("/")[-1]), img=image_resized)

    _resize_image(images, img_size=(28, 28), target_directory=TARGET_DIRECTORY)

    # generate train, val, test labels
    generate_dataset(RAW_DIRECTORY, TARGET_DIRECTORY, PREPROCESS_DIRECTORY)


def color_detector(image_path: str) -> np.ndarray:
    """Detects if poster image contains pixel content of given color above threshold.

    Args:
        image_path (str): path to image

    Returns:
        np.ndarray: colored pixel proportion against threshold
    """
    # pylint: disable=no-member
    # read image
    image = cv2.imread(image_path)
    # lower bound for red color
    lower_red = np.array(PreprocessParams.LOWER_BOUND_COLOR, dtype="uint8")
    # upper bound for red color
    upper_red = np.array(PreprocessParams.UPPER_BOUND_COLOR, dtype="uint8")
    # detect pixels in red range
    mask = cv2.inRange(image, lower_red, upper_red)
    # keep pixels in the range
    detected_output = cv2.bitwise_and(image, image, mask=mask)
    # sum total number of red pixels
    sum_red = np.sum(detected_output, axis=2)
    # count non zeros
    red_pixel = np.count_nonzero(sum_red)
    # get total number of pixel
    total_pixels = detected_output.size
    # get proportion of detected pixel
    amount_detected = np.round(red_pixel / total_pixels, 5)
    # return bool comparison with threshold
    return amount_detected < PreprocessParams.THRESHOLD


if __name__ == "__main__":
    main()
