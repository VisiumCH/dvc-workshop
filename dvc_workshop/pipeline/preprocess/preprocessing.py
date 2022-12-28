import numpy as np

from dvc_workshop.params import PreprocessParams
from dvc_workshop.pipeline.preprocess.constants import (
    PREPROCESS_DIRECTORY,
    RAW_DIRECTORY,
    SOURCE_DIRECTORY,
    TARGET_DIRECTORY,
)
from dvc_workshop.pipeline.preprocess.io import generate_dataset, read_images, save_images


def main():
    # read image data
    images = read_images(SOURCE_DIRECTORY)
    # filter on color content
    color_filtered = filter(color_detector, images)
    # save result
    save_images(color_filtered, TARGET_DIRECTORY)
    # generate train, val, test labels
    generate_dataset(RAW_DIRECTORY, TARGET_DIRECTORY, PREPROCESS_DIRECTORY)


def color_detector(image_path: str) -> bool:
    """Detects if poster image contains pixel content of given color above threshold

    Args:
        image_path (str): path to image

    Returns:
        bool: colored pixel proportion against threshold
    """
    import cv2

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
