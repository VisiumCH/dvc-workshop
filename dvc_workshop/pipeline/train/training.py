"""Module for training the model."""
import os
from typing import Tuple

import tensorflow as tf

from dvc_workshop.models.efficient_net import EfficientNet, EfficientNetSmall
from dvc_workshop.models.tiny_model import TinyModel
from dvc_workshop.params import ModelParams, TrainingParams
from dvc_workshop.pipeline.preprocess.constants import PREPROCESS_DIRECTORY
from dvc_workshop.pipeline.train.constants import MODEL_NAME, SAVE_MODEL, TRAIN_HISTORY, TUNE_HISTORY
from dvc_workshop.pipeline.train.io import csv_to_image_data_gen, save_history, save_model


def train_model(  # pylint: disable = too-many-arguments, too-many-locals
    model_type: str,
    csv_train_path: str,
    csv_valid_path: str,
    image_path: str,
    target: str,
) -> Tuple[tf.keras.Model, dict]:
    """Train multilabel classifier based on images from csv path.

    Load csv files as dataframe containg two columns. Train model based on labels and paths
    in the datafram.
    This function returns Tensorflow model and dictionary with test results

    Args:
        model_type (str): type of model
        csv_train_path (str): path to the csv file with train set
        csv_valid_path (str): path to the csv file with validation set
        image_path (str): name of the column with paths to the images in a csv file
        target (str): name of the column with images labels in a csv file

    Returns:
        tuple(tf.keras.Model, dict): 2 variables model and a dictionary with performance results

    Raises:
        ValueError:  If a model_type that does not exist is used.
    """
    # GENERATE TRAIN AND VALIDATION DATAGENERATOR FROM COLUMNS
    train = csv_to_image_data_gen(csv_train_path, image_path, target)
    val = csv_to_image_data_gen(csv_valid_path, image_path, target)

    # load the pre-trained model
    if model_type == "efficientnetlarge":
        model = EfficientNet(
            ModelParams.IMAGE_HEIGHT,
            ModelParams.IMAGE_WIDTH,
            ModelParams.NUMBER_CHANNELS,
            ModelParams.POOLING,
            ModelParams.TOP,
            ModelParams.ACTIVATION,
            train.class_indices,
        )
    if model_type == "efficientnetsmall":
        model = EfficientNetSmall(
            ModelParams.IMAGE_HEIGHT,
            ModelParams.IMAGE_WIDTH,
            ModelParams.NUMBER_CHANNELS,
            ModelParams.POOLING,
            ModelParams.TOP,
            ModelParams.ACTIVATION,
            train.class_indices,
        )
    if model_type == "tinymodel":
        model = TinyModel(
            ModelParams.IMAGE_HEIGHT,
            ModelParams.IMAGE_WIDTH,
            ModelParams.NUMBER_CHANNELS,
            ModelParams.ACTIVATION,
            train.class_indices,
        )
    else:
        raise ValueError("This model_type does not exist, the possible models are currently: efficientnetlarge")

    model_history = model.train(train, val, TrainingParams)

    results_dict = model.evaluate(train, verbose=0, return_dict=True)

    print(results_dict)

    results_dict.update(model_history)

    # results_dict["history_training"] = history_training
    # results_dict["history_finetuning"] = history_finetuning

    return model, results_dict


def evaluate_model(
    model: tf.keras.Model,
    csv_test_path: str,
    image_path: str,
    target: str,
) -> dict:
    """Evaluate model with test set data.

    Args:
        model (tf.keras.Model): trained model
        csv_test_path (str): path to dataframe containing the file names of the test images
        image_path (str): path to the folder containing all of the images
        target (str): name of the column with images labels in a csv file
    """
    test = csv_to_image_data_gen(csv_test_path, image_path, target)
    results_dict = model.evaluate(test, verbose=0, return_dict=True)
    print(results_dict)
    return results_dict


def main() -> None:
    """Train the model, save it and its training history (train loss and accuracy per iteration)."""
    model, results_dict = train_model(
        model_type="tinymodel",
        csv_train_path=os.path.join(PREPROCESS_DIRECTORY, "train.csv"),
        csv_valid_path=os.path.join(PREPROCESS_DIRECTORY, "valid.csv"),
        image_path="Paths",
        target="Labels",
    )

    save_model(model, os.path.join(SAVE_MODEL, MODEL_NAME))

    save_history(results_dict["history_training"].history, SAVE_MODEL, TRAIN_HISTORY)
    save_history(results_dict["history_finetuning"].history, SAVE_MODEL, TUNE_HISTORY)


if __name__ == "__main__":
    main()
