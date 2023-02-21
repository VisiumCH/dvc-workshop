"""Module for training the model."""
import os
from typing import Tuple

import tensorflow as tf

from dvc_workshop.models.classifier import Classifier
from dvc_workshop.pipeline.preprocess.constants import PREPROCESS_DIRECTORY
from dvc_workshop.pipeline.train.constants import SAVE_MODEL, TRAIN_HISTORY, TUNE_HISTORY
from dvc_workshop.pipeline.train.io import save_history, save_model
from dvc_workshop.utils.csv_to_image_data_gen import csv_to_image_data_gen
from params import ModelParams, PreprocessParams, TrainingParams


def train_model(  # pylint: disable = too-many-arguments, too-many-locals
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

    model = Classifier(
        PreprocessParams.IMAGE_HEIGHT,
        PreprocessParams.IMAGE_WIDTH,
        ModelParams.NUMBER_CHANNELS,
        ModelParams.ACTIVATION,
        train.class_indices,
    )
    model_history = model.train(train, val, TrainingParams)

    results_dict = model.evaluate(train, verbose=0, return_dict=True)

    print(results_dict)

    results_dict.update(model_history)

    return model, results_dict


def main() -> None:
    """Train the model, save it and its training history (train loss and accuracy per iteration)."""
    model, results_dict = train_model(
        csv_train_path=os.path.join(PREPROCESS_DIRECTORY, "train.csv"),
        csv_valid_path=os.path.join(PREPROCESS_DIRECTORY, "valid.csv"),
        image_path="Paths",
        target="Labels",
    )

    save_model(model.model, os.path.join(SAVE_MODEL, ModelParams.MODEL_NAME))

    save_history(results_dict["history_training"].history, SAVE_MODEL, TRAIN_HISTORY)
    save_history(results_dict["history_finetuning"].history, SAVE_MODEL, TUNE_HISTORY)


if __name__ == "__main__":
    main()
