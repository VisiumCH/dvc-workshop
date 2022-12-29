"""Module for training the model."""
import os

import tensorflow as tf

from dvc_workshop.models.efficientnet import EfficentNet
from dvc_workshop.params import ModelParams, TrainingParams
from dvc_workshop.pipeline.preprocess.constants import PREPROCESS_DIRECTORY
from dvc_workshop.pipeline.train.constants import SAVE_MODEL,MODEL_NAME, TRAIN_HISTORY,TUNE_HISTORY
from dvc_workshop.pipeline.train.io import csv_to_image_data_gen, save_model, save_history


def train_model(
    image_height: int,
    image_width: int,
    csv_train_path: str,
    csv_valid_path: str,
    csv_test_path: str,
    image_path: str,
    target: str,
) -> tuple[tf.keras.Model, dict]:
    """Train multilabel classifier based on images from csv path.
    Load csv files as dataframe containg two columns. Train model based on labels and paths
    in the datafram.
    This function returns Tensorflow model and dictionary with test results

    Args:
        image_height (int): image horizontal dimensions in pixels
        image_width (int): image vertical dimensions in pixels
        csv_train_path (str): path to the csv file with train set
        csv_valid_path (str): path to the csv file with validation set
        csv_test_path (str): path to the csv file with test set
        image_path (str): name of the column with paths to the images in a csv file
        target (str): name of the column with images labels in a csv file

    Returns:
        tuple(tf.keras.Model, dict): 2 variables model and a dictionary with performance results
    """

    # GENERATE TRAIN AND VALIDATION DATAGENERATOR FROM COLUMNS
    train = csv_to_image_data_gen(csv_train_path, image_path, target)
    val = csv_to_image_data_gen(csv_valid_path, image_path, target)
    model = EfficentNet(
        ModelParams.IMAGE_HEIGHT,
        ModelParams.IMAGE_WIDTH,
        ModelParams.NUMBER_CHANNELS,
        ModelParams.POOLING,
        ModelParams.TOP,
        ModelParams.ACTIVATION,
        train.class_indices,
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(TrainingParams.TRAINING_LR),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=[
            tf.keras.metrics.Precision(),
            tf.keras.metrics.Recall(),
            tf.keras.metrics.BinaryAccuracy(),
        ],
    )
    # train the output layer
    train_history = model.fit(
        train,
        validation_data=val,
        epochs=TrainingParams.TRAINING_EPOCHS,
        callbacks=[tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)],
    )
    """
    # fine-tune the model
    model.set_trainable()

    model.compile(
        optimizer=tf.keras.optimizers.Adam(TrainingParams.TUNING_LR),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=[
            tf.keras.metrics.Precision(),
            tf.keras.metrics.Recall(),
            tf.keras.metrics.BinaryAccuracy(),
        ],
    )
    tune_history = model.fit(
        train,
        validation_data=val,
        epochs=TrainingParams.TUNING_EPOCHS,
        callbacks=[tf.keras.callbacks.EarlyStopping(patience=TrainingParams.PATIENCE, restore_best_weights=True)],
    )

"""    

    save_model(model, os.path.join(SAVE_MODEL, MODEL_NAME))
    save_history(train_history.history,SAVE_MODEL,TRAIN_HISTORY)
   # save_history(tune_history.history,SAVE_MODEL,TUNE_HISTORY)

    return model


def main():
    model = train_model(
        image_height=ModelParams.IMAGE_HEIGHT,
        image_width=ModelParams.IMAGE_WIDTH,
        csv_train_path=os.path.join(PREPROCESS_DIRECTORY, "train.csv"),
        csv_valid_path=os.path.join(PREPROCESS_DIRECTORY, "valid.csv"),
        csv_test_path=os.path.join(PREPROCESS_DIRECTORY, "test.csv"),
        image_path="Paths",
        target="Labels",
    )


if __name__ == "__main__":
    main()
