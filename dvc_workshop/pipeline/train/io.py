"""Module for loading images and labels from csv file."""
import ast
import os

import pandas as pd
import tensorflow as tf
from keras_preprocessing.image import ImageDataGenerator


def csv_to_image_data_gen(file_path: str, paths_columns: str, labels_columns: str) -> ImageDataGenerator():
    """Load csv file from the path with two columns paths_column and labels_column.
    Path can be relative or full. If csv contains full path remove "directory" parameter.
    If csv contains just filenames add directory pointing to data directory.
    Args:
        file_path (str): path to a csv file with columns
        paths_columns (str): name of column with path
        labels_columns (str): name of column with labels

    Returns:
        ImageDataGenerator: DataGenerator with loaded Images and Labels
    """
    # Use ast.eval to convert string '[label1, label2, label3]' to python list
    path_label_df = pd.read_csv(
        file_path,
        usecols=[paths_columns, labels_columns],
        converters={labels_columns: ast.literal_eval},
    )
    # all unique labels: {label1, label2, label3 .... }
    classes = set(sum(path_label_df[labels_columns].to_list(), []))
    datagen = ImageDataGenerator()
    generator = datagen.flow_from_dataframe(
        dataframe=path_label_df,
        x_col=paths_columns,
        y_col=labels_columns,
        batch_size=32,
        seed=42,
        shuffle=True,
        class_mode="categorical",
        classes=classes,
    )
    return generator


def save_model(save_path: str, model: tf.keras.Model):
    # SAVE MODE TO A FILE

    if not os.path.exists(save_path):
        #  Create a new directory because it does not exist
        os.makedirs(save_path)
        print("The new directory has been created!")

    # # save the model
    model.save(f"{save_path}")
