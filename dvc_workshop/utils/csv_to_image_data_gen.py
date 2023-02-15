"""Module to create a data generatore from csv files."""
import ast

import pandas as pd
from keras_preprocessing.image import ImageDataGenerator

from dvc_workshop.params import ModelParams, TrainingParams


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
    print(classes)
    datagen = ImageDataGenerator()
    generator = datagen.flow_from_dataframe(
        dataframe=path_label_df,
        x_col=paths_columns,
        y_col=labels_columns,
        target_size=(ModelParams.IMAGE_HEIGHT, ModelParams.IMAGE_WIDTH),
        batch_size=TrainingParams.BACTH_SIZE,
        seed=TrainingParams.SEED,
        shuffle=True,
        class_mode="categorical",
        classes=classes,
        color_mode=ModelParams.COLOR_TYPE,
    )
    # generator = tf.keras.utils.image_dataset_from_directory()
    return generator
