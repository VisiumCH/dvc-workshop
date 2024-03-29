"""Module used to evaluate the model."""
import tensorflow as tf

from dvc_workshop.utils.csv_to_image_data_gen import csv_to_image_data_gen


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
