"""Module to evaluate the trained model."""
import os

import tensorflow as tf

from dvc_workshop.pipeline.evaluate.constants import RESULTS_FILE, SAVE_RESULTS
from dvc_workshop.pipeline.evaluate.io import load_model, save_json
from dvc_workshop.pipeline.preprocess.constants import PREPROCESS_DIRECTORY
from dvc_workshop.pipeline.train.constants import MODEL_NAME, SAVE_MODEL
from dvc_workshop.utils.data_gen import csv_to_image_data_gen


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
    """Load and evaluate the model, then save the results to a json file."""
    model_path = os.path.join(SAVE_MODEL, MODEL_NAME)

    # load model
    model = load_model(model_path)

    results_dict = evaluate_model(
        model=model,
        csv_test_path=os.path.join(PREPROCESS_DIRECTORY, "test.csv"),
        image_path="Paths",
        target="Labels",
    )

    # save results
    save_json(results_dict, SAVE_RESULTS, RESULTS_FILE)


if __name__ == "__main__":
    main()
