"""Module for training the model."""
import os

from image_classification_autotrain.multilabel_classifier import train_model

from dvc_workshop.params import ModelParams
from dvc_workshop.pipeline.preprocess.constants import PREPROCESS_DIRECTORY
from dvc_workshop.pipeline.train.constants import MODEL_NAME, SAVE_MODEL, TRAIN_HISTORY, TUNE_HISTORY
from dvc_workshop.pipeline.train.io import save_history, save_model


def main():
    model, results_dict = train_model(
        image_height=ModelParams.IMAGE_HEIGHT,
        image_width=ModelParams.IMAGE_WIDTH,
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
