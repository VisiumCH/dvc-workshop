"""Plotting DVC step."""
import os

from image_classification_autotrain.plots import plot_train_and_finetune

from dvc_workshop.pipeline.plot.constants import SAVE_PLOT
from dvc_workshop.pipeline.plot.io import read_history
from dvc_workshop.pipeline.train.constants import SAVE_MODEL, TRAIN_HISTORY, TUNE_HISTORY


def main() -> None:
    """Main function used to create and save training and finetuning loss and accuracy curves on the train split."""
    os.makedirs(SAVE_MODEL, exist_ok=True)

    train_history = read_history(os.path.join(SAVE_MODEL, TRAIN_HISTORY))

    tune_history = read_history(os.path.join(SAVE_MODEL, TUNE_HISTORY))

    plot_train_and_finetune(
        train_history,
        tune_history,
        output_folder=SAVE_PLOT,
    )


if __name__ == "__main__":
    main()
