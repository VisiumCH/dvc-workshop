"""Plotting DVC step."""
import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from dvc_workshop.params import GlobalParams
from dvc_workshop.pipeline.plot.constants import SAVE_PLOT
from dvc_workshop.pipeline.train.constants import SAVE_MODEL, TRAIN_HISTORY, TUNE_HISTORY


def plot_training_history_loss_acc(
    history: dict,
    path_to_save: Path,
    filename: str = "training_history.png",
    save_plot: bool = True,
) -> None:
    """Plot the training and validation accuracies and losses for a given model.

    Args:
        history (dict): Dictionnary output by the model.fit(...) function.
        path_to_save (Path): Where to save the plot.
        filename (str): Plot filename.
        save_plot (boolean): Wether or not to save the plot.
    """
    _, axs = plt.subplots(1, 2, sharex=True, figsize=(10, 3))
    ((losses, accuracy)) = axs

    # Loss curve
    (train_loss,) = losses.plot(history["loss"], label="train_loss")
    (val_loss,) = losses.plot(history["val_loss"], label="val_loss")
    losses.legend(handles=[train_loss, val_loss])
    losses.set_xlabel("Epochs")
    losses.set_ylabel("Loss")

    # Accuracy curve
    (train_acc,) = accuracy.plot(history["binary_accuracy"], label="train_acc")
    (val_acc,) = accuracy.plot(history["val_binary_accuracy"], label="val_acc")
    accuracy.legend(handles=[train_acc, val_acc])
    accuracy.set_xlabel("Epochs")
    accuracy.set_ylabel("Accuracy")

    plt.suptitle("Loss and Accuracy during training")
    plt.tight_layout()
    plt.show()
    # Save Plot
    if save_plot:
        plt.savefig(Path(path_to_save) / filename, format="png", dpi=400)


def plot_train_and_finetune(
    history_training: pd.DataFrame, history_finetuning: pd.DataFrame, output_folder: Path
) -> None:
    """Plot the train and test loss and accuracy cuves during training and finetuning.

    Args:
        history_training (pd.DataFrame): training history
        history_finetuning (pd.DataFrame): finetuning history
        output_folder (Path): folder where to save the plots
    """
    os.makedirs(output_folder, exist_ok=True)

    plot_training_history_loss_acc(
        history_training, path_to_save=output_folder, filename="history_training.png", save_plot=True
    )
    plot_training_history_loss_acc(
        history_finetuning, path_to_save=output_folder, filename="history_finetuning.png", save_plot=True
    )


def main() -> None:
    """Main function used to create and save training and finetuning loss and accuracy curves on the train split."""
    os.makedirs(SAVE_PLOT, exist_ok=True)
    train_history = pd.read_csv(os.path.join(SAVE_MODEL, TRAIN_HISTORY))

    if GlobalParams.MODEL_TYPE == "tinymodel":
        plot_training_history_loss_acc(
            train_history, path_to_save=SAVE_PLOT, filename="history_training.png", save_plot=True
        )
    elif GlobalParams.MODEL_TYPE in ["efficientnetlarge", "efficientnetsmall"]:
        tune_history = pd.read_csv(os.path.join(SAVE_MODEL, TUNE_HISTORY))
        plot_train_and_finetune(train_history, tune_history, SAVE_PLOT)
    else:
        raise ValueError(f"Plot function not implemented for the model type: {GlobalParams.MODEL_TYPE}")


if __name__ == "__main__":
    main()
