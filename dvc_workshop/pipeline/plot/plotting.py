"""Plotting DVC step."""
import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from dvc_workshop.pipeline.plot.constants import SAVE_PLOT
from dvc_workshop.pipeline.train.constants import SAVE_MODEL, TRAIN_HISTORY


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
    # pylint: disable=all
    _, axs = plt.subplots(1, 2, sharex=True, figsize=(10, 3))
    ((losses, accuracy)) = axs

    # Loss curve
    (train_loss,) = losses.plot(history["loss"], label="train_loss")

    raise NotImplementedError("Implement validation loss curve here")
    losses.set_xlabel("Epochs")
    losses.set_ylabel("Loss")

    # Accuracy curve
    (train_acc,) = accuracy.plot(history["binary_accuracy"], label="train_acc")

    raise NotImplementedError("Implement validation accuracy curve here")
    accuracy.set_xlabel("Epochs")
    accuracy.set_ylabel("Accuracy")

    plt.suptitle("Loss and Accuracy during training")
    plt.tight_layout()
    plt.show()
    # Save Plot
    if save_plot:
        plt.savefig(Path(path_to_save) / filename, format="png", dpi=400)


def main() -> None:
    """Main function used to create and save training and finetuning loss and accuracy curves on the train split."""
    os.makedirs(SAVE_PLOT, exist_ok=True)
    train_history = pd.read_csv(os.path.join(SAVE_MODEL, TRAIN_HISTORY))

    plot_training_history_loss_acc(
        train_history, path_to_save=SAVE_PLOT, filename="history_training.png", save_plot=True
    )


if __name__ == "__main__":
    main()
