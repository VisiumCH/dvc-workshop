import os
from typing import List

import pandas as pd
import visplotlib as vpl
from visplotlib.pyplot import VISIUM_CLASSIC, VISIUM_DARK, mpl, plt
from visplotlib.seaborn import sns

from dvc_workshop.pipeline.plot.constants import METRICS, PLOT_LABELS, SAVE_PLOT, TITLES, TRAIN_PLOT, TUNE_PLOT
from dvc_workshop.pipeline.plot.io import read_history, save_plot
from dvc_workshop.pipeline.train.constants import SAVE_MODEL, TRAIN_HISTORY, TUNE_HISTORY


def main():
    train_history = read_history(os.path.join(SAVE_MODEL, TRAIN_HISTORY))
    train_plot = plot_history(train_history, METRICS, TITLES, PLOT_LABELS, True)
    save_plot(train_plot, SAVE_PLOT, TRAIN_PLOT)

    tune_history = read_history(os.path.join(SAVE_MODEL, TUNE_HISTORY))
    tune_plot = plot_history(tune_history, METRICS, TITLES, PLOT_LABELS, False)
    save_plot(tune_plot, SAVE_PLOT, TUNE_PLOT)


def plot_history(
    history: pd.DataFrame, metrics: List[str], titles: List[str], labels: List[str], training: bool = True
):
    """plot training and tuning history

    Args:
        history (pd.DataFrame): Training history
        metrics (List[str]): metrics to plot
        titles (List[str]): plot titles
        labels (List[str]): metric label
        training (bool, optional): wether training or tuning history is plot. Defaults to True.

    Returns:
        _type_: _description_
    """
    # get plot type
    fit = "Training" if training else "Tuning"
    # plot history
    g = sns.lineplot(data=history[metrics])
    # set plot title
    g.set_title(f"{fit} " + " and ".join(titles))
    # set axis labels
    g.set(xlabel="epochs", ylabel="metrics")
    # Remove the normal legend and create a horizontal one on top
    g.legend(loc="upper left", bbox_to_anchor=(1.05, 1), labels=labels, fontsize="10")
    # format plot visium way
    plt.format()
    return g.get_figure()


if __name__ == "__main__":
    main()
