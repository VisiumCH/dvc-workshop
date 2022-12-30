import os

import pandas as pd
import visplotlib as vpl
from visplotlib.pyplot import mpl, plt
from visplotlib.seaborn import sns

from dvc_workshop.pipeline.plot.constants import SAVE_PLOT, TRAIN_PLOT, TUNE_PLOT
from dvc_workshop.pipeline.plot.io import read_history, save_plot
from dvc_workshop.pipeline.train.constants import SAVE_MODEL, TRAIN_HISTORY


def main():
    train_history = read_history(os.path.join(SAVE_MODEL, TRAIN_HISTORY))
    # tune_history = read_history(os.path.join(SAVE_MODEL, SAVE_TUNE_PLOT))
    train_plot = plot_history(train_history)
    # tune_plot = plot_history(tune_history)
    save_plot(train_plot, SAVE_PLOT, TRAIN_PLOT)
    # save_plot(tune_plot,SAVE_PLOT, TUNE_PLOT)


def plot_history(history: pd.DataFrame):
    plot = sns.lineplot(data=history.iloc[:, 1:])
    return plot.get_figure()


if __name__ == "__main__":
    main()
