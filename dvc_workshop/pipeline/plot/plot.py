import os
import visplotlib as vpl
from visplotlib.pyplot import mpl, plt 
from visplotlib.seaborn import sns
from dvc_workshop.pipeline.plot.io import read_history, save_history
from dvc_workshop.pipeline.train.constants import SAVE_MODEL, SAVE_TRAIN_PLOT, SAVE_TUNE_PLOT
from dvc_workshop.pipeline.plot.constants import SAVE_TRAIN_PLOT, SAVE_TUNE_PLOT

def main():

    train_history = read_history(os.path.join(SAVE_MODEL,SAVE_TRAIN_PLOT))
    tune_history = read_history(os.path.join(SAVE_MODEL,SAVE_TUNE_PLOT))
    train_plot = plot_history(train_history)
    tune_plot = plot_history(tune_history)
    save_history(train_plot,SAVE_TRAIN_PLOT)
    save_history(tune_plot,SAVE_TUNE_PLOT)


def plot_history(save_path : str, hisotry : pd.DataFrame) :
    plot = sns.lineplot(data=history)
    return plot.get_figure()


if __name__ == "__main__":
    main()
