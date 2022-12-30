import pandas as pd
from visplotlib.pyplot import mpl, plt

from dvc_workshop.pipeline.preprocess.utils import create_path


def read_history(read_path: str):
    return pd.read_csv(read_path)


def save_plot(fig: plt.Axes, save_path: str, file_name: str):
    create_path(save_path)
    fig.savefig(save_path)
