import os

import pandas as pd
from visplotlib.pyplot import plt

from dvc_workshop.pipeline.preprocess.utils import create_path


def read_history(read_path: str):
    """load history into Dataframe

    Args:
        read_path (str): file path

    Returns:
        _type_: history Dataframe
    """
    return pd.read_csv(read_path)


def save_plot(fig: plt.Axes, save_path: str, file_name: str):
    """save plot results

    Args:
        fig (plt.Axes): figure to save
        save_path (str): figure save path
        file_name (str): figure file name
    """
    # check if path exists else create it
    create_path(save_path)
    # save file
    fig.savefig(os.path.join(save_path, file_name))
    # close figure
    plt.close(fig)
