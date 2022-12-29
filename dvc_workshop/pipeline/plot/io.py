import pandas as pd
from visplotlib.pyplot import mpl, plt


def read_history(read_path: str, save_path: str):
    return pd.read_csv(path)


def save_plot(fig: plt.figure.Figure, save_path: str):
    fig.savefig(save_path)
