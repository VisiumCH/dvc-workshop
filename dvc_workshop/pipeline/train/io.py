"""Module for loading images and labels from csv file."""

import os
from typing import Dict

import pandas as pd
import tensorflow as tf

from dvc_workshop.pipeline.preprocess.utils import create_path


def save_model(model: tf.keras.Model, save_path: str) -> None:
    """Save model weights for evaluation step.

    Args:
        model (tf.keras.Model): model to save
        save_path (str): save path
    """
    # check path exists else create it
    create_path(save_path)
    # save model
    model.save(f"{save_path}")


def save_history(history: Dict, save_path: str, title: str) -> None:
    """Save model history for plot step.

    Args:
        history (Dict): _description_
        save_path (str): _description_
        title (str): _description_
    """
    # convert to DF
    hist_df = pd.DataFrame(history)
    # create save path
    hist_csv_file = os.path.join(save_path, title)
    # save history
    with open(hist_csv_file, mode="w", encoding="utf-8") as f:
        hist_df.to_csv(f)
