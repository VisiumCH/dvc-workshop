"""IO module for the plotting step."""

import pandas as pd


def read_history(read_path: str) -> pd.DataFrame:
    """Load the training history.

    Args:
        read_path (str): file path

    Returns:
        pd.DataFrame: history Dataframe
    """
    return pd.read_csv(read_path)
