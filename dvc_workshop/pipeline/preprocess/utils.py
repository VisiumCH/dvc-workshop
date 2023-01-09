"""perform stratification on data frame with labels and paths"""
import os

import numpy as np
import pandas as pd
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
from sklearn.preprocessing import MultiLabelBinarizer


def handle_no_classes_in_test_and_valid_after_stratification(
    train: pd.DataFrame, test: pd.DataFrame, valid: pd.DataFrame, mlb: any
) -> tuple([pd.DataFrame, pd.DataFrame, pd.DataFrame]):
    word_to_remove = None
    for i in [valid, test, train]:
        results = ~np.all(mlb.transform(i["Labels"]) == 0, axis=0)
        index_to_remove = (results == False).nonzero()
        if mlb.classes_[index_to_remove] != None:
            word_to_remove = mlb.classes_[index_to_remove]

    search_for_word = lambda x: [w for w in x if w != word_to_remove]
    valid["Labels"] = valid["Labels"].apply(search_for_word)
    test["Labels"] = test["Labels"].apply(search_for_word)
    train["Labels"] = train["Labels"].apply(search_for_word)

    # Check for equal size
    assert (
        mlb.transform(test["Labels"]).shape[1]
        == mlb.transform(train["Labels"]).shape[1]
        == mlb.transform(valid["Labels"]).shape[1]
    )

    return train, valid, test


def perform_stratification(
    data_df: pd.DataFrame, test_size: float, valid_size: float, random_state: int
) -> tuple([pd.DataFrame, pd.DataFrame, pd.DataFrame]):
    """perform stratification on data frame with labels and paths
    Args:
        df (pd.DataFrame): data frame
        test_size (float): test size
        valid_size (float): validation size
        random_state (int): random state
    Returns:
        tuple([pd.DataFrame, pd.DataFrame, pd.DataFrame]): train, test, valid
    """

    mlb = MultiLabelBinarizer()
    mlb_transformed_labels = mlb.fit_transform(data_df["Labels"].values)

    train = None
    test = None
    valid = None
    ##Iterative Stratification
    msss = MultilabelStratifiedShuffleSplit(n_splits=10, test_size=test_size, random_state=random_state)
    for train_index, test_index in msss.split(data_df["Paths"].values, mlb_transformed_labels):
        train, test = (
            data_df.iloc[train_index],
            data_df.iloc[test_index],
        )
    train = train.reset_index(drop=True)
    test = test.reset_index(drop=True)
    test_valid_features = test["Paths"].values
    test_valid_labels = mlb.fit_transform(test["Labels"].values)
    n_msss = MultilabelStratifiedShuffleSplit(
        n_splits=10,
        test_size=test_size / (valid_size + test_size),
        random_state=random_state,
    )
    # print(test_valid_labels)
    for test_index, valid_index in n_msss.split(test_valid_features, test_valid_labels):
        try:
            test, valid = (
                test.iloc[test_index],
                test.iloc[valid_index],
            )
        except IndexError:
            continue

    train, test, valid = handle_no_classes_in_test_and_valid_after_stratification(train, test, valid, mlb)

    return train, test, valid


def create_path(target_directory: str):
    """create save path

    Args:
        target_directory (str): directory path
    """
    # check if directory exists
    target_exist = os.path.exists(target_directory)
    if not target_exist:
        # Create a new directory because it does not exist
        os.makedirs(target_directory)
