"""Utilities used during the dataset creation and data splitting."""
import os

import numpy as np
import pandas as pd
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
from sklearn.preprocessing import MultiLabelBinarizer


def handle_no_classes_in_test_and_valid_after_stratification(
    train: pd.DataFrame, test: pd.DataFrame, valid: pd.DataFrame, mlb: any
) -> tuple([pd.DataFrame, pd.DataFrame, pd.DataFrame]):
    """Handle cases where there is no labels in test and validation after stratification."""
    word_to_remove = None

    for i in [valid, test, train]:
        results = ~np.all(mlb.transform(i["Labels"]) == 0, axis=0)
        index_to_remove = (results == False).nonzero()  # pylint: disable=singleton-comparison

        if mlb.classes_[index_to_remove] != None:  # pylint: disable=singleton-comparison
            word_to_remove = mlb.classes_[index_to_remove]

    def search_for_word(x: list, word_to_remove: str) -> list:
        """Search for a word."""
        return [w for w in x if w != word_to_remove]

    valid["Labels"] = valid["Labels"].apply(lambda x: search_for_word(x, word_to_remove))
    test["Labels"] = test["Labels"].apply(lambda x: search_for_word(x, word_to_remove))
    train["Labels"] = train["Labels"].apply(lambda x: search_for_word(x, word_to_remove))

    # Check for equal size
    assert (
        mlb.transform(test["Labels"]).shape[1]
        == mlb.transform(train["Labels"]).shape[1]
        == mlb.transform(valid["Labels"]).shape[1]
    )

    return train, valid, test


def are_classes_same(test: pd.DataFrame, valid: pd.DataFrame, train: pd.DataFrame) -> bool:
    """Check if train test and valid contain the same amount of classes."""
    classes_test = set(test.Labels.explode().unique())
    classes_valid = set(valid.Labels.explode().unique())
    classes_train = set(train.Labels.explode().unique())
    is_same_classes = (classes_test == classes_valid) & (classes_train == classes_valid)
    return is_same_classes


def drop_rows_with_no_labels_(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows that have no labels."""
    index_of_rows_with_no_labels = df[df["Labels"].apply(len) == 0].index
    df.drop(index=index_of_rows_with_no_labels, inplace=True)
    return df


def replace_na_and_remove_empty_rows(
    train: pd.DataFrame, test: pd.DataFrame, valid: pd.DataFrame
) -> tuple([pd.DataFrame, pd.DataFrame, pd.DataFrame]):
    """Replace NA values and remove rows with no labels."""

    def replace_na_(label: str) -> str:
        """Replace float nan values."""
        if label == float("nan"):  # pylint: disable = W0177
            label = "N/A"
        return label

    train["Labels"] = train["Labels"].apply(replace_na_)
    test["Labels"] = test["Labels"].apply(replace_na_)
    valid["Labels"] = valid["Labels"].apply(replace_na_)
    train = drop_rows_with_no_labels_(train)
    test = drop_rows_with_no_labels_(test)
    valid = drop_rows_with_no_labels_(valid)
    return train, test, valid


def perform_stratification(
    data_df: pd.DataFrame, test_size: float, valid_size: float, random_state: int
) -> tuple([pd.DataFrame, pd.DataFrame, pd.DataFrame]):
    """Perform stratification on data frame with labels and paths.

    Args:
        data_df (pd.DataFrame): data frame
        test_size (float): test size
        valid_size (float): validation size
        random_state (int): random state

    Returns:
        tuple([pd.DataFrame, pd.DataFrame, pd.DataFrame]): train, test, valid
    """
    # pylint: disable=R0914
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

    train, test, valid = replace_na_and_remove_empty_rows(train, test, valid)

    assert are_classes_same(
        test, valid, train
    ), """ There aren't the same number of classes in train test and validation"""

    return train, test, valid


def create_path(target_directory: str) -> None:
    """Create save paths.

    Args:
        target_directory (str): directory path
    """
    # check if directory exists
    target_exist = os.path.exists(target_directory)
    if not target_exist:
        # Create a new directory because it does not exist
        os.makedirs(target_directory)
