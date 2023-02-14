"""Download the MNIST dataset, save the images to png files, save the labels and paths to a dataframe."""
import os
from pathlib import Path

import keras
import numpy as np
import pandas as pd
import tensorflow as tf
import typer
from PIL import Image


def download_mnist_keras() -> None:
    tf.keras.datasets.mnist.load_data(path="mnist.npz")
    (x_train, y_train), (_, _) = keras.datasets.mnist.load_data()
    return x_train, y_train


def save_mnist_to_png(x_train: np.array, y_train: np.array, output_image_path: Path) -> pd.DataFrame:
    """Convert MNIST from numpy arrays to png pictures and prepare a dataframe with the paths and labels."""
    # initialize empty pandas dataframe
    mnist_df = pd.DataFrame.from_dict({"Labels": [], "Paths": []})

    # TODO: take random sample instead of first 1000
    for i in range(x_train[:1000, :, :].shape[0]):
        # get the image
        image = x_train[i, :, :]
        # get the label
        label = y_train[i]
        # create the file path
        file_path = output_image_path / f"image_{str(i)}.png"
        # conver the image from numpy array to .png using PIL
        im = Image.fromarray(image)
        im.save(file_path)
        # add a row to the dataframe
        mnist_df = pd.concat([mnist_df, pd.DataFrame.from_dict({"Labels": [int(label)], "Paths": [str(file_path)]})])
    mnist_df.reset_index(drop=True, inplace=True)

    return mnist_df


def main(output_image_path: Path = typer.Option(...), output_df_path: Path = typer.Option(...)) -> None:
    """Download the MNIST dataset, save a sample to png files, save the labels and paths in a dataframe."""
    os.makedirs(output_image_path, exist_ok=True)
    os.makedirs(output_df_path, exist_ok=True)

    x_train, y_train = download_mnist_keras()
    mnist_df = save_mnist_to_png(x_train, y_train, output_image_path)

    mnist_df.to_csv(output_df_path / "train.csv")


if __name__ == "__main__":
    typer.run(main)
