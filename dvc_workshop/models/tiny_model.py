"""Model to instantiate and train a small Image classification model."""
import tensorflow as tf
from keras import Sequential
from keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPooling2D
from keras_preprocessing.image import ImageDataGenerator

from dvc_workshop.params import TrainingParams


class TinyModel(tf.keras.Model):
    """Small convolutional model for image classification."""

    def __init__(
        self,
        image_height: int,
        image_width: int,
        channels: int,
        activation: str,
        class_indices: dict,
    ):  # pylint: disable=R0913
        super().__init__()
        # initiate model classes
        self.class_indices = class_indices

        self.model = Sequential()
        # input layer
        self.model.add(
            Conv2D(8, kernel_size=(3, 3), activation="relu", input_shape=(image_height, image_width, channels))
        )
        self.model.add(Conv2D(16, kernel_size=(3, 3), activation="relu"))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Dropout(0.25))

        self.model.add(Conv2D(32, kernel_size=(3, 3), activation="relu"))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Conv2D(32, kernel_size=(3, 3), activation="relu"))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Dropout(0.25))

        self.model.add(Flatten())
        self.model.add(Dense(64, activation="relu"))
        self.model.add(Dropout(0.5))
        # output layer
        self.model.add(Dense(len(class_indices), activation=activation))
        print(f"Current model has {self.model.count_params()} params.")

    def call(self, inputs: tf.Tensor) -> tf.Tensor:  # pylint: disable=W0221
        """Override call function, forward pass.

        Args:
            inputs (tf.Tensor): model input
        Returns:
            tf.Tensor: model multilabel prediction
        """
        return self.model(inputs)

    def set_trainable(self, trainable: bool = True) -> None:
        """Freeze/unfreeze feature extractor layers.

        Args:
            trainable (bool, optional):  Defaults to True.
        """
        self.base_model.trainable = trainable

    def train(self, train: ImageDataGenerator, val: ImageDataGenerator, trainingparams: TrainingParams) -> dict:
        """Train the classification head, then finetune the model."""
        # pylint: disable=R0801
        # load the model
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(),
            loss=tf.keras.losses.BinaryCrossentropy(),
            metrics=[
                tf.keras.metrics.Precision(),
                tf.keras.metrics.Recall(),
                tf.keras.metrics.BinaryAccuracy(),
            ],
        )

        # train the model
        history_training = self.model.fit(
            train,
            validation_data=val,
            epochs=trainingparams.TRAINING_EPOCHS,
            callbacks=[tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)],
        )

        return {"history_training": history_training, "history_finetuning": None}
