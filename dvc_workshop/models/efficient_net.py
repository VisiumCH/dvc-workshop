"""Defines how to instantiate and train the efficient net models."""
import tensorflow as tf
from keras_preprocessing.image import ImageDataGenerator

from dvc_workshop.params import TrainingParams


class EfficientNet(tf.keras.Model):
    """Class that can instantiate and train Efficient net V2."""

    def __init__(
        self,
        image_height: int,
        image_width: int,
        channels: int,
        pooling: bool,
        include_top: bool,
        activation: str,
        class_indices: dict,
    ):  # pylint: disable=R0913
        super().__init__()
        # initiate model classes
        self.class_indices = class_indices
        # load efficinet model from keras api
        self.base_model = tf.keras.applications.efficientnet_v2.EfficientNetV2B0(
            include_top=include_top,
            pooling=pooling,
            input_shape=(image_height, image_width, channels),
        )
        # freeze lower layers
        self.set_trainable(False)
        # declare trainable neck layer (predictor)
        self.output_layer = tf.keras.layers.Dense(len(class_indices), activation=activation)

    def call(self, inputs: tf.Tensor) -> tf.Tensor:  # pylint: disable=W0221
        """Override call function, forward pass.

        Args:
            inputs (tf.Tensor): model input
        Returns:
            tf.Tensor: model multilabel prediction
        """
        x = self.base_model(inputs)
        return self.output_layer(x)

    def set_trainable(self, trainable: bool = True) -> None:
        """Freeze/unfreeze feature extractor layers.

        Args:
            trainable (bool, optional):  Defaults to True.
        """
        self.base_model.trainable = trainable

    def train(self, train: ImageDataGenerator, val: ImageDataGenerator, trainingparams: TrainingParams) -> dict:
        """Train the classification head, then finetune the model."""
        return train_efficient_net(self, train, val, trainingparams)


class EfficientNetSmall(tf.keras.Model):
    """Class that can instantiate and train a smaller version of Efficient net V2."""

    def __init__(
        self,
        image_height: int,
        image_width: int,
        channels: int,
        pooling: bool,
        include_top: bool,
        activation: str,
        class_indices: dict,
    ):  # pylint: disable=R0913
        super().__init__()
        # initiate model classes
        self.class_indices = class_indices
        # load efficinet model from keras api
        self.base_model = tf.keras.applications.efficientnet_v2.EfficientNetV2S(
            include_top=include_top,
            pooling=pooling,
            input_shape=(image_height, image_width, channels),
        )
        # freeze lower layers
        self.set_trainable(False)
        # declare trainable neck layer (predictor)
        self.output_layer = tf.keras.layers.Dense(len(class_indices), activation=activation)

    def call(self, inputs: tf.Tensor) -> tf.Tensor:  # pylint: disable=W0221
        """Override call function, forward pass.

        Args:
            inputs (tf.Tensor): model input
        Returns:
            tf.Tensor: model multilabel prediction
        """
        x = self.base_model(inputs)
        return self.output_layer(x)

    def set_trainable(self, trainable: bool = True) -> None:
        """Freeze/unfreeze feature extractor layers.

        Args:
            trainable (bool, optional):  Defaults to True.
        """
        self.base_model.trainable = trainable

    def train(self, train: ImageDataGenerator, val: ImageDataGenerator, trainingparams: TrainingParams) -> dict:
        """Train the classification head, then finetune the model."""
        return train_efficient_net(self, train, val, trainingparams)


def train_efficient_net(
    model: tf.keras.Model, train: ImageDataGenerator, val: ImageDataGenerator, trainingparams: TrainingParams
) -> dict:
    """Train the classification head, then finetune the model."""
    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=[
            tf.keras.metrics.Precision(),
            tf.keras.metrics.Recall(),
            tf.keras.metrics.BinaryAccuracy(),
        ],
    )

    # train the output layer
    history_training = model.fit(
        train,
        validation_data=val,
        epochs=trainingparams.TRAINING_EPOCHS,
        callbacks=[tf.keras.callbacks.EarlyStopping(patience=trainingparams.TUNING_LR, restore_best_weights=True)],
    )

    # fine-tune the model
    model.set_trainable()

    model.compile(
        optimizer=tf.keras.optimizers.Adam(trainingparams.TUNING_LR),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=[
            tf.keras.metrics.Precision(),
            tf.keras.metrics.Recall(),
            tf.keras.metrics.BinaryAccuracy(),
        ],
    )

    history_finetuning = model.fit(
        train,
        validation_data=val,
        epochs=trainingparams.TUNING_EPOCHS,
        callbacks=[tf.keras.callbacks.EarlyStopping(patience=trainingparams.PATIENCE, restore_best_weights=True)],
    )

    return {"history_training": history_training, "history_finetuning": history_finetuning}
