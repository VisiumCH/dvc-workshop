import tensorflow as tf


class EfficentNet(tf.keras.Model):
    def __init__(self, image_height, image_width, channels, pooling, include_top, activation, class_indices):
        super(EfficentNet, self).__init__()
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

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        """override call function, forward pass

        Args:
            inputs (tf.Tensor): model input

        Returns:
            tf.Tensor: model multilabel prediction
        """
        x = self.base_model(inputs)
        return self.output_layer(x)

    def set_trainable(self, trainable=True):
        """freeze/unfreeze feature extractor layers

        Args:
            trainable (bool, optional):  Defaults to True.
        """
        self.base_model.trainable = trainable
